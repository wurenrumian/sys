#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo5: CPU 喂不饱 GPU —— 数据流水线的重叠、木桶与抖动

对应表格: "数据处理流程与模型训练脱节, ETL 环节耗时长, CPU-GPU 协同效率低下" /
          "通过对数据湖、缓存、分布式计算和 GPU IO 的协同优化"

一次训练 step 要做两件事:
  CPU 侧: 读样本 -> protobuf 反序列化 -> 特征拼接/哈希 -> padding -> pin memory
  GPU 侧: 前向 + 反向 + 优化器
这两件事**可以重叠**(GPU 算第 i 批时, CPU 准备第 i+1 批). 能不能真的重叠起来,
决定了你买的 GPU 有多少时间在干活.

本 demo 用一个精确的流水线模型回答三个问题:
  1. 加 dataloader worker, 加到几个就没用了?
  2. 平均 ETL 时间明明比 GPU 快, 为什么 GPU 还是有气泡?
  3. 手里有一份优化预算, 该花在"CPU 更快"还是"worker 更多"还是"队列更深"?
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.params import env_int, env_float, banner  # noqa: E402

np.seterr(all="ignore")

N_STEPS = env_int("N_STEPS", 3000, "模拟多少个训练 step")
ETL_MS = env_float("ETL_MS", 90.0, "单批样本的 CPU 侧处理耗时均值(ms)")
GPU_MS = env_float("GPU_MS", 100.0, "单批的 GPU 计算耗时(ms)")
ETL_CV = env_float("ETL_CV", 0.6, "ETL 耗时的变异系数(抖动有多大)")
WORKERS = env_int("WORKERS", 4, "dataloader worker 数")
QUEUE = env_int("QUEUE", 2, "prefetch 队列槽位数(1=完全不重叠)")


def gen_times(seed=0):
    """生成每个 step 的 CPU/GPU 耗时。

    CPU 侧用对数正态: 真实 ETL 的耗时是长尾的 —— 大多数批很快, 偶尔撞上一个
    超长序列的批、一次 page cache miss、一次 GC, 就会慢几倍。
    GPU 侧近似恒定(同样的 kernel、同样的 shape), 只给一点点抖动。
    """
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(np.log(1 + ETL_CV ** 2))
    mu = np.log(ETL_MS) - sigma ** 2 / 2          # 让均值正好等于 ETL_MS
    etl = rng.lognormal(mu, sigma, N_STEPS)
    gpu = GPU_MS * (1 + 0.02 * rng.standard_normal(N_STEPS))
    return etl, np.maximum(gpu, 0.1)


def simulate(etl, gpu, workers, queue):
    """流水线模型。

    第 i 批的生产必须同时满足两个条件才能开始:
      (a) 有空闲的 worker;
      (b) prefetch 队列有空槽 —— 队列有 queue 个槽位, 所以要等第 i-queue 批被消费掉。
    GPU 按顺序消费, 第 i 批必须等第 i-1 批算完、且自己已经被准备好。

    queue=1 表示只有一个槽位, 于是"生产一批 -> GPU 算完 -> 再生产下一批", 完全不重叠。
    queue=2 就是经典的双缓冲。
    """
    n = len(etl)
    queue = max(1, queue)
    worker_free = np.zeros(max(1, workers))
    prod_done = np.zeros(n)
    cons_done = np.zeros(n)
    gpu_free = 0.0
    gpu_busy = 0.0

    for i in range(n):
        w = int(np.argmin(worker_free))
        # 队列有空位的时刻: 第 i-queue 批被消费完的时刻
        slot_free = cons_done[i - queue] if i - queue >= 0 else 0.0
        start = max(worker_free[w], slot_free)
        finish = start + etl[i]
        worker_free[w] = finish
        prod_done[i] = finish

        g_start = max(gpu_free, prod_done[i])
        gpu_free = g_start + gpu[i]
        gpu_busy += gpu[i]
        cons_done[i] = gpu_free

    total = cons_done[-1]
    return {
        "total_s": total / 1000.0,
        "util": gpu_busy / total,          # GPU 有多少时间在真干活
        "step_ms": total / n,
        "ideal_ms": gpu.mean(),            # GPU 从不等数据时的理想 step 时间
        "bubble_ms": total / n - gpu.mean(),
    }


def row(tag, r, width=22):
    print(f"  {tag:<{width}}{r['step_ms']:>10.1f}ms{r['ideal_ms']:>11.1f}ms"
          f"{r['bubble_ms']:>11.1f}ms{r['util']:>11.1%}")


def head(first="配置", width=22):
    print(f"  {first:<{width}}{'实际step':>12}{'理想step':>13}{'GPU气泡':>13}{'GPU利用率':>13}")
    print("  " + "-" * (width + 46))


def main():
    banner(84)
    etl, gpu = gen_times()
    print("=" * 84)
    print(f"训练流水线: {N_STEPS:,} 个 step | CPU 侧 ETL 均值 {etl.mean():.1f}ms "
          f"(p99 {np.percentile(etl,99):.0f}ms, 变异系数 {ETL_CV:g}) | GPU 计算 {gpu.mean():.1f}ms")
    print(f"关键比值: ETL/GPU = {etl.mean()/gpu.mean():.2f} —— "
          f"单看均值, {int(np.ceil(etl.mean()/gpu.mean()))} 个 worker 就该喂饱 GPU 了")
    print("=" * 84)

    # ---------------------------------------------------------- 1. 重叠的价值
    print("\n[1] 重叠与否: 同样的工作量, 只是安排方式不同")
    head()
    row("串行(1 槽位, 不重叠)", simulate(etl, gpu, 1, 1))
    row("双缓冲(2 槽位)", simulate(etl, gpu, 1, 2))
    row(f"{WORKERS} worker + {QUEUE} 槽位", simulate(etl, gpu, WORKERS, QUEUE))
    ser = simulate(etl, gpu, 1, 1)
    best = simulate(etl, gpu, WORKERS, QUEUE)
    print(f"  -> 串行时 step = ETL + GPU = {ser['step_ms']:.0f}ms, "
          f"GPU 有 {1-ser['util']:.0%} 的时间在发呆.")
    print(f"     重叠之后 step 逼近 max(ETL, GPU) = {max(etl.mean(), gpu.mean()):.0f}ms, "
          f"端到端快了 {ser['step_ms']/best['step_ms']:.2f}x.")
    print("     注意: **没有任何一行代码变快**, 快的只是安排 —— 这是流水线的全部价值.")

    # ---------------------------------------------------------- 2. worker 扫描
    # 把 ETL 调重到明显慢于 GPU, 否则 1 个 worker 就够了, 看不到拐点
    heavy = etl * 3
    print(f"\n[2] 加 worker 加到几个才够? (这里把 ETL 调重到 {heavy.mean():.0f}ms = "
          f"GPU 的 {heavy.mean()/gpu.mean():.1f} 倍, 队列 8 槽位)")
    head("worker 数")
    prev = None
    for w in [1, 2, 3, 4, 6, 8, 12, 16]:
        r = simulate(heavy, gpu, w, 8)
        gain = "" if prev is None else f"   (比上一档快 {prev['step_ms']/r['step_ms']-1:+.1%})"
        print(f"  {w:<22}{r['step_ms']:>10.1f}ms{r['ideal_ms']:>11.1f}ms"
              f"{r['bubble_ms']:>11.1f}ms{r['util']:>11.1%}{gain}")
        prev = r
    print("  -> 典型的**木桶效应**: worker 加到某个点之后, 瓶颈从 CPU 换成了 GPU,")
    print("     再加 worker 收益归零 —— 但它们仍然在占内存、占 CPU 核、抢总线带宽.")
    print("     '把 num_workers 调大一点总没坏处'是错的: 过了拐点纯粹是浪费,")
    print("     而且每个 worker 都持有一份预取数据, 内存占用是线性增长的.")
    q2 = simulate(heavy, gpu, 16, 2)
    q8 = simulate(heavy, gpu, 16, 8)
    print(f"\n     顺带一个容易踩的坑: 队列槽位数是 worker 并行度的**硬上限**.")
    print(f"     同样 16 个 worker, 队列只给 2 个槽位时 step = {q2['step_ms']:.0f}ms,")
    print(f"     给 8 个槽位时 = {q8['step_ms']:.0f}ms. 因为在途的批数不可能超过槽位数,")
    print("     多出来的 worker 全程在阻塞等空槽 —— 只调 num_workers 不调 prefetch,")
    print("     是最常见的'调了参数却没效果'的原因.")

    # ---------------------------------------------------------- 3. 抖动
    print("\n[3] 最反直觉的一点: 平均值够快, 不代表 GPU 不挨饿")
    print(f"  下面固定 {WORKERS} 个 worker, 只改 ETL 的抖动程度(均值始终 {ETL_MS:g}ms, "
          f"永远快于 GPU 的 {GPU_MS:g}ms)")
    print(f"\n  {'变异系数':<12}{'ETL p99':>10}{'队列槽位':>10}{'GPU利用率':>12}{'实际step':>12}")
    print("  " + "-" * 58)
    for cv in [0.0, 0.3, 0.6, 1.0, 1.5]:
        sigma = np.sqrt(np.log(1 + max(cv, 1e-9) ** 2))
        mu = np.log(ETL_MS) - sigma ** 2 / 2
        e = np.random.default_rng(1).lognormal(mu, sigma, N_STEPS) if cv > 0 \
            else np.full(N_STEPS, ETL_MS)
        for q in [2, 8]:
            r = simulate(e, gpu, WORKERS, q)
            print(f"  {cv if q == 2 else '':<12}{np.percentile(e,99):>9.0f}ms"
                  f"{q:>10}{r['util']:>12.1%}{r['step_ms']:>10.1f}ms")
    print("\n  -> 抖动为 0 时, GPU 利用率接近 100%; 抖动一大, 即使**均值完全没变**,")
    print("     GPU 利用率就开始掉 —— 因为流水线是同步的: 一个慢批到不了,")
    print("     GPU 就得干等, 而它前面跑得再快也**存不下来**. 快的时候省下的时间,")
    print("     并不能拿去补慢的时候的坑 —— 除非有队列把它存起来.")
    print("     这就是 prefetch 队列的真正作用: 它不是让 CPU 更快, 而是给流水线加**缓冲**,")
    print("     把'瞬时的慢'摊平成'平均的快'. 看每组的两行: 槽位从 2 加到 8,")
    print("     没多花一分钱 CPU, 利用率却明显回升.")

    # ---------------------------------------------------------- 4. 队列深度
    print(f"\n[4] 队列槽位扫描 ({WORKERS} worker, 变异系数 {ETL_CV:g})")
    head("队列槽位")
    for q in [1, 2, 3, 4, 8, 16, 32]:
        row(str(q), simulate(etl, gpu, WORKERS, q))
    print("  -> 收益很快饱和: 队列的作用是吸收抖动, 抖动被吸收完就没用了.")
    print("     代价是内存(每个槽位都持有一批完整的样本, 推荐场景一批可能几百 MB)")
    print("     和**数据新鲜度**(队列越深, GPU 吃到的样本越旧 —— 在线学习场景要命).")

    # ---------------------------------------------------------- 5. 花钱的地方
    print("\n" + "=" * 84)
    print("[5] 一份优化预算该花在哪? (基线 = "
          f"{WORKERS} worker / {QUEUE} 槽位 / ETL {ETL_MS:g}ms)")
    base = simulate(etl, gpu, WORKERS, QUEUE)
    plans = [
        ("把 ETL 做快 2x (改格式/GPU解码)", simulate(etl / 2, gpu, WORKERS, QUEUE)),
        ("worker 数翻倍 (加 CPU 核)", simulate(etl, gpu, WORKERS * 2, QUEUE)),
        ("队列槽位翻倍 (纯改配置)", simulate(etl, gpu, WORKERS, QUEUE * 2)),
        ("消除 ETL 抖动 (方差归零)", simulate(np.full(N_STEPS, etl.mean()), gpu, WORKERS, QUEUE)),
        ("GPU 换成快 2x 的卡", simulate(etl, gpu / 2, WORKERS, QUEUE)),
    ]
    best_cpu_side = min(r["step_ms"] for _, r in plans[:4])
    print(f"\n  {'方案':<32}{'step':>10}{'相对基线':>12}{'GPU利用率':>12}")
    print("  " + "-" * 66)
    print(f"  {'基线':<32}{base['step_ms']:>8.1f}ms{'—':>12}{base['util']:>12.1%}")
    for name, r in plans:
        print(f"  {name:<30}{r['step_ms']:>8.1f}ms"
              f"{base['step_ms']/r['step_ms']:>11.2f}x{r['util']:>12.1%}")
    print("\n  -> 这张表要看三件事:")
    print("     1. **优化非瓶颈的收益精确等于 0**: 'worker 数翻倍'是 1.00x. 一分不赚.")
    print("        基线的 4 个 worker 早就够用了, 加到 8 个只是多占 4 份内存.")
    print("     2. **收益有天花板**: 基线 GPU 利用率已经 "
          f"{base['util']:.0%}, 所以任何 CPU 侧的优化")
    print(f"        (ETL 做快 2x / 加深队列 / 消抖动)最多也只能拿到 "
          f"{base['step_ms']/best_cpu_side:.2f}x —— 它们都在抢同一块 "
          f"{base['bubble_ms']:.0f}ms 的气泡.")
    print("        这三件事**不可叠加**, 做了其中最便宜的一个, 另外两个就没价值了.")
    print("     3. **优化瓶颈只是把瓶颈推给下一个环节**: 换快 2x 的卡, 端到端只快 "
          f"{base['step_ms']/plans[-1][1]['step_ms']:.2f}x 而不是 2x,")
    print(f"        而且 GPU 利用率反而掉到 {plans[-1][1]['util']:.0%} —— 瓶颈翻转到了 CPU 侧.")
    print("        真实世界里的性能优化永远是这样: 一次只能拔掉一个瓶颈,")
    print("        拔掉之后必须**重新测量**, 因为整张表已经变了.")
    print("\n     所以做 infra 的第一动作永远是先测出瓶颈在哪, 而不是先优化 ——")
    print("     而且要在每一轮优化之后重测一次.")

    print("\n[对应到表格的课题]")
    print("  * 'GPU/DPU 等异构计算资源利用率不足' —— 上表第 1 行就是这条: 把解压、")
    print("    解析、特征变换从 CPU 挪到 GPU/DPU 上, 直接压缩 ETL 时间.")
    print("    (GPUDirect Storage 让数据从 NVMe 直达显存, 绕过 CPU 和系统内存;")
    print("     nvCOMP 在 GPU 上解压; DALI 在 GPU 上做预处理)")
    print("  * '数据处理流程与模型训练脱节' —— 脱节的代价就是上面的串行那一行.")
    print("    训练和 ETL 是两套系统、两个团队时, 没人有动力去做重叠.")
    print("  * 推荐场景的特殊难点: 样本大(KB 级)、特征多(几百个)、要做哈希和 padding,")
    print("    所以 ETL/GPU 的比值天然比 CV/NLP 高得多 —— 别的领域不用操心的事,")
    print("    在推荐里是头号瓶颈.")


if __name__ == "__main__":
    main()
