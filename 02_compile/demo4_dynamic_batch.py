#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo4: 动态批处理 —— 吞吐与 P99 的取舍, 以及为什么必须"自适应"

对应表格: "构建基于推荐请求特征(如用户活跃度、请求峰值)的动态批处理阈值自适应模型,
          实时调整批处理尺寸"

批处理是提升 GPU 利用率的第一手段, 但**攒批要等, 等就是延迟**.
本 demo 用离散事件模拟, 在"正弦波动 + 突发尖峰"的真实流量下对比三种策略.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.params import env_int, env_float, banner  # noqa: E402

np.seterr(all="ignore")

SIM_MS = env_int("SIM_MS", 60_000, "模拟时长(ms)")
BASE_QPS = env_int("BASE_QPS", 800, "平均 QPS")
SPIKE_X = env_float("SPIKE_X", 3.0, "尖峰倍数")
GPU_FIXED_MS = env_float("GPU_FIXED_MS", 2.0, "单批的固定启动开销(ms)")
GPU_PER_REQ_MS = env_float("GPU_PER_REQ_MS", 0.012, "每条请求的边际计算耗时(ms)")
SPIKE_AT = (SIM_MS // 2, SIM_MS // 2 + SIM_MS // 15)   # 中段一个尖峰
EPS = 1e-9                    # 浮点容差, 防止离散事件模拟里时间无法推进


# ---------------------------------------------------------------- GPU 成本模型
def batch_time_ms(bs):
    """一次 batch 推理的耗时.

    真实 GPU kernel 的典型形态: 固定启动开销 + 随 batch 线性增长的计算.
    正是这个固定开销让"攒批"有意义 —— 批越大, 每条请求摊到的开销越小.
    """
    return GPU_FIXED_MS + GPU_PER_REQ_MS * bs


def gen_arrivals(seed=0):
    """泊松到达 + 正弦日内波动 + 一段突发尖峰."""
    rng = np.random.default_rng(seed)
    times = []
    t = 0.0
    while t < SIM_MS:
        qps = BASE_QPS * (1 + 0.4 * np.sin(2 * np.pi * t / 20_000))
        if SPIKE_AT[0] <= t < SPIKE_AT[1]:
            qps *= SPIKE_X
        t += rng.exponential(1000.0 / qps)
        if t < SIM_MS:
            times.append(t)
    return np.array(times)


# ---------------------------------------------------------------- 三种策略
def simulate(arrivals, policy, max_bs=256, fixed_bs=64, timeout_ms=5.0):
    """单 GPU worker 的离散事件模拟.

    policy:
      'fixed_bs'      —— 攒够 fixed_bs 条才发车 (吞吐优先, 低谷会饿死请求)
      'fixed_timeout' —— 攒够 max_bs 或等满 timeout_ms 就发车 (延迟有上界)
      'adaptive'      —— 按队列积压动态调整批大小和等待时间
    """
    n = len(arrivals)
    latencies = np.empty(n)
    q = []                    # 队列中请求的到达时间(索引)
    i = 0                     # 下一个未入队的请求
    now = 0.0
    gpu_busy_ms = 0.0
    batch_sizes = []

    while i < n or q:
        # 把当前时刻之前到达的请求全部入队
        while i < n and arrivals[i] <= now:
            q.append(i)
            i += 1

        if not q:
            now = arrivals[i]          # 空闲, 直接跳到下一个到达
            continue

        oldest_wait = now - arrivals[q[0]]

        # ---- 决定这一批发多大车, 或者继续等 ----
        if policy == "fixed_bs":
            if len(q) < fixed_bs and i < n:
                now = arrivals[i]      # 没攒够就继续等下一个请求
                continue
            bs = min(len(q), fixed_bs)

        elif policy == "fixed_timeout":
            # EPS 不可省: oldest_wait 是 now - arrivals[q[0]] 算出来的, 浮点误差会让它
            # 略小于 timeout_ms, 而 arrivals[q[0]] + timeout_ms 又恰好等于 now,
            # 于是时间无法推进 -> 死循环. 离散事件模拟里这是个经典坑.
            if len(q) < max_bs and oldest_wait < timeout_ms - EPS and i < n:
                # 等到 "攒满" 或 "超时" 两者中先发生的那个
                now = min(arrivals[i], arrivals[q[0]] + timeout_ms)
                continue
            bs = min(len(q), max_bs)

        elif policy == "adaptive":
            # 核心思想: 积压越多 -> 说明流量高, 越该用大批(摊薄固定开销、提高吞吐);
            #          积压越少 -> 说明流量低, 越该早发车(此时 GPU 反正闲着, 别让请求干等).
            backlog = len(q)
            if backlog >= max_bs:
                bs = max_bs                      # 高峰: 直接满批
            else:
                # 动态超时: 积压少时几乎不等, 积压多时愿意多等一点凑批
                dyn_timeout = 0.5 + 4.0 * (backlog / max_bs)
                if oldest_wait < dyn_timeout - EPS and i < n:
                    now = min(arrivals[i], arrivals[q[0]] + dyn_timeout)
                    continue
                bs = backlog
        else:
            raise ValueError(policy)

        # ---- 发车 ----
        dur = batch_time_ms(bs)
        batch_sizes.append(bs)
        gpu_busy_ms += dur
        finish = now + dur
        for idx in q[:bs]:
            latencies[idx] = finish - arrivals[idx]
        q = q[bs:]
        now = finish

    return {
        "p50": np.percentile(latencies, 50),
        "p99": np.percentile(latencies, 99),
        "p999": np.percentile(latencies, 99.9),
        "max": latencies.max(),
        "avg_bs": np.mean(batch_sizes),
        "n_batch": len(batch_sizes),
        # 分母必须是实际跑完的时长, 不能写死 SIM_MS —— 队列积压时最后一批会
        # 结束在 SIM_MS 之后, 用固定分母会算出 >100% 的利用率.
        "gpu_util": gpu_busy_ms / max(SIM_MS, now),
        "throughput": n / (SIM_MS / 1000),
        "latencies": latencies,
    }


def main():
    banner(84)
    arrivals = gen_arrivals()
    print("=" * 84)
    print(f"动态批处理演示: {len(arrivals):,} 个请求 / {SIM_MS/1000:.0f} 秒, "
          f"均值 {BASE_QPS} QPS")
    print(f"流量形态: 正弦波动(±40%) + 第 {SPIKE_AT[0]/1000:.0f}~{SPIKE_AT[1]/1000:.0f} 秒 "
          f"{SPIKE_X:g} 倍尖峰")
    print(f"GPU 成本模型: 单批耗时 = {GPU_FIXED_MS:g}ms 固定开销 + "
          f"{GPU_PER_REQ_MS:g}ms x batch_size")
    print("=" * 84)

    configs = [
        ("固定批 bs=16", dict(policy="fixed_bs", fixed_bs=16)),
        ("固定批 bs=64", dict(policy="fixed_bs", fixed_bs=64)),
        ("固定批 bs=256", dict(policy="fixed_bs", fixed_bs=256)),
        ("固定超时 5ms", dict(policy="fixed_timeout", timeout_ms=5.0)),
        ("固定超时 20ms", dict(policy="fixed_timeout", timeout_ms=20.0)),
        ("自适应", dict(policy="adaptive")),
    ]

    print(f"\n{'策略':<18}{'P50':>9}{'P99':>9}{'P999':>10}{'最大':>10}"
          f"{'平均批':>9}{'GPU利用率':>11}")
    print("-" * 84)
    results = {}
    for name, kw in configs:
        r = simulate(arrivals, **kw)
        results[name] = r
        print(f"{name:<18}{r['p50']:>8.1f}ms{r['p99']:>8.1f}ms{r['p999']:>9.1f}ms"
              f"{r['max']:>9.1f}ms{r['avg_bs']:>9.1f}{r['gpu_util']:>10.1%}")

    # ------------------------------------------------------------ 解读
    print("\n" + "-" * 84)
    print("[固定批的问题] 它把'什么时候发车'完全交给了流量:")
    r16, r256 = results["固定批 bs=16"], results["固定批 bs=256"]
    print(f"  bs=16 : 批小 -> 发车频繁 -> 固定开销占比高 -> GPU 利用率 {r16['gpu_util']:.1%}")
    print(f"  bs=256: 批大 -> 低谷期要等很久才凑满 -> P999 高达 {r256['p999']:.1f}ms")
    print("  固定批的致命问题: 流量低谷时请求会被**饿死在队列里**等后来者凑批,")
    print("  而这恰恰是 GPU 最闲、本该最快响应的时候. 完全是反的.")

    print("\n[固定超时的改进] 给等待加了一个硬上界, 延迟可控了:")
    rt5, rt20 = results["固定超时 5ms"], results["固定超时 20ms"]
    print(f"  5ms 超时 : P999 {rt5['p999']:.1f}ms, 平均批 {rt5['avg_bs']:.1f}, "
          f"GPU 利用率 {rt5['gpu_util']:.1%}")
    print(f"  20ms 超时: P999 {rt20['p999']:.1f}ms, 平均批 {rt20['avg_bs']:.1f}, "
          f"GPU 利用率 {rt20['gpu_util']:.1%}")
    print("  但超时值本身是个**静态**参数: 调小则高峰期批太小、吞吐不够;")
    print("  调大则低谷期白白多等. 一个值没法同时服务两种流量状态.")

    ra = results["自适应"]
    print("\n[自适应] 让等待时间随队列积压走:")
    print(f"  P50 {ra['p50']:.1f}ms / P99 {ra['p99']:.1f}ms / P999 {ra['p999']:.1f}ms, "
          f"平均批 {ra['avg_bs']:.1f}, GPU 利用率 {ra['gpu_util']:.1%}")
    print("  规则很简单: 积压少 -> 几乎不等就发车(反正 GPU 闲着, 何必让人干等);")
    print("             积压多 -> 愿意多等一点凑大批(此时凑批几乎不花时间, 白赚吞吐).")
    print("  它在低谷期像'小超时', 在高峰期像'大批量', 两种状态自动切换.")

    # ------------------------------------------------------------ 尖峰段
    print("\n" + "-" * 84)
    spike = (arrivals >= SPIKE_AT[0]) & (arrivals < SPIKE_AT[1])
    trough = ~spike
    print(f"[分段看] 尖峰期({spike.sum():,} 条) vs 非尖峰期({trough.sum():,} 条) 分别的表现:")
    print(f"\n  {'策略':<18}{'低谷P99':>10}{'尖峰P99':>10}{'低谷P999':>11}{'尖峰P999':>11}")
    print("  " + "-" * 62)
    for name, _ in configs:
        lat = results[name]["latencies"]
        print(f"  {name:<18}"
              f"{np.percentile(lat[trough], 99):>9.1f}ms"
              f"{np.percentile(lat[spike], 99):>9.1f}ms"
              f"{np.percentile(lat[trough], 99.9):>10.1f}ms"
              f"{np.percentile(lat[spike], 99.9):>10.1f}ms")
    print("\n  -> 注意固定批策略的**低谷 P99 反而比尖峰更差** —— 这是反直觉但正确的:")
    print("     尖峰期请求多, 一眨眼就凑满一批发车了; 低谷期请求稀疏, 先到的请求")
    print("     要干等后来者凑够批. 也就是说, 系统最闲的时候, 用户等得最久.")
    print("     固定超时策略修掉了这一点, 但延迟被超时值死死锁住(5ms 配置就是 ~7ms).")
    print("     自适应的价值: 低谷期不等(P99 4.0ms, 优于任何静态配置),")
    print("     高峰期又能自动放大批量 —— **两段都不吃亏**.")

    print("\n" + "=" * 84)
    print("[对应到表格的课题]")
    print("  '构建基于推荐请求特征(如用户活跃度、请求峰值)的动态批处理阈值自适应模型'")
    print("  本 demo 的自适应规则只用了'队列积压'这一个信号, 已经明显优于静态策略.")
    print("  真实系统还能用上更多信号, 这正是课题的空间:")
    print("    - 请求特征: 用户序列长度(决定这条请求的实际计算量, 不是所有请求都等价)")
    print("    - 时间特征: 日内周期、活动日历(可以**预测**尖峰而不是被动响应)")
    print("    - SLA 反馈: 实时 P99 距离 SLA 还有多少余量, 据此决定要不要更激进地凑批")
    print("    - 显存约束: 大批需要更多 KV Cache, 要和 demo3 的显存分配协同决策")
    print("  更进一步: 这本质上是个在'吞吐-延迟'帕累托前沿上的实时控制问题,")
    print("  可以上强化学习/模型预测控制(MPC), 而不只是手写规则.")


if __name__ == "__main__":
    main()
