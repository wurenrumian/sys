#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo1: 一次训练 step 的时间预算 —— 把三个方向串成一张账单

这个 demo 不属于任何一个团队, 它是前面所有 demo 的**收束**:

    一次训练 step = 数据搬进来(01 存储) + 算子算完(02 编译) + 梯度同步完(03 网络)

前面每个 demo 都在回答"某一段能快多少". 这个 demo 回答两个更重要的问题:
    1. 这三段各占多少? 也就是**瓶颈在哪**、该先学/先做哪一段?
    2. 瓶颈会随什么迁移? (模型形态、卡数、batch —— 换个场景答案就变了)

所有数字都由下面这几个硬件常数和模型规模**算出来**, 而不是拍脑袋写死的.
改参数重跑, 就能看到瓶颈是怎么迁移的.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.params import env_int, env_float, banner  # noqa: E402

np.seterr(all="ignore")

# ------------------------------------------------------------------ 硬件
GPU_TFLOPS = env_float("GPU_TFLOPS", 989.0, "单卡峰值算力(TFLOPS, bf16 稠密)")
GPU_BW_GBS = env_float("GPU_BW_GBS", 3350.0, "单卡 HBM 带宽(GB/s)")
NET_GBS = env_float("NET_GBS", 25.0, "机间网络单向带宽(GB/s)")
STORAGE_GBS = env_float("STORAGE_GBS", 2.0, "单卡能分到的样本读取带宽(GB/s)")

# ------------------------------------------------------------------ 模型与负载
N_GPU = env_int("N_GPU", 512, "训练卡数")
DENSE_PARAMS_M = env_float("DENSE_PARAMS_M", 300.0, "稠密部分参数量(百万)")
SPARSE_PARAMS_B = env_float("SPARSE_PARAMS_B", 100.0, "稀疏 Embedding 参数量(十亿)")
BATCH_PER_GPU = env_int("BATCH_PER_GPU", 1024, "单卡 batch size")
SEQ_LEN = env_int("SEQ_LEN", 128, "平均用户行为序列长度")
EMB_DIM = env_int("EMB_DIM", 128, "Embedding 维度")
SAMPLE_KB = env_float("SAMPLE_KB", 4.0, "单条样本在磁盘上的大小(KB)")
ETL_MS_PER_1K = env_float("ETL_MS_PER_1K", 8.0, "CPU 每处理 1000 条样本的耗时(ms)")


# ================================================================ 1. Roofline
def roofline():
    """算术强度决定了一个算子最多能跑到峰值算力的百分之几。

    ridge point(脊点) = 峰值算力 / 峰值带宽. 算术强度低于它 -> 带宽说了算(memory-bound),
    高于它 -> 算力说了算(compute-bound). 这是**在写任何优化之前**就该做的判断。
    """
    ridge = GPU_TFLOPS * 1e12 / (GPU_BW_GBS * 1e9)
    ops = [
        # (名字, FLOP, 访存字节, 说明)
        ("Embedding lookup", 0, EMB_DIM * 4 * 2, "只是搬数据, 一次乘法都没有"),
        ("逐元素 (加/门控/激活)", 1, 3 * 4, "读两个写一个, 做一次运算"),
        ("LayerNorm", 8, 2 * 4, "几次归约, 仍然是带宽主导"),
        ("GEMM (M=N=K=1024)", 2 * 1024 ** 3, 3 * 1024 ** 2 * 2, "方阵矩阵乘, bf16"),
        ("GEMM (M=N=K=8192)", 2 * 8192 ** 3, 3 * 8192 ** 2 * 2, "大矩阵乘: 唯一真正吃算力的"),
        ("小 GEMM (batch=8)", 2 * 8 * 1024 * 1024, (8 * 1024 + 1024 * 1024 + 8 * 1024) * 2,
         "在线推理的小 batch: 权重搬进来只用一次"),
    ]
    print(f"\n[1] Roofline: 这台机器的脊点 = {GPU_TFLOPS:.0f}TFLOPS / "
          f"{GPU_BW_GBS:.0f}GB/s = {ridge:.0f} FLOP/Byte")
    print("    算术强度低于脊点的算子, 无论如何都吃不满算力 —— 它在等内存.")
    print(f"\n  {'算子':<26}{'算术强度':>12}{'可达峰值':>12}{'类型':>14}   说明")
    print("  " + "-" * 86)
    for name, flop, byts, note in ops:
        ai = flop / byts
        frac = min(1.0, ai / ridge)
        kind = "compute-bound" if ai >= ridge else "memory-bound"
        print(f"  {name:<24}{ai:>11.1f}{frac:>11.1%}{kind:>16}   {note}")
    print("\n  -> 推荐模型的算子谱系里, **只有大 GEMM 一项是 compute-bound**,")
    print("     而它在推荐模型里的占比远小于 LLM(推荐的稠密部分通常只有几百 M 参数).")
    print("     Embedding lookup 的算术强度是 **0** —— 它根本不算数, 纯粹是搬运.")
    print("     这解释了一个反复被问到的问题: 为什么推荐模型的 MFU(算力利用率)")
    print("     普遍只有百分之几, 而 LLM 能到 40~50%? 不是工程没做好,")
    print("     **是模型形态决定的**: 一个以搬运为主的负载, 本来就不该用算力去衡量.")
    print("     推论: 推荐场景砸钱买更强的算力卡, 收益远小于 LLM 场景;")
    print("     真正该买/该优化的是**带宽**(HBM、网络、存储)—— 也就是这三个方向.")
    return ridge


# ================================================================ 2. step 预算
def step_budget(n_gpu=None, batch=None, dense_m=None, overlap="realistic"):
    """把一次 step 拆成数据/计算/通信三段, 每段都由上面的硬件常数算出来。"""
    n = N_GPU if n_gpu is None else n_gpu
    b = BATCH_PER_GPU if batch is None else batch
    dense = (DENSE_PARAMS_M if dense_m is None else dense_m) * 1e6

    # ---- 数据段: 读样本 + CPU 侧 ETL (两者本身可以重叠, 取较大者) ----
    read_s = b * SAMPLE_KB * 1024 / (STORAGE_GBS * 1e9)
    etl_s = b / 1000 * ETL_MS_PER_1K / 1e3
    t_data = max(read_s, etl_s)

    # ---- 计算段 ----
    # 稠密部分: 前向+反向 ≈ 6 * 参数量 * 样本数 FLOP (标准估算)
    dense_flop = 6 * dense * b
    # Embedding: 查表 + 反向散射, 纯访存, 用带宽算
    emb_bytes = b * SEQ_LEN * EMB_DIM * 4 * 3
    t_dense = dense_flop / (GPU_TFLOPS * 1e12 * 0.45)   # 大 GEMM 实测约 45% 峰值
    t_emb = emb_bytes / (GPU_BW_GBS * 1e9)
    t_compute = t_dense + t_emb

    # ---- 通信段 ----
    # 稠密梯度 AllReduce: Ring, 单卡传输 2(n-1)/n * D
    dense_grad_bytes = dense * 4
    t_ar = 2 * (n - 1) / n * dense_grad_bytes / (NET_GBS * 1e9)
    # 稀疏 Embedding 的 AllToAll: 每卡要把自己 batch 需要的 embedding 从别的卡取回来
    a2a_bytes = b * SEQ_LEN * EMB_DIM * 4 * (n - 1) / n * 2
    t_a2a = a2a_bytes / (NET_GBS * 1e9)
    t_comm = t_ar + t_a2a

    ideal = t_compute
    if overlap == "none":
        step = t_data + t_compute + t_comm
    elif overlap == "perfect":
        # 上界: 三段完全并行. 注意这是**理论下界**, 通信仍然至少要 t_comm 这么久.
        step = max(t_data, t_compute, t_comm)
    else:
        # 现实: 数据靠 prefetch 完全重叠; 通信最多只有一半能被拆成梯度桶去和反向重叠,
        # **而且藏起来的部分不可能超过计算本身的时长** —— 这一条最容易被忽略:
        # 通信比计算长得多时, 再怎么"重叠"也藏不下去.
        hidden = min(0.5 * t_comm, t_compute)
        step = max(t_data, t_compute + t_comm - hidden)

    return {
        "t_data": t_data, "t_compute": t_compute, "t_comm": t_comm,
        "t_dense": t_dense, "t_emb": t_emb, "t_ar": t_ar, "t_a2a": t_a2a,
        "step": step, "mfu": dense_flop / step / (GPU_TFLOPS * 1e12),
        "ideal": ideal,
        "samples_per_s": b / step * n,
    }


def bar(frac, width=28):
    k = int(round(frac * width))
    return "█" * k + "·" * (width - k)


def main():
    global STORAGE_GBS, NET_GBS, GPU_TFLOPS, GPU_BW_GBS
    banner(88)
    print("=" * 88)
    print(f"场景: {N_GPU} 卡 | 稠密 {DENSE_PARAMS_M:g}M 参数 + 稀疏 {SPARSE_PARAMS_B:g}B 参数 "
          f"| 单卡 batch {BATCH_PER_GPU} | 序列 {SEQ_LEN} x {EMB_DIM} 维")
    print(f"硬件: 单卡 {GPU_TFLOPS:g}TFLOPS / HBM {GPU_BW_GBS:g}GB/s | "
          f"网络 {NET_GBS:g}GB/s | 存储 {STORAGE_GBS:g}GB/s")
    print("=" * 88)

    roofline()

    # ---------------------------------------------------------- 2. 时间账单
    r = step_budget()
    print("\n" + "-" * 88)
    print("[2] 一次 step 的时间账单 (单卡视角)")
    tot = r["t_data"] + r["t_compute"] + r["t_comm"]
    rows = [
        ("数据: 读样本 + ETL", r["t_data"], "01_data_storage"),
        ("计算: 稠密 GEMM", r["t_dense"], "02_compile"),
        ("计算: Embedding 访存", r["t_emb"], "02_compile"),
        ("通信: 梯度 AllReduce", r["t_ar"], "03_network"),
        ("通信: Embedding AllToAll", r["t_a2a"], "03_network"),
    ]
    print(f"\n  {'环节':<26}{'耗时':>10}{'占比':>8}  {'':<28}  对应目录")
    print("  " + "-" * 84)
    for name, t, d in rows:
        print(f"  {name:<24}{t*1e3:>9.1f}ms{t/tot:>8.1%}  {bar(t/tot)}  {d}")
    print(f"  {'合计(纯串行)':<24}{tot*1e3:>9.1f}ms")

    print(f"\n  {'重叠程度':<26}{'step 耗时':>12}{'MFU':>10}{'全局吞吐(样本/s)':>20}")
    print("  " + "-" * 70)
    for mode, tag in [("none", "完全不重叠"), ("realistic", "数据全叠+通信尽量叠"),
                      ("perfect", "完美重叠(理论下界)")]:
        x = step_budget(overlap=mode)
        print(f"  {tag:<24}{x['step']*1e3:>10.1f}ms{x['mfu']:>10.1%}"
              f"{x['samples_per_s']:>19,.0f}")
    print(f"\n  -> **重叠不是万能的**: 这个场景里通信 {r['t_comm']*1e3:.0f}ms 远大于计算 "
          f"{r['t_compute']*1e3:.1f}ms,")
    print("     而'通信藏在计算背后'最多只能藏起计算那么长的一段. 通信一旦成为")
    print("     绝对大头, 重叠就基本没有空间了 —— 这时唯一的出路是**减少通信量本身**")
    print("     (梯度压缩/量化、分层 AllReduce、更大的 batch 摊薄同步频率).")
    print("\n  -> MFU 只有百分之几, 而且**这不是 bug**: 见 [1], 推荐模型本来就是搬运型负载.")
    print("     用 MFU 考核推荐训练是个常见的管理错误 —— 它会逼着团队去优化")
    print("     一个不该被优化的指标. 更合适的指标是'每张卡每秒能吃多少样本'.")

    # ---------------------------------------------------------- 3. 敏感度
    print("\n" + "-" * 88)
    print("[3] 把某一段做快 2x, 端到端能快多少? (木桶效应的量化版)")
    base = step_budget()["step"]
    print(f"\n  {'优化':<34}{'step':>11}{'端到端收益':>13}{'谁的活':>16}")
    print("  " + "-" * 76)
    variants = [
        ("存储/ETL 快 2x (列存+GPU解码)", dict(), "STORAGE_GBS", "01 存储"),
        ("算子快 2x (融合/编译)", dict(), "COMPUTE", "02 编译"),
        ("网络快 2x (更好的拓扑/算法)", dict(), "NET_GBS", "03 网络"),
        ("batch 减半 (降低单步负载)", dict(batch=BATCH_PER_GPU // 2), "", "调参"),
    ]
    for name, kw, knob, owner in variants:
        if knob == "STORAGE_GBS":
            STORAGE_GBS *= 2
            s = step_budget()["step"]
            STORAGE_GBS /= 2
        elif knob == "NET_GBS":
            NET_GBS *= 2
            s = step_budget()["step"]
            NET_GBS /= 2
        elif knob == "COMPUTE":
            GPU_TFLOPS *= 2
            GPU_BW_GBS *= 2
            s = step_budget()["step"]
            GPU_TFLOPS /= 2
            GPU_BW_GBS /= 2
        else:
            s = step_budget(**kw)["step"] * 2   # batch 减半, 要跑两倍的 step 才等价
        print(f"  {name:<32}{s*1e3:>9.1f}ms{base/s:>12.2f}x{owner:>16}")
    print("\n  -> 收益完全由**当前占比**决定, 和这件事本身有多难、多前沿毫无关系.")
    print("     这是选择研究方向时最该先算的一笔账: 一个 5% 占比的环节,")
    print("     就算你做到无穷快, 端到端也只有 1.05x. 论文可以发, 但线上没人感谢你.")

    # ---------------------------------------------------------- 4. 规模扫描
    print("\n" + "-" * 88)
    print("[4] 瓶颈会迁移: 卡数扫描 (单卡 batch 不变, 即 weak scaling)")
    print(f"\n  {'卡数':>7}{'数据':>10}{'计算':>10}{'通信':>10}{'step':>11}"
          f"{'并行效率':>11}{'当前瓶颈':>12}")
    print("  " + "-" * 74)
    base1 = step_budget(n_gpu=1)
    for n in [1, 8, 64, 512, 4096]:
        x = step_budget(n_gpu=n)
        eff = base1["step"] / x["step"]
        who = max([("数据", x["t_data"]), ("计算", x["t_compute"]),
                   ("通信", x["t_comm"])], key=lambda z: z[1])[0]
        print(f"  {n:>7}{x['t_data']*1e3:>9.1f}ms{x['t_compute']*1e3:>9.1f}ms"
              f"{x['t_comm']*1e3:>9.1f}ms{x['step']*1e3:>10.1f}ms{eff:>11.1%}{who:>12}")
    print("\n  -> 卡数增加时, 单卡的计算量**一点没变**, 通信量却在涨(Ring 的 2(n-1)/n 项")
    print("     很快趋于 2, 但 AllToAll 的 (n-1)/n 同理), 于是通信占比单调上升.")
    print("     这就是'千卡之后效率变低'的算术来源(展开细节见 03/demo2).")
    print("     实践含义: **小规模时该优化计算, 大规模时该优化通信** ——")
    print("     同一个团队在不同阶段, 正确答案是不一样的.")

    # ---------------------------------------------------------- 5. 模型形态
    print("\n" + "-" * 88)
    print("[5] 换一种模型形态, 三段的占比完全不同")
    print("    先做一步纸面推导: 计算量 ∝ 参数量 x batch, 而梯度通信量 ∝ 参数量.")
    print("    两者一比, **参数量约掉了** —— 纯数据并行下, 计算/通信的平衡点")
    print("    只由**单卡 batch**决定, 和模型多大根本没关系. 下表验证这个推导.")
    print(f"\n  {'场景':<30}{'单卡batch':>11}{'数据':>8}{'计算':>8}{'通信':>8}{'瓶颈':>8}")
    print("  " + "-" * 74)
    scenes = [
        ("推荐: 300M 稠密 / batch 1k", dict(dense_m=300, batch=1024)),
        ("中等: 1B 稠密 / batch 8k", dict(dense_m=1000, batch=8192)),
        ("类 LLM: 7B / batch 32k token", dict(dense_m=7000, batch=32768)),
        ("大 LLM: 70B / batch 256k token", dict(dense_m=70000, batch=262144)),
    ]
    for tag, kw in scenes:
        x = step_budget(**kw)
        t = x["t_data"] + x["t_compute"] + x["t_comm"]
        who = max([("数据", x["t_data"]), ("计算", x["t_compute"]),
                   ("通信", x["t_comm"])], key=lambda z: z[1])[0]
        print(f"  {tag:<28}{kw['batch']:>11,}{x['t_data']/t:>8.1%}{x['t_compute']/t:>8.1%}"
              f"{x['t_comm']/t:>8.1%}{who:>8}")
    print("\n  -> 推导被验证了: 瓶颈从通信翻到计算, 唯一的驱动因素是 batch 变大,")
    print("     参数量从 300M 到 70B 涨了 200 多倍, 却没有改变这个平衡.")
    print("     (只有'数据'那一段受参数量影响 —— 它 ∝ batch 而与参数无关, 所以占比在缩)")
    print("\n     这解释了一件很多人想不通的事: 为什么 LLM 那套 infra 经验搬到推荐上")
    print("     会水土不服. 不是因为'我们模型小所以不用操心通信' —— 恰恰相反,")
    print("     模型小反而更该操心, 因为**决定权在 batch 手里, 而推荐的 batch 天生小**.")
    print("     LLM 一个 step 喂进去几十万 token, 计算量把通信压得死死的;")
    print("     推荐一个 step 只有一两千条样本, 时间全花在**搬运**上")
    print("     (读样本、查 embedding、AllToAll、同步梯度).")
    print("     所以推荐 infra 的重心是存储、访存和网络, 也就是这个项目的三个目录.")
    print("\n     顺带解释了另一件事: 为什么推荐训练不能靠'无脑加大 batch'来提效率.")
    print("     加大 batch 确实能把这个比值拉上去, 但推荐是**在线学习**的 ——")
    print("     batch 越大, 模型对最新行为的响应越慢, 而这直接损失线上效果.")
    print("     LLM 没有这个约束, 所以它可以自由地把 batch 堆到百万 token.")
    print("     (本模型假设纯数据并行 + fp32 梯度; 真实 LLM 还有 TP/PP、bf16 梯度、")
    print("      梯度累积、ZeRO —— 这些手段做的事都是在压同一个比值)")

    print("\n" + "=" * 88)
    print("[这个项目的一句话总结]")
    print("  三个方向不是三块并列的知识, 而是**一次 step 里的三段时间**:")
    print("      01 存储  = 把数据搬到 GPU 门口")
    print("      02 编译  = 让 GPU 门内的那段代码跑得像样")
    print("      03 网络  = 让几百上千张卡的结果对齐")
    print("  而它们共同的母题只有一个: **在异构硬件上, 把'搬运'的成本压到接近'计算'.**")
    print("  搜广推让这个母题格外尖锐, 因为它同时要 (a) 万亿级流量 (b) 毫秒级 P99")
    print("  (c) 稀疏不定长数据 —— 去掉任何一条, 通用方案就够用了.")
    print("\n  学习建议: 别按目录顺序学, 按**这张账单的占比**学. 先跑这个 demo,")
    print("  看看你关心的那个场景里瓶颈在哪一段, 再回去读对应目录的 README 和 demo.")


if __name__ == "__main__":
    main()
