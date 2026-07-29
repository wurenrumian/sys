#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo2: AllReduce 通信模型 —— "千卡规模后通信效率变低"具体是怎么发生的

对应表格:
  - TikTok: "训练任务达到千卡规模后(H100/B200), IO 和通信效率变低 ...
             卡间通信开销会随着卡数变多而效率变低"
  - 网络方向: "异构网络环境中(NVLink/InfiniBand/RoCE以太网)的带宽差异达数十倍且拓扑层级复杂"

用经典的 alpha-beta 模型(T = 轮数 x 延迟 + 传输量 / 带宽)算清楚三种算法的代价.
"""
import numpy as np

np.seterr(all="ignore")

# ---------------------------------------------------------------- 硬件参数
LINKS = {
    #  名称           单向带宽GB/s   单跳延迟us
    "NVLink (机内)":      (450.0,      1.0),
    "InfiniBand (机间)":   (25.0,      2.5),
    "RoCE 以太网":         (12.5,      5.0),
}


def ring_allreduce(n, bytes_, bw, lat):
    """Ring AllReduce: reduce-scatter + all-gather, 各 (n-1) 步.

    每步每卡收发 D/n 字节 -> 单卡总传输量 2*(n-1)/n*D, **与 n 几乎无关**(趋近 2D).
    但要走 2(n-1) 步, 所以延迟项随 n **线性**增长 —— 这就是千卡下的痛点.
    """
    steps = 2 * (n - 1)
    per_step_bytes = bytes_ / n
    return steps * lat * 1e-6 + steps * per_step_bytes / (bw * 1e9)


def tree_allreduce(n, bytes_, bw, lat):
    """二项树 Reduce + Broadcast: 2*log2(n) 步, **每步传输完整的 D 字节**.

    延迟项只随 log n 增长(优于 Ring 的线性), 但带宽项是 2*log2(n)*D —— 随 n 增长!
    所以: 小数据量被延迟支配 -> Tree 赢; 大数据量被带宽支配 -> Ring 赢.
    这才是教科书里 Ring vs Tree 的经典权衡.
    """
    steps = 2 * int(np.ceil(np.log2(n)))
    return steps * lat * 1e-6 + steps * bytes_ / (bw * 1e9)


def halving_doubling(n, bytes_, bw, lat):
    """递归折半-加倍(Recursive Halving-Doubling): 2*log2(n) 步.

    每步传输量减半(D/2, D/4, ...)再加倍, 总传输量 ~2*(n-1)/n*D —— 和 Ring 一样是
    带宽最优的, 同时步数只有 log n —— 也就是说它在**两项上同时最优**.

    那为什么工业界还大量用 Ring? 因为这个理想模型忽略了:
      1. HD 要求卡数是 2 的幂, 否则要补零或退化;
      2. HD 的通信对象是"距离 2^k 的对端", 在分层拓扑上会**跨交换机**,
         而 Ring 的通信永远只发生在相邻节点, 对物理拓扑更友好;
      3. Ring 实现简单、对非均匀带宽更鲁棒.
    真实的 NCCL 会综合数据量、卡数、拓扑在这几种算法间自动选择.
    """
    steps = 2 * int(np.ceil(np.log2(n)))
    total_bytes = 2 * bytes_ * (n - 1) / n
    return steps * lat * 1e-6 + total_bytes / (bw * 1e9)


def naive_allreduce(n, bytes_, bw, lat):
    """朴素中心汇聚: 所有卡把数据发给 rank0, rank0 归约后广播回去.

    中心节点要收发 (n-1)*D 字节 -> **单点带宽瓶颈**, 完全不可扩展.
    """
    return 2 * lat * 1e-6 + 2 * (n - 1) * bytes_ / (bw * 1e9)


ALGOS = [("Naive(中心汇聚)", naive_allreduce),
         ("Ring", ring_allreduce),
         ("二项树", tree_allreduce),
         ("递归折半HD", halving_doubling)]


def main():
    print("=" * 86)
    print("AllReduce 通信模型:  T = 通信轮数 x 单跳延迟  +  传输字节 / 带宽")
    print("=" * 86)

    # ---------------------------------------------------------- 1. 规模扩展
    bw, lat = LINKS["InfiniBand (机间)"]
    D = 1e9  # 1GB 梯度, 相当于 250M 参数的 fp32 梯度
    print(f"\n[1] 固定数据量 {D/1e9:.0f}GB (约 250M 参数的 fp32 梯度), "
          f"链路 = InfiniBand {bw:.0f}GB/s / {lat:.1f}us")
    print(f"\n  {'卡数':>7}" + "".join(f"{n:>16}" for n, _ in ALGOS)
          + f"{'Ring带宽占比':>14}")
    print("  " + "-" * 84)
    for n in [8, 64, 256, 1024, 4096]:
        row = f"  {n:>7}"
        ts = []
        for _, fn in ALGOS:
            t = fn(n, D, bw, lat)
            ts.append(t)
            row += f"{t*1e3:>13.1f}ms"
        # Ring 的时间由"带宽项 + 延迟项"组成; 带宽项占比越低, 说明越被延迟拖累
        ring_bw_term = 2 * (n - 1) / n * D / (bw * 1e9)
        row += f"{ring_bw_term/ts[1]:>13.1%}"
        print(row)

    print("\n  -> Naive 完全不可扩展: 中心节点传输量随卡数线性增长, 4096 卡要 5 分钟.")
    print("     Ring 的**带宽项**与卡数几乎无关(趋近 2D, 这是它的精髓),")
    print("     但它要走 2(n-1) 步, 延迟项随卡数**线性**增长.")
    print("     看最后一列: 8 卡时 Ring 的耗时几乎全是有效数据传输(带宽占比高),")
    print("     到 4096 卡时带宽占比降到 80% 以下 —— 有 20% 的时间纯粹耗在通信轮次的")
    print("     延迟累积上. 这就是表格里'卡间通信开销随卡数变多而效率变低'的数学来源.")
    print("     二项树因为每步都要传完整的 D, 带宽项随 log n 增长, 大数据量下最差.")

    # ---------------------------------------------------------- 2. 交叉点
    print("\n" + "-" * 86)
    print(f"[2] Ring vs Tree 的交叉点: 取决于数据量 (1024 卡, InfiniBand)")
    n = 1024
    print(f"\n  {'数据量':>10}{'Ring':>13}{'二项树':>13}{'递归折半HD':>14}{'Ring vs 树':>13}")
    print("  " + "-" * 64)
    for D2 in [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]:
        tr = ring_allreduce(n, D2, bw, lat)
        tt = tree_allreduce(n, D2, bw, lat)
        th = halving_doubling(n, D2, bw, lat)
        label = f"{D2/1e6:.0f}MB" if D2 >= 1e6 else f"{D2/1e3:.0f}KB"
        who = "Ring 赢" if tr < tt else "树赢"
        print(f"  {label:>10}{tr*1e6:>11.0f}us{tt*1e6:>11.0f}us{th*1e6:>12.0f}us{who:>13}")

    print("\n  -> 交叉点清晰可见:")
    print("     * 小数据量: 被**延迟**支配 -> 树赢(2*log n 轮 vs Ring 的 2(n-1) 轮).")
    print("       1024 卡下 Ring 要走 2046 轮, 光延迟累积就 5.1ms; 树只要 20 轮.")
    print("     * 大数据量: 被**带宽**支配 -> Ring 赢(单卡传输量趋近 2D 且与 n 无关,")
    print("       而二项树每步传完整 D, 总量达 2*log2(n)*D = 20D).")
    print("\n     注意第三列: 递归折半(HD)在这个理想模型里**全程最优** —— 它同时拿到了")
    print("     树的 log n 轮数和 Ring 的最优传输量. 但它要求卡数为 2 的幂, 且通信对端")
    print("     跨越拓扑层级, 在真实分层网络里未必划算(详见函数注释).")
    print("     NCCL 会根据数据量、卡数、拓扑在这几种算法间自动选择.")
    print("     推荐场景的特殊性: 稀疏参数的 AllToAll 通信量既大又不规则,")
    print("     以上算法都不直接适用 —— 这正是可以做文章的地方.")

    # ---------------------------------------------------------- 3. 异构网络
    print("\n" + "-" * 86)
    print("[3] 异构网络: 同样 1024 卡 1GB, 不同链路差多少")
    print(f"\n  {'链路类型':<22}{'带宽':>12}{'延迟':>10}{'Ring 耗时':>14}{'相对最快':>10}")
    print("  " + "-" * 68)
    best = min(ring_allreduce(1024, D, b, l) for b, l in LINKS.values())
    for name, (b, l) in LINKS.items():
        t = ring_allreduce(1024, D, b, l)
        print(f"  {name:<20}{b:>10.1f}GB/s{l:>9.1f}us{t*1e3:>12.1f}ms{t/best:>9.1f}x")
    bw_ratio = max(b for b, _ in LINKS.values()) / min(b for b, _ in LINKS.values())
    t_ratio = (ring_allreduce(1024, D, *LINKS["RoCE 以太网"])
               / ring_allreduce(1024, D, *LINKS["NVLink (机内)"]))
    print(f"\n  -> 带宽差 {bw_ratio:.0f} 倍, 通信耗时差 {t_ratio:.0f} 倍 "
          f"(不完全成比例, 因为延迟项也在变). 而真实集群是**分层**的:")
    print("     机内 NVLink 快, 机间 IB 慢 -> 必须用**分层 AllReduce**")
    print("     (先机内规约, 再机间规约, 最后机内广播), 把慢链路上的数据量降到 1/8.")
    print("     表格里'异构网络环境中带宽差异达数十倍且拓扑层级复杂'说的就是这件事.")

    # ---------------------------------------------------------- 4. 分层
    print("\n" + "-" * 86)
    print("[4] 分层 AllReduce 的收益 (1024 卡 = 128 机 x 8 卡)")
    nvb, nvl = LINKS["NVLink (机内)"]
    ibb, ibl = LINKS["InfiniBand (机间)"]
    G, M = 8, 128        # 每机 8 卡, 共 128 机

    def reduce_scatter(n, bytes_, bw, lat):
        """(n-1) 步, 每步传 bytes_/n; 结束后每卡持有 1/n 的规约结果."""
        return (n - 1) * lat * 1e-6 + (n - 1) * (bytes_ / n) / (bw * 1e9)

    flat = ring_allreduce(G * M, D, ibb, ibl)

    # 分层的关键: 机内先 reduce-scatter, 之后每卡只持有 D/8,
    # 机间 AllReduce 就只需要在 D/8 上做 —— 慢链路上的数据量直接降到 1/8.
    t_rs = reduce_scatter(G, D, nvb, nvl)          # 机内 reduce-scatter
    t_inter = ring_allreduce(M, D / G, ibb, ibl)   # 机间 AllReduce, 只对 D/8
    t_ag = reduce_scatter(G, D, nvb, nvl)          # 机内 all-gather(对称, 同代价)
    hier = t_rs + t_inter + t_ag

    print(f"\n  扁平 Ring (1024 卡全走 IB)        : {flat*1e3:>8.2f} ms")
    print(f"  分层 AllReduce                    : {hier*1e3:>8.2f} ms  "
          f"({flat/hier:.2f}x 加速)")
    print(f"    ├─ 机内 reduce-scatter (NVLink) : {t_rs*1e3:>8.2f} ms")
    print(f"    ├─ 机间 AllReduce   (IB, 仅 D/{G}) : {t_inter*1e3:>8.2f} ms")
    print(f"    └─ 机内 all-gather     (NVLink) : {t_ag*1e3:>8.2f} ms")
    print(f"\n  收益的全部来源: 机间(慢链路)上的数据量从 D 降到了 D/{G}.")
    print(f"  扁平方案里 {flat*1e3:.0f}ms 几乎全部花在 IB 上; 分层后 IB 只需搬 1/{G} 的数据,")
    print(f"  剩下的规约工作交给快 {nvb/ibb:.0f} 倍的机内 NVLink 完成.")
    print("  这就是'拓扑感知'的价值: 同样的数学结果, 把数据搬运安排在正确的链路上.")

    print("\n" + "=" * 86)
    print("[对应到表格的课题]")
    print("  TikTok 那条列的探索方向, 现在可以对号入座了:")
    print("  * '算子级通信/计算重叠' —— 上面算的都是**纯通信**时间. 如果能让通信")
    print("    藏在计算的空隙里, 这些时间就等于免费. 这是收益最大的方向.")
    print("  * '有损压缩通信(梯度量化/稀疏化)' —— 直接减小上面公式里的 D.")
    print("    fp32->fp8 就是 4x, 但要保证 AUC 不掉(和 01 目录的 DCAI 是同一类问题).")
    print("  * 'Asynchronous Pipeline Parallelism(允许微小版本差异)' —— 打破同步屏障,")
    print("    让通信不再是关键路径上的阻塞点.")
    print("  * '自适应多路径路由(Fat-Tree/Torus 避热点)' —— 上面模型假设带宽是常数,")
    print("    真实网络里链路会拥塞, 有效带宽远低于标称值.")
    print("\n  注意本 demo 的模型只是**下界**: 它假设了完美的带宽、零拥塞、零抖动.")
    print("  真实千卡训练还要面对慢节点(straggler)、网络抖动、故障重传 ——")
    print("  表格里'单卡出问题都会导致训练不稳定, 卡多后概率也更高'说的就是这个.")


if __name__ == "__main__":
    main()
