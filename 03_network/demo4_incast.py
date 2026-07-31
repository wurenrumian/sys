#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo4: 扇出(fan-out)的两个杀手 —— Incast 吞吐崩塌 与 尾延迟放大

对应表格: "微服务 RPC 请求以小包、高频、低延迟流量为主" /
          "在推理集群中, 用户请求 RPC 流量的 P999 延迟要求通常低于 100ms"

前面 demo1 讲的是"两类流量互相干扰". 但推荐系统里还有一类问题, 是**它自己造成的**:
一次推荐请求要扇出到几十上百个分片(召回分片、embedding 参数服务器、粗排/精排),
然后等**全部**返回才能出结果. 这个 scatter-gather 结构自带两个杀手:

  杀手一 (Incast):    N 个响应几乎同时回到同一个网口 -> 瞬时突发压垮交换机缓冲
                      -> 丢包 -> TCP 超时重传(几百 ms) -> P999 灾难.
  杀手二 (尾延迟放大): 请求延迟 = max(N 个分片的延迟). 分片越多, 撞上慢分片的概率越大.
                      单分片 P99=1% 时, 扇出 100 -> 有 63% 的请求会撞上至少一个慢分片.

这两个杀手都**不是**链路带宽不够造成的 —— 平均利用率可能只有百分之几.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.params import env_int, env_float, banner  # noqa: E402

np.seterr(all="ignore")

N_REQ = env_int("N_REQ", 20_000, "模拟多少次推荐请求")
FANOUT = env_int("FANOUT", 64, "一次请求扇出到几个分片")
RESP_KB = env_float("RESP_KB", 32.0, "单个分片的响应大小(KB)")
LINK_GBPS = env_float("LINK_GBPS", 25.0, "接收端网口速率(Gbps)")
BUFFER_KB = env_float("BUFFER_KB", 1024.0, "交换机出端口的缓冲区(KB)")
RTO_MS = env_float("RTO_MS", 200.0, "TCP 超时重传的最小等待(ms)")
RTT_US = env_float("RTT_US", 50.0, "机架内往返延迟(us)")
SHARD_P99_X = env_float("SHARD_P99_X", 10.0, "单分片延迟 p99 是中位数的几倍")

rng = np.random.default_rng(0)


# ================================================================ 杀手一: Incast
def incast(n_req, fanout, resp_kb, buffer_kb, batch=None):
    """模拟 scatter-gather 的响应汇聚。

    batch=None : 所有分片同时返回(最自然的实现, 也是最糟的)
    batch=W    : 应用层限制并发, 一次只向 W 个分片发请求, 分多轮完成

    建模要点:
      * 缓冲区可用量随背景流量波动(0.4~1.0 倍), 所以同样的扇出有时丢有时不丢 ——
        这正是为什么 incast 表现为"偶发的、难复现的" P999 毛刺.
      * 一旦丢包, 这个分片要等一个 RTO 才重传. RTO 是**毫秒级**的,
        而正常传输是**微秒级**的 —— 差三个数量级, 所以只要丢一个包, 这次请求就完了.
    """
    w = fanout if batch is None else min(batch, fanout)
    rounds = int(np.ceil(fanout / w))
    link_bps = LINK_GBPS * 1e9
    burst_kb = w * resp_kb                       # 一轮里同时涌向网口的数据量

    # 每轮的可用缓冲(背景流量占掉了一部分)
    avail = buffer_kb * rng.uniform(0.4, 1.0, (n_req, rounds))
    # 网口在一个 RTT 内能排掉的量, 也算作可吸收的容量
    drain_kb = link_bps * (RTT_US * 1e-6) / 8 / 1024
    overflow = np.maximum(0.0, burst_kb - avail - drain_kb)
    lost_shards = np.ceil(overflow / resp_kb)     # 溢出多少就丢多少个响应

    # 一轮的耗时: 传输时间 + 一个 RTT; 若这一轮有丢包, 还要加一个 RTO
    tx_ms = burst_kb * 1024 * 8 / link_bps * 1e3
    per_round = tx_ms + RTT_US / 1e3 + np.where(lost_shards > 0, RTO_MS, 0.0)
    lat = per_round.sum(axis=1)

    good_kb = n_req * fanout * resp_kb
    return {
        "p50": np.percentile(lat, 50),
        "p99": np.percentile(lat, 99),
        "p999": np.percentile(lat, 99.9),
        "avg": lat.mean(),
        "loss_req": float((lost_shards.sum(axis=1) > 0).mean()),   # 有多少比例的请求踩到丢包
        "loss_shard": float(lost_shards.sum() / (n_req * fanout)),
        "goodput_gbps": good_kb * 1024 * 8 / (lat.sum() / 1e3) / 1e9,
        "rounds": rounds,
    }


def show_incast(tag, r, width=26):
    print(f"  {tag:<{width}}{r['p50']:>9.2f}ms{r['p99']:>10.2f}ms{r['p999']:>11.2f}ms"
          f"{r['loss_req']:>11.1%}{r['goodput_gbps']:>12.2f}")


def head_incast(first="配置", width=26):
    print(f"  {first:<{width}}{'P50':>11}{'P99':>10}{'P999':>11}{'踩到丢包':>13}{'有效吞吐Gbps':>14}")
    print("  " + "-" * (width + 56))


# ================================================================ 杀手二: 尾延迟放大
def shard_latency(n, seed=1):
    """单个分片的服务延迟: 对数正态, 中位数 1ms, p99 = SHARD_P99_X 倍中位数.

    真实系统里这个长尾来自: GC、page cache miss、锁竞争、被同机邻居抢 CPU、
    磁盘/网卡队列、以及最烦人的"这台机器就是有点慢".
    """
    sigma = np.log(SHARD_P99_X) / 2.326      # p99 = median * exp(2.326*sigma)
    return np.random.default_rng(seed).lognormal(0.0, sigma, n)


def fanout_latency(n_req, fanout, hedge_pct=None):
    """扇出请求的端到端延迟 = max(所有分片).

    hedge_pct 不为空时启用**对冲请求(hedged request)**: 某个分片超过 hedge_pct 分位
    还没返回, 就再向另一个副本发一份, 取先回来的那个.
    这是 Google《The Tail at Scale》里最有效也最便宜的一招.
    """
    lat = shard_latency(n_req * fanout).reshape(n_req, fanout)
    extra_load = 0.0
    if hedge_pct is not None:
        thr = np.percentile(lat, hedge_pct)
        second = shard_latency(n_req * fanout, seed=2).reshape(n_req, fanout)
        slow = lat > thr
        extra_load = slow.mean()                       # 多打出去的请求量占比
        # 慢分片的实际延迟 = min(原来的, 等到阈值后重发的那份)
        lat = np.where(slow, np.minimum(lat, thr + second), lat)
    req = lat.max(axis=1)
    return req, extra_load


def main():
    banner(88)
    print("=" * 88)
    print(f"scatter-gather: 每次请求扇出 {FANOUT} 个分片, 单分片响应 {RESP_KB:g}KB, "
          f"接收网口 {LINK_GBPS:g}Gbps")
    print(f"交换机出端口缓冲 {BUFFER_KB:g}KB | TCP 超时重传 RTO {RTO_MS:g}ms | "
          f"机架内 RTT {RTT_US:g}us")
    print("=" * 88)

    # ---------------------------------------------------------- 1. 带宽账
    burst = FANOUT * RESP_KB
    print(f"\n[0] 先算账: 一次请求的响应总量 = {FANOUT} x {RESP_KB:g}KB = {burst:g}KB")
    print(f"    而交换机出端口缓冲只有 {BUFFER_KB:g}KB —— "
          f"突发量是缓冲的 {burst/BUFFER_KB:.1f} 倍.")
    print(f"    传完这 {burst:g}KB 只要 {burst*1024*8/(LINK_GBPS*1e9)*1e3:.2f}ms, "
          f"而丢一个包要等 {RTO_MS:g}ms 才重传 —— 差 "
          f"{RTO_MS/(burst*1024*8/(LINK_GBPS*1e9)*1e3):.0f} 倍.")
    print("    **整个 incast 问题就浓缩在这两个数字的对比里**: 传输是微秒级的,")
    print("    而丢包恢复是毫秒级的. 所以哪怕只有万分之一的包被丢, 尾延迟也会爆炸.")

    # ---------------------------------------------------------- 2. 扇出扫描
    print("\n[1] 扇出度扫描 (无任何控制, 所有分片同时返回)")
    head_incast("扇出度")
    for f in [8, 16, 32, 64, 128, 256]:
        show_incast(str(f), incast(N_REQ, f, RESP_KB, BUFFER_KB))
    print("  -> 这就是经典的 **TCP incast 吞吐崩塌**: 扇出度越过某个临界点后,")
    print("     有效吞吐不是缓慢下降, 而是**断崖式**掉下去 —— 因为所有请求都在等 RTO,")
    print("     而等 RTO 的时候链路是**空着**的. 带宽没被用完, 时间却全花光了.")
    print("     这是所有 scatter-gather 系统(推荐召回、分布式存储、MapReduce shuffle)")
    print("     共同的坑, 而推荐系统的扇出度天然就高.")

    # ---------------------------------------------------------- 3. 四种解法
    print(f"\n[2] 四种解法, 同样的扇出度 {FANOUT}")
    head_incast("方案")
    base = incast(N_REQ, FANOUT, RESP_KB, BUFFER_KB)
    show_incast("不做任何控制", base)
    show_incast("缓冲区加大 4x (换硬件)", incast(N_REQ, FANOUT, RESP_KB, BUFFER_KB * 4))
    show_incast("响应压缩到 1/2", incast(N_REQ, FANOUT, RESP_KB / 2, BUFFER_KB))
    show_incast("应用层限并发 W=16", incast(N_REQ, FANOUT, RESP_KB, BUFFER_KB, batch=16))
    show_incast("应用层限并发 W=8", incast(N_REQ, FANOUT, RESP_KB, BUFFER_KB, batch=8))
    print("\n  -> **反直觉的地方**: '加大缓冲区'看起来救回来了 —— P50 从 200ms 变成 0.72ms.")
    print("     但看 P99 那一列: 仍然是 200ms. 它只是把临界点往后推了一点,")
    print("     把'每次都炸'变成了'偶尔炸' —— 而 SLA 恰恰是按偶尔那次签的.")
    print("     更糟的是缓冲变大本身会抬高排队延迟(bufferbloat, 见 demo3): 花钱换硬件,")
    print("     买到的是一个更难复现的故障. 压缩响应同理, 只是把临界扇出度乘了 2.")
    print("     真正有效的是**限制并发扇出** —— 它从源头上不让突发发生,")
    print("     P50/P99/P999 三条线全部落在 1ms 以内, 而且一行业务代码就能改.")
    print("     代价是多跑几轮, 单看 P50 比'缓冲够大且没丢包'的理想情况略慢一点点.")
    print("     一般原则: **突发问题要在源头解决, 在下游用缓冲去接是接不住的.**")
    print("     (工业界的完整方案还包括: 让分片错峰返回、用 DCTCP/ECN 提前降速、")
    print("      把 RTO_min 从 200ms 调到微秒级、以及 RDMA 的无损网络 PFC —— 各有代价)")

    # ---------------------------------------------------------- 4. 尾延迟放大
    print("\n" + "=" * 88)
    print("[3] 杀手二: 就算一个包都不丢, 扇出本身也会放大尾延迟")
    print(f"    单分片延迟: 中位数 1.00ms, p99 = {SHARD_P99_X:g}x 中位数")
    print(f"\n  {'扇出度':>8}{'请求P50':>11}{'请求P99':>11}{'请求P999':>12}"
          f"{'撞上慢分片的概率':>18}")
    print("  " + "-" * 60)
    for f in [1, 5, 20, 50, 100, 300]:
        req, _ = fanout_latency(N_REQ, f)
        p_slow = 1 - 0.99 ** f
        print(f"  {f:>8}{np.percentile(req,50):>9.2f}ms{np.percentile(req,99):>9.2f}ms"
              f"{np.percentile(req,99.9):>10.2f}ms{p_slow:>17.1%}")
    print("\n  -> 注意 **P50 那一列**: 扇出 100 时请求的**中位数**延迟, 已经接近")
    print("     单分片的 p99 了. 也就是说, '一半的请求都很慢'成了常态.")
    print("     数学上很简单: P(所有 N 个分片都不慢) = 0.99^N, N=100 时只剩 37%.")
    print("     **这意味着单分片的 p99 优化, 在高扇出系统里比平均值重要得多** ——")
    print("     一台机器偶尔慢一下, 在扇出 1 的系统里没人察觉, 在扇出 100 的系统里")
    print("     是 63% 的请求都会碰到.")

    # ---------------------------------------------------------- 5. 对冲请求
    print(f"\n[4] 对冲请求(hedged request): 等到 p95 还没回来, 就再发一份给副本")
    print(f"\n  {'方案':<26}{'请求P50':>11}{'请求P99':>11}{'请求P999':>12}{'额外负载':>12}")
    print("  " + "-" * 72)
    for tag, hp in [("不对冲", None), ("p95 后对冲", 95.0), ("p90 后对冲", 90.0)]:
        req, extra = fanout_latency(N_REQ, FANOUT, hedge_pct=hp)
        print(f"  {tag:<24}{np.percentile(req,50):>9.2f}ms{np.percentile(req,99):>9.2f}ms"
              f"{np.percentile(req,99.9):>10.2f}ms{extra:>12.1%}")
    print("\n  -> 只多打 5% 的请求量, 尾延迟就大幅下降 —— 这是《The Tail at Scale》里")
    print("     性价比最高的一招. 它的本质是**用一点冗余算力去买确定性**.")
    print("     前提条件有两个, 缺一不可:")
    print("       1. 请求必须是**幂等**的(重复执行不产生副作用) —— 读请求天然满足;")
    print("       2. 系统要有**余量**. 如果本来就跑在 90% 负载上, 对冲会引发雪崩:")
    print("          多打的请求让系统更慢, 更慢又触发更多对冲 —— 正反馈.")
    print("     所以工业实现都会给对冲加一个全局速率上限(比如'对冲量不超过总量的 5%').")

    print("\n" + "=" * 88)
    print("[和其他 demo 的关系 / 对应到表格的课题]")
    print("  * demo1 说'大包堵住小包', 本 demo 说'小包自己堵住自己' ——")
    print("    两者都指向同一个结论: **尾延迟由突发和不可抢占单元决定, 与平均利用率无关**.")
    print("  * demo3 的 DCTCP 正是为 incast 设计的: 在缓冲被打满之前就让发送方降速.")
    print("    但它需要一个 RTT 才能生效, 而 incast 的突发在一个 RTT 内就完成了 ——")
    print("    这是拥塞控制对 incast 天然乏力的原因, 也是为什么要在应用层限并发.")
    print("  * 表格里'跨层协同的标准化信令体系': 应用层**知道**自己要扇出 100 个分片,")
    print("    网络层却要等丢包才发现. 如果这个信息能提前传下去(哪怕只是一个'我要发")
    print("    一次 4MB 突发'的提示), 交换机就能提前预留缓冲或调度错峰 ——")
    print("    这正是 incast 这个几十年的老问题至今没有干净解法的症结所在.")


if __name__ == "__main__":
    main()
