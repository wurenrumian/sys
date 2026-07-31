#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo1: RPC 小包 vs 集合通信大包 —— 队头阻塞与调度策略

对应表格: "在推理集群中, 用户请求 RPC 流量的 P999 延迟要求通常低于 100ms,
          而 KV Cache 分布式同步、多卡推理的张量传输流量占比超过 60%,
          两类流量的性能冲突直接影响服务 SLA 达标率"

一句话: 一根网线上同时跑'搬家卡车'(AllReduce 大包)和'救护车'(RPC 小包).
"""
import os
import sys
from collections import deque
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.params import env_int, env_float, banner  # noqa: E402

np.seterr(all="ignore")

LINK_GBPS = env_float("LINK_GBPS", 100.0, "链路带宽(Gbps)")
SIM_MS = env_float("SIM_MS", 2000.0, "模拟时长(ms)")
RPC_QPS = env_int("RPC_QPS", 20_000, "RPC 请求速率")
RPC_BYTES = env_int("RPC_BYTES", 512, "RPC 小包大小(字节)")
BULK_MB = env_float("BULK_MB", 4.0, "单个集合通信块大小(MB)")
BULK_PERIOD_MS = env_float("BULK_PERIOD_MS", 2.0, "集合通信发起周期(ms)")
CHUNK_KB = env_float("CHUNK_KB", 4.0, "分片策略的片大小(KB)")
CHUNK_LABEL = f"优先级+分片{CHUNK_KB:g}KB"


def tx_time_ms(nbytes):
    """在链路上发送 nbytes 需要的时间(ms)."""
    return nbytes * 8 / (LINK_GBPS * 1e9) * 1e3


def tx_us(nbytes):
    """同上, 单位微秒 —— 报告里统一用 us, 避免单位换算出错."""
    return tx_time_ms(nbytes) * 1e3


def gen_events(seed=0, bulk_period_ms=BULK_PERIOD_MS, bulk_bursty=False):
    """生成两类流量的到达事件: (到达时间, 类型, 字节数).

    bulk_bursty=False: 集合通信严格周期性到达(纯训练集群的真实形态 —— 每个 step
                       同步一次, 节奏由计算决定, 非常规整).
    bulk_bursty=True : 集合通信按泊松过程到达, 平均速率相同但有突发
                       (弹性调度/多任务混部的形态: 多个训练任务互不协调地共用链路).
    两者**平均带宽完全一样**, 差别只在突发性 —— 而这个差别对尾延迟是决定性的.
    """
    rng = np.random.default_rng(seed)
    ev = []
    # RPC: 泊松到达
    t = 0.0
    while t < SIM_MS:
        t += rng.exponential(1000.0 / RPC_QPS)
        if t < SIM_MS:
            ev.append((t, "rpc", RPC_BYTES))
    # 集合通信
    t = 0.0
    while t < SIM_MS:
        ev.append((t, "bulk", int(BULK_MB * 1e6)))
        t += rng.exponential(bulk_period_ms) if bulk_bursty else bulk_period_ms
    ev.sort()
    return ev


def simulate(events, policy, chunk_kb=None, wrr_weights=(1, 1)):
    """单链路的调度模拟.

    policy:
      'fifo'      —— 单队列先进先出
      'priority'  —— RPC 绝对优先
      'wrr'       —— 加权轮转
      'priority'+chunk_kb —— RPC 优先 + 大包分片

    关键建模点: 一个包一旦开始发送就**不可抢占**(store-and-forward),
    所以大包的发送时长直接决定了后来者的最坏等待时间. 这就是队头阻塞.
    """
    # 用 deque 而不是 list: 分片开启后队列里可能有百万级的片,
    # list.pop(0) 是 O(n), 会把模拟拖成平方复杂度.
    rpc_q, bulk_q = deque(), deque()
    i, n = 0, len(events)
    now = 0.0
    rpc_lat, bulk_done = [], 0.0
    wrr_state = 0

    while True:
        # 入队所有已到达的事件
        while i < n and events[i][0] <= now:
            t, kind, nb = events[i]
            if kind == "rpc":
                rpc_q.append((t, nb))
            else:
                if chunk_kb:                       # 分片: 大包切成小片再入队
                    csz = int(chunk_kb * 1024)
                    for off in range(0, nb, csz):
                        bulk_q.append((t, min(csz, nb - off)))
                else:
                    bulk_q.append((t, nb))
            i += 1

        if not rpc_q and not bulk_q:
            if i >= n:
                break
            now = events[i][0]
            continue

        # ---- 选择下一个要发的包 ----
        if policy == "fifo":
            # 单队列: 谁先到谁先发
            if rpc_q and bulk_q:
                pick = "rpc" if rpc_q[0][0] <= bulk_q[0][0] else "bulk"
            else:
                pick = "rpc" if rpc_q else "bulk"
        elif policy == "priority":
            pick = "rpc" if rpc_q else "bulk"
        elif policy == "wrr":
            # 加权轮转: 按权重轮流服务两个队列
            wr, wb = wrr_weights
            if rpc_q and bulk_q:
                pick = "rpc" if (wrr_state % (wr + wb)) < wr else "bulk"
                wrr_state += 1
            else:
                pick = "rpc" if rpc_q else "bulk"
        else:
            raise ValueError(policy)

        q = rpc_q if pick == "rpc" else bulk_q
        arrive, nb = q.popleft()
        start = max(now, arrive)
        dur = tx_time_ms(nb)
        now = start + dur
        if pick == "rpc":
            rpc_lat.append(now - arrive)
        else:
            bulk_done += nb

    lat = np.array(rpc_lat) if rpc_lat else np.array([0.0])
    return {
        "p50": np.percentile(lat, 50),
        "p99": np.percentile(lat, 99),
        "p999": np.percentile(lat, 99.9),
        "max": lat.max(),
        "n_rpc": len(rpc_lat),
        "bulk_gbps": bulk_done * 8 / (now / 1e3) / 1e9,
        "makespan": now,
    }


def main():
    banner(88)
    print("=" * 88)
    print(f"链路 {LINK_GBPS:.0f}Gbps | RPC: {RPC_QPS:,}QPS x {RPC_BYTES}B "
          f"| 集合通信: 每 {BULK_PERIOD_MS}ms 一个 {BULK_MB}MB 块")
    print("=" * 88)

    # ---- 先算清楚各自要占多少带宽 ----
    rpc_gbps = RPC_QPS * RPC_BYTES * 8 / 1e9
    bulk_gbps = (BULK_MB * 1e6) / (BULK_PERIOD_MS / 1e3) * 8 / 1e9
    print(f"\n[带宽账]")
    print(f"  RPC  需求: {rpc_gbps:>6.2f} Gbps ({rpc_gbps/LINK_GBPS:>5.1%} 链路)")
    print(f"  集合通信  : {bulk_gbps:>6.2f} Gbps ({bulk_gbps/LINK_GBPS:>5.1%} 链路)")
    print(f"  合计      : {rpc_gbps+bulk_gbps:>6.2f} Gbps "
          f"({(rpc_gbps+bulk_gbps)/LINK_GBPS:>5.1%} 链路利用率)")
    print(f"\n  单个 {BULK_MB}MB 大包的发送耗时 = {tx_us(BULK_MB*1e6):>8.1f} us")
    print(f"  单个 {RPC_BYTES}B 小包的发送耗时 = {tx_us(RPC_BYTES):>8.3f} us")
    print(f"  -> 一个大包占住链路的时间, 相当于 "
          f"{tx_us(BULK_MB*1e6)/tx_us(RPC_BYTES):,.0f} 个 RPC 包的发送时间.")
    print("     这就是队头阻塞的量级: RPC 排在大包后面, 就要等这么久.")

    events = gen_events()

    configs = [
        ("FIFO 单队列", dict(policy="fifo")),
        ("优先级(RPC优先)", dict(policy="priority")),
        ("加权轮转 1:1", dict(policy="wrr", wrr_weights=(1, 1))),
        ("加权轮转 4:1", dict(policy="wrr", wrr_weights=(4, 1))),
        (f"优先级+分片{CHUNK_KB*16:g}KB", dict(policy="priority", chunk_kb=CHUNK_KB * 16)),
        (CHUNK_LABEL, dict(policy="priority", chunk_kb=CHUNK_KB)),
    ]

    print(f"\n{'调度策略':<22}{'RPC P50':>11}{'RPC P99':>11}{'RPC P999':>12}"
          f"{'RPC 最大':>11}{'训练吞吐':>12}")
    print("-" * 88)
    results = {}
    for name, kw in configs:
        r = simulate(events, **kw)
        results[name] = r
        print(f"{name:<20}{r['p50']*1e3:>9.1f}us{r['p99']*1e3:>9.1f}us"
              f"{r['p999']*1e3:>10.1f}us{r['max']*1e3:>9.1f}us"
              f"{r['bulk_gbps']:>10.1f}Gbps")

    # ------------------------------------------------------------ 解读
    print("\n" + "-" * 88)
    f, p = results["FIFO 单队列"], results["优先级(RPC优先)"]
    c = results[CHUNK_LABEL]
    bulk_us = tx_us(BULK_MB * 1e6)
    print(f"[1] 队头阻塞: FIFO 下 RPC 的 P999 = {f['p999']*1e3:.1f}us, "
          f"最大 {f['max']*1e3:.1f}us")
    print(f"    而 RPC 包自身只需要 {tx_us(RPC_BYTES):.3f}us 就能发完, "
          f"链路利用率也只有 {(rpc_gbps+bulk_gbps)/LINK_GBPS:.0%}.")
    print(f"    尾延迟几乎精确等于一个大包的发送时间 {bulk_us:.1f}us —— 这不是巧合:")
    print("    RPC 的尾延迟完全由'倒霉撞上一个正在发的大包'决定, 与平均负载无关.")

    print(f"\n[2] 反直觉的结果: 优先级队列 P999 = {p['p999']*1e3:.1f}us, "
          f"相比 FIFO 改善 {(1-p['p999']/f['p999']):.1%} —— 几乎**没有用**.")
    print("    为什么? 因为包一旦开始发送就**不可抢占**(store-and-forward).")
    print("    优先级只能决定'下一个发谁', 决定不了'打断正在发的那个'.")
    print(f"    在 {(rpc_gbps+bulk_gbps)/LINK_GBPS:.0%} 这种低利用率下, 队列里通常只有一个大包,")
    print("    根本没有'下一个'可选 —— 于是优先级完全失效, 尾延迟仍是那 320us.")
    print("    (等到高利用率、队列里堆积多个大包时, 优先级才开始起作用, 见 [4])")

    print(f"\n[3] 真正治本的是分片: 切成 4KB 后 P999 = {c['p999']*1e3:.2f}us, "
          f"比优先级好 {(1-c['p999']/p['p999']):.1%}")
    print(f"    因为不可抢占的时间窗口从 {bulk_us:.1f}us 缩到了 {tx_us(CHUNK_KB*1024):.3f}us,")
    print(f"    整整小了 {bulk_us/tx_us(CHUNK_KB*1024):,.0f} 倍. 尾延迟随之等比例下降.")
    print("    这条结论的一般形式: **尾延迟的下界 = 最大不可抢占单元的传输时间.**")
    print("    想降尾延迟, 就得缩小这个单元, 而不是调整排队顺序.")
    print("    代价: 分片增加包头开销和 CPU 中断次数(本 demo 未建模这部分成本),")
    print("    片太小会让 CPU/网卡成为新瓶颈, 所以片大小本身是个权衡.")

    # ------------------------------------------------------------ 利用率扫描
    print("\n" + "-" * 88)
    print("[4] 利用率推高时会发生什么? 取决于流量的**突发性**, 而不只是平均利用率")
    print("    左半张表: 集合通信严格周期性到达(纯训练集群)")
    print("    右半张表: 集合通信泊松到达(弹性调度混部, 平均带宽完全相同)")
    print(f"\n  {'链路利用率':>10}|{'周期性 FIFO':>13}{'周期性 优先级':>15}|"
          f"{'突发 FIFO':>12}{'突发 优先级':>14}{'突发 分片'+f'{CHUNK_KB:g}KB':>15}")
    print("  " + "-" * 82)
    for period in [2.0, 1.0, 0.5, 0.4, 0.36, 0.345]:
        util = (rpc_gbps + (BULK_MB * 1e6) / (period / 1e3) * 8 / 1e9) / LINK_GBPS
        ev_p = gen_events(bulk_period_ms=period, bulk_bursty=False)
        ev_b = gen_events(bulk_period_ms=period, bulk_bursty=True)
        a1 = simulate(ev_p, policy="fifo")["p999"] * 1e3
        a2 = simulate(ev_p, policy="priority")["p999"] * 1e3
        b1 = simulate(ev_b, policy="fifo")["p999"] * 1e3
        b2 = simulate(ev_b, policy="priority")["p999"] * 1e3
        b3 = simulate(ev_b, policy="priority", chunk_kb=CHUNK_KB)["p999"] * 1e3
        print(f"  {util:>9.0%}|{a1:>11.0f}us{a2:>13.0f}us|"
              f"{b1:>10.0f}us{b2:>12.0f}us{b3:>13.1f}us")

    print("\n  -> 这张表推翻了一个常见的直觉('利用率高 = 尾延迟差'):")
    print("     * 周期性流量下, 即使把利用率推到 93%, P999 依然稳定在一个大包的时间(320us).")
    print("       因为到达是规整的(D/D/1 排队), 队列根本不会堆积.")
    print("     * 突发流量下, 同样的平均利用率, P999 却随利用率显著恶化 ——")
    print("       泊松到达会让多个大包挤在一起, RPC 要等的就不止一个包了.")
    print("\n     所以真正决定尾延迟的是**突发性**, 不是平均利用率. 这恰好解释了")
    print("     表格里为什么单独把'弹性调度场景'列为更难的一类: 把在线集群的空闲资源")
    print("     分给训练, 引入的不只是更高的利用率, 更是**多个互不协调的任务**")
    print("     所带来的突发性 —— 后者才是 SLA 杀手.")
    print("     注意最后一列: 无论突发与否, 分片都把 P999 稳定压在 1us 以内.")

    print("\n" + "=" * 88)
    print("[对应到表格的课题]")
    print("  本 demo 只做了**单点链路**的调度. 表格里的课题要难得多:")
    print("  1. '差异化负载的统一性能建模' —— 要能**预测**上面这张表, 而不是模拟出来;")
    print("     难在异构网络(NVLink/IB/RoCE 带宽差数十倍)和复杂拓扑层级.")
    print("  2. '跨层协同的标准化信令' —— 上面我是**假设**调度器知道哪个包是 RPC.")
    print("     真实系统里网络层根本不知道! 应用层的 QoS 优先级传不到转发层,")
    print("     这个信息断层才是最难的部分.")
    print("  3. '无侵入落地' —— 分片和优先级都要求改协议栈或换交换机配置,")
    print("     而课题要求'不修改业务代码、不更换现有硬件'.")


if __name__ == "__main__":
    main()
