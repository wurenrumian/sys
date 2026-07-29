#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo3: 拥塞控制 —— 为什么一个算法服务不了两类流量

对应表格: "现有拥塞控制算法要么针对 RPC 低延迟优化(如 DCTCP), 要么针对大模型高吞吐
          优化(如 HPCC、AICC), 缺乏同时满足两类需求的通用算法"

核心矛盾: 拥塞控制要同时追求 (a) 高链路利用率 和 (b) 低队列深度.
         而队列深度 ≈ 排队延迟, 所以 (b) 就是尾延迟.
         填满队列才能确保链路不空闲 -> 两个目标天然冲突.
"""
import numpy as np

np.seterr(all="ignore")

# ---------------------------------------------------------------- 链路参数
LINK_PKT_PER_MS = 100.0     # 瓶颈链路: 每 ms 能发 100 个包
RTT_MS = 1.0                # 基础往返延迟
BDP = LINK_PKT_PER_MS * RTT_MS   # 带宽时延积 = 100 包(在途数据的理想量)
QUEUE_CAP = 500             # 交换机队列容量(包), 超了就丢
SIM_MS = 4000
DCTCP_K = 20                # DCTCP 的 ECN 标记阈值(队列超过 K 就打标)


class Reno:
    """丢包驱动: 加性增、乘性减(AIMD).

    致命弱点: 它**只能靠丢包**感知拥塞, 而丢包意味着队列已经满了.
    所以 Reno 的稳态就是"把队列填满 -> 丢包 -> 减半 -> 再填满", 队列常年很深.
    """
    name = "Reno (丢包驱动)"

    def __init__(self):
        self.cwnd = 10.0

    def on_ack(self, marked, lost):
        if lost:
            self.cwnd = max(2.0, self.cwnd / 2)       # 乘性减
        else:
            self.cwnd += 1.0 / self.cwnd              # 加性增(每 RTT +1)


class DCTCP:
    """ECN 驱动: 交换机队列超过阈值 K 就给包打标记,
    发送方按**标记比例** alpha 温和降速: cwnd *= (1 - alpha/2).

    关键改进: 它在队列**变深之前**就收到信号, 而不是等丢包.
    所以能把队列稳定在 K 附近的低水位 -> 排队延迟低.
    """
    name = "DCTCP (ECN驱动)"

    def __init__(self, g=0.0625):
        self.cwnd = 10.0
        self.alpha = 0.0
        self.g = g

    def on_ack(self, marked, lost):
        # alpha 是被标记比例的指数加权移动平均
        self.alpha = (1 - self.g) * self.alpha + self.g * (1.0 if marked else 0.0)
        if lost:
            self.cwnd = max(2.0, self.cwnd / 2)
        elif marked:
            self.cwnd = max(2.0, self.cwnd * (1 - self.alpha / 2))
        else:
            self.cwnd += 1.0 / self.cwnd


class BBRLike:
    """带宽时延积驱动(BBR 思路的极简版):
    估计瓶颈带宽和最小 RTT, 把在途数据量控制在 BDP 附近, 而不是去填队列.

    理想情况下能同时拿到高利用率和空队列 —— 但代价是要准确估计 BDP,
    估高了就会自己制造队列, 估低了就浪费带宽.
    """
    name = "BBR式 (BDP驱动)"

    def __init__(self, gain_cycle=(1.25, 0.75, 1, 1, 1, 1, 1, 1)):
        self.cwnd = 10.0
        self.bw_est = 1.0          # 每 ms 能发多少包
        self.min_rtt = RTT_MS
        self.cycle = gain_cycle
        self.t = 0

    def on_round(self, delivered_per_ms, rtt):
        self.bw_est = max(self.bw_est * 0.9, delivered_per_ms)  # 取近期最大交付率
        self.min_rtt = min(self.min_rtt * 1.001, rtt)           # 缓慢遗忘
        gain = self.cycle[self.t % len(self.cycle)]
        self.t += 1
        self.cwnd = max(4.0, gain * self.bw_est * self.min_rtt)

    def on_ack(self, marked, lost):
        pass                        # BBR 不响应单个丢包/标记


def simulate(cc_cls, n_flows=4, sim_ms=SIM_MS, k=DCTCP_K):
    """极简的流体近似模拟: 按 ms 推进, 每个 ms 结算一次链路和队列.

    这不是包级仿真, 但足以复现拥塞控制的核心动力学:
    在途数据 > BDP 的部分会堆在队列里, 队列深度直接决定排队延迟.
    """
    flows = [cc_cls() for _ in range(n_flows)]
    queue = 0.0
    hist_q, hist_util, hist_cwnd, drops = [], [], [], 0

    for t in range(sim_ms):
        # 每个流按 cwnd/RTT 的速率注入数据
        offered = sum(f.cwnd for f in flows) / RTT_MS
        served = min(offered + queue, LINK_PKT_PER_MS)     # 链路本 ms 的发送能力
        queue = max(0.0, queue + offered - LINK_PKT_PER_MS)

        lost = queue > QUEUE_CAP
        if lost:
            drops += queue - QUEUE_CAP
            queue = QUEUE_CAP
        marked = queue > k

        for f in flows:
            if isinstance(f, BBRLike):
                # BBR 按"轮"更新: 用实际交付率和当前 RTT(含排队)
                rtt_now = RTT_MS + queue / LINK_PKT_PER_MS
                f.on_round(served / n_flows, rtt_now)
            else:
                f.on_ack(marked, lost)

        hist_q.append(queue)
        hist_util.append(served / LINK_PKT_PER_MS)
        hist_cwnd.append([f.cwnd for f in flows])

    q = np.array(hist_q[sim_ms // 4:])       # 丢掉前 1/4 的启动瞬态
    u = np.array(hist_util[sim_ms // 4:])
    cw = np.array(hist_cwnd[sim_ms // 4:])
    # 公平性: Jain's fairness index, 1.0 = 完全公平
    last = cw[-1]
    jain = last.sum() ** 2 / (len(last) * (last ** 2).sum())
    return {
        "util": u.mean(),
        "q_avg": q.mean(),
        "q_p99": np.percentile(q, 99),
        "delay_avg": RTT_MS + q.mean() / LINK_PKT_PER_MS,
        "delay_p99": RTT_MS + np.percentile(q, 99) / LINK_PKT_PER_MS,
        "drops": drops,
        "jain": jain,
    }


def main():
    print("=" * 84)
    print(f"瓶颈链路 {LINK_PKT_PER_MS:.0f} 包/ms | 基础 RTT {RTT_MS}ms | "
          f"BDP {BDP:.0f} 包 | 队列容量 {QUEUE_CAP} 包")
    print(f"DCTCP 的 ECN 标记阈值 K = {DCTCP_K} 包")
    print("=" * 84)

    print(f"\n{'算法':<20}{'链路利用率':>12}{'平均队列':>11}{'P99队列':>10}"
          f"{'平均延迟':>11}{'P99延迟':>11}{'公平性':>9}")
    print("-" * 84)
    results = {}
    for cls in (Reno, DCTCP, BBRLike):
        r = simulate(cls)
        results[cls.name] = r
        print(f"{cls.name:<18}{r['util']:>12.1%}{r['q_avg']:>10.0f}包"
              f"{r['q_p99']:>9.0f}包{r['delay_avg']:>10.2f}ms"
              f"{r['delay_p99']:>10.2f}ms{r['jain']:>9.3f}")

    print("\n" + "-" * 84)
    print("[核心矛盾] 拥塞控制要同时满足两个冲突的目标:")
    print("  (a) 链路利用率高 —— 要求随时有数据在途, 不让链路空闲")
    print("  (b) 队列深度低   —— 队列深度 / 链路速率 = 排队延迟, 直接决定尾延迟")
    print("  填满队列是确保 (a) 的最简单办法, 但它直接毁掉 (b).")

    rn, dc = results["Reno (丢包驱动)"], results["DCTCP (ECN驱动)"]
    print(f"\n[Reno] 利用率 {rn['util']:.1%} 很高, 但平均队列 {rn['q_avg']:.0f} 包")
    print(f"       -> 排队延迟把 RTT 从 {RTT_MS}ms 抬到了 {rn['delay_avg']:.2f}ms "
          f"({rn['delay_avg']/RTT_MS:.1f} 倍)")
    print("       这就是著名的 **bufferbloat**: 为了不丢包而配的大缓冲区,")
    print("       被丢包驱动的算法当成了'可用容量'填满, 结果延迟灾难.")
    print("       对 RPC 这类小包流量, 这是致命的.")

    print(f"\n[DCTCP] 利用率 {dc['util']:.1%}, 而平均队列只有 {dc['q_avg']:.0f} 包")
    print(f"        -> 平均延迟 {dc['delay_avg']:.2f}ms, "
          f"比 Reno 低 {(1-dc['delay_avg']/rn['delay_avg']):.0%}")
    print("        关键在于它用 ECN 在队列**变深之前**就拿到了拥塞信号,")
    print(f"        并且是按标记比例**温和**降速(而不是 Reno 的粗暴减半).")
    print("        代价: 需要交换机支持 ECN 标记, 且阈值 K 要针对场景调优.")

    # ---------------------------------------------------------- K 的取舍
    print("\n" + "-" * 84)
    print("[DCTCP 的阈值 K 怎么选] —— 这正是'一个算法服务不了两类流量'的具体体现")
    print(f"\n  {'K(包)':>8}{'利用率':>11}{'平均队列':>11}{'P99延迟':>11}   适合谁")
    print("  " + "-" * 58)
    rows = []
    for k in [5, 10, 20, 50, 100, 200]:
        r = simulate(DCTCP, k=k)
        rows.append((k, r))
        fit = "延迟偏高, RPC 吃亏" if r["delay_p99"] > 2 * RTT_MS else "延迟可接受"
        print(f"  {k:>8}{r['util']:>11.1%}{r['q_avg']:>10.0f}包"
              f"{r['delay_p99']:>10.2f}ms   {fit}")

    umin = min(r["util"] for _, r in rows)
    print(f"\n  -> 先说这张表**没有**显示什么: 所有 K 的链路利用率都在 {umin:.0%} 以上.")
    print("     也就是说在这个理想化模型里(单瓶颈、同构 RTT、恒定负载),")
    print("     把 K 调小几乎是**免费**的 —— 队列变浅、延迟变好, 吞吐却没损失.")
    print("     如果只看这个模型, 结论会是'K 越小越好', 那 DCTCP 就没有取舍可言了.")
    print("\n     但现实中 K 不能无限小, 原因是这个流体模型**没有建模**的两件事:")
    print("     1. **突发吸收**: 真实流量是突发的, 浅队列没有缓冲余量, 突发直接变成丢包,")
    print("        而丢包重传的代价远大于排队. 队列的存在本就是为了吸收突发.")
    print("     2. **反馈时延**: ECN 信号要一个 RTT 才能传回发送方. 在这一个 RTT 里")
    print("        队列仍在增长, K 太小会导致来不及反应就已经溢出, 引发剧烈震荡.")
    print("     这两件事都需要包级仿真才能复现 —— 本 demo 的流体近似做不到.")
    print("\n     所以表格里'一个算法服务不了两类流量'的真正含义, 不是'K 调不好',")
    print("     而是: RPC 要浅队列(低延迟), 训练大流要深队列(吸收突发、保吞吐),")
    print("     **同一个队列没法同时既浅又深** —— 这是队列本身的物理约束,")
    print("     不是参数调优能绕过去的. 出路只能是给两类流量分配不同的队列,")
    print("     而这又要求网络能**区分**它们(回到 demo1 和跨层信令的问题).")

    print("\n" + "=" * 84)
    print("[对应到表格的课题]")
    print("  表格提出的解法方向是'通用自适应传输与拥塞控制策略设计',")
    print("  即算法要能'自动识别负载类型, 动态调整拥塞窗口、速率控制、队列调度策略'.")
    print("  拆开来看有三个真问题:")
    print("  1. **怎么识别流量类型**? 包长分布、到达周期性、连接时长都是线索,")
    print("     但要在**线**上实时判断, 且不能误判(误判的代价是 SLA 违约).")
    print("  2. **识别之后怎么隔离**? 光有分类没用, 还要能给不同类流量")
    print("     分配不同的队列/阈值 —— 这需要交换机侧的多队列支持(回到 demo1).")
    print("  3. **信号怎么跨层传递**? 应用层知道'这是延迟敏感的 RPC',")
    print("     但这个信息传不到交换机; 交换机知道'这条链路拥塞了',")
    print("     但这个信息传不回去指导并行策略调整. 这就是'跨层信令'课题.")
    print("\n  注: 本 demo 是流体近似, 不是包级仿真. 它能复现队列深度与利用率的")
    print("  基本动力学和各算法的定性差异, 但不要把具体数值当作真实测量.")


if __name__ == "__main__":
    main()
