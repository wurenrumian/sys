#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo2: 多级存储 + Embedding Cache —— 长尾分布下, 缓存到底值多少钱?

对应表格:
  - 数据中台: "通过对数据湖、缓存、分布式计算和 GPU IO 的协同优化"
  - TikTok:  "设计高性能的分布式特征索引缓存(Embedding Cache), 减少 RDMA 远程访问次数"
  - TikTok:  "Sparse/Dense 参数多级存储流水线协同: 模拟并优化从冷存储到 HBM 之间的多级 Buffer 穿透率"

结论预览: 推荐场景的访问是极度 Zipf 的, 1% 容量的缓存就能吃掉大半流量;
         而存储层级之间的延迟差是数量级的, 所以"提升命中率"永远比"让某层更快"划算.
"""
from collections import OrderedDict, defaultdict
import numpy as np

N_KEYS = 1_000_000       # Embedding 表里的 item 数
N_REQ = 400_000          # 请求数
ZIPF_A = 1.15            # 推荐场景实测的头部集中度大致在这个量级

# 存储金字塔: (名字, 单次访问延迟 us, 单位容量相对成本)
TIERS = [
    ("GPU HBM",      0.2,   100.0),
    ("本机 DRAM",     2.0,    10.0),
    ("远程 DRAM/RDMA", 10.0,   8.0),
    ("本地 NVMe",    100.0,    1.0),
    ("对象存储",   10_000.0,   0.05),
]


def gen_trace(churn_every=None):
    """生成访问序列.

    churn_every 不为空时, 每隔这么多请求就把 key 空间重新映射一次, 模拟推荐场景里
    真实存在的"热点churn": 新视频不断爆火、旧热点迅速过气(大促/春晚/热搜更迭).
    注意分布形状不变, 变的是"谁是热点" —— 这正是缓存策略要应对的.
    """
    rng = np.random.default_rng(7)
    keys = (rng.zipf(ZIPF_A, N_REQ) % N_KEYS).astype(np.int64)
    if churn_every:
        for seg, start in enumerate(range(0, N_REQ, churn_every)):
            if seg == 0:
                continue
            offset = int(rng.integers(1, N_KEYS))  # 每段换一批热点
            end = min(start + churn_every, N_REQ)
            keys[start:end] = (keys[start:end] + offset) % N_KEYS
    return keys


class LRU:
    name = "LRU"

    def __init__(self, cap):
        self.cap, self.d = cap, OrderedDict()

    def get(self, k):
        if k in self.d:
            self.d.move_to_end(k)
            return True
        if len(self.d) >= self.cap:
            self.d.popitem(last=False)
        self.d[k] = 1
        return False


class LFU:
    """近似 LFU: 频次计数 + 定期老化 + 采样淘汰.

    工业实现(Redis allkeys-lfu / Caffeine 的 TinyLFU)都不会做全量 min 扫描 ——
    那是 O(容量) 的. 这里用 Redis 的做法: 随机采样 K 个候选, 淘汰其中频次最低的.
    K=8 时就已经非常接近精确 LFU 的效果, 而单次淘汰是 O(K).
    """
    name = "LFU"
    SAMPLE = 8

    def __init__(self, cap):
        self.cap = cap
        self.d = {}                     # key -> 1, 同时用 list 维护可采样的 key 集合
        self.keys = []                  # 采样池(允许有已删除的陈旧项, 惰性清理)
        self.freq = defaultdict(int)
        self.n = 0
        self.rng = np.random.default_rng(11)

    def get(self, k):
        self.n += 1
        self.freq[k] += 1
        if self.n % 200_000 == 0:       # 老化, 否则永远淘汰不掉过气热点
            for x in list(self.freq):
                self.freq[x] >>= 1
        if k in self.d:
            return True
        if len(self.d) >= self.cap:
            self._evict()
        self.d[k] = 1
        self.keys.append(k)
        return False

    def _evict(self):
        best, best_f = None, None
        tries = 0
        while len(self.keys) and tries < self.SAMPLE * 4:
            i = int(self.rng.integers(0, len(self.keys)))
            cand = self.keys[i]
            if cand not in self.d:                       # 陈旧项, 惰性剔除
                self.keys[i] = self.keys[-1]
                self.keys.pop()
                continue
            f = self.freq[cand]
            if best_f is None or f < best_f:
                best, best_f = cand, f
            tries += 1
            if tries >= self.SAMPLE:
                break
        if best is not None:
            del self.d[best]


def simulate(trace, policy_cls, cap):
    p = policy_cls(cap)
    hits = sum(1 for k in trace if p.get(int(k)))
    return hits / len(trace)


def effective_latency(hit_rates):
    """端到端平均延迟.

    hit_rates[i] = 在第 i 层命中的请求 **占总流量的绝对比例**(不是条件概率),
    与 TIERS 前若干层一一对应; 剩下的 1-sum(hit_rates) 即"穿透率", 落到最后一层兜底.
    """
    assert sum(hit_rates) <= 1.0 + 1e-9, "各层命中率之和不能超过 1"
    lat = sum(h * l for (_, l, _), h in zip(TIERS, hit_rates))
    lat += (1.0 - sum(hit_rates)) * TIERS[-1][1]
    return lat


def main():
    print("=" * 78)
    print(f"Embedding 表 {N_KEYS:,} key, 请求 {N_REQ:,} 条, Zipf(a={ZIPF_A}) 长尾分布")
    print("=" * 78)

    trace = gen_trace()
    uniq, cnt = np.unique(trace, return_counts=True)
    order = np.argsort(-cnt)
    top1pct = cnt[order][:max(1, len(uniq) // 100)].sum() / len(trace)
    print(f"\n[0] 分布有多偏: 访问过的 key 共 {len(uniq):,} 个, "
          f"其中最热的 1% 承接了 {top1pct:.1%} 的请求")
    print("    -> 这就是缓存在推荐场景性价比极高的根本原因")

    # ---------- 1. 容量 vs 命中率 ----------
    print("\n[1] 缓存容量 vs 命中率 (缓存条目数 / 全表)")
    print(f"  {'容量占比':<10}{'条目数':>10}{'LRU':>10}{'LFU':>10}")
    caps = [0.001, 0.005, 0.01, 0.05, 0.10]
    for r in caps:
        cap = max(1, int(N_KEYS * r))
        print(f"  {r:<10.1%}{cap:>10,}{simulate(trace, LRU, cap):>10.1%}"
              f"{simulate(trace, LFU, cap):>10.1%}")
    print("  -> 命中率对容量是强凹的: 前 1% 的容量买到了绝大部分收益, 之后急剧衰减")

    # ---------- 2. 热点 churn 时 LRU/LFU 谁更稳 ----------
    print("\n[2] 热点 churn (每 5 万请求换一批热点, 模拟新内容爆火/旧热点过气)")
    tr2 = gen_trace(churn_every=50_000)
    cap = int(N_KEYS * 0.01)
    print(f"  {'策略':<8}{'稳态流量':>12}{'churn流量':>12}{'掉幅':>10}")
    res = {}
    for cls in (LRU, LFU):
        a, b = simulate(trace, cls, cap), simulate(tr2, cls, cap)
        res[cls.name] = (a, b)
        print(f"  {cls.name:<8}{a:>12.1%}{b:>12.1%}{b-a:>10.1%}")
    print("  -> 稳态下 LFU 更强(记得住长期热点); churn 下 LFU 的优势被吃掉甚至反转,")
    print("     因为陈旧的高频次计数会挡住新热点进入缓存(cache pollution).")
    print("     工业方案 W-TinyLFU: 小 LRU 窗口吸收突发 + 主体 TinyLFU 保稳态 + 计数老化.")

    # ---------- 3. 有效延迟: 命中率 vs 单层提速 ----------
    print("\n[3] 端到端有效延迟 —— '提命中率' 和 '让某层变快' 哪个更值?")
    print(f"  {'层级':<16}{'延迟(us)':>10}{'相对HBM':>10}")
    for name, l, _ in TIERS:
        print(f"  {name:<16}{l:>10.1f}{l/TIERS[0][1]:>9.0f}x")

    def compare(base, tag):
        """同样的工程投入, 花在不同地方的回报对比."""
        l0 = effective_latency(base)
        leak = 1 - sum(base)
        # 投入一: 上层调优 —— 把 1 个百分点的流量从 DRAM 提到 HBM
        plus_hbm = effective_latency([base[0] + 0.01, base[1] - 0.01] + base[2:])
        # 投入二: 硬件升级 —— NVMe 层延迟降到 1/10
        backup = TIERS[3]
        TIERS[3] = (backup[0], backup[1] / 10, backup[2])
        faster = effective_latency(base)
        TIERS[3] = backup
        # 投入三: 兜底 —— 把穿透到对象存储的流量砍一半, 由 NVMe 层接住
        less_leak = effective_latency(base[:3] + [base[3] + leak / 2])
        print(f"\n  {tag}   穿透到对象存储 = {leak:.3%}, 基线有效延迟 = {l0:.2f} us")
        print(f"    {'投入':<26}{'延迟(us)':>10}{'改善':>10}")
        print(f"    {'上层调优: HBM +1pt':<26}{plus_hbm:>10.2f}{(l0-plus_hbm)/l0:>10.1%}")
        print(f"    {'硬件升级: NVMe 提速 10x':<24}{faster:>10.2f}{(l0-faster)/l0:>10.1%}")
        print(f"    {'压穿透: 穿透率减半':<25}{less_leak:>10.2f}{(l0-less_leak)/l0:>10.1%}")
        return l0

    compare([0.30, 0.40, 0.20, 0.08], "场景A 穿透率高:")
    l_good = compare([0.30, 0.40, 0.20, 0.0995], "场景B 穿透率低:")

    print("\n  -> 场景A 里穿透率虽只有 2%, 却贡献了绝大部分平均延迟, 此时上层调优和硬件升级")
    print("     基本是噪声, 唯一有效的动作是压穿透率; 等穿透率压到万分之几(场景B),")
    print("     压穿透的边际收益迅速衰减, 上层的命中率分布才开始成为主要矛盾.")
    print("     这就是 TikTok 那条课题写'优化多级 Buffer 穿透率'而不是'把某层做快'的原因 ——")
    print("     优化顺序是被这条延迟-穿透率曲线决定的, 不是拍脑袋定的.")

    # ---------- 4. 成本 ----------
    print("\n[4] 分层的成本意义 (相对单位成本; 200TB Sparse 参数不可能全放 HBM)")
    dist = [0.005, 0.05, 0.15, 0.3, 1.0]  # 各层容量占全表比例
    total = sum(d * c for (_, _, c), d in zip(TIERS, dist))
    print(f"  {'层级':<16}{'容量占比':>10}{'相对成本':>12}")
    for (name, _, c), d in zip(TIERS, dist):
        print(f"  {name:<16}{d:>10.1%}{d*c:>12.3f}")
    print(f"  {'合计':<16}{'':>10}{total:>12.3f}   (全量放 HBM = {TIERS[0][2]:.1f})")
    print(f"  -> 存储成本降到全 HBM 方案的 {total/TIERS[0][2]:.2%};")
    print(f"     在低穿透率下(场景B), 代价是有效延迟 {l_good:.1f}us vs 全 HBM 的 {TIERS[0][1]:.1f}us.")
    print(f"     即: 成本降到 1/{TIERS[0][2]/total:.0f}, 换来延迟涨 {l_good/TIERS[0][1]:.0f} 倍.")
    print("     分层存储的全部设计工作, 就是在这条兑换曲线上找业务能接受的那个点.")


if __name__ == "__main__":
    main()
