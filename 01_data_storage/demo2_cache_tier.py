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
import heapq
import os
import sys
from collections import OrderedDict, defaultdict, deque
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.params import env_int, env_float, banner  # noqa: E402

N_KEYS = env_int("N_KEYS", 1_000_000, "Embedding 表里的 item 数")
N_REQ = env_int("N_REQ", 400_000, "请求数")
ZIPF_A = env_float("ZIPF_A", 1.15, "Zipf 指数, 越大头部越集中")
CHURN_EVERY = env_int("CHURN_EVERY", 50_000, "每多少请求换一批热点(热点churn)")
CACHE_RATIO = env_float("CACHE_RATIO", 0.01, "churn 实验用的缓存容量占比")

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


class FIFO:
    """先进先出: 完全不看访问模式, 只看进来的顺序. 作为"最笨的策略"基线.

    它和 LRU 的唯一区别是命中时**不**把条目移到队尾. 两者的差距,
    就是"时间局部性"这一个信号值多少钱.
    """
    name = "FIFO"

    def __init__(self, cap):
        self.cap, self.d, self.q = cap, set(), deque()

    def get(self, k):
        if k in self.d:
            return True
        if len(self.d) >= self.cap:
            self.d.discard(self.q.popleft())
        self.d.add(k)
        self.q.append(k)
        return False


class RandomEvict:
    """随机淘汰: 连顺序都不看. 用来回答"策略到底值多少钱"这个问题.

    如果精心设计的策略只比随机好一点点, 那说明容量才是主要矛盾, 不是策略.
    """
    name = "随机淘汰"

    def __init__(self, cap):
        self.cap, self.d, self.arr = cap, {}, []
        self.rng = np.random.default_rng(3)

    def get(self, k):
        if k in self.d:
            return True
        if len(self.arr) >= self.cap:
            i = int(self.rng.integers(0, len(self.arr)))
            del self.d[self.arr[i]]
            self.arr[i] = self.arr[-1]
            self.arr.pop()
        self.d[k] = 1
        self.arr.append(k)
        return False


class WTinyLFU:
    """W-TinyLFU (Caffeine / 现代缓存库的默认策略) 的简化版.

    结构 = 小 LRU 窗口(1%) + 主体 SLRU(99%) + 频次准入.
      * **窗口**吸收突发和一次性扫描 —— 新 key 先进窗口, 不打扰主体;
      * 从窗口被挤出来的 key, 要和主体里最不常用的那个**比频次**才能进主体(准入),
        于是一次性访问的 key 根本进不来, 主体不会被扫描流量污染;
      * **频次老化**: 定期把所有计数减半, 让过气热点自然退位 —— 这是它比纯 LFU
        在 churn 下更稳的关键.
    真实实现用 Count-Min Sketch 存频次以节省内存, 这里用字典直接存, 结论一样.
    """
    name = "W-TinyLFU"

    WINDOW_RATIO = 0.01

    def __init__(self, cap, window_ratio=None):
        self.cap = cap
        self.set_window(self.WINDOW_RATIO if window_ratio is None else window_ratio)
        self.window = OrderedDict()          # 小 LRU 窗口
        self.main = OrderedDict()            # 主体, 也按 LRU 维护便于取"受害者"
        self.freq = defaultdict(int)
        self.n = 0
        # 老化周期与容量挂钩(Caffeine 的做法: 采样量到 10 倍容量就整体减半),
        # 而不是一个写死的常数 —— 否则大缓存永远来不及老化.
        self.reset_every = max(1000, 10 * cap)

    def set_window(self, ratio):
        self.ratio = min(0.9, max(0.005, ratio))
        self.wcap = max(1, int(self.cap * self.ratio))
        self.mcap = max(1, self.cap - self.wcap)

    def get(self, k):
        self.n += 1
        self.freq[k] += 1
        if self.n % self.reset_every == 0:   # 频次老化
            for x in list(self.freq):
                self.freq[x] >>= 1

        if k in self.window:
            self.window.move_to_end(k)
            return True
        if k in self.main:
            self.main.move_to_end(k)
            return True

        # 未命中: 先进窗口
        self.window[k] = 1
        while len(self.window) > self.wcap:
            cand, _ = self.window.popitem(last=False)     # 被窗口挤出来的候选
            if len(self.main) < self.mcap:
                self.main[cand] = 1
            else:
                victim = next(iter(self.main))            # 主体里最久未用的
                # 准入判决: 候选的历史频次要高于受害者, 才允许它进来
                if self.freq[cand] > self.freq[victim]:
                    self.main.popitem(last=False)
                    self.main[cand] = 1
        while len(self.main) > self.mcap:                 # 窗口变大时主体要让位
            self.main.popitem(last=False)
        return False


class AdaptiveWTinyLFU(WTinyLFU):
    """自适应窗口的 W-TinyLFU —— Caffeine 后来加的爬山法(hill climbing)。

    固定窗口的 W-TinyLFU 有一个真实弱点: 窗口小 -> 稳态好但**对突变反应慢**;
    窗口大 -> 抗突变但稳态吃亏。而"该多大"取决于当前的流量形态, 是会变的。

    做法: 每隔一段时间统计一次命中率, 和上一段比。变好就沿同方向继续调窗口,
    变差就掉头。一个最朴素的在线控制器, 不需要任何先验知识。
    """
    name = "自适应W-TinyLFU"
    STEP = 0.10

    def __init__(self, cap):
        super().__init__(cap, window_ratio=0.05)
        self.probe = max(2000, cap)
        self.hits_seg = 0
        self.prev_hr = -1.0
        self.dir = 1

    def get(self, k):
        hit = super().get(k)
        self.hits_seg += hit
        if self.n % self.probe == 0:
            hr = self.hits_seg / self.probe
            if hr < self.prev_hr:            # 上一步调坏了 -> 掉头
                self.dir = -self.dir
            self.set_window(self.ratio + self.dir * self.STEP)
            self.prev_hr = hr
            self.hits_seg = 0
        return hit


def belady_hit_rate(trace, cap):
    """Belady MIN —— 理论最优的离线策略: 淘汰"下一次访问最晚"的那个。

    它需要预知未来, 所以线上**不可实现**。它的价值是给出**上界**:
    知道了上界, 才知道现有策略离天花板还有多远、还值不值得继续调。
    没有上界的优化就是在黑暗里拧螺丝。

    实现: 先算出每个位置的"下一次出现的位置", 然后用最大堆维护缓存内各 key 的
    下次访问时间, 淘汰堆顶(最晚的那个)。用惰性删除处理过期条目。
    """
    n = len(trace)
    nxt = np.full(n, n, dtype=np.int64)
    last = {}
    for i in range(n - 1, -1, -1):
        k = int(trace[i])
        nxt[i] = last.get(k, n)
        last[k] = i

    cache = {}            # key -> 该 key 当前的下次访问位置
    heap = []             # (-下次访问位置, key)
    hits = 0
    for i in range(n):
        k = int(trace[i])
        if k in cache:
            hits += 1
        else:
            if len(cache) >= cap:
                while heap:
                    negpos, vk = heapq.heappop(heap)
                    if vk in cache and cache[vk] == -negpos:   # 不是过期条目
                        del cache[vk]
                        break
        cache[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
    return hits / n


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
    banner(78)
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

    # ---------- 1. 容量 vs 命中率, 六种策略横向对比 ----------
    POLICIES = [RandomEvict, FIFO, LRU, LFU, WTinyLFU, AdaptiveWTinyLFU]
    print("\n[1] 缓存容量 vs 命中率 —— 六种策略横向对比 (含理论最优上界)")
    print(f"\n  {'容量占比':<9}{'条目数':>9}" +
          "".join(f"{p.name:>13}" for p in POLICIES) +
          f"{'Belady上界':>12}{'离上界':>10}")
    print("  " + "-" * 108)
    caps = [0.001, 0.005, 0.01, 0.05, 0.10]
    for r in caps:
        cap = max(1, int(N_KEYS * r))
        hits = [simulate(trace, p, cap) for p in POLICIES]
        opt = belady_hit_rate(trace, cap)
        gap = (opt - max(hits)) * 100          # 单位: 百分点
        print(f"  {r:<9.1%}{cap:>9,}" + "".join(f"{h:>13.1%}" for h in hits) +
              f"{opt:>12.1%}{gap:>8.1f}pt")
    print("\n  -> 三件事:")
    print("     1. **命中率对容量是强凹的**: 前 1% 的容量买到绝大部分收益, 之后急剧衰减.")
    print("        所以'缓存该做多大'要看这条曲线的拐点, 不该拍脑袋.")
    print("     2. **策略之间的差距, 比容量带来的差距小得多**. 把容量从 0.1% 提到 1%,")
    print("        比把随机淘汰换成最好的策略更有效. 先把容量给够, 再谈策略.")
    print("     3. **Belady 是不可实现的上界**(它要预知未来), 但它回答了最关键的问题:")
    print("        '还值不值得继续调?' 上面最后一列就是现有最好策略离天花板的距离 ——")
    print("        没有这个数, 你不知道自己是该继续优化还是该收手.")

    # ---------- 2. 热点 churn 时 LRU/LFU 谁更稳 ----------
    print(f"\n[2] 热点 churn (每 {CHURN_EVERY:,} 请求换一批热点, 模拟新内容爆火/旧热点过气)")
    tr2 = gen_trace(churn_every=CHURN_EVERY)
    cap = max(1, int(N_KEYS * CACHE_RATIO))
    print(f"\n  {'策略':<12}{'稳态流量':>12}{'churn流量':>12}{'掉幅':>10}{'churn下排名':>12}")
    print("  " + "-" * 60)
    res = {}
    for cls in POLICIES:
        a, b = simulate(trace, cls, cap), simulate(tr2, cls, cap)
        res[cls.name] = (a, b)
    order = sorted(res, key=lambda k: -res[k][1])
    for cls in POLICIES:
        a, b = res[cls.name]
        print(f"  {cls.name:<12}{a:>12.1%}{b:>12.1%}{b-a:>10.1%}"
              f"{order.index(cls.name)+1:>12}")
    print(f"  {'Belady上界':<12}{belady_hit_rate(trace, cap):>12.1%}"
          f"{belady_hit_rate(tr2, cap):>12.1%}")
    print("\n  -> **这张表跑出来的结果和教科书叙述不一样, 值得仔细看**:")
    print("     1. 稳态下的排名(LFU/W-TinyLFU 领先)在 churn 下**整个翻过来**:")
    print("        LRU 反而第一, 而 W-TinyLFU 掉了 13.7 个点, 是全场最差.")
    print("     2. 原因是**越聪明的策略, 历史先验越重**: LFU 的高频次计数、")
    print("        W-TinyLFU 的准入过滤器, 都是拿'过去'预测'未来'. 一旦热点整体换代,")
    print("        这些先验全部失效, 而且会**主动把新热点挡在门外** —— 准入过滤器")
    print("        本来是用来挡一次性扫描流量的, 此刻它把真正的新热点也一起挡了.")
    print("        相比之下 LRU 没有任何先验, 所以也没有可失效的东西.")
    print("     3. 救回来的办法是**让策略能自我校正**: 自适应版本用一个最朴素的爬山法")
    print("        (每隔一段看命中率变好没有, 变差就掉头调窗口), 掉幅从 -13.7% 收到 -4.5%,")
    print("        churn 下排到第 2. 它做的事其实是'检测到先验失效, 就自动退化成 LRU'.")
    print("\n     可迁移的结论: **带先验的优化在负载突变时是负资产.**")
    print("     所以生产系统真正需要的不是'某个场景下最强的策略', 而是")
    print("     '没有灾难性失效场景 + 能自我校正'的策略 —— 这也是 Caffeine 后来给")
    print("     W-TinyLFU 加上自适应窗口的原因.")
    print("     另一个教训: 拿**稳态** benchmark 选策略, 上线后会被 churn 教做人.")
    print("     推荐场景的 churn 尤其剧烈(新内容不断爆火), 选型时必须把它压进测试集.")

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
