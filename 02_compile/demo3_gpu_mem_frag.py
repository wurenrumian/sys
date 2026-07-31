#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo3: 显存碎片 —— 为什么显存"还剩很多"却 OOM

对应表格: "设计显存碎片实时清理与内存复用策略, 减少推荐大模型推理过程中的显存碎片,
          提升显存利用率; 构建基于显存使用情况与推理任务特征的智能分配模型,
          实现显存的动态按需分配"

推荐场景的特殊性: 用户行为序列**长短不一**, 每个请求要的 KV Cache 大小都不同,
                这是产生外部碎片的最佳温床(LLM 定长 batch 反而没这么严重).

对比 4 种分配器在同一段不定长分配/释放序列上的表现.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.params import env_int, env_float, banner  # noqa: E402

np.seterr(all="ignore")

GPU_MB = env_int("GPU_MB", 4096, "模拟可用于 KV Cache 的显存(MB)")
N_OPS = env_int("N_OPS", 20_000, "分配/释放操作数")
PAGE_MB = env_float("PAGE_MB", 4, "分页分配器的页大小(MB)")
PARETO_A = env_float("PARETO_A", 1.2, "请求大小的 Pareto 指数, 越小尾巴越长")


# ================================================================ 分配器实现
class FreeListAllocator:
    """连续分配器: 维护一个空闲区间列表. first-fit / best-fit 只差在选哪个洞."""

    def __init__(self, total, policy="first"):
        self.total = total
        self.policy = policy
        self.free = [(0, total)]       # (起址, 大小), 保持按地址有序
        self.alloc = {}                # id -> (起址, 大小)
        self.oom = 0
        self.name = f"{'First-Fit' if policy == 'first' else 'Best-Fit'}(连续)"

    def malloc(self, aid, size):
        cand = None
        for i, (off, sz) in enumerate(self.free):
            if sz >= size:
                if self.policy == "first":
                    cand = i
                    break
                if cand is None or sz < self.free[cand][1]:
                    cand = i                       # best-fit: 选最小的够用的洞
        if cand is None:
            self.oom += 1
            return False
        off, sz = self.free[cand]
        self.alloc[aid] = (off, size)
        if sz == size:
            self.free.pop(cand)
        else:
            self.free[cand] = (off + size, sz - size)
        return True

    def free_block(self, aid):
        if aid not in self.alloc:
            return
        off, size = self.alloc.pop(aid)
        self.free.append((off, size))
        self.free.sort()
        merged = [self.free[0]]                    # 合并相邻空闲区间
        for o, s in self.free[1:]:
            po, ps = merged[-1]
            if po + ps == o:
                merged[-1] = (po, ps + s)
            else:
                merged.append((o, s))
        self.free = merged

    def stats(self):
        used = sum(s for _, s in self.alloc.values())
        free_total = self.total - used
        largest = max((s for _, s in self.free), default=0)
        # 外部碎片率: 空闲内存中, 无法用于一次"最大连续申请"的比例
        frag = 1 - largest / free_total if free_total > 0 else 0
        return used, free_total, largest, frag, len(self.free)


class BuddyAllocator:
    """伙伴系统: 把大小向上取整到 2 的幂, 换来 O(log n) 的合并和零外部碎片(块级),
    代价是**内部碎片** —— 申请 33MB 会实际占用 64MB."""

    def __init__(self, total):
        self.order = int(np.ceil(np.log2(total)))
        self.total = 2 ** self.order
        self.free = {o: set() for o in range(self.order + 1)}
        self.free[self.order].add(0)
        self.alloc = {}
        self.oom = 0
        self.name = "Buddy(2的幂)"

    def _order_of(self, size):
        return max(0, int(np.ceil(np.log2(max(1, size)))))

    def malloc(self, aid, size):
        need = self._order_of(size)
        o = need
        while o <= self.order and not self.free[o]:
            o += 1
        if o > self.order:
            self.oom += 1
            return False
        off = self.free[o].pop()
        while o > need:                            # 逐级劈半
            o -= 1
            self.free[o].add(off + 2 ** o)
        self.alloc[aid] = (off, need, size)
        return True

    def free_block(self, aid):
        if aid not in self.alloc:
            return
        off, o, _ = self.alloc.pop(aid)
        while o < self.order:                      # 和伙伴合并
            buddy = off ^ (2 ** o)
            if buddy in self.free[o]:
                self.free[o].remove(buddy)
                off = min(off, buddy)
                o += 1
            else:
                break
        self.free[o].add(off)

    def stats(self):
        used_real = sum(2 ** o for _, o, _ in self.alloc.values())   # 实际占用
        used_req = sum(s for _, _, s in self.alloc.values())         # 申请量
        free_total = self.total - used_real
        largest = max((2 ** o for o in self.free if self.free[o]), default=0)
        frag = 1 - largest / free_total if free_total > 0 else 0
        self.internal = 1 - used_req / used_real if used_real else 0
        return used_real, free_total, largest, frag, sum(len(v) for v in self.free.values())


class PagedAllocator:
    """分页分配器(vLLM PagedAttention 的思路):
    把显存切成固定大小的页, 一次申请 = 一组**不必连续**的页.
    外部碎片被彻底消灭, 只剩下最后一页的内部碎片(平均半页)."""

    def __init__(self, total, page=PAGE_MB):
        self.page = float(page)
        self.npages = int(total / self.page)
        self.total = self.npages * self.page
        self.free = list(range(self.npages))
        self.alloc = {}
        self.oom = 0
        label = f"{page}MB" if page >= 1 else f"{int(page*1024)}KB"
        self.name = f"Paged({label}/页)"

    def malloc(self, aid, size):
        need = int(np.ceil(size / self.page))
        if need > len(self.free):
            self.oom += 1
            return False
        self.alloc[aid] = ([self.free.pop() for _ in range(need)], size)
        return True

    def free_block(self, aid):
        if aid not in self.alloc:
            return
        pages, _ = self.alloc.pop(aid)
        self.free.extend(pages)

    def stats(self):
        used_real = sum(len(p) for p, _ in self.alloc.values()) * self.page
        used_req = sum(s for _, s in self.alloc.values())
        free_total = self.total - used_real
        # 分页没有外部碎片: 任意一次申请都能用上全部空闲页
        largest = free_total
        self.internal = 1 - used_req / used_real if used_real else 0
        return used_real, free_total, largest, 0.0, len(self.free)


# ================================================================ 负载生成
WATERMARK = env_float("WATERMARK", 0.65, "存活数据量占显存的上限水位")


def gen_workload(seed=0, mode="skewed", watermark=None):
    """生成不定长的 KV Cache 分配序列.

    mode='skewed': 序列长度长尾分布(真实推荐场景) -> 大小差异极大, 最易碎片化
    mode='uniform': 长度均匀 -> 对照组

    **关键**: 存活数据总量被限制在显存的 65% 以内. 这样一来, 一个理想的分配器
    应该**永远不 OOM** —— 于是任何 OOM 都只可能是碎片造成的, 而不是真的不够用.
    (如果让负载打满显存, OOM 次数就只反映'谁装得下更多', 无法区分碎片问题.)
    """
    rng = np.random.default_rng(seed)
    cap = GPU_MB * (WATERMARK if watermark is None else watermark)
    ops, live, nid, live_mb = [], [], 0, 0.0
    for _ in range(N_OPS):
        if mode == "skewed":
            # 用户序列长度: 多数很短, 少数极长 (对应 KV Cache 大小)
            size = float(np.clip(rng.pareto(PARETO_A) * 6 + 1, 1, 400))
        else:
            size = float(rng.uniform(1, 60))
        # 只要还在水位以下就分配, 否则释放一个存活块
        if live and (live_mb + size > cap or rng.random() >= 0.6):
            i = int(rng.integers(0, len(live)))
            aid, sz = live.pop(i)
            live_mb -= sz
            ops.append(("f", aid, 0))
        else:
            ops.append(("m", nid, size))
            live.append((nid, size))
            live_mb += size
            nid += 1
    return ops


def run(alloc, ops):
    peak_used, frag_samples = 0, []
    for i, (kind, aid, size) in enumerate(ops):
        if kind == "m":
            alloc.malloc(aid, size)
        else:
            alloc.free_block(aid)
        if i % 200 == 0:
            used, free_t, largest, frag, nfree = alloc.stats()
            peak_used = max(peak_used, used)
            frag_samples.append(frag)
    used, free_t, largest, frag, nfree = alloc.stats()
    return {
        "oom": alloc.oom,
        "peak_used": peak_used,
        "frag_avg": float(np.mean(frag_samples)),
        "frag_end": frag,
        "nfree": nfree,
        "internal": getattr(alloc, "internal", 0.0),
    }


def main():
    banner(84)
    print("=" * 84)
    print(f"显存碎片演示: {GPU_MB}MB 显存, {N_OPS:,} 次不定长分配/释放")
    print("=" * 84)

    for mode, desc in [("skewed", "长尾分布(真实推荐: 用户序列长短差异极大)"),
                       ("uniform", "均匀分布(对照组)")]:
        ops = gen_workload(mode=mode)
        sizes = [s for k, _, s in ops if k == "m"]
        print(f"\n### 负载: {desc}")
        print(f"    申请大小: 中位数 {np.median(sizes):.1f}MB, "
              f"p99 {np.percentile(sizes, 99):.1f}MB, 最大 {max(sizes):.1f}MB")

        print(f"\n  {'分配器':<18}{'OOM次数':>9}{'峰值占用':>10}{'外部碎片率':>12}"
              f"{'内部碎片':>10}{'空闲块数':>10}")
        print("  " + "-" * 76)
        for maker in [lambda: FreeListAllocator(GPU_MB, "first"),
                      lambda: FreeListAllocator(GPU_MB, "best"),
                      lambda: BuddyAllocator(GPU_MB),
                      lambda: PagedAllocator(GPU_MB)]:
            al = maker()
            r = run(al, ops)
            print(f"  {al.name:<18}{r['oom']:>9,}{r['peak_used']:>8.0f}MB"
                  f"{r['frag_avg']:>11.1%}{r['internal']:>10.1%}{r['nfree']:>10,}")

    # ------------------------------------------------------------ 水位扫描
    print("\n" + "=" * 84)
    print("[水位扫描] 显存用到几成时, 碎片开始真正伤人? (长尾负载, OOM 次数)")
    print("  每一列的 OOM 都**不是**因为显存真的不够 —— 存活数据始终在水位以下,")
    print("  一个理想分配器应当全程 0 OOM. 所以下面每一个非零数字都是碎片的代价.")
    makers = [("First-Fit", lambda: FreeListAllocator(GPU_MB, "first")),
              ("Best-Fit", lambda: FreeListAllocator(GPU_MB, "best")),
              ("Buddy", lambda: BuddyAllocator(GPU_MB)),
              ("Paged/4MB", lambda: PagedAllocator(GPU_MB, 4)),
              ("Paged/1MB", lambda: PagedAllocator(GPU_MB, 1)),
              ("Paged/256KB", lambda: PagedAllocator(GPU_MB, 0.25))]
    print(f"\n  {'水位':<7}" + "".join(f"{n:>12}" for n, _ in makers))
    print("  " + "-" * 79)
    for wm in [0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        ops = gen_workload(mode="skewed", watermark=wm)
        row = f"  {wm:<7.0%}"
        for _, mk in makers:
            al = mk()
            row += f"{run(al, ops)['oom']:>12,}"
        print(row)

    print("\n  -> 三件事同时发生, 都值得注意:")
    print("     1. 连续分配器(First/Best-Fit)在 70~80% 水位就开始因**外部碎片** OOM,")
    print("        意味着显存的最后 20~30% 你根本用不上.")
    print("     2. Buddy 最早垮, 因为向上取整到 2 的幂制造了约 30% 的**内部碎片**.")
    print("     3. Paged/4MB 在 80% 以上同样 OOM —— 但原因完全不同: 它没有外部碎片,")
    print("        垮掉是因为 4MB 的页对中位数仅 ~5MB 的请求太粗, 内部碎片高达 ~20%.")
    print("        把页减小到 1MB / 256KB, 内部碎片随之下降, 抗压水位显著提高.")
    print("\n     结论不是'分页万能', 而是: **分页把难以控制的外部碎片, 换成了**")
    print("     **可以用页大小这一个旋钮精确调节的内部碎片.** 这才是它的真正价值 ——")
    print("     从一个无法预测的问题, 变成一个可以定量权衡的问题.")
    print("     (代价: 页越小, 页表越大、间接寻址开销越高 —— 见下面的页大小权衡)")

    # ------------------------------------------------------------ 解读
    print("\n" + "=" * 84)
    print("[怎么读这张表]")
    print("  外部碎片率 = 1 - 最大连续空闲块 / 全部空闲显存")
    print("    它回答的是: '我还剩 1GB 显存, 但我能一次性申请到多大?'")
    print("    这个值越高, 就越容易出现'显存明明够, 却 OOM'的诡异现象.")
    print("  内部碎片 = 1 - 实际申请量 / 实际占用量  (向上取整浪费掉的部分)")
    print()
    print("[结论]")
    print("  1. First-Fit / Best-Fit 都在和外部碎片搏斗. Best-Fit 挑最小的洞,")
    print("     反而制造出大量用不上的小碎屑, 空闲块数量爆炸.")
    print("  2. Buddy 用'向上取整到 2 的幂'换来了快速合并, 但代价是显著的内部碎片 ——")
    print("     申请 33MB 实占 64MB, 这在长尾分布下浪费极大.")
    print("  3. Paged 把外部碎片**直接降为 0**: 因为申请不再要求物理连续,")
    print("     任何一页都能被任何请求使用. 内部碎片只剩最后一页的零头(平均半页).")
    print()
    print("[对推荐场景的意义]")
    print("  - 这正是 vLLM 的 PagedAttention 之所以是范式级改进的原因;")
    print("    它把操作系统里成熟了几十年的虚拟内存分页思想搬进了 KV Cache 管理.")
    print("  - 推荐场景比 LLM **更**需要它: 用户行为序列的长度分布是长尾的(见上表),")
    print("    大小差异越大, 连续分配器的碎片就越严重.")
    print("  - 页大小是个可调参数: 页越小内部碎片越少, 但页表越大、间接寻址开销越高.")
    print("    表格里'构建基于显存使用情况与推理任务特征的智能分配模型'指的就是")
    print("    根据实时负载特征动态选择这类参数.")
    print("  - 分页还顺带解锁了**前缀共享**: 多个请求的公共前缀可以指向同一批物理页,")
    print("    这直接对应电商推荐那条课题里的'用户多刷之间 KvCache 的共享'.")


if __name__ == "__main__":
    main()
