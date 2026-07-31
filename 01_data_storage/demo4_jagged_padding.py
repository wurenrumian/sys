#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo4: 不定长序列 —— padding 到底浪费了多少, 以及为什么"排序分桶"几乎是免费的午餐

对应表格: "高效数据格式 / 不定长序列(Jagged Tensor)" (数据中台 + 编译方向都点了这条)

推荐样本和 LLM 样本最大的结构差异: **用户行为序列长短差着两三个数量级**.
一个新用户可能只有 3 个行为, 一个重度用户有 2000 个. 而 GPU 只吃规整的稠密张量,
于是要 padding 到统一长度 —— 短序列的那些空位, 显存、带宽、算力全都要照付.

本 demo 对比四种布局在同一批真实形态的长度分布上的代价, 并实测 padding 的算力浪费.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.params import env_int, env_float, banner  # noqa: E402

np.seterr(all="ignore")

N_USERS = env_int("N_USERS", 200_000, "样本数(用户数)")
MAX_LEN = env_int("MAX_LEN", 1024, "序列最大长度(全局 padding 的目标长度)")
BATCH = env_int("BATCH", 512, "训练 batch size")
SIGMA = env_float("SIGMA", 1.6, "长度分布的对数正态 sigma, 越大长尾越夸张")

rng = np.random.default_rng(0)


def gen_lengths():
    """用户行为序列长度: 对数正态 —— 这是真实推荐场景里长度分布的典型形状.

    多数用户行为很少, 少数重度用户的序列长得离谱, 而且是**连续的长尾**,
    不存在一个"自然的"截断点. 这正是问题的根源.
    """
    x = rng.lognormal(mean=3.0, sigma=SIGMA, size=N_USERS)
    return np.clip(x, 1, MAX_LEN).astype(np.int64)


# ---------------------------------------------------------------- 四种布局
def layout_costs(lens):
    """返回四种布局各自要**分配**多少个元素槽位(useful 之外的都是 padding)."""
    n = len(lens)
    nb = int(np.ceil(n / BATCH))
    useful = int(lens.sum())

    # 1. 全局 padding: 所有样本都补到 MAX_LEN. 最简单, 也最浪费.
    global_pad = n * MAX_LEN

    # 2. 动态 padding: 只补到**本 batch 内**的最大长度. 几乎零成本的改进.
    batches = [lens[i * BATCH:(i + 1) * BATCH] for i in range(nb)]
    dyn_pad = sum(len(b) * b.max() for b in batches)

    # 3. 排序分桶(length bucketing): 先按长度排序再切 batch,
    #    于是同一批里的长度彼此接近, 本批 max 逼近本批 mean.
    #    代价: 破坏了 batch 内的随机性(同一批用户的活跃度高度相关),
    #    工业上的折中是"局部排序" —— 在一个几千样本的窗口内排序后再切批.
    srt = np.sort(lens)
    sb = [srt[i * BATCH:(i + 1) * BATCH] for i in range(nb)]
    bucket_pad = sum(len(b) * b.max() for b in sb)

    # 4. Jagged / 变长布局: values 一维紧密排列 + offsets 索引, 零 padding.
    #    代价是每个 kernel 都要处理不规则边界, 不能直接用现成的稠密算子.
    jagged = useful

    return useful, [("全局 padding (补到 MAX_LEN)", global_pad),
                    ("动态 padding (补到批内 max)", dyn_pad),
                    ("排序分桶 + 动态 padding", bucket_pad),
                    ("Jagged (values+offsets)", jagged)]


# ---------------------------------------------------------------- 实测: 浪费的算力是真的
def measure_pooling(lens):
    """对每个用户的行为序列做一次 sum-pooling, 对比 padded 稠密算子 vs jagged 算子.

    这是推荐模型里最基础的操作(把用户的历史行为聚合成一个向量).
    padded 版本要读整块 (B, L) 的内存, 其中绝大部分是 0.
    """
    b = min(BATCH * 8, len(lens))          # 取若干个 batch 的量, 让计时稳定
    sub = lens[:b]
    L = int(sub.max())

    # ---- padded: (b, L) 稠密矩阵, 大部分是 0 ----
    dense = np.zeros((b, L), dtype=np.float32)
    vals = rng.random(int(sub.sum())).astype(np.float32)
    off = np.concatenate([[0], np.cumsum(sub)])
    for i in range(b):
        dense[i, :sub[i]] = vals[off[i]:off[i + 1]]

    t0 = time.perf_counter()
    for _ in range(5):
        r_dense = dense.sum(axis=1)
    t_dense = (time.perf_counter() - t0) / 5

    # ---- jagged: values 一维 + offsets, 用 reduceat 做分段求和 ----
    starts = off[:-1].astype(np.int64)
    t0 = time.perf_counter()
    for _ in range(5):
        r_jag = np.add.reduceat(vals, starts)
    t_jag = (time.perf_counter() - t0) / 5

    assert np.allclose(r_dense, r_jag, atol=1e-3), "两种布局必须算出同样的结果"
    return b, L, dense.nbytes, vals.nbytes, t_dense, t_jag


def main():
    banner(80)
    print("=" * 80)
    print(f"不定长序列: {N_USERS:,} 个用户, 长度上限 {MAX_LEN}, batch={BATCH}")
    print("=" * 80)

    lens = gen_lengths()
    qs = [50, 75, 90, 95, 99, 99.9]
    print("\n[0] 长度分布 (这是整个问题的根源)")
    print(f"  均值 {lens.mean():.1f}  " +
          "  ".join(f"p{q:g}={np.percentile(lens, q):.0f}" for q in qs) +
          f"  max={lens.max()}")
    print(f"  -> p50 只有 {np.percentile(lens, 50):.0f}, p99 却有 "
          f"{np.percentile(lens, 99):.0f}, 差 {np.percentile(lens,99)/max(1,np.percentile(lens,50)):.0f} 倍.")
    print("     稠密张量的尺寸由**最长的那个样本**决定, 而绝大多数样本用不上.")

    # ---------------------------------------------------------- 1. 四种布局
    useful, rows = layout_costs(lens)
    print("\n[1] 四种布局的槽位分配量 (有效数据 = "
          f"{useful/1e6:.1f}M 个元素)")
    print(f"  {'布局':<30}{'分配槽位':>12}{'放大倍数':>10}{'padding占比':>12}")
    print("  " + "-" * 64)
    for name, alloc in rows:
        print(f"  {name:<28}{alloc/1e6:>10.1f}M{alloc/useful:>10.2f}x"
              f"{1-useful/alloc:>12.1%}")

    g = rows[0][1]
    d = rows[1][1]
    s = rows[2][1]
    print(f"\n  -> **第一个反直觉结论**: 动态 padding 几乎白干 —— 只拿到 {g/d:.2f}x.")
    print(f"     直觉是'补到批内最大值总该比补到全局最大值省不少', 但实测几乎没差别.")
    print(f"     原因是极值统计: 一批 {BATCH} 个样本里, 只要有**一个**接近 {MAX_LEN} 的")
    print("     重度用户, 整批就被撑满. 长尾分布下这件事几乎必然发生 ——")
    print(f"     实测各批最大值的中位数 = {np.median([b.max() for b in [lens[i*BATCH:(i+1)*BATCH] for i in range(int(np.ceil(len(lens)/BATCH)))]]):.0f}, "
          f"而全局最大值 = {lens.max()}.")
    print(f"\n     真正有效的是**按长度排序分桶**: {g/s:.1f}x, "
          f"距离 Jagged 的理论下限只差 {s/useful:.2f}x.")
    print("     因为排序把'一个长样本污染整批'变成了'长样本只和长样本同批'.")
    print("     也就是说: 不改任何 kernel、不引入 Jagged Tensor, 光靠数据侧的排布")
    print("     就能吃掉几乎全部浪费 —— 这是性价比最高的一步, 也是最容易被跳过的一步.")
    print("     代价是 batch 内样本不再独立同分布(同一批全是重度用户), 会影响 BN/采样;")
    print("     工业做法是'局部排序': 在几千样本的窗口内排序后再切批, 兼顾随机性与整齐度.")

    # ---------------------------------------------------------- 2. batch 越大越亏
    print("\n[2] 反直觉: batch 越大, 动态 padding 的浪费**越严重**")
    print(f"  {'batch':>8}{'动态padding放大':>16}{'排序分桶放大':>14}")
    print("  " + "-" * 38)
    for bs in [32, 128, 512, 2048, 8192]:
        nb = int(np.ceil(len(lens) / bs))
        dyn = sum(len(b) * b.max() for b in
                  (lens[i * bs:(i + 1) * bs] for i in range(nb)))
        srt = np.sort(lens)
        buc = sum(len(b) * b.max() for b in
                  (srt[i * bs:(i + 1) * bs] for i in range(nb)))
        print(f"  {bs:>8}{dyn/useful:>15.2f}x{buc/useful:>13.2f}x")
    print("  -> 批越大, 批里出现一个超长序列的概率就越高, 而它会把整批都撑到那么长.")
    print("     这是极值统计: n 个样本的最大值随 n 增长, 长尾分布下增长得还很快.")
    print("     所以'加大 batch 提高 GPU 利用率'这个常识, 在不定长场景下要打折扣 ——")
    print("     真实吃进去的算力有很大一部分是在算 padding. 排序分桶能把这条曲线压平.")

    # ---------------------------------------------------------- 3. 截断
    print("\n[3] 另一条路: 直接截断长序列 (只保留最近 K 个行为)")
    print(f"  {'截断 K':>10}{'保留的行为占比':>16}{'受影响用户':>12}{'相对全局padding':>16}")
    print("  " + "-" * 56)
    total = lens.sum()
    for q in [50, 75, 90, 95, 99, 100]:
        K = int(np.percentile(lens, q))
        kept = np.minimum(lens, K).sum()
        affected = (lens > K).mean()
        tag = f"p{q:g}={K}"
        print(f"  {tag:<10}{kept/total:>15.1%}{affected:>12.1%}"
              f"{K/MAX_LEN:>15.1%}")
    print("  -> 截到 p95 就保住了绝大部分行为 token, 显存却降到很小的比例.")
    print("     但注意: 被截掉的是**重度用户的长期兴趣**, 而重度用户往往贡献了")
    print("     大部分的消费时长和收入. 这个取舍不是纯 infra 决定, 要和算法一起做.")
    print("     这也是为什么大家最终还是要往 Jagged 走: 截断是拿效果换成本,")
    print("     而 Jagged 是不换效果地省成本.")

    # ---------------------------------------------------------- 4. 实测
    b, L, dense_bytes, jag_bytes, t_dense, t_jag = measure_pooling(lens)
    print(f"\n[4] 实测: 对 {b:,} 个序列做 sum-pooling (padded 稠密算子 vs jagged 分段算子)")
    print(f"  {'实现':<24}{'占用内存':>12}{'耗时':>12}{'相对':>10}")
    print("  " + "-" * 58)
    print(f"  {'padded (b x '+str(L)+')':<24}{dense_bytes/1e6:>10.1f}MB"
          f"{t_dense*1e3:>10.2f}ms{1.0:>9.2f}x")
    print(f"  {'jagged (values+offsets)':<24}{jag_bytes/1e6:>10.1f}MB"
          f"{t_jag*1e3:>10.2f}ms{t_dense/t_jag:>9.2f}x")
    print(f"  -> 内存差 {dense_bytes/jag_bytes:.1f}x, 耗时差 {t_dense/t_jag:.1f}x.")
    print("     浪费的不只是显存: padding 位置的 0 也要被读进 cache、也要参与运算,")
    print("     在 memory-bound 的算子上, 浪费的访存带宽就等比例地变成了浪费的时间.")
    print("     (注: reduceat 是 numpy 里的分段归约, 相当于一个最朴素的 jagged kernel;")
    print("      真实的 Jagged Tensor 库还要处理反向传播、多维 embedding、GPU 并行度)")

    # ---------------------------------------------------------- 5. 研究点
    print("\n" + "=" * 80)
    print("[为什么这条课题难]")
    print("  1. **生态不兼容**: 所有现成算子(matmul/attention/norm)都假设稠密规整张量.")
    print("     一旦走 Jagged, 每个算子都要重写一遍, 还要处理反向. 这是巨大的工程量,")
    print("     也是 torchrec / FBGEMM / NestedTensor 这类库存在的理由.")
    print("  2. **和编译对冲**: 变长意味着 shape 动态, 而编译器最喜欢静态 shape.")
    print("     Jagged 省下的访存, 可能被'无法特化编译'吃回去 —— 见 02/demo5.")
    print("  3. **负载不均**: 一个 warp 里各线程处理的序列长度不同, GPU 会等最慢的那个.")
    print("     所以 Jagged kernel 通常还要做负载均衡的重排(如按长度分组调度).")
    print("\n[和其他 demo 的关系]")
    print("  * 01/demo1 (列存): 都是'布局决定 I/O 量'. Jagged 就是列存思想在序列维度上的延伸.")
    print("  * 02/demo3 (显存碎片): 不定长是碎片的根源, 分页是显存侧的解法,")
    print("    Jagged 是张量侧的解法 —— 同一个问题的两个层次.")
    print("  * 02/demo5 (动态 shape): 分桶 padding 在编译侧是同一个权衡的另一面.")


if __name__ == "__main__":
    main()
