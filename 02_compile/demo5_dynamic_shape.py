#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo5: 动态 shape —— 编译器最喜欢静态 shape, 而推荐系统最没有静态 shape

对应表格: "增量编译与热更新机制, 避免全量编译带来的高时间成本" /
          "差异化编译策略(离线重吞吐、在线重延迟)" /
          "轻量化编译产物部署"

前四个 demo 讲的都是"编译能带来多少收益". 这个 demo 讲**收益为什么拿不到**.

编译加速的前提是能为具体的 shape 生成特化代码(循环边界是常数、能展开、能向量化、
能选最优的 tile 大小). 但在线推理的请求里, 用户行为序列长度千变万化 ——
每来一个新长度就是一个新 shape, 就要重新编译一次, 一次几百毫秒.

于是出现了一个典型的系统权衡, 本 demo 把它量化:
    不编译(慢但稳)  vs  每 shape 特化(快但编译风暴)  vs  分桶 padding(折中)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.params import env_int, env_float, banner  # noqa: E402

np.seterr(all="ignore")

N_REQ = env_int("N_REQ", 20_000, "在线请求数")
MAX_LEN = env_int("MAX_LEN", 1024, "序列最大长度")
SIGMA = env_float("SIGMA", 1.2, "长度分布的对数正态 sigma")
COMPILE_MS = env_float("COMPILE_MS", 400.0, "编译一个新 shape 的耗时(ms)")
MS_INTERP = env_float("MS_INTERP", 0.020, "未编译时每个元素的耗时(ms)")
MS_COMPILED = env_float("MS_COMPILED", 0.004, "特化编译后每个元素的耗时(ms)")
DYN_PENALTY = env_float("DYN_PENALTY", 1.5, "动态 shape 编译相对特化编译的减速比")
RELOAD_EVERY = env_int("RELOAD_EVERY", 0, "每多少个请求做一次模型热更新(0=不更新)")

rng = np.random.default_rng(0)


def gen_lengths(n=None):
    n = N_REQ if n is None else n
    x = rng.lognormal(mean=3.2, sigma=SIGMA, size=n)
    return np.clip(x, 1, MAX_LEN).astype(np.int64)


# ---------------------------------------------------------------- 分桶方案
def buckets_pow2():
    """2 的幂分桶: 1,2,4,...,MAX_LEN. 实现最简单, 最坏 padding 浪费 2x."""
    b, x = [], 1
    while x < MAX_LEN:
        b.append(x)
        x *= 2
    b.append(MAX_LEN)
    return b


def buckets_quantile(lens, k):
    """分位数分桶: 按**实际流量分布**切 k 个桶, 让每个桶里的请求数大致相等.

    直觉上这比 2 的幂更聪明 —— 密集的地方桶就密. 是不是真的更好? 见下面的对比.
    """
    qs = np.percentile(lens, np.linspace(0, 100, k + 1)[1:])
    return sorted({int(np.ceil(q)) for q in qs} | {MAX_LEN})


def buckets_geometric(k):
    """几何分桶: 桶边界按等比数列排布(2 的幂是它 k=log2(MAX_LEN) 的特例).

    为什么是等比而不是等距/等频? 因为 padding 浪费是**相对量** ——
    把 10 补到 20 和把 500 补到 1000 都是浪费一倍算力. 要让最坏相对浪费一致,
    桶边界就必须等比排布. 这条推理下面会被数据验证。
    """
    r = MAX_LEN ** (1.0 / k)
    return sorted({int(np.ceil(r ** i)) for i in range(1, k + 1)} | {MAX_LEN})


def bucket_of(x, edges):
    i = int(np.searchsorted(edges, x, side="left"))
    return edges[min(i, len(edges) - 1)]


# ---------------------------------------------------------------- 模拟
def run(lens, mode, edges=None, reload_every=0):
    """返回每个请求的端到端延迟(ms).

    mode:
      'interp'   —— 完全不编译, 直接解释执行. 零编译开销, 但每个元素都慢.
      'exact'    —— 每个出现过的 shape 都特化编译一次, 结果进缓存.
      'bucket'   —— 先把长度向上取整到桶边界再编译, 于是只有 len(edges) 种 shape.
                    代价: padding 到桶边界的那部分算力是白烧的.
      'dynamic'  —— 只编译一份"支持任意长度"的动态 shape 版本(编译一次),
                    但因为循环边界不是常数, 生成的代码比特化版慢 DYN_PENALTY 倍.
    """
    compiled = set()
    lat = np.empty(len(lens), dtype=float)
    n_compile = 0
    compile_ms = 0.0
    padded_total = 0
    for i, L in enumerate(lens):
        if reload_every and i and i % reload_every == 0:
            compiled.clear()          # 模型热更新: 编译缓存整体失效

        if mode == "interp":
            lat[i] = MS_INTERP * L
            padded_total += L
            continue

        if mode == "dynamic":
            key = "*"
            eff = L
            per = MS_COMPILED * DYN_PENALTY
        elif mode == "exact":
            key = int(L)
            eff = L
            per = MS_COMPILED
        else:
            eff = bucket_of(int(L), edges)
            key = eff
            per = MS_COMPILED

        extra = 0.0
        if key not in compiled:
            compiled.add(key)
            n_compile += 1
            compile_ms += COMPILE_MS
            extra = COMPILE_MS       # 编译是同步阻塞的: 这个倒霉的请求要等编译完
        lat[i] = extra + per * eff
        padded_total += eff

    useful = int(lens.sum())
    return {
        "lat": lat,
        "avg": lat.mean(),
        "p50": np.percentile(lat, 50),
        "p99": np.percentile(lat, 99),
        "p999": np.percentile(lat, 99.9),
        "n_compile": n_compile,
        "compile_ms": compile_ms,
        "hit": 1 - n_compile / len(lens),
        "pad_waste": padded_total / useful - 1,
        "n_shape": len(compiled),
    }


def show(tag, r, width=26):
    print(f"  {tag:<{width}}{r['avg']:>9.2f}ms{r['p50']:>9.2f}ms{r['p99']:>10.1f}ms"
          f"{r['p999']:>10.1f}ms{r['n_compile']:>9,}{r['pad_waste']:>10.1%}")


def header(first="策略", width=26):
    print(f"  {first:<{width}}{'平均':>11}{'P50':>11}{'P99':>10}{'P999':>10}"
          f"{'编译次数':>11}{'算力浪费':>11}")
    print("  " + "-" * (width + 62))


def main():
    banner(88)
    lens = gen_lengths()
    print("=" * 88)
    print(f"在线推理 {N_REQ:,} 个请求 | 序列长度 p50={np.percentile(lens,50):.0f} "
          f"p99={np.percentile(lens,99):.0f} max={lens.max()} | "
          f"出现过 {len(set(lens.tolist()))} 种不同长度")
    print(f"成本模型: 编译一个 shape {COMPILE_MS:g}ms | 解释执行 {MS_INTERP:g}ms/元素 | "
          f"编译后 {MS_COMPILED:g}ms/元素 ({MS_INTERP/MS_COMPILED:.0f}x)")
    print("=" * 88)

    pw = buckets_pow2()
    header()
    r_int = run(lens, "interp")
    r_exa = run(lens, "exact")
    r_dyn = run(lens, "dynamic")
    r_p2 = run(lens, "bucket", pw)
    show("不编译(解释执行)", r_int)
    show("每 shape 特化编译", r_exa)
    show("动态 shape 编译一次", r_dyn)
    show(f"2的幂分桶({len(pw)} 个桶)", r_p2)

    print(f"\n  -> **第一个反直觉**: '每 shape 特化'的平均延迟 {r_exa['avg']:.1f}ms 看着不错,")
    print(f"     但它的 P99 是 {r_exa['p99']:.0f}ms, P999 是 {r_exa['p999']:.0f}ms —— "
          f"比不编译还差 {r_exa['p999']/r_int['p999']:.0f} 倍.")
    print(f"     因为它编译了 {r_exa['n_compile']:,} 次, 每次 {COMPILE_MS:g}ms 全部同步砸在")
    print("     某个倒霉请求头上. 平均值把这些完全抹平了, 这正是只看平均值的危险:")
    print("     **编译开销不会消失, 它只是全部集中到了尾延迟里**.")
    print(f"     缓存命中率 {r_exa['hit']:.2%} 听起来很高 —— 但 SLA 是按 P99 签的, 不是按命中率.")

    print(f"\n  -> **第二个反直觉**: 分桶把编译次数从 {r_exa['n_compile']:,} 压到 "
          f"{r_p2['n_compile']}, ")
    print(f"     代价只是 {r_p2['pad_waste']:.1%} 的算力浪费(padding 到桶边界),")
    print(f"     P999 从 {r_exa['p999']:.0f}ms 降到 {r_p2['p999']:.1f}ms.")
    print("     用一点确定的浪费, 换掉一大堆不确定的抖动 —— 这是系统设计里反复出现的模式.")

    # ---------------------------------------------------------- 桶数扫描
    print("\n" + "-" * 88)
    print("[桶数扫描] 桶越多 padding 越省, 但编译次数越多. 最优点在中间")
    header("分桶方案", 26)
    for k in [2, 4, 8, 16, 32, 64]:
        e = buckets_geometric(k)
        show(f"几何分桶 k={k} (实{len(e)}桶)", run(lens, "bucket", e))
    print("  " + "." * 88)
    for k in [4, 16, 64]:
        e = buckets_quantile(lens, k)
        show(f"分位数分桶 k={k} (实{len(e)}桶)", run(lens, "bucket", e))
    print("  " + "." * 88)
    show(f"2的幂分桶 ({len(pw)} 桶)", r_p2)
    show("每 shape 特化(桶数=∞)", r_exa)

    g16 = run(lens, "bucket", buckets_geometric(16))
    q16 = run(lens, "bucket", buckets_quantile(lens, 16))
    print("\n  -> 先看几何分桶那组: 一条典型的 U 形曲线, 两端各有一个失效原因.")
    print("     桶太少 -> padding 浪费主导(算力白烧, 平均延迟上去了);")
    print("     桶太多 -> 编译次数主导(尾延迟被冷启动打爆). 最优点在中间.")
    print(f"\n  -> **第三个反直觉**: '按流量分布切等频桶'听起来明显更聪明, 实测却更差 ——")
    print(f"     同样 16 个桶, 几何分桶浪费 {g16['pad_waste']:.1%}, "
          f"分位数分桶浪费 {q16['pad_waste']:.1%}, 差 {q16['pad_waste']/g16['pad_waste']:.0f} 倍.")
    print("     原因是 padding 浪费是**相对量**: 把 10 补到 20、把 500 补到 1000,")
    print("     都是白烧一倍算力. 等频分桶把桶全堆在了请求密集的短序列区,")
    print("     长尾区只剩一两个巨宽的桶 —— 于是一个长度 400 的请求被补到 1024.")
    print("     而恰恰是长序列贡献了绝大部分算力. 要让最坏相对浪费一致, 桶必须**等比**排布.")
    print("     (2 的幂分桶就是等比的特例, 所以它表现接近最优 —— 老经验往往有它的道理)")
    print("\n     一般结论: 分桶要按**算力分布**优化, 而不是按**请求数分布**.")
    print("     这就是表格里'基于推荐请求特征构建自适应模型'在编译侧的具体含义 ——")
    print("     并且它提醒我们: '自适应'不等于'照着直方图切', 目标函数得先写对.")

    # ---------------------------------------------------------- 热更新
    print("\n" + "-" * 88)
    print("[热更新] 推荐模型天天上线, 而每次上线都会让编译缓存整体失效")
    print(f"\n  {'上线频率':<22}{'平均':>11}{'P99':>11}{'P999':>11}{'编译次数':>11}{'编译总耗时':>13}")
    print("  " + "-" * 78)
    for every, tag in [(0, "不更新(理想)"), (5000, "每 5000 请求一次"),
                       (2000, "每 2000 请求一次"), (500, "每 500 请求一次"),
                       (100, "每 100 请求一次")]:
        r = run(lens, "bucket", pw, reload_every=every)
        print(f"  {tag:<20}{r['avg']:>9.2f}ms{r['p99']:>9.1f}ms{r['p999']:>9.1f}ms"
              f"{r['n_compile']:>11,}{r['compile_ms']/1000:>11.1f}s")
    print("\n  -> 上线越频繁, 编译成本被摊薄的机会越少, 最后编译时间反而成了主要成本.")
    print("     这就是表格里'增量编译与热更新机制, 避免全量编译带来的高时间成本'的由来:")
    print("     推荐系统的迭代速度(一天几次上线)和编译器的假设(编译一次跑很久)是冲突的.")
    print("     可行的方向:")
    print("       * 编译产物**持久化**并跨实例共享(一台机器编译, 全集群复用);")
    print("       * **函数级**缓存 + 稳定 ABI: 只有改动的子图重新编译, 而不是整个模型;")
    print("       * **分层编译**(像 JIT 那样): 先用解释/动态版本顶上, 后台异步编译特化版本,")
    print("         编译好了再切过去 —— 于是编译再也不会阻塞任何一个请求.")

    # ---------------------------------------------------------- 异步编译
    print("\n" + "-" * 88)
    print("[异步编译] 把上面第三条实现出来: 编译不阻塞请求, 先用动态版本顶着")
    async_lat = np.minimum(r_dyn["lat"], r_p2["lat"])   # 编译好之前走 dynamic, 好了走特化
    print(f"  {'方案':<26}{'平均':>11}{'P99':>11}{'P999':>11}")
    print("  " + "-" * 60)
    for tag, l in [("同步编译 + 分桶", r_p2["lat"]), ("动态 shape 兜底", r_dyn["lat"]),
                   ("异步编译(两者取快)", async_lat)]:
        print(f"  {tag:<24}{l.mean():>9.2f}ms{np.percentile(l,99):>9.2f}ms"
              f"{np.percentile(l,99.9):>9.2f}ms")
    print("  -> 异步编译拿到了'动态版本的稳定尾延迟'和'特化版本的平均性能',")
    print("     代价是要同时维护两套产物、以及运行时的版本切换机制.")
    print("     (这里的建模很粗: 假设两个版本随时可切、切换零成本.")
    print("      真实系统里还要处理编译资源竞争、切换时的一致性、以及内存里存两份代码)")

    print("\n" + "=" * 88)
    print("[这个 demo 在整个项目里的位置]")
    print("  01/demo4 说: 不定长序列在**数据侧**造成 padding 浪费, 解法是 Jagged.")
    print("  本 demo 说: 同一个不定长, 在**编译侧**造成的是编译风暴, 解法是分桶.")
    print("  而这两个解法是**互相冲突**的 —— Jagged 消灭了 padding, 却也消灭了静态 shape,")
    print("  于是编译器更难特化. 这个张力是推荐系统 infra 最核心的难题之一,")
    print("  也是为什么不能把 LLM 那套(定长 batch + 静态图)直接搬过来.")


if __name__ == "__main__":
    main()
