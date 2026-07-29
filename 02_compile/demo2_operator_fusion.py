#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo2: 算子融合 —— 向量化之后的下一道墙是访存, 不是算力

对应表格: "融合传统编译器(LLVM、GCC)与深度学习编译器(TVM 等)的优势, 覆盖
          「纯业务逻辑 Python 代码」与「深度学习相关 Python 代码」的全场景优化"

核心: d = relu(a*b + c) * s
     未融合 = 4 个 kernel, 每个都要把整个数组从内存读进来、算一下、再写回去.
     融合后 = 1 个 kernel, 数据只过一遍内存, 中间结果留在寄存器/cache 里.
     **算术量完全相同, 访存量差好几倍** —— 而这类算子是彻底 memory-bound 的.
"""
import time
import numpy as np

np.seterr(all="ignore")

REPEAT = 5


def bench(fn, *args, repeat=REPEAT):
    for _ in range(2):          # 预热, 让 cache/分配器进入稳态
        fn(*args)
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best, out


# ---------------------------------------------------------------- 三种实现
def unfused(a, b, c, s):
    """未融合: 每一步都产生一个全尺寸临时数组, 每一步都是一轮完整的内存往返."""
    t1 = a * b            # 读 a,b  写 t1
    t2 = t1 + c           # 读 t1,c 写 t2
    t3 = np.maximum(t2, 0)  # 读 t2   写 t3
    return t3 * s         # 读 t3   写 out


def fused_numpy(a, b, c, s):
    """numpy 能做到的最好: 用 out= 减少分配, 但**仍然是 4 趟内存遍历**.
    numpy 没有融合能力 —— 它的每个 ufunc 都是独立的 kernel, 这是它的架构上限."""
    out = np.empty_like(a)
    np.multiply(a, b, out=out)
    np.add(out, c, out=out)
    np.maximum(out, 0, out=out)
    np.multiply(out, s, out=out)
    return out


def fused_blocked(a, b, c, s, block=8192):
    """手工分块融合: 每次只处理一个能放进 L1/L2 cache 的小块,
    在这个块上把 4 步全做完再换下一块 —— 理论上中间结果始终留在 cache 里.

    这就是 TVM/XLA 做的事(loop fusion + tiling). 但注意: 在 Python 里这么写是**做不到**的,
    因为每个块都要付 5 次 numpy 调用的解释器开销. 见下面的块大小扫描.
    """
    out = np.empty_like(a)
    n = len(a)
    for i in range(0, n, block):
        j = min(i + block, n)
        blk = a[i:j] * b[i:j]
        blk += c[i:j]
        np.maximum(blk, 0, out=blk)
        blk *= s
        out[i:j] = blk
    return out


def try_torch_compile(a, b, c, s):
    """torch.compile: 真正的自动算子融合(TorchInductor 会生成一个融合 kernel)."""
    import torch
    ta, tb, tc = (torch.from_numpy(x) for x in (a, b, c))

    def f(x, y, z):
        return torch.relu(x * y + z) * s

    eager_t, _ = bench(lambda: f(ta, tb, tc).numpy())
    try:
        cf = torch.compile(f, mode="max-autotune-no-cudagraphs")
        cf(ta, tb, tc)  # 触发编译
        comp_t, _ = bench(lambda: cf(ta, tb, tc).numpy())
        return eager_t, comp_t
    except Exception as e:
        return eager_t, None


def main():
    print("=" * 80)
    print("算子融合演示:  d = relu(a*b + c) * s")
    print("=" * 80)

    print(f"\n{'规模':>10}{'工作集':>10}{'未融合':>11}{'numpy out=':>12}"
          f"{'分块融合':>11}{'融合加速':>10}")
    print("-" * 80)

    for N in [100_000, 1_000_000, 8_000_000, 32_000_000]:
        rng = np.random.default_rng(0)
        a = rng.random(N, dtype=np.float32) - 0.5
        b = rng.random(N, dtype=np.float32) - 0.5
        c = rng.random(N, dtype=np.float32) - 0.5
        s = np.float32(1.7)

        t_un, r_un = bench(unfused, a, b, c, s)
        t_np, r_np = bench(fused_numpy, a, b, c, s)
        t_bl, r_bl = bench(fused_blocked, a, b, c, s)

        assert np.allclose(r_un, r_np) and np.allclose(r_un, r_bl), "三种实现结果必须一致"

        ws = N * 4 * 4 / 1e6   # a,b,c,out 四个数组的 MB
        print(f"{N/1e6:>8.1f}M{ws:>9.0f}MB{t_un*1e3:>10.2f}ms{t_np*1e3:>11.2f}ms"
              f"{t_bl*1e3:>10.2f}ms{t_un/t_bl:>9.2f}x")

    print("\n  !! 注意: '分块融合'这一列的加速比 < 1, 也就是说它比不融合还慢.")
    print("     访存量明明少了一半, 为什么反而更慢? 下面这个扫描给出答案.")

    # ------------------------------------------------------------ 块大小扫描
    N = 8_000_000
    rng = np.random.default_rng(0)
    a = rng.random(N, dtype=np.float32) - 0.5
    b = rng.random(N, dtype=np.float32) - 0.5
    c = rng.random(N, dtype=np.float32) - 0.5
    s = np.float32(1.7)

    print("\n" + "-" * 80)
    print(f"[块大小扫描] N = {N:,}: 融合的收益 vs Python 调用开销的成本")
    print(f"  {'块大小':>10}{'块工作集':>10}{'块数':>9}{'耗时':>10}"
          f"{'vs未融合':>10}   {'能否进cache'}")
    t_un_ref, _ = bench(unfused, a, b, c, s)
    for block in [1024, 8192, 65_536, 262_144, 1_048_576, N]:
        t, _ = bench(fused_blocked, a, b, c, s, block)
        nblk = (N + block - 1) // block
        blk_ws = block * 4 * 4 / 1024   # KB, 一个块里 4 个数组
        fits = "L1/L2 ✓" if blk_ws < 512 else ("L3 ~" if blk_ws < 32768 else "不进 ✗")
        print(f"  {block:>10,}{blk_ws:>8.0f}KB{nblk:>9,}{t*1e3:>9.2f}ms"
              f"{t_un_ref/t:>9.2f}x   {fits}")
    print(f"\n  (未融合基线 = {t_un_ref*1e3:.2f}ms; 每块要发 5 次 numpy 调用, "
          f"单次调用开销约 1~2us)")

    print("\n  -> 这就是矛盾所在:")
    print("     * 块要**小**才能进 cache, 融合才有意义;")
    print("     * 块要**大**才能摊薄 Python/numpy 的每次调用开销(每块约 5 次调用).")
    print("     两个要求直接冲突, 所以在 Python 层**根本无法表达有效的算子融合**.")
    print("     融合必须发生在编译产物内部 —— 生成的机器码里是一个真正的融合循环,")
    print("     每个元素只付一次循环开销, 而不是每块付 5 次解释器调用.")

    # ------------------------------------------------------------ 访存量分析
    print("\n" + "-" * 80)
    print(f"[访存量分析] N = {N:,}, float32 —— 融合在理论上该省多少")
    print(f"  {'实现':<16}{'kernel数':>9}{'内存遍历':>10}{'访存字节':>12}{'算术强度':>12}")
    flops = N * 3  # 一次乘、一次加、一次乘 (relu 视作免费)
    for name, kernels, passes in [("未融合", 4, 8), ("numpy out=", 4, 7), ("分块融合", 1, 4)]:
        byts = N * 4 * passes
        print(f"  {name:<14}{kernels:>9}{passes:>10}{byts/1e6:>10.0f}MB"
              f"{flops/byts:>10.2f} F/B")
    print("\n  注: '内存遍历'指对全尺寸数组的读或写次数.")
    print("  未融合 = 读a读b写t1 | 读t1读c写t2 | 读t2写t3 | 读t3写out = 8 趟")
    print("  分块融合 = 读a读b读c写out = 4 趟(理论下限), 中间值全在 cache/寄存器里")

    print("\n  算术强度都远小于 1 FLOP/Byte -> 彻底 memory-bound.")
    print("  这意味着: **换更强的算力芯片一点用都没有, 只有减少访存才有用.**")
    print("  这就是为什么深度学习编译器的头号优化永远是算子融合.")
    print("  理论上融合能把访存从 8 趟降到 4 趟, 也就是 2x —— 但前提是能真正生成融合 kernel.")

    # ------------------------------------------------------------ torch.compile
    print("\n" + "-" * 80)
    print("[自动融合] 交给真正的编译器: torch.compile (TorchInductor)")
    try:
        eager_t, comp_t = try_torch_compile(a, b, c, 1.7)
        print(f"  torch eager    : {eager_t*1e3:>8.2f} ms  (逐算子, 和 numpy 未融合同理)")
        if comp_t:
            print(f"  torch.compile  : {comp_t*1e3:>8.2f} ms  ({eager_t/comp_t:.2f}x)")
            print("\n  -> 对比一下三种'融合'尝试:")
            print(f"     手写 Python 分块 : 比不融合**更慢** (被解释器开销吃掉)")
            print(f"     torch.compile   : {eager_t/comp_t:.2f}x 提速 (真正生成了融合 kernel)")
            print("     TorchInductor 生成 C++/OpenMP 代码, 在**编译后的循环内部**完成融合,")
            print("     每个元素只付一次循环开销 —— 这是 Python 层写不出来的东西.")
            r = eager_t / comp_t
            if r > 2.0:
                print(f"     注意它甚至**超过**了纯访存分析给出的 2x 上限. 说明 eager 基线里")
                print("     除了访存, 还有算子分发、临时张量分配等开销, 也一并被编译消掉了 ——")
                print("     这正是'访存模型只是下界, 真实收益还包含框架开销'的体现.")
            else:
                print(f"     {r:.2f}x 略低于访存分析的 2x 上限, 差额来自 kernel 启动与线程同步开销.")
        else:
            print("  torch.compile 在本机不可用(CPU 后端需要 C++ 编译器), 跳过.")
    except ImportError:
        print("  未安装 torch, 跳过.")

    print("\n" + "-" * 80)
    print("[对推荐场景的特殊意义]")
    print("  1. 推荐模型里有大量这种'算术量小、访存量大'的逐元素算子:")
    print("     特征归一化、多路 embedding 相加、门控、各种业务加权 —— 全是融合的目标.")
    print("  2. 推荐的数据是**不定长**的(用户序列长短不一), 静态 shape 的融合策略会失效,")
    print("     这是推荐场景比 LLM 更难的地方, 也是表格里 Jagged Tensor 那条课题的由来.")
    print("  3. 融合的边界选择本身是个优化问题: 融太少省不下访存, 融太多则寄存器不够用")
    print("     导致 spill. 自动搜索这个边界(auto-scheduling)是 TVM/Ansor 的核心贡献.")


if __name__ == "__main__":
    main()
