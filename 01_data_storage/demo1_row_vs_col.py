#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo1: 行存 vs 列存 —— 训练样本只用一小部分字段时, I/O 量差多少?

对应表格: "负责我们分布式存储与处理栈的核心组件, 涵盖从文件格式、流数据压缩到元数据管理"

模拟一批推荐训练样本(40 个字段), 训练任务只投影其中 3 个字段, 对比:
  1. 扫描字节量   (行存必须读全部, 列存只读投影列)
  2. 反序列化耗时
  3. 压缩比       (列存同质数据熵更低)
"""
import os
import sys
import time
import zlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.params import env_int, banner  # noqa: E402

N_ROWS = env_int("N_ROWS", 200_000, "样本行数")
N_COLS = env_int("N_COLS", 40, "字段数(宽表有多宽)")
N_PROJ = env_int("N_PROJ", 3, "训练实际用到几个字段(投影度)")

# 投影哪几列: 用一个固定步长挑, 保证覆盖到不同类型的字段(时间戳/稀疏ID/枚举/浮点)
PROJECT = sorted({(i * 7 + 3) % N_COLS for i in range(N_PROJ)})

rng = np.random.default_rng(0)


def make_table():
    """构造一张宽表, 模拟真实推荐样本的字段异质性."""
    cols = []
    for c in range(N_COLS):
        if c % 4 == 0:  # 时间戳类: 单调、局部性强 -> 极易压缩
            cols.append((np.arange(N_ROWS) // 100 + 1_700_000_000).astype(np.int64))
        elif c % 4 == 1:  # 稀疏 ID 类: 长尾分布
            cols.append(rng.zipf(1.3, N_ROWS).clip(0, 10_000_000).astype(np.int64))
        elif c % 4 == 2:  # 低基数枚举: 场景/设备/国家
            cols.append(rng.integers(0, 16, N_ROWS).astype(np.int64))
        else:  # 稠密浮点: 统计特征 / embedding 分量
            cols.append((rng.normal(0, 1, N_ROWS) * 1000).astype(np.int64))
    return cols


def as_row_format(cols):
    """行存: 每行的 40 个字段紧挨着放 -> 内存布局是 (N_ROWS, N_COLS)"""
    return np.stack(cols, axis=1).copy(order="C")


def as_col_format(cols):
    """列存: 每列各自连续 -> N_COLS 个独立的一维数组"""
    return [c.copy() for c in cols]


def scan_row(row_tbl, project):
    """行存读投影: 数据在磁盘/网络上无法跳过, 必须把整行搬进来再丢掉不要的字段."""
    touched = row_tbl.nbytes                       # 全表都要过一遍 I/O
    t0 = time.perf_counter()
    out = row_tbl[:, project].sum(axis=0)          # strided 访问, cache 不友好
    return touched, time.perf_counter() - t0, out


def scan_col(col_tbl, project):
    """列存读投影: 只需要读 3 个列文件, 其余列的字节根本不落到 I/O 上."""
    touched = sum(col_tbl[c].nbytes for c in project)
    t0 = time.perf_counter()
    out = np.array([col_tbl[c].sum() for c in project])
    return touched, time.perf_counter() - t0, out


def compress_ratio(buf: bytes) -> float:
    return len(buf) / max(1, len(zlib.compress(buf, 6)))


# ---------------------------------------------------------------- 逐列专用编码
# 列存真正的杀手锏不是"通用压缩器压得更好", 而是**分列之后每列可以各自选最优编码**.
# 行存做不到这一点 —— 一行里各字段类型是混的, 只能一视同仁地丢给 zlib.
# 下面四种编码各有各的适用场景, 后面的表会把这个"各有所长"跑出来.

def enc_raw(col):
    """不编码, 直接存 int64. 作为基线."""
    return col.nbytes


def enc_zlib(col):
    """通用压缩: 什么列都能用, 但对谁都不是最优."""
    return len(zlib.compress(col.tobytes(), 6))


def enc_delta_zlib(col):
    """Delta 编码 + 通用压缩: 只存相邻差值.

    对单调/近似单调的列(时间戳、自增 ID、有序主键)效果极好 ——
    原值域可能是 1.7e9 量级, 差值却只有 0 或 1, 熵接近 0.
    对无序列则毫无用处(差值反而比原值更随机).
    """
    d = np.diff(col, prepend=col[:1])
    return len(zlib.compress(d.tobytes(), 6))


def enc_dict_rle(col):
    """字典编码 + 游程编码(RLE): 值 -> 小整数, 再把连续相同的值折叠成 (值, 次数).

    对低基数枚举列(场景/设备/国家/性别)是量身定做的.
    高基数列上会退化 —— 字典本身就和原数据一样大.
    """
    uniq, codes = np.unique(col, return_inverse=True)
    if len(uniq) > 65535:                    # 基数太高, 字典编码不适用
        return None
    codes = codes.astype(np.uint16)
    # RLE: 找出游程边界
    change = np.flatnonzero(np.diff(codes)) + 1
    runs = len(change) + 1
    dict_bytes = uniq.nbytes                 # 字典本身也要存
    rle_bytes = runs * (2 + 4)               # 每个游程: 2 字节码 + 4 字节长度
    return dict_bytes + min(rle_bytes, codes.nbytes)   # RLE 划不来时就存原始码


def enc_bitpack(col):
    """Bit-packing: 值域已知且小时, 用 ceil(log2(值域)) 位而不是 64 位.

    对低基数、值域紧凑的列有效; 对稀疏 ID 这类值域巨大的列毫无帮助.
    """
    span = int(col.max() - col.min()) + 1
    bits = max(1, int(np.ceil(np.log2(span))))
    return int(np.ceil(len(col) * bits / 8)) + 16      # +16: 存 min 和 bits 的元数据


ENCODINGS = [("原始int64", enc_raw), ("zlib", enc_zlib),
             ("delta+zlib", enc_delta_zlib), ("字典+RLE", enc_dict_rle),
             ("bit-pack", enc_bitpack)]


def main():
    banner(74)
    print("=" * 74)
    print(f"样本: {N_ROWS:,} 行 x {N_COLS} 字段 (int64), 训练只投影字段 {PROJECT}")
    print("=" * 74)

    cols = make_table()
    row_tbl = as_row_format(cols)
    col_tbl = as_col_format(cols)

    # ---------- 1. 投影扫描 ----------
    r_bytes, r_time, r_out = scan_row(row_tbl, PROJECT)
    c_bytes, c_time, c_out = scan_col(col_tbl, PROJECT)
    assert np.array_equal(r_out, c_out), "两种布局必须算出同样的结果"

    print(f"\n[1] 投影扫描 (读 {len(PROJECT)}/{N_COLS} 个字段)")
    print(f"  {'':<10}{'扫描字节':>14}{'耗时(ms)':>12}")
    print(f"  {'行存':<10}{r_bytes/1e6:>12.1f}MB{r_time*1e3:>12.2f}")
    print(f"  {'列存':<10}{c_bytes/1e6:>12.1f}MB{c_time*1e3:>12.2f}")
    print(f"  -> I/O 量降为 {c_bytes/r_bytes:.1%} (理论下限 {len(PROJECT)/N_COLS:.1%}), "
          f"CPU 耗时 {r_time/max(c_time,1e-9):.1f}x 加速")

    # ---------- 2. 压缩率 ----------
    row_cr = compress_ratio(row_tbl.tobytes())
    col_crs = [compress_ratio(col_tbl[c].tobytes()) for c in range(N_COLS)]
    col_total_raw = sum(col_tbl[c].nbytes for c in range(N_COLS))
    col_total_zip = sum(len(zlib.compress(col_tbl[c].tobytes(), 6)) for c in range(N_COLS))

    print("\n[2] 压缩比 (zlib-6, 越大越省钱)")
    print(f"  行存整表          : {row_cr:>6.2f}x")
    print(f"  列存整表          : {col_total_raw/col_total_zip:>6.2f}x")
    print("  列存分列看 (前 4 列, 展示不同字段类型的可压缩性差异):")
    kind = ["时间戳(单调)", "稀疏ID(长尾)", "低基数枚举", "稠密浮点"]
    for c in range(min(4, N_COLS)):
        print(f"    col{c:<3}{kind[c]:<16}{col_crs[c]:>6.2f}x")
    print("  -> 列存把同质数据放一起, 熵更低; 低基数枚举列还能进一步用字典/RLE 编码")

    # ---------- 2b. 逐列专用编码对比 ----------
    print("\n[2b] 分列之后, 每列可以各自选最优编码 (压缩比, 越大越好)")
    print("     行存做不到这件事 —— 一行里各字段类型是混的, 只能一视同仁丢给 zlib.")
    print(f"\n  {'列类型':<16}" + "".join(f"{n:>13}" for n, _ in ENCODINGS) + f"{'最优编码':>14}")
    print("  " + "-" * (16 + 13 * len(ENCODINGS) + 14))
    kind_names = ["时间戳(单调)", "稀疏ID(长尾)", "低基数枚举", "稠密浮点"]
    best_of = {}
    for c in range(min(4, N_COLS)):
        col = col_tbl[c]
        row = f"  {kind_names[c]:<14}"
        best, best_name = 1.0, "-"
        for name, fn in ENCODINGS:
            nb = fn(col)
            if nb is None:
                row += f"{'不适用':>13}"
                continue
            r = col.nbytes / max(1, nb)
            row += f"{r:>11.1f}x"
            if name != "原始int64" and r > best:
                best, best_name = r, name
        best_of[kind_names[c]] = (best_name, best)
        row += f"{best_name:>14}"
        print(row)

    print("\n  -> 四种列类型的最优编码**各不相同**:")
    for k, (name, r) in best_of.items():
        print(f"     {k:<14} -> {name:<12} ({r:.1f}x)")
    print("     这就是列存第二个、也是更值钱的收益: **编码可以按列裁剪**.")
    print("     通用 zlib 是所有列的'及格线', 而专用编码在对的列上能再翻几倍到几十倍.")
    print("     Parquet/ORC 的编码选择器做的就是这件事: 扫一遍列的统计信息")
    print("     (是否有序、基数多少、值域多大), 自动挑一种编码.")
    print("\n     两个跑出来才知道的细节:")
    print("     1. **字典+RLE 在低基数枚举列上输给了 bit-pack**. 因为这里的枚举值是随机排列的,")
    print("        游程长度平均只有 1, RLE 完全没东西可折叠. RLE 的前提是**有序**.")
    print("        这解释了一件真实的工程实践: 数仓会按低基数列做排序/聚簇(sort key、")
    print("        clustering、Z-order) —— 排序本身不改变数据, 却能让 RLE 从失效变成暴击.")
    print("        也就是说, **编码的效果取决于数据的物理顺序**, 而顺序也是可以设计的.")
    print("     2. **稠密浮点对谁都不敏感** —— 近似随机的数据压不动.")
    print("        这也是为什么 embedding 这类稠密向量的存储成本只能靠量化(fp16/int8)去降,")
    print("        而不是靠压缩. 压缩解决不了'信息量本来就大'的问题.")

    # ---------- 3. 存储成本外推 ----------
    print("\n[3] 外推到千亿级/日 (假设单样本 1KB, 1000 亿条/日 = 100 PB/日 原始)")
    for name, cr, io in [("行存", row_cr, 1.0),
                         ("列存", col_total_raw / col_total_zip, len(PROJECT) / N_COLS)]:
        print(f"  {name}: 落盘 {100/cr:>6.1f} PB/日, 一次实验需扫 {100/cr*io:>6.2f} PB")
    print("  -> 存储成本和训练 I/O 同时下降; 一天省的钱 >> 一台训练机的成本")

    print("\n注意: 列存的代价是写入要攒批(column chunk), 批越大压缩越好但数据越不新鲜.")
    print("      流式 Lakehouse(Iceberg/Hudi/Paimon) 的核心矛盾就在这里.")


if __name__ == "__main__":
    main()
