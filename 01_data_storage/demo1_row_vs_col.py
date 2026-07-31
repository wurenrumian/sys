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
