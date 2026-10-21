#!/usr/bin/env python3
"""生成合成振动信号（固定随机种子，可完全复现）。

这个脚本存在的意义：让仓库里的 `synthetic_signal_1025.csv` **可被验证**——
在记录的 NumPy/Pandas 版本下运行本脚本，可得到逐字节相同的 CSV，确认它由数学函数生成，
而不是从真实数据中截取的。

生成规则
--------
- 长度: 1025 个采样点（第 1 行是表头，实际数据 1025 行 → 取前 1024 点作为窗口）
- 分布: 标准正态噪声 N(0, 1)
- 随机源: numpy.random.default_rng(20260930)
- 输出: 单列 CSV，列名 signal

校验
----
运行后应打印与下面一致的 SHA256（已实测，两次运行相同）：

    71ce1eef099ca3e43ba5241c46d0e1e1cf15e52db942027a8107c508122268a9

本版固定 LF 换行。原说明中的 01e3ca0838b935c1257f8c98e56966015578dab6b41132c15bbfe03e644d92ed
对应相同内容的 CRLF 版本。已在 Python 3.12、NumPy 2.3.5、Pandas 2.2.3 下两次复现 LF 版本。

仓库中的 `synthetic_signal_1025.csv` 就是本脚本的输出，**逐字节一致**。

用法
----
    python examples/generate_synthetic_signal.py
    python examples/generate_synthetic_signal.py --out ./check.csv
"""
from __future__ import annotations

import argparse
import hashlib
import os

import numpy as np
import pandas as pd

# ---- 生成参数（改动任何一项都会改变输出，请同时更新此处与 README）----
SEED = 20260930
N_POINTS = 1025
DISTRIBUTION = "normal(0, 1)"
COLUMN_NAME = "signal"


def generate(seed: int = SEED, n: int = N_POINTS) -> np.ndarray:
    """按固定种子生成合成噪声信号。"""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, n).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成合成振动信号（可复现）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--points", type=int, default=N_POINTS)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "synthetic_signal_1025.csv"))
    args = ap.parse_args()

    data = generate(args.seed, args.points)
    pd.DataFrame({COLUMN_NAME: data}).to_csv(args.out, index=False, lineterminator="\n")

    blob = open(args.out, "rb").read()
    print("已生成: %s" % args.out)
    print("  点数     : %d" % args.points)
    print("  分布     : %s" % DISTRIBUTION)
    print("  随机种子 : %d  (numpy.random.default_rng)" % args.seed)
    print("  均值     : %.6f" % data.mean())
    print("  标准差   : %.6f" % data.std())
    print("  csv 字节 : %d" % len(blob))
    print("  csv sha256 = %s" % hashlib.sha256(blob).hexdigest())
    print()
    print("声明：本文件由纯数学函数生成，不含任何真实测量数据。")


if __name__ == "__main__":
    main()
