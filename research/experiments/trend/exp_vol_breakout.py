"""ボラティリティ・ブレイクアウト族の固定グリッド検証(trend_lab 共有基盤)。

構成(固定グリッド、追加探索なし):
  1-4. bb_breakout      : H4/D1 x (period,mult) {(20,2.0),(60,1.5)}  side=both
  5-6. squeeze_breakout : H4/D1 既定 PARAMS (period=20, mult=2.0, squeeze=100)
  7-8. range_squeeze    : close版「レンジ収縮→拡張」 H4/D1

range_squeeze(close ベース、クロスにも正しく適用される):
  - width = (close の20本 rolling max - rolling min) / close
  - 収縮: width < width の過去100本 25%分位 …を .shift(1) で「直前バーが収縮」判定
  - レンジ極値も .shift(1)(自バー不含)。上抜けロング / 下抜けショート
  - 手仕舞い: 20本レンジ中央(prev_high+prev_low)/2 への回帰

実行: PYTHONPATH=. uv run python research/experiments/trend/exp_vol_breakout.py
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c/research/lab")
import trend_lab as tl  # noqa: E402

from strategies.bb_breakout import generate_signals as bb_gen  # noqa: E402
from strategies.squeeze_breakout import generate_signals as sq_gen  # noqa: E402


def range_squeeze_signals(data, period: int = 20, lookback: int = 100, q: float = 0.25):
    """レンジ収縮→拡張ブレイク(close 版)。全て確定バー(.shift(1))で判定。"""
    close = data["close"]
    roll_high = close.rolling(period).max()
    roll_low = close.rolling(period).min()
    width = (roll_high - roll_low) / close

    # 直前バー時点で「収縮」していたか(分位も自バー含み rolling → shift で確定化)
    is_squeeze = (width < width.rolling(lookback).quantile(q)).shift(1)
    is_squeeze = is_squeeze.fillna(False).astype(bool)

    prev_high = roll_high.shift(1)  # 自バーを含まない直近20本高値
    prev_low = roll_low.shift(1)
    mid = (prev_high + prev_low) / 2.0

    long_entries = is_squeeze & (close > prev_high)
    short_entries = is_squeeze & (close < prev_low)
    long_exits = close < mid
    short_exits = close > mid
    return (long_entries.fillna(False), long_exits.fillna(False),
            short_entries.fillna(False), short_exits.fillna(False))


CONFIGS = [
    # label, gen, params, tf
    ("bb_p20m2.0_H4_both", bb_gen, {"period": 20, "mult": 2.0}, "H4"),
    ("bb_p60m1.5_H4_both", bb_gen, {"period": 60, "mult": 1.5}, "H4"),
    ("bb_p20m2.0_D1_both", bb_gen, {"period": 20, "mult": 2.0}, "D1"),
    ("bb_p60m1.5_D1_both", bb_gen, {"period": 60, "mult": 1.5}, "D1"),
    ("squeeze_p20m2.0s100_H4_both", sq_gen, {"period": 20, "mult": 2.0, "squeeze": 100}, "H4"),
    ("squeeze_p20m2.0s100_D1_both", sq_gen, {"period": 20, "mult": 2.0, "squeeze": 100}, "D1"),
    ("rangesq_p20lb100q25_H4_both", range_squeeze_signals, {"period": 20, "lookback": 100, "q": 0.25}, "H4"),
    ("rangesq_p20lb100q25_D1_both", range_squeeze_signals, {"period": 20, "lookback": 100, "q": 0.25}, "D1"),
]


def main() -> None:
    results = []
    for label, gen, params, tf in CONFIGS:
        pool = tl.build_pool(gen, params, tf=tf, side="both")
        st = tl.pool_stats(pool)
        st["label"] = label
        st["tf"] = tf
        st["params"] = json.dumps(params)
        results.append(st)
        print(label, "->", st, flush=True)
    print("\n=== JSON ===")
    print(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
