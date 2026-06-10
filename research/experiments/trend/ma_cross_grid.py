"""移動平均クロス族の固定グリッド検証(trend_lab 共有基盤)。

11構成:
  H4 both: (10,50), (20,100), (50,200)
  D1 both: (5,20), (10,50), (20,100), (50,200)
  W1 both: (4,13), (10,40)
  D1 スロープフィルタ版: (10,50), (20,100)
    — slow MA の直近10本傾きが正のときのみロング許可 / 負のときのみショート許可

実行: PYTHONPATH=. uv run python research/experiments/trend/ma_cross_grid.py
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c/research/lab")
import trend_lab as tl  # noqa: E402

import pandas as pd  # noqa: E402
import vectorbt as vbt  # noqa: E402


def ma_cross(data: pd.DataFrame, fast: int = 20, slow: int = 50):
    """strategies/ma_cross.py と同一ロジック(close のみ使用=クロスでも正しい)。"""
    close = data["close"]
    fast_ma = vbt.MA.run(close, fast).ma
    slow_ma = vbt.MA.run(close, slow).ma
    golden = (fast_ma > slow_ma) & (fast_ma.shift() <= slow_ma.shift())
    dead = (fast_ma < slow_ma) & (fast_ma.shift() >= slow_ma.shift())
    return golden, dead, dead, golden


def ma_cross_slope(data: pd.DataFrame, fast: int = 20, slow: int = 50, slope_bars: int = 10):
    """スロープフィルタ版: slow MA の直近 slope_bars 本の傾きが
    正のときのみロングエントリー許可 / 負のときのみショートエントリー許可。
    判定は確定バーの MA 値のみ(close ベース、先読みなし)。出口は素のクロス。"""
    close = data["close"]
    fast_ma = vbt.MA.run(close, fast).ma
    slow_ma = vbt.MA.run(close, slow).ma
    golden = (fast_ma > slow_ma) & (fast_ma.shift() <= slow_ma.shift())
    dead = (fast_ma < slow_ma) & (fast_ma.shift() >= slow_ma.shift())
    slope = slow_ma.diff(slope_bars)
    long_entries = golden & (slope > 0)
    short_entries = dead & (slope < 0)
    return long_entries, dead, short_entries, golden


GRID = [
    # (label, gen, params, tf)
    ("ma_f10s50_H4_both", ma_cross, {"fast": 10, "slow": 50}, "H4"),
    ("ma_f20s100_H4_both", ma_cross, {"fast": 20, "slow": 100}, "H4"),
    ("ma_f50s200_H4_both", ma_cross, {"fast": 50, "slow": 200}, "H4"),
    ("ma_f5s20_D1_both", ma_cross, {"fast": 5, "slow": 20}, "D1"),
    ("ma_f10s50_D1_both", ma_cross, {"fast": 10, "slow": 50}, "D1"),
    ("ma_f20s100_D1_both", ma_cross, {"fast": 20, "slow": 100}, "D1"),
    ("ma_f50s200_D1_both", ma_cross, {"fast": 50, "slow": 200}, "D1"),
    ("ma_f4s13_W1_both", ma_cross, {"fast": 4, "slow": 13}, "W1"),
    ("ma_f10s40_W1_both", ma_cross, {"fast": 10, "slow": 40}, "W1"),
    ("ma_f10s50_slope10_D1_both", ma_cross_slope, {"fast": 10, "slow": 50, "slope_bars": 10}, "D1"),
    ("ma_f20s100_slope10_D1_both", ma_cross_slope, {"fast": 20, "slow": 100, "slope_bars": 10}, "D1"),
]


def main() -> None:
    results = []
    for label, gen, params, tf in GRID:
        pool = tl.build_pool(gen, params, tf=tf, side="both")
        st = tl.pool_stats(pool)
        st["label"] = label
        st["tf"] = tf
        st["side"] = "both"
        st["params"] = json.dumps(params)
        results.append(st)
        print(json.dumps(st, ensure_ascii=False), flush=True)
    out = pd.DataFrame(results)
    out_path = tl.ROOT / "research" / "outputs" / "trend_ma_cross_grid.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
