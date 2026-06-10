"""tsmom 族の固定グリッド検証(trend_lab 基盤・19対象プール)。

構成(固定・追い込み禁止):
  D1: lookback {20, 60, 120, 250}  side=both
  H4: lookback {120, 360}          side=both
  W1: lookback {12, 26}            side=both
  D1 lookback {60, 120} ボラ正規化版: |lookbackリターン| / (20本ボラ) > 0.5 のときだけ建玉

シグナルは close ベースのみ(クロスの合成 close にも正しく適用される)。
過去リターン・rolling ボラとも確定バーの値のみ使用(先読みなし)。

実行: PYTHONPATH=. uv run python research/experiments/trend/tsmom_grid.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c/research/lab")

import pandas as pd  # noqa: E402

import trend_lab as tl  # noqa: E402

from strategies.tsmom import generate_signals as tsmom_signals  # noqa: E402


def tsmom_voln_signals(data: pd.DataFrame, lookback: int = 60, vol_window: int = 20,
                       threshold: float = 0.5):
    """ボラ正規化 tsmom: |lookbackリターン|/(20本リターンボラ) > threshold のときだけ建玉。

    条件が外れたら(強度低下 or 符号反転)手仕舞い。全て確定バーの close のみ使用。
    """
    close = data["close"]
    mom = close / close.shift(lookback) - 1.0
    vol = close.pct_change().rolling(vol_window).std()
    strength = mom.abs() / vol
    active = strength > threshold

    long_state = (mom > 0) & active
    short_state = (mom < 0) & active

    long_entries = long_state & ~long_state.shift(fill_value=False)
    long_exits = ~long_state & long_state.shift(fill_value=False)
    short_entries = short_state & ~short_state.shift(fill_value=False)
    short_exits = ~short_state & short_state.shift(fill_value=False)
    return long_entries, long_exits, short_entries, short_exits


CONFIGS = [
    # (label, gen, params, tf)
    ("tsmom_lb20_D1_both", tsmom_signals, {"lookback": 20, "band": 0.0}, "D1"),
    ("tsmom_lb60_D1_both", tsmom_signals, {"lookback": 60, "band": 0.0}, "D1"),
    ("tsmom_lb120_D1_both", tsmom_signals, {"lookback": 120, "band": 0.0}, "D1"),
    ("tsmom_lb250_D1_both", tsmom_signals, {"lookback": 250, "band": 0.0}, "D1"),
    ("tsmom_lb120_H4_both", tsmom_signals, {"lookback": 120, "band": 0.0}, "H4"),
    ("tsmom_lb360_H4_both", tsmom_signals, {"lookback": 360, "band": 0.0}, "H4"),
    ("tsmom_lb12_W1_both", tsmom_signals, {"lookback": 12, "band": 0.0}, "W1"),
    ("tsmom_lb26_W1_both", tsmom_signals, {"lookback": 26, "band": 0.0}, "W1"),
    ("tsmom_voln_lb60_D1_both", tsmom_voln_signals,
     {"lookback": 60, "vol_window": 20, "threshold": 0.5}, "D1"),
    ("tsmom_voln_lb120_D1_both", tsmom_voln_signals,
     {"lookback": 120, "vol_window": 20, "threshold": 0.5}, "D1"),
]


def main() -> None:
    rows = []
    for label, gen, params, tf in CONFIGS:
        pool = tl.build_pool(gen, params, tf=tf, side="both")
        st = tl.pool_stats(pool)
        st["label"], st["tf"], st["params"] = label, tf, params
        rows.append(st)
        print(f"\n=== {label} (tf={tf}, params={params}) ===")
        for k, v in st.items():
            if k not in ("label", "tf", "params"):
                print(f"  {k}: {v}")

    df = pd.DataFrame(rows)
    cols = ["label", "tf", "n", "trades_per_year", "sum_ret", "pool_pf", "is_pf",
            "oos_pf", "is_sum", "oos_sum", "mean_bps", "win_rate", "avg_bars",
            "yearly_pos", "worst_year"]
    df = df[[c for c in cols if c in df.columns]]
    print("\n\n==== SUMMARY ====")
    print(df.to_string(index=False))
    out = "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c/research/outputs/tsmom_grid.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
