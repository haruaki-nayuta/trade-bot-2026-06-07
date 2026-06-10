"""tsmom ヒステリシス(不感帯)つき短期足検証(trend_lab 基盤・FX19 プール)。

シグナル(close のみ・確定バー・先読みなし):
  r   = close/close.shift(lb) - 1
  v   = close.pct_change().rolling(20).std()
  thr = b * v * sqrt(lb)
  ロング建玉: r > thr / 手仕舞い: r < 0。ショートは対称(r < -thr 建玉 / r > 0 手仕舞い)。
  b=0 は素の tsmom(フリップ)。レベルシグナルを from_signals の状態機械に渡すことで
  0 < r < thr の不感帯では「建玉中なら保持・ノーポジなら様子見」のヒステリシスになる。

固定グリッド(追い込み禁止)・計18構成・side=both:
  H1:  lb {24, 48, 120, 360} × b {0, 0.5}
  M30: lb {48, 96, 240}      × b {0, 0.5}
  M15: lb {96, 192}          × b {0, 0.5}

対象: FX19(XAUUSD 除く)。net(コスト計上)と gross(コスト0)の両方を構築。

実行: PYTHONPATH=. uv run python research/experiments/trend2/tsmom_hysteresis_grid.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c/research/lab")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import trend_lab as tl  # noqa: E402


def hyst_tsmom_signals(data: pd.DataFrame, lb: int = 24, b: float = 0.0,
                       vol_window: int = 20):
    """ヒステリシスつき時系列モメンタム。全て確定バーの close のみ使用。"""
    close = data["close"]
    r = close / close.shift(lb) - 1.0
    v = close.pct_change().rolling(vol_window).std()
    thr = b * v * float(np.sqrt(lb))
    long_entries = r > thr
    long_exits = r < 0
    short_entries = r < -thr
    short_exits = r > 0
    return long_entries, long_exits, short_entries, short_exits


GRID = {
    "H1": ([24, 48, 120, 360], [0.0, 0.5]),
    "M30": ([48, 96, 240], [0.0, 0.5]),
    "M15": ([96, 192], [0.0, 0.5]),
}

CONFIGS = [
    (f"hyst_tsmom_{tf}_lb{lb}_b{b:g}", {"lb": lb, "b": b}, tf)
    for tf, (lbs, bs) in GRID.items()
    for lb in lbs
    for b in bs
]


def run_all(instruments: list[str], tag: str) -> pd.DataFrame:
    rows = []
    for label, params, tf in CONFIGS:
        pool = tl.build_pool(hyst_tsmom_signals, params, tf=tf, side="both",
                             instruments=instruments)
        st = tl.pool_stats(pool)
        st["label"], st["tf"], st["params"] = label, tf, str(params)
        rows.append(st)
        print(f"[{tag}] {label}: n={st.get('n')} pf={st.get('pool_pf')} "
              f"mean_bps={st.get('mean_bps')} oos_pf={st.get('oos_pf')}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    fx19 = [i for i in tl.default_instruments() if i != "XAUUSD"]
    print(f"instruments ({len(fx19)}): {fx19}", flush=True)

    # --- net(コスト計上) ---
    net = run_all(fx19, "net")

    # --- gross(コスト0)診断 ---
    tl.register_spreads()
    from fxlab import config
    for k in list(config.SPREADS_PIPS):
        config.SPREADS_PIPS[k] = 0.0
    _orig = tl.register_spreads
    tl.register_spreads = lambda: None
    try:
        gross = run_all(fx19, "gross")
    finally:
        tl.register_spreads = _orig
        tl.register_spreads()  # スプレッドを元に戻す

    g = gross.set_index("label")
    net["gross_mean_bps"] = net["label"].map(g["mean_bps"])
    net["gross_pf"] = net["label"].map(g["pool_pf"])
    net["gross_sum_ret"] = net["label"].map(g["sum_ret"])

    cols = ["label", "tf", "params", "n", "trades_per_year", "sum_ret", "pool_pf",
            "is_pf", "oos_pf", "mean_bps", "gross_mean_bps", "gross_pf",
            "gross_sum_ret", "win_rate", "avg_bars", "yearly_pos", "worst_year"]
    net = net[[c for c in cols if c in net.columns]]
    print("\n==== SUMMARY (net + gross diagnostics) ====", flush=True)
    print(net.to_string(index=False), flush=True)
    out = ("/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c/"
           "research/outputs/trend2_tsmom_hysteresis.csv")
    net.to_csv(out, index=False)
    print(f"\nsaved: {out}", flush=True)


if __name__ == "__main__":
    main()
