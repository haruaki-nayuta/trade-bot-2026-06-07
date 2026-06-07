"""イテレーション17: confluence(毎年プラス) + ペアトレード(高PF) の2スリーブ合成。

confluence_meanrev は毎年プラス100%だがPF1.83。ペアトレード(exp16)はPF中央2.05だが82%年。
両者はリターン源(方向性 vs スプレッド)が異なり悪い年も異なる。等加重で合成すれば、
confluence が毎年プラスを担保しつつ、ペアトレードの高PFトレードが集計PFを押し上げ、
取引数も合算で増える可能性がある(非相関スリーブ合成=非カーブフィット)。

全トレードをリターン率(割合)に正規化して等加重プールし、決済年でPF/毎年プラス/取引を集計。
実行: uv run python exp17.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import universe as uni
from fxlab.backtest import run
from fxlab.trades import trade_table
from strategies.confluence_meanrev import generate_signals as g
from exp16 import pair_trades, PAIRS

TF = "H4"
P = {"window": 50, "entry_z": 2.0, "exit_z": 0.5, "rsi_p": 14, "rsi_low": 35, "rsi_high": 65,
     "vol_win": 100, "vol_pct": 0.70, "slow_win": 250, "slow_z": 1.75}


def confluence_returns():
    """confluence champion の全トレードを (year, ret割合) で返す。"""
    uni.register_cross_spreads(3.0)
    out = []
    for nm in [x for x in uni.universe() if x != "AUDJPY"]:
        data = uni.instrument_data(nm, TF)
        tt = trade_table(run(nm, TF, g, P, data=data, size_mode="value"), data)
        for _, r in tt.iterrows():
            out.append((pd.Timestamp(r["exit"]).year, r["return_pct"] / 100.0))
    return out


def pairs_returns(entry=2.5, exit=0.5):
    out = []
    for a, b in PAIRS:
        out += pair_trades(a, b, entry=entry, exit=exit)
    return out


def summarize(label, trades):
    df = pd.DataFrame(trades, columns=["year", "ret"])
    rows = {}
    for y, gd in df.groupby("year"):
        gp = gd["ret"][gd["ret"] > 0].sum(); gl = -gd["ret"][gd["ret"] < 0].sum()
        rows[int(y)] = {"trades": len(gd), "pf": gp/gl if gl > 0 else np.inf, "ret%": gd["ret"].sum()*100}
    t = pd.DataFrame(rows).T
    pf = t["pf"].replace(np.inf, np.nan)
    star = " ★達成" if (t["ret%"] > 0).all() and pf.median() >= 2.0 and t["trades"].mean() >= 100 else ""
    print(f"\n=== {label}{star} ===")
    print(t.round(2).to_string())
    print(f"  → 毎年プラス {(t['ret%']>0).mean():.0%} / PF中央 {pf.median():.2f} / "
          f"PF最小 {pf.min():.2f} / 年取引 {int(t['trades'].mean())}")
    return t


def main():
    conf = confluence_returns()
    print(f"confluence トレード数: {len(conf)} (年{len(conf)/11:.0f})")
    summarize("confluence のみ", conf)
    for e in [2.0, 2.5]:
        pr = pairs_returns(entry=e)
        summarize(f"ペアトレードのみ (entry={e})", pr)
        summarize(f"合成: confluence + ペアトレード(entry={e})", conf + pr)


if __name__ == "__main__":
    main()
