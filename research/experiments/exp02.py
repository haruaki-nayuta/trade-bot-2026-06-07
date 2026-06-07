"""イテレーション2 実験: 平均回帰の磨き込み / 非対称出口 / 複数戦略の非相関合成。

目的:
  (1) レンジフィルタ(meanrev_range)が素の rsi_meanrev を改善するか
  (2) トレーリング/リスク%サイジングで PF・年次安定が上がるか
  (3) 平均回帰 + モメンタム の合成ポートフォリオが「毎年プラス」に近づくか

実行: uv run python exp02.py
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from fxlab import yearly as ylib
from fxlab.data import available_pairs

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)

PAIRS = available_pairs()


def _gen(name):
    m = importlib.import_module(f"strategies.{name}")
    return m.generate_signals, getattr(m, "PARAMS", {})


def combo_yearly(specs, tf):
    """複数の (戦略名, params, run_kw) を独立に回し、決済年で合算したポートフォリオ年次成績。"""
    accum = {}
    for name, params, kw in specs:
        gen, _p = _gen(name)
        params = params or _p
        for pair in PAIRS:
            try:
                y = ylib.yearly(pair, tf, gen, params, **kw)
            except Exception:  # noqa: BLE001
                continue
            for year, r in y.iterrows():
                a = accum.setdefault(int(year), [0.0, 0.0, 0.0, 0.0])
                a[0] += r["gross_profit"]; a[1] += r["gross_loss"]
                a[2] += r["pnl"]; a[3] += r["trades"]
    rows = {}
    for year, (gp, gl, pnl, trades) in sorted(accum.items()):
        rows[year] = {
            "trades": int(trades),
            "profit_factor": round(gp / gl, 2) if gl > 0 else float("inf"),
            "pnl": round(pnl, 0),
            "positive": pnl > 0,
        }
    return pd.DataFrame(rows).T


def summarize(label, port):
    pf = port["profit_factor"].replace(np.inf, np.nan)
    pos = (port["pnl"] > 0).mean() if "pnl" in port else np.nan
    print(f"\n--- {label} ---")
    print(port.to_string())
    print(f"  → プラス年率 {pos:.0%} / PF中央値 {pf.median():.2f} / PF最小 {pf.min():.2f} / 年平均取引 {int(port['trades'].mean())}")


def main():
    tf = "H4"
    print(f"通貨ペア: {PAIRS}  時間足: {tf}\n")

    # (1) レンジフィルタの効果
    for name in ["rsi_meanrev", "meanrev_range", "rsi2_pullback"]:
        port = combo_yearly([(name, None, {"size_mode": "value"})], tf)
        summarize(f"{name}  (value)", port)

    # (2) トレーリング / リスク%サイジング
    summarize("meanrev_range + TSL2%  (value)",
              combo_yearly([("meanrev_range", None, {"size_mode": "value", "tsl_stop": 0.02})], tf))
    summarize("meanrev_range  (risk1%)",
              combo_yearly([("meanrev_range", None, {"size_mode": "risk", "size_value": 0.01})], tf))

    # (3) 合成ポートフォリオ: 平均回帰(レンジ) + モメンタム(トレンド)
    summarize("COMBO: meanrev_range + tsmom  (value)",
              combo_yearly([("meanrev_range", None, {"size_mode": "value"}),
                            ("tsmom", None, {"size_mode": "value"})], tf))
    summarize("COMBO: rsi_meanrev + tsmom + breakout_trend  (value)",
              combo_yearly([("rsi_meanrev", None, {"size_mode": "value"}),
                            ("tsmom", None, {"size_mode": "value"}),
                            ("breakout_trend", None, {"size_mode": "value"})], tf))


if __name__ == "__main__":
    main()
