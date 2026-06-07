"""イテレーション4 実験: レジーム切替(レンジ=Zスコア回帰 / トレンド=ブレイク)。

「悪い年(エッジがレジームで同時消失)」に効くか検証。ADX で局面を判定し、
レンジではZスコア平均回帰、トレンドではドンチャン・ブレイクを別スリーブで建て、
決済年で合算する(出口の競合を避けるため2スリーブ構成)。

実行: uv run python exp04.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.trend import ADXIndicator

from fxlab import yearly as ylib
from fxlab.data import available_pairs

pd.set_option("display.width", 200)
PAIRS = available_pairs()


def _adx(data, period=14):
    return ADXIndicator(data["high"], data["low"], data["close"], window=period).adx()


def zmr_range_gen(data, window=50, entry_z=2.0, exit_z=0.5, adx_max=25):
    """レンジ(ADX<adx_max)限定のZスコア平均回帰。"""
    close = data["close"]
    z = (close - close.rolling(window).mean()) / close.rolling(window).std()
    ranging = _adx(data) < adx_max
    le = ranging & (z < -entry_z) & (z.shift() >= -entry_z)
    se = ranging & (z > entry_z) & (z.shift() <= entry_z)
    lx = z > -exit_z
    sx = z < exit_z
    return le.fillna(False), lx.fillna(False), se.fillna(False), sx.fillna(False)


def trend_break_gen(data, entry=40, exit=20, trend=200, adx_min=25):
    """トレンド(ADX>adx_min)限定のドンチャン・ブレイク(SMA方向)。"""
    high, low, close = data["high"], data["low"], data["close"]
    upper = high.rolling(entry).max().shift()
    lower = low.rolling(entry).min().shift()
    ex_lo = low.rolling(exit).min().shift()
    ex_hi = high.rolling(exit).max().shift()
    sma = close.rolling(trend).mean()
    trending = _adx(data) > adx_min
    le = trending & (close > upper) & (close > sma)
    se = trending & (close < lower) & (close < sma)
    lx = close < ex_lo
    sx = close > ex_hi
    return le.fillna(False), lx.fillna(False), se.fillna(False), sx.fillna(False)


def combo_yearly(specs, tf):
    """specs: list of (gen, params, run_kw)。決済年でポートフォリオ合算。"""
    accum = {}
    for gen, params, kw in specs:
        for pair in PAIRS:
            try:
                y = ylib.yearly(pair, tf, gen, params, **kw)
            except Exception as e:  # noqa: BLE001
                print("skip", pair, e); continue
            for year, r in y.iterrows():
                a = accum.setdefault(int(year), [0.0, 0.0, 0.0, 0.0])
                a[0] += r["gross_profit"]; a[1] += r["gross_loss"]
                a[2] += r["pnl"]; a[3] += r["trades"]
    rows = {}
    for year, (gp, gl, pnl, trades) in sorted(accum.items()):
        rows[year] = {"trades": int(trades),
                      "profit_factor": round(gp / gl, 2) if gl > 0 else float("inf"),
                      "pnl": round(pnl, 0), "positive": pnl > 0}
    return pd.DataFrame(rows).T


def summarize(label, port):
    pf = port["profit_factor"].replace(np.inf, np.nan)
    pos = (port["pnl"] > 0).mean()
    print(f"\n--- {label} ---")
    print(port.to_string())
    print(f"  → プラス年率 {pos:.0%} / PF中央値 {pf.median():.2f} / PF最小 {pf.min():.2f} / 年平均取引 {int(port['trades'].mean())}")


def main():
    tf = "H4"
    print(f"ペア: {PAIRS}  足: {tf}")
    summarize("レンジ限定Zスコア回帰のみ",
              combo_yearly([(zmr_range_gen, {}, {"size_mode": "value"})], tf))
    summarize("トレンド限定ブレイクのみ",
              combo_yearly([(trend_break_gen, {}, {"size_mode": "value"})], tf))
    summarize("レジーム切替(回帰+ブレイク)",
              combo_yearly([(zmr_range_gen, {}, {"size_mode": "value"}),
                            (trend_break_gen, {}, {"size_mode": "value"})], tf))


if __name__ == "__main__":
    main()
