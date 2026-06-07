"""イテレーション6 実験: ボラ・レジームフィルタで「平均回帰が轢かれる年(2020,2024)」を抑制。

仮説: 平均回帰はボラが急拡大する局面(危機/強トレンド)で大きく負ける。実現ボラがその
通貨自身の過去分布の高位(percentile)にある間はエントリーを止めれば、悪い年の損失を圧縮できる。
20対象(メジャー+クロス)ポートフォリオで、フィルタ有無の年次を比較する。
実行: uv run python exp06.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import config
from fxlab import yearly as ylib
from fxlab.data import available_pairs
from exp05 import build_crosses, synth_ohlc, portfolio_from

pd.set_option("display.width", 200)
TF = "H4"
BASE = {"window": 50, "entry_z": 2.0, "exit_z": 0.5, "rsi_p": 14, "rsi_low": 35, "rsi_high": 65}


def cmr_volfilter(data, window=50, entry_z=2.0, exit_z=0.5, rsi_p=14, rsi_low=35, rsi_high=65,
                  vol_win=100, vol_pct=0.85):
    """confluence_meanrev + ボラフィルタ(実現ボラが過去 vol_win の vol_pct 分位超ならエントリー停止)。"""
    import vectorbt as vbt
    close = data["close"]
    z = (close - close.rolling(window).mean()) / close.rolling(window).std()
    rsi = vbt.RSI.run(close, rsi_p).rsi
    ret = close.pct_change()
    vol = ret.rolling(20).std()                       # 直近20本の実現ボラ
    vol_thresh = vol.rolling(vol_win).quantile(vol_pct)
    calm = (vol <= vol_thresh)                         # 平穏レジームのみ建玉
    le = (z < -entry_z) & (z.shift() >= -entry_z) & (rsi < rsi_low) & calm
    se = (z > entry_z) & (z.shift() <= entry_z) & (rsi > rsi_high) & calm
    lx = z > -exit_z
    sx = z < exit_z
    return le.fillna(False), lx.fillna(False), se.fillna(False), sx.fillna(False)


def tables_for(gen, params):
    from strategies.confluence_meanrev import generate_signals as cmr
    g = gen
    out = {}
    for p in available_pairs():
        out[p] = ylib.yearly(p, TF, g, params, size_mode="value")
    for nm, cl in build_crosses().items():
        out[nm] = ylib.yearly(nm, TF, g, params, data=synth_ohlc(cl), size_mode="value")
    return out


def report(label, tables):
    port = portfolio_from(tables)
    pf = port["profit_factor"].replace(np.inf, np.nan)
    print(f"\n=== {label} ===")
    print(port.to_string())
    print(f"  → プラス年率 {(port['pnl']>0).mean():.0%} / PF中央 {pf.median():.2f} / PF最小 {pf.min():.2f} / 年取引 {int(port['trades'].mean())}")


def main():
    for nm in ["EURGBP","EURCHF","EURAUD","EURCAD","GBPAUD","GBPCHF","AUDNZD","AUDCAD",
               "NZDCAD","AUDCHF","EURJPY","GBPJPY","AUDJPY"]:
        config.SPREADS_PIPS[nm] = 2.0

    from strategies.confluence_meanrev import generate_signals as cmr
    report("ベース(フィルタ無し)20対象", tables_for(cmr, BASE))
    report("ボラフィルタ pct=0.85", tables_for(cmr_volfilter, {**BASE, "vol_win": 100, "vol_pct": 0.85}))
    report("ボラフィルタ pct=0.70", tables_for(cmr_volfilter, {**BASE, "vol_win": 100, "vol_pct": 0.70}))


if __name__ == "__main__":
    main()
