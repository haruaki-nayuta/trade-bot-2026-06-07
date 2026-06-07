"""イテレーション7: (A)ボラフィルタの高原性確認 (B)多時間足(MTF)合流でPF底上げ。

(A) vol_pct を広く振り、プラス年率/PF が滑らかに変化する(=偶然の1点でない)かを見る。
(B) H4 のエントリーに「日足でも同方向に行き過ぎ」を要求し、確信度を上げて PF が上がるか。
    日足Zは H4 データを 1D リサンプル→1日シフト(前日確定値)で算出=先読みなし。
実行: uv run python exp07.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt

from fxlab import config
from fxlab import yearly as ylib
from fxlab.data import available_pairs
from exp05 import build_crosses, synth_ohlc, portfolio_from

pd.set_option("display.width", 200)
TF = "H4"
CROSSES = ["EURGBP","EURCHF","EURAUD","EURCAD","GBPAUD","GBPCHF","AUDNZD","AUDCAD",
           "NZDCAD","AUDCHF","EURJPY","GBPJPY","AUDJPY"]


def _zscore(s, w):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def cmr(data, window=50, entry_z=2.0, exit_z=0.5, rsi_p=14, rsi_low=35, rsi_high=65,
        vol_win=100, vol_pct=0.70, slow_win=20, slow_z=0.0):
    """confluence + ボラフィルタ + (slow_z>0 なら)日足MTF合流。"""
    close = data["close"]
    z = _zscore(close, window)
    rsi = vbt.RSI.run(close, rsi_p).rsi
    if vol_pct >= 1.0:
        calm = pd.Series(True, index=close.index)
    else:
        vol = close.pct_change().rolling(20).std()
        calm = vol <= vol.rolling(vol_win).quantile(vol_pct)
    if slow_z > 0:
        # 長期ウィンドウZ(同一系列・先読みなし)= より大きな時間軸の行き過ぎ代理
        zslow = _zscore(close, slow_win)
        long_ok = (zslow < -slow_z).fillna(False)
        short_ok = (zslow > slow_z).fillna(False)
    else:
        long_ok = short_ok = pd.Series(True, index=close.index)
    le = (z < -entry_z) & (z.shift() >= -entry_z) & (rsi < rsi_low) & calm & long_ok
    se = (z > entry_z) & (z.shift() <= entry_z) & (rsi > rsi_high) & calm & short_ok
    return le.fillna(False), (z > -exit_z).fillna(False), se.fillna(False), (z < exit_z).fillna(False)


def tables(params):
    out = {p: ylib.yearly(p, TF, cmr, params, size_mode="value") for p in available_pairs()}
    for nm, cl in build_crosses().items():
        out[nm] = ylib.yearly(nm, TF, cmr, params, data=synth_ohlc(cl), size_mode="value")
    return out


def stats(params):
    port = portfolio_from(tables(params))
    if port.empty or "profit_factor" not in port:
        return float("nan"), float("nan"), float("nan"), 0, port
    pf = port["profit_factor"].replace(np.inf, np.nan)
    return (port["pnl"] > 0).mean(), pf.median(), pf.min(), int(port["trades"].mean()), port


def main():
    for nm in CROSSES:
        config.SPREADS_PIPS[nm] = 2.0

    print("=== (A) ボラフィルタ vol_pct の高原性(20対象ポート)===")
    print(f"{'vol_pct':>8} {'プラス年率':>10} {'PF中央':>8} {'PF最小':>8} {'年取引':>7}")
    for vp in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 1.0]:
        pos, med, mn, tr, _ = stats({"vol_pct": vp})
        print(f"{vp:>8.2f} {pos:>10.0%} {med:>8.2f} {mn:>8.2f} {tr:>7d}")

    print("\n=== (B) マルチウィンドウ合流(vol_pct=0.70, slow_win=250 の長期Zも要求)===")
    print(f"{'slow_z':>8} {'プラス年率':>10} {'PF中央':>8} {'PF最小':>8} {'年取引':>7}")
    for sz in [0.0, 0.5, 1.0, 1.5]:
        pos, med, mn, tr, port = stats({"vol_pct": 0.70, "slow_win": 250, "slow_z": sz})
        poss = f"{pos:>9.0%}" if pos == pos else f"{'n/a':>9}"
        print(f"{sz:>8.1f} {poss} {med:>8.2f} {mn:>8.2f} {tr:>7d}")


if __name__ == "__main__":
    main()
