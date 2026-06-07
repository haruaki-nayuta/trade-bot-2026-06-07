"""イテレーション8: マルチウィンドウ合流の頑健性確認(100%プラス年が高原か検証)。

exp07(B) で「短期Z<-2 かつ 長期Z(250)<-1.5」が全年プラス・PF中央1.58を出した。
これが偶然の1点(=カーブフィット)でなく、slow_win × slow_z の広い領域で安定するかを走査する。
実行: uv run python exp08.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import config
from fxlab import yearly as ylib
from fxlab.data import available_pairs
from exp05 import build_crosses, synth_ohlc, portfolio_from
from exp07 import cmr, CROSSES

pd.set_option("display.width", 200)
TF = "H4"


def tables(params):
    out = {p: ylib.yearly(p, TF, cmr, params, size_mode="value") for p in available_pairs()}
    for nm, cl in build_crosses().items():
        out[nm] = ylib.yearly(nm, TF, cmr, params, data=synth_ohlc(cl), size_mode="value")
    return out


def stat(params):
    port = portfolio_from(tables(params))
    if port.empty or "profit_factor" not in port:
        return float("nan"), float("nan"), float("nan"), 0, port
    pf = port["profit_factor"].replace(np.inf, np.nan)
    return (port["pnl"] > 0).mean(), pf.median(), pf.min(), int(port["trades"].mean()), port


def main():
    for nm in CROSSES:
        config.SPREADS_PIPS[nm] = 2.0

    print("=== 頑健性走査: slow_win × slow_z (vol_pct=0.70 固定) → プラス年率 / PF中央 / PF最小 / 年取引 ===")
    header = f"{'slow_z\\win':>10}" + "".join(f"{w:>16}" for w in [200, 250, 300])
    print(header)
    for sz in [1.0, 1.25, 1.5, 1.75, 2.0]:
        cells = []
        for w in [200, 250, 300]:
            pos, med, mn, tr, _ = stat({"vol_pct": 0.70, "slow_win": w, "slow_z": sz})
            cells.append(f"{pos:.0%}/{med:.2f}/{mn:.2f}/{tr}")
        print(f"{sz:>10.2f}" + "".join(f"{c:>16}" for c in cells))

    # 代表設定の年次詳細 + vol_pct も少し振って二重に高原確認
    print("\n=== 代表 slow_win=250, slow_z=1.5 の年次(vol_pct 別)===")
    for vp in [0.65, 0.70, 0.75]:
        pos, med, mn, tr, port = stat({"vol_pct": vp, "slow_win": 250, "slow_z": 1.5})
        print(f"\n-- vol_pct={vp}: プラス年率 {pos:.0%} / PF中央 {med:.2f} / PF最小 {mn:.2f} / 年取引 {tr}")
        print(port.to_string())


if __name__ == "__main__":
    main()
