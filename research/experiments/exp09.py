"""イテレーション9: チャンピオン最終検証 — コスト感応度 + 銘柄別の成績。

champion: confluence_meanrev(Z×RSI×ボラ×マルチウィンドウ) on 20対象(メジャー+クロス) H4。
params: vol_pct=0.70, slow_win=250, slow_z=1.5。
(1) クロスのスプレッドを 2/3/4 pips と厳しくしても全年プラスが保たれるか。
(2) 銘柄ごとの通算PF・プラス年率・年取引(どの対象が支えているか/弱いか)。
実行: uv run python exp09.py
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
P = {"vol_pct": 0.70, "slow_win": 250, "slow_z": 1.5}


def all_tables():
    out = {p: ylib.yearly(p, TF, cmr, P, size_mode="value") for p in available_pairs()}
    for nm, cl in build_crosses().items():
        out[nm] = ylib.yearly(nm, TF, cmr, P, data=synth_ohlc(cl), size_mode="value")
    return out


def main():
    print("=== (1) コスト感応度(クロスのスプレッド pips を変えて全20対象ポート)===")
    for sp in [2.0, 3.0, 4.0]:
        for nm in CROSSES:
            config.SPREADS_PIPS[nm] = sp
        port = portfolio_from(all_tables())
        pf = port["profit_factor"].replace(np.inf, np.nan)
        print(f"  cross_spread={sp}pips: プラス年率 {(port['pnl']>0).mean():.0%} / "
              f"PF中央 {pf.median():.2f} / PF最小 {pf.min():.2f} / 年取引 {int(port['trades'].mean())}")

    print("\n=== (2) 銘柄別の通算(cross_spread=3.0pips=厳しめ)===")
    for nm in CROSSES:
        config.SPREADS_PIPS[nm] = 3.0
    tabs = all_tables()
    rows = {}
    for nm, t in tabs.items():
        if t.empty:
            continue
        gp, gl = t["gross_profit"].sum(), t["gross_loss"].sum()
        rows[nm] = {
            "PF": round(gp / gl, 2) if gl > 0 else np.inf,
            "pos_yr%": round((t["pnl"] > 0).mean() * 100),
            "trades/yr": round(t["trades"].mean()),
            "total_pnl": round(t["pnl"].sum()),
        }
    board = pd.DataFrame(rows).T.sort_values("PF", ascending=False)
    print(board.to_string())
    print(f"\n  通算プラス銘柄: {int((board['total_pnl']>0).sum())}/{len(board)}")


if __name__ == "__main__":
    main()
