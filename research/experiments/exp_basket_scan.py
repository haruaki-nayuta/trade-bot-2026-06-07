"""ドルバスケット・トレンド(単一lookback)のロバスト性スキャン + vol-target変種。

scout/gated の知見:
  - アンサンブル(常時建玉)は単体赤字。単一3mo(D1)は単体+1684・corr0.06。
  - ERゲートは失血窓保護を壊す(早期トレンド捕捉が保護の源)。
狙い: 単一lookbackのまま「どのlookbackでも単体プラス & 失血窓で稼ぐ」高原があるか。
       長lookback=長セグメント=コスト効率↑+持続ドルトレンド(2021-22ラリー等)を捕捉。
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from fxlab import config, universe as uni
import bleed_lab as bl
from research.experiments.exp_usdbasket_gated import basket_tsmom_gated, to_monthly, metrics

pd.set_option("display.width", 260)


def main():
    eqm, eqr, pool, _ = bl.champion_mtm(max_pos=8)
    mask, dd = bl.bleed_mask_monthly(eqm)
    cm = pool.copy()
    cm["m"] = pd.PeriodIndex(pd.to_datetime(cm["exit"]).dt.to_period("M"))
    NOTIONAL = 10_000.0
    champ_monthly = cm.groupby("m")["ret"].sum().reindex(mask.index).fillna(0.0) * NOTIONAL

    print("=== 単一lookback ドルバスケットTSMOM ロバスト性スキャン ===")
    for tf, grid in [("D1", [21, 42, 63, 84, 126, 189, 252]),
                     ("H4", [120, 240, 360, 480, 720, 1080])]:
        rows = []
        for L in grid:
            tr = basket_tsmom_gated(tf, [L], consensus_min=1, er_gate=None)
            mo = to_monthly(tr)
            if len(mo) == 0:
                continue
            m = metrics(mo, mask, champ_monthly)
            rows.append({"tf": tf, "lookback": L, "total": round(m["total"], 0),
                         "pos_yr": f"{m['pos_yr']:.0%}", "PF": round(m["PF"], 2),
                         "IS": round(m["IS"], 0), "OOS": round(m["OOS"], 0),
                         "worst_yr": round(m["worst_yr"], 0), "corr": round(m["corr"], 3),
                         "bleedIS": round(m["bleed_IS"], 0), "bleedOOS": round(m["bleed_OOS"], 0)})
        print(f"\n--- {tf} (lookback単位=バー; D1: 21≈1mo,63≈3mo,126≈6mo,252≈12mo)---")
        print(pd.DataFrame(rows).to_string(index=False))
    print("\n判定: total>0 & pos_yr>=55% & IS/OOS両プラス が広いlookback帯で成立する高原か?")
    print("      bleedIS/bleedOOS両プラスなら失血窓保護も(xsecに無い性質)。")


if __name__ == "__main__":
    main()
