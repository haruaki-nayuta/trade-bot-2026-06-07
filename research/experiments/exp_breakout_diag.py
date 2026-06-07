"""診断: 失血窓では勝つのに統合DDが悪化する理由を分解する。

conditional_score(月次realized) は「窓内で稼ぐ」を示すのに、integrated_dd_test(バーMtM)では
CAGR が落ち p95(理論DD)が悪化した。仮説:
 (1) overlay 単体の月次vol が大きく、平時の負け月のバラつきが統合DDを押し上げる。
 (2) 窓内の勝ちが「窓の月」には立つが、champion のDDの谷(バー単位)と時間的にズレる。
 (3) breakout は塩漬けではないが MtM の含み損スイングが大きい。

ここでは:
 - 最良 overlay の月次PnL系列を champion月次MtM変化と相関(全期間/窓内)。
 - overlay単体の月次vol / 最大連敗 を champion と比較。
 - 失血窓の各月で overlay が実際にプラスだったか(個別月の貢献)を並べる。
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

import bleed_lab as bl
import mm_lab as mm

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 60)


def main():
    eqm, eqr, pool, closes = bl.champion_mtm(max_pos=8)
    mask, dd = bl.bleed_mask_monthly(eqm)

    # champion 月次MtM equity変化(失血のタイミング源)
    champ_m = eqm.groupby(eqm.index.to_period("M")).last()
    champ_chg = champ_m.diff()

    bt = importlib.import_module("strategies.breakout_trend")
    mp = bl.strategy_monthly_pnl("breakout_trend",
                                 params={"entry": 80, "exit": 20, "trend": 200},
                                 side="short", tf="H4")
    mp = mp.reindex(mask.index).fillna(0.0)

    # 月次相関(全期間 / 窓内)
    al = pd.concat([champ_chg.reindex(mask.index), mp], axis=1).dropna()
    al.columns = ["champ_chg", "ovl"]
    corr_all = al["champ_chg"].corr(al["ovl"])
    inb = al[mask.reindex(al.index).fillna(False).values]
    corr_bleed = inb["champ_chg"].corr(inb["ovl"]) if len(inb) > 2 else float("nan")
    print(f"月次相関 overlay vs champion月次MtM変化: 全期間={corr_all:+.3f} 窓内={corr_bleed:+.3f}")
    print(f"  (負ければ良いヘッジ: champ が下げる月に overlay が上げる)\n")

    print(f"overlay月次PnL: mean={mp.mean():+.1f} std={mp.std():.1f} min={mp.min():+.1f} max={mp.max():+.1f}")
    print(f"  負け月率={ (mp<0).mean():.0%}  最大連敗(月)={_max_streak(mp<0)}\n")

    # 失血窓の各月: champ DD と overlay PnL
    print("=== 失血窓の各月: champ月末DD と overlay月次PnL ===")
    rows = []
    for m in mask.index[mask.values]:
        rows.append({"month": str(m), "champ_dd": dd.get(m, np.nan), "ovl_pnl": mp.get(m, 0.0)})
    bdf = pd.DataFrame(rows)
    print(bdf.round(1).to_string(index=False))
    print(f"\n  窓内 overlAY 合計={bdf['ovl_pnl'].sum():+.0f}  プラス月数={(bdf['ovl_pnl']>0).sum()}/{len(bdf)}")

    # overlay 単体の通算(保険コスト)
    print(f"\noverlay 単体 通算PnL(全月合算) = {mp.sum():+.0f}  "
          f"(value$10k/銘柄・19銘柄合算, 11年)")


def _max_streak(boolser):
    m = 0; c = 0
    for v in boolser:
        c = c + 1 if v else 0
        m = max(m, c)
    return m


if __name__ == "__main__":
    main()
