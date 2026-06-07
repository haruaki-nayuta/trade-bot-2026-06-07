"""trend_insurance のパラメータ探索 — 失血窓貢献(IS/OOS両プラス)を最大化。

bleed_lab の共有基盤を使い、各(er_win, er_hi, er_lo, ma_win, side)構成について
チャンピオン失血窓での条件付きスコアを測る。吟味の核は conditional_score:
  mean_in_bleed     : 失血窓の平均月次PnL($, value 10k/銘柄合算)
  hedge_edge        : mean_in_bleed - mean_normal(窓で平時より稼ぐか)
  mean_in_bleed_IS  : 2022年より前の失血窓での平均(=2022一発まぐれ排除の核)
  mean_in_bleed_OOS : 2022年以降の失血窓での平均
  winrate_in_bleed  : 失血窓で正だった月の割合
持続性の合格条件: mean_in_bleed_IS と mean_in_bleed_OOS が **両方プラス**。

実行: uv run python ti_sweep.py
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

import bleed_lab as bl

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 200)

NAME = "trend_insurance"


def run_grid():
    # チャンピオンの失血窓マスク(一度だけ計算)
    eqm, eqr, pool, closes = bl.champion_mtm()
    mask, dd = bl.bleed_mask_monthly(eqm)

    grid = {
        "er_win": [20, 40, 60],
        "er_hi": [0.35, 0.45, 0.55],
        "er_lo": [0.25],
        "ma_win": [50, 100, 200],
    }
    sides = ["both", "long", "short"]

    combos = list(itertools.product(grid["er_win"], grid["er_hi"], grid["er_lo"],
                                    grid["ma_win"], sides))
    rows = []
    for er_win, er_hi, er_lo, ma_win, side in combos:
        params = {"er_win": er_win, "er_hi": er_hi, "er_lo": er_lo, "ma_win": ma_win}
        mp = bl.strategy_monthly_pnl(NAME, params=params, side=side)
        if mp.empty:
            continue
        sc = bl.conditional_score(mp, mask)
        rows.append({
            "er_win": er_win, "er_hi": er_hi, "er_lo": er_lo, "ma_win": ma_win, "side": side,
            "mean_in_bleed": sc["mean_in_bleed"], "mean_normal": sc["mean_normal"],
            "hedge_edge": sc["hedge_edge"], "winrate_in_bleed": sc["winrate_in_bleed"],
            "IS": sc["mean_in_bleed_IS"], "OOS": sc["mean_in_bleed_OOS"],
            "total_in_bleed": sc["total_in_bleed"], "total_all": sc["total_all"],
        })
    df = pd.DataFrame(rows)
    df["persist"] = (df["IS"] > 0) & (df["OOS"] > 0)
    return df, mask, dd


def main():
    df, mask, dd = run_grid()
    df.to_csv("ti_sweep_results.csv", index=False)
    print(f"=== trend_insurance スイープ({len(df)}構成)失血窓 {int(mask.sum())}ヶ月 ===\n")

    print(">>> 持続性合格(IS>0 かつ OOS>0)構成を hedge_edge 降順:")
    ok = df[df["persist"]].sort_values("hedge_edge", ascending=False)
    if ok.empty:
        print("  なし(持続的に失血窓で稼ぐ構成は見つからず)")
    else:
        print(ok.round(2).to_string(index=False))

    print("\n>>> 全構成 hedge_edge 降順 トップ15:")
    print(df.sort_values("hedge_edge", ascending=False).head(15).round(2).to_string(index=False))

    print("\n>>> 全構成 mean_in_bleed 降順 トップ15:")
    print(df.sort_values("mean_in_bleed", ascending=False).head(15).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
