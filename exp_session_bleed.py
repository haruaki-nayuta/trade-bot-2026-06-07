"""セッション・ブレイクアウト(session_breakout)の失血窓貢献スイープ。

チャンピオンv2(H4・無ストップ・乖離回帰)が失血する21ヶ月で、日中レンジ放れ
(アジアレンジ→ロンドン放れ)が稼げるか=保険になるかを評価する。

評価軸(bleed_lab.conditional_score):
  mean_in_bleed  : 失血窓の月次平均PnL($, value $10k/銘柄合算)
  mean_normal    : 平時の月次平均PnL
  hedge_edge     : mean_in_bleed - mean_normal(>0 で「失血窓により強い」)
  IS/OOS         : 2022年で分割。両方プラス=持続ヘッジ(2022一発のまぐれでない)

session_breakout は時刻粒度が要るので H1 で実行(チャンピオンはH4のまま)。
月次集計するのでTFの差は問題にならない。

振るパラメータ: asian_end / entry_end / exit_hour / side。
"""
from __future__ import annotations

import itertools
import warnings

import pandas as pd

import bleed_lab as bl

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

TF = "H1"  # session_breakout は時刻粒度が必要

# 振る範囲(PARAM_GRID をベースに少し広げる)
ASIAN_END = [6, 7, 8]
ENTRY_END = [11, 12, 14]
EXIT_HOUR = [18, 20, 22]
SIDES = ["both", "long", "short"]


def main():
    print("=== チャンピオンv2 失血窓マスク取得 ===")
    eqm, eqr, pool, closes = bl.champion_mtm()
    mask, dd = bl.bleed_mask_monthly(eqm)
    nb = int(mask.sum())
    print(f"月数 {len(mask)} / 失血窓 {nb}ヶ月\n")

    rows = []
    combos = list(itertools.product(ASIAN_END, ENTRY_END, EXIT_HOUR, SIDES))
    print(f"探索 {len(combos)} 構成 (session_breakout, {TF}) ...\n")
    for ae, ee, eh, side in combos:
        if ee <= ae:
            continue
        params = {"asian_end": ae, "entry_end": ee, "exit_hour": eh}
        try:
            mp = bl.strategy_monthly_pnl("session_breakout", params=params,
                                         tf=TF, side=side)
        except Exception as e:  # noqa: BLE001
            print(f"  skip ae{ae} ee{ee} eh{eh} {side}: {e}")
            continue
        if mp.empty:
            continue
        sc = bl.conditional_score(mp, mask)
        rows.append({
            "asian_end": ae, "entry_end": ee, "exit_hour": eh, "side": side,
            "mean_in_bleed": sc["mean_in_bleed"],
            "mean_normal": sc["mean_normal"],
            "hedge_edge": sc["hedge_edge"],
            "winrate_in_bleed": sc["winrate_in_bleed"],
            "in_IS": sc["mean_in_bleed_IS"],
            "in_OOS": sc["mean_in_bleed_OOS"],
            "total_in_bleed": sc["total_in_bleed"],
            "total_all": sc["total_all"],
            "n_trades_months": int((mp != 0).sum()),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("結果なし")
        return
    df = df.sort_values("mean_in_bleed", ascending=False).reset_index(drop=True)
    cols = ["asian_end", "entry_end", "exit_hour", "side", "mean_in_bleed",
            "mean_normal", "hedge_edge", "winrate_in_bleed", "in_IS", "in_OOS",
            "total_in_bleed", "total_all"]
    print("=== 失血窓貢献ランキング(mean_in_bleed 降順)===")
    print(df[cols].round(2).to_string())

    # 持続ヘッジ条件: in_IS>0 かつ in_OOS>0
    persist = df[(df["in_IS"] > 0) & (df["in_OOS"] > 0)].copy()
    print(f"\n=== 持続ヘッジ候補(in_IS>0 かつ in_OOS>0): {len(persist)}件 ===")
    if not persist.empty:
        persist = persist.sort_values("mean_in_bleed", ascending=False)
        print(persist[cols].round(2).to_string())

    df.to_csv("results/exp_session_bleed.csv", index=False)
    print("\nsaved -> results/exp_session_bleed.csv")


if __name__ == "__main__":
    main()
