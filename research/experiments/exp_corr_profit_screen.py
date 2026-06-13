"""全戦略の「単体収益性 vs チャンピオン相関」網羅スキャン — 壁を主張でなく実測で証明。

仮説: 2016-26のFXメジャーで正のリスクプレミアムは平均回帰(MR)だけ。
  → 単体黒字の戦略は全てMR系 → チャンピオンと正相関 → 分散にならない。
  → トレンド/ブレイク系は低相関だが単体赤字。
反証(=探していた補完): 「単体黒字 かつ 低/負相関」の外れ値が1つでもあれば、それが候補。

各戦略: 単体total/プラス年率/PF + チャンピオン月次相関。黒字×低相関なら2スリーブrobust便益も。
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from fxlab import config, universe as uni
import mm_lab as mm
import bleed_lab as bl

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
NOTIONAL = 10_000.0
OOS = pd.Period("2022-01", "M")

# 通貨・非chaos・非champion家系の全戦略(MR系 / トレンド系 / ブレイク系 / その他)
STRATS = {
    "rsi_meanrev": "MR", "zscore_meanrev": "MR", "meanrev_range": "MR",
    "rsi2_pullback": "MR", "nextbar_revscalp": "MR",
    "tsmom": "TREND", "adx_trend": "TREND", "ma_cross": "TREND",
    "breakout_trend": "TREND", "donchian_breakout": "BREAK",
    "bb_breakout": "BREAK", "squeeze_breakout": "BREAK", "session_breakout": "BREAK",
}


def main():
    # チャンピオン基準
    eqm, eqr, pool, _ = bl.champion_mtm(max_pos=8)
    mask, dd = bl.bleed_mask_monthly(eqm)
    cm = pool.copy()
    cm["m"] = pd.PeriodIndex(pd.to_datetime(cm["exit"]).dt.to_period("M"))
    champ_monthly = cm.groupby("m")["ret"].sum().reindex(mask.index).fillna(0.0) * NOTIONAL

    rows = []
    for name, fam in STRATS.items():
        try:
            mo = bl.strategy_monthly_pnl(name, side="both")
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {name}: {e}")
            continue
        if mo is None or len(mo) == 0:
            continue
        mo = mo.reindex(mask.index).fillna(0.0)
        yr = mo.groupby([p.year for p in mo.index]).sum()
        pos = mo[mo > 0].sum(); neg = -mo[mo < 0].sum()
        cr = float(np.corrcoef(mo.values, champ_monthly.values)[0, 1])
        inb = mo[mask.values]
        rows.append({
            "strategy": name, "family": fam,
            "total": round(float(mo.sum()), 0),
            "pos_yr": f"{float((yr > 0).mean()):.0%}",
            "PF": round(float(pos / neg), 2) if neg > 0 else np.nan,
            "IS": round(float(mo[mo.index < OOS].sum()), 0),
            "OOS": round(float(mo[mo.index >= OOS].sum()), 0),
            "corr_champ": round(cr, 3),
            "bleed_mean": round(float(inb.mean()), 1),
            "profitable": float(mo.sum()) > 0,
            "low_corr": abs(cr) < 0.40,
        })
    df = pd.DataFrame(rows).sort_values(["family", "corr_champ"])
    print("\n=== 全戦略: 単体収益性 vs チャンピオン相関(H4, 19銘柄, value$10k)===")
    show = df.drop(columns=["profitable", "low_corr"])
    print(show.to_string(index=False))

    print("\n=== 仮説検定 ===")
    prof = df[df["profitable"]]
    print(f"  単体黒字の戦略: {list(prof['strategy'])}")
    print(f"    → その family: {list(prof['family'])}")
    print(f"    → その corr_champ: {[round(c,2) for c in prof['corr_champ']]}")
    cand = df[df["profitable"] & df["low_corr"]]
    print(f"\n  ★候補(黒字 かつ |corr|<0.40): {list(cand['strategy']) if len(cand) else 'なし'}")
    if len(cand):
        print("    → 該当あり。これらは2スリーブrobust評価に進める価値がある。")
    else:
        print("    → 該当なし=「単体黒字⟺正相関(MR系)」を実測で確認。壁は構造的(主張でなく実測)。")
    # 黒字戦略の相関分布
    if len(prof):
        print(f"\n  黒字戦略の corr_champ: 最小={prof['corr_champ'].min():.2f} / "
              f"最大={prof['corr_champ'].max():.2f} / 中央={prof['corr_champ'].median():.2f}")


if __name__ == "__main__":
    main()
