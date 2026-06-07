"""イテレーション21: 時系列モメンタム(tsmom)の「失血窓ヘッジ」探索。

戦略フレーム(reports/09 後):補完エッジは「平均相関の低さ」でなく
**チャンピオンv2が失血しているまさにその窓で稼ぐ**かで評価する。
tsmom はトレンド継続レジームで稼ぐ素直な候補(チャンピオンの失血窓=高ER=トレンド継続と一致するはず)。

手順:
  1. チャンピオンv2(z-size mp8, 20%較正)の MtM equity から失血窓マスク(月次)を取得。
  2. tsmom を lookback∈{50,100,150,200} × band∈{0.0,0.002,0.005} × side∈{both,long,short}
     × tf∈{H4,D1} で走査し、各構成の月次PnLストリームを得る。
  3. conditional_score で「失血窓 vs 平時」の貢献を測る。**単体PFは無視**。
     最優先 = mean_in_bleed_IS と mean_in_bleed_OOS が両方プラス(=2022一発でない持続ヘッジ)。
  4. 上位構成を hedge_edge / mean_in_bleed で並べ、ベスト構成を報告。
  5. ベスト構成は integrated_dd_test で最終確認(任意, champion+overlay@20%DDのCAGR)。

実行: uv run python exp21_tsmom_bleed.py
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

import bleed_lab as bl

warnings.simplefilter("ignore")
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


def main():
    print("=== チャンピオンv2 失血窓マスクの構築 ===")
    eqm, eqr, pool, closes = bl.champion_mtm()
    mask, dd = bl.bleed_mask_monthly(eqm)
    nb = int(mask.sum())
    print(f"失血窓 {nb}ヶ月 / 全{len(mask)}ヶ月  (OOS境界=2022-01)")
    print(f"失血窓の年別: {dict(pd.Series(mask.values, index=[p.year for p in mask.index]).groupby(level=0).sum())}\n")

    lookbacks = [50, 100, 150, 200]
    bands = [0.0, 0.002, 0.005]
    sides = ["both", "long", "short"]
    tfs = ["H4", "D1"]

    rows = []
    for tf, lb, band, side in itertools.product(tfs, lookbacks, bands, sides):
        params = {"lookback": lb, "band": band}
        try:
            mp = bl.strategy_monthly_pnl("tsmom", params=params, side=side, tf=tf)
        except Exception as e:  # noqa: BLE001
            print(f"  skip tf={tf} lb={lb} band={band} side={side}: {e}")
            continue
        if mp.empty:
            continue
        sc = bl.conditional_score(mp, mask)
        rows.append({
            "tf": tf, "lb": lb, "band": band, "side": side,
            "mean_bleed": sc["mean_in_bleed"],
            "mean_norm": sc["mean_normal"],
            "edge": sc["hedge_edge"],
            "wr_bleed": sc["winrate_in_bleed"],
            "IS": sc["mean_in_bleed_IS"],
            "OOS": sc["mean_in_bleed_OOS"],
            "tot_bleed": sc["total_in_bleed"],
            "tot_all": sc["total_all"],
        })
        flag = "  <-- IS/OOS両プラス" if (sc["mean_in_bleed_IS"] > 0 and sc["mean_in_bleed_OOS"] > 0) else ""
        print(f"tf={tf} lb={lb:>3} band={band:.3f} side={side:>5}: "
              f"bleed={sc['mean_in_bleed']:+8.1f} norm={sc['mean_normal']:+8.1f} "
              f"edge={sc['hedge_edge']:+8.1f} wr={sc['winrate_in_bleed']:.2f} "
              f"IS={sc['mean_in_bleed_IS']:+8.1f} OOS={sc['mean_in_bleed_OOS']:+8.1f} "
              f"tot_all={sc['total_all']:+9.0f}{flag}")

    res = pd.DataFrame(rows)
    if res.empty:
        print("結果なし")
        return

    res["persist"] = (res["IS"] > 0) & (res["OOS"] > 0)

    print("\n" + "=" * 100)
    print("=== 持続ヘッジ(IS>0 かつ OOS>0)のみ, mean_bleed 降順 ===")
    persist = res[res["persist"]].sort_values("mean_bleed", ascending=False)
    if persist.empty:
        print("  該当なし(全構成でIS/OOSのどちらかが失血窓でマイナス)")
    else:
        print(persist.round(2).to_string(index=False))

    print("\n=== 全構成 hedge_edge 降順 トップ15 ===")
    print(res.sort_values("edge", ascending=False).head(15).round(2).to_string(index=False))

    print("\n=== 全構成 mean_in_bleed 降順 トップ15 ===")
    print(res.sort_values("mean_bleed", ascending=False).head(15).round(2).to_string(index=False))

    # ベスト構成の選定: 持続ヘッジの中で mean_bleed 最大。なければ全体 mean_bleed 最大。
    pool_for_best = persist if not persist.empty else res
    best = pool_for_best.sort_values("mean_bleed", ascending=False).iloc[0]
    print("\n" + "=" * 100)
    print("=== ベスト構成(持続ヘッジ中の mean_in_bleed 最大)===")
    print(best.to_string())

    # 最終確認: integrated_dd_test(任意・重い)
    print("\n=== integrated_dd_test(ベスト構成を overlay として統合, DD=20%較正) ===")
    try:
        import importlib
        import mm_lab as mm
        mod = importlib.import_module("strategies.tsmom")
        ovl_params = {"lookback": int(best["lb"]), "band": float(best["band"])}
        ovl_pool = mm.build_pool_for(mod, ovl_params, tf=best["tf"], side=best["side"],
                                     tag=f"tsmom_lb{int(best['lb'])}_b{best['band']}_{best['side']}")
        print(f"  overlay pool: {len(ovl_pool)} trades")
        for w in [0.5, 1.0]:
            r = bl.integrated_dd_test(ovl_pool, overlay_weight=w)
            print(f"  weight={w}: CAGR={r['cagr']:+.1%} maxdd={r['maxdd_mtm']:+.1%} "
                  f"sharpe={r['sharpe']:.2f} boot_p95={r['boot_p95']:+.1%} pos_year={r['pos_year_rate']:.0%}")
        print("  比較: champion単独 CAGR +21.6% / Sharpe 1.21 / 100%プラス年 / boot_p95 -28.5%")
    except Exception as e:  # noqa: BLE001
        print(f"  integrated_dd_test skip: {e}")


if __name__ == "__main__":
    main()
