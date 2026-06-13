"""通貨内最強の単体MR(zscore_meanrev 等)を独立スリーブで2スリーブrobust評価。

スクリーニングで単体黒字だった通貨戦略3本(全てMR系・正相関0.13-0.33)を、
champion と独立スリーブ合成→p95=20%較正で robust ΔCAGR を実測。
同ファミリー=失血窓共有なので期待薄だが、「納得のため数字で」確認する最後の石。
"""
from __future__ import annotations

import warnings
import importlib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from fxlab import config, universe as uni
import mm_lab as mm
from research.experiments.exp_two_sleeve import lever_to_dd, p95_at_lever, series_stats, TF

pd.set_option("display.width", 240)


def sleeve_equity(pool, closes, max_pos, label):
    def mk(k):
        base = k / max_pos
        return lambda ctx: ctx["equity_real"] * base
    k, eqm, eqr, info = mm.calibrate(pool, closes, mk, target_dd=0.20, max_pos=max_pos)
    s = mm.stats(eqm, eqr, info)
    print(f"  {label}: k={k:.2f} CAGR={s['cagr']:.1%} Sharpe={s['sharpe']:.2f} "
          f"pos_yr={s['pos_year_rate']:.0%} worst_yr={s['worst_year']:.1%} maxconc={s['max_conc']}")
    return eqm


def main():
    closes = mm.load_closes(TF)
    from mm_production import champion_sizing
    pool_c = mm.build_pool()
    kc, eqm_c, eqr_c, info_c = mm.calibrate(pool_c, closes, champion_sizing(pool_c, max_pos=8),
                                            target_dd=0.20, max_pos=8)
    sc = mm.stats(eqm_c, eqr_c, info_c)
    print("=== スリーブ(20%経験DD較正)===")
    print(f"  champion: CAGR={sc['cagr']:.1%} Sharpe={sc['sharpe']:.2f} pos_yr={sc['pos_year_rate']:.0%}")

    r_c = eqm_c.pct_change().fillna(0.0)
    Lc_rob, cagr_c_rob = lever_to_dd(r_c, 0.20, robust=True)
    p95_c = p95_at_lever(r_c, lever_to_dd(r_c, 0.20)[0])

    cands = ["zscore_meanrev", "meanrev_range", "rsi_meanrev"]
    rows = []
    for nm in cands:
        mod = importlib.import_module(f"strategies.{nm}")
        params = dict(getattr(mod, "PARAMS", {}))
        pool = mm.build_pool_for(mod, params, tag=f"blend_{nm}", side="both", cache=False)
        if pool.empty:
            print(f"  [skip] {nm}: empty pool"); continue
        eqx = sleeve_equity(pool, closes, max_pos=12, label=nm)
        r_x = eqx.reindex(eqm_c.index).pct_change().fillna(0.0)
        mc = eqm_c.resample("ME").last().pct_change().dropna()
        mx = eqx.reindex(eqm_c.index).resample("ME").last().pct_change().dropna()
        j = mc.index.intersection(mx.index)
        mcorr = float(np.corrcoef(mc.reindex(j), mx.reindex(j))[0, 1])
        best = None
        for w in (0.1, 0.15, 0.2, 0.25, 0.3):
            rp = (1 - w) * r_c + w * r_x
            Lr, cgr = lever_to_dd(rp, 0.20, robust=True)
            Le, _ = lever_to_dd(rp, 0.20, robust=False)
            cand = dict(w=w, robCAGR=cgr, drob=cgr - cagr_c_rob,
                        p95e=p95_at_lever(rp, Le), worst=series_stats(rp)["worst_yr"])
            if best is None or cand["robCAGR"] > best["robCAGR"]:
                best = cand
        rows.append({"complement": nm, "mcorr": round(mcorr, 3), "best_w": best["w"],
                     "robCAGR": round(best["robCAGR"], 4), "Δrob_pp": round(best["drob"] * 100, 2),
                     "p95@empL": round(best["p95e"], 3),
                     "leverdisguise?": "YES" if best["p95e"] > p95_c + 0.005 else "no",
                     "worst_yr": round(best["worst"], 3)})
    print(f"\n  champion robCAGR={cagr_c_rob:.3f}  p95@empL={p95_c:.3f}")
    print("\n=== 通貨内MR補完の2スリーブ robust 結果 ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  Δrob_pp>~0.5 & leverdisguise=no & worst_yr>=0 なら採用候補。全て該当外なら通貨内の壁確定。")


if __name__ == "__main__":
    main()
