"""xsec(クロスセクションMR)の原理的ロバスト化 → 合成robust便益が有意になるか。

base xsec の弱点(docstring + 二スリーブ評価で実測):
  - Sharpe 0.43(低リターン資産)=合成便益のボトルネック。
  - 太い尾(p95を悪化させ、合成のレバ偽装の原因)。
  - hold感応・利益後半偏在(IS弱)・pos_yr 73%。

原理的改善(curve-fitでなく、既知弱点への一般的処方):
  ① オーバーラップ・コホート: step<holdで複数コホートを重ねる→「リバランス日の運」を除去・PnL平滑化。
  ② vol-target レッグ: 各脚を当該ペアの直近volに反比例サイズ→高vol脚の尾を抑制→robust較正改善。

検証: 単体ロバスト(pos_yr/IS-OOS/PF/p95)の改善 + 二スリーブrobust ΔCAGR(レバ偽装署名込み)。
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from fxlab import config, universe as uni
import mm_lab as mm
from research.experiments.exp_two_sleeve import lever_to_dd, p95_at_lever, series_stats, TF, BPY

pd.set_option("display.width", 240)
NOTIONAL = 10_000.0


def build_xsec_pool_v2(tf=TF, lookback=9, hold=24, step=6, max_legs=4, vol_win=50,
                       vol_target=True) -> pd.DataFrame:
    import xsec_meanrev as xs
    uni.register_cross_spreads(3.0)
    close = xs.universe_close(tf)
    names = list(close.columns)
    mom = close.pct_change(lookback)
    vol = close.pct_change().rolling(vol_win).std()
    mp = close.mean()
    hs = {p: config.spread_pips(p) * config.pip_size(p) / 2.0 / mp[p] for p in names}
    med_vol = vol.median(axis=1)   # 横断中央vol(その時点)
    rows = []
    start = max(lookback, vol_win) + 1
    for t in range(start, len(close) - hold, step):     # ← step刻み(オーバーラップ)
        score = mom.iloc[t] / vol.iloc[t]
        if score.isna().any():
            continue
        score = score - score.mean()
        s = score.sort_values()
        longs = s[s < 0].index[:max_legs]
        shorts = s[s > 0].index[-max_legs:]
        ets, xts = close.index[t], close.index[t + hold]
        mv = med_vol.iloc[t]
        for grp, d in ((longs, 1), (shorts, -1)):
            for p in grp:
                fwd = close[p].iloc[t + hold] / close[p].iloc[t] - 1.0
                w = float(np.clip(mv / vol[p].iloc[t], 0.3, 3.0)) if vol_target else 1.0
                rows.append((p, ets, xts, d, float(close[p].iloc[t]),
                             float(d * fwd - 2 * hs[p]), hold, w))
    pool = pd.DataFrame(rows, columns=["instr", "entry", "exit", "dir", "entry_price",
                                       "ret", "bars_held", "wsize"])
    pool["z_entry"] = 1.0
    pool["vol_entry"] = 0.01
    return pool.sort_values("entry").reset_index(drop=True)


def sleeve_eval(pool, closes, max_pos, label):
    # wsize を使った固定比率サイジング(vol-target反映)
    wmap = {}
    if "wsize" in pool.columns:
        instr = pool["instr"].to_numpy(); ret = pool["ret"].to_numpy()
        bh = pool["bars_held"].to_numpy(); ws = pool["wsize"].to_numpy()
        for i in range(len(pool)):
            wmap[(instr[i], round(float(ret[i]), 12), int(bh[i]))] = float(ws[i])

    def mk(k):
        base = k / max_pos
        def sizing(ctx):
            w = wmap.get((ctx["instr"], round(float(ctx["ret"]), 12), int(ctx["bars_held"])), 1.0)
            return ctx["equity_real"] * base * w
        return sizing
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

    variants = {
        "xsec_base(step24)":      dict(step=24, vol_target=False, max_pos=8),
        "xsec_overlap(step6)":    dict(step=6, vol_target=False, max_pos=40),
        "xsec_overlap+voltgt":    dict(step=6, vol_target=True, max_pos=40),
        "xsec_voltgt(step24)":    dict(step=24, vol_target=True, max_pos=8),
    }
    eqs = {}
    for name, cfg in variants.items():
        mp = cfg.pop("max_pos")
        pool = build_xsec_pool_v2(**cfg)
        eqs[name] = sleeve_eval(pool, closes, mp, name)

    print("\n=== 二スリーブ robust 評価(champion + 各xsec変種, p95=20%較正)===")
    idx = eqm_c.index
    r_c = eqm_c.pct_change().fillna(0.0)
    Lc_rob, cagr_c_rob = lever_to_dd(r_c, 0.20, robust=True)
    p95_c = p95_at_lever(r_c, lever_to_dd(r_c, 0.20)[0])
    print(f"  champion robCAGR={cagr_c_rob:.3f}  p95@empL={p95_c:.3f}\n")
    rows = []
    for name, eqx in eqs.items():
        r_x = eqx.reindex(idx).pct_change().fillna(0.0)
        mc = eqm_c.resample("ME").last().pct_change().dropna()
        mx = eqx.reindex(idx).resample("ME").last().pct_change().dropna()
        j = mc.index.intersection(mx.index)
        mcorr = float(np.corrcoef(mc.reindex(j), mx.reindex(j))[0, 1])
        best = None
        for w in (0.1, 0.15, 0.2, 0.25, 0.3):
            rp = (1 - w) * r_c + w * r_x
            Lr, cgr = lever_to_dd(rp, 0.20, robust=True)
            Le, _ = lever_to_dd(rp, 0.20, robust=False)
            p95e = p95_at_lever(rp, Le)
            st = series_stats(rp)
            cand = dict(w=w, robCAGR=cgr, drob=cgr - cagr_c_rob, p95_at_empL=p95e,
                        worst_yr=st["worst_yr"], pos_yr=st["pos_yr"])
            if best is None or cand["robCAGR"] > best["robCAGR"]:
                best = cand
        rows.append({"variant": name, "mcorr": round(mcorr, 3),
                     "best_w": best["w"], "robCAGR": round(best["robCAGR"], 4),
                     "Δrob_pp": round(best["drob"] * 100, 2),
                     "p95@empL": round(best["p95_at_empL"], 3),
                     "leverdisguise?": "YES" if best["p95_at_empL"] > p95_c + 0.005 else "no",
                     "worst_yr": round(best["worst_yr"], 3), "pos_yr": best["pos_yr"]})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n  判定: Δrob_pp が有意(>~0.5pp)& leverdisguise=no & worst_yr>=0 なら採用候補。")


if __name__ == "__main__":
    main()
