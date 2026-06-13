"""USDJPY ~1日モメンタム(半スプレッド)の robust 2スリーブ監査 — レバ偽装でないか決着。

月次Sharpe +0.13 改善は本物(robust CAGR増)か、xsecの+1.75ppのようなレバ偽装(boot p95悪化)か。
champion(H4 MtM, 20%DD)と USDJPYモメンタム(H1, 半スプレッド)を日次リターンで合成し、
経験的20%DD と robust(boot p95=20%)の両方にレバ調整して ΔCAGR と p95@empL を出す。
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from fxlab import load, run
import fxlab.config as C
from strategies.tsmom import generate_signals as tsmom_sig
import mm_lab as mm
from research.experiments.exp_two_sleeve import lever_to_dd, p95_at_lever

ORIG = dict(C.SPREADS_PIPS)


def champion_daily_returns():
    closes = mm.load_closes("H4")
    from mm_production import champion_sizing
    pool = mm.build_pool()
    k, eqm, eqr, info = mm.calibrate(pool, closes, champion_sizing(pool, max_pos=8),
                                     target_dd=0.20, max_pos=8)
    d = eqm.resample("D").last().dropna()
    return d.pct_change().dropna()


def mom_daily_returns(pairs, lookback=24, spread_mult=0.5):
    C.SPREADS_PIPS = {k: v * spread_mult for k, v in ORIG.items()}
    eqs = []
    for p in pairs:
        data = load(p, "H1")
        pf = run(p, "H1", tsmom_sig, {"lookback": lookback, "band": 0.0},
                 data=data, size_mode="value", side="both")
        v = pf.value()
        if hasattr(v, "columns"):
            v = v.iloc[:, 0]
        eqs.append(v.resample("D").last())
    C.SPREADS_PIPS = dict(ORIG)
    eq = pd.concat(eqs, axis=1).ffill().dropna()
    port = eq.sum(axis=1)              # 等notional合算
    return port.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def main():
    r_c = champion_daily_returns()
    print("champion daily returns:", r_c.index[0].date(), "..", r_c.index[-1].date(), len(r_c))

    Lc_e, cagr_c_e = lever_to_dd(r_c, 0.20, robust=False)
    Lc_r, cagr_c_r = lever_to_dd(r_c, 0.20, robust=True)
    p95_c = p95_at_lever(r_c, Lc_e)
    print(f"champion: empCAGR={cagr_c_e:.3f} robCAGR={cagr_c_r:.3f} p95@empL={p95_c:.3f}\n")

    combos = [("USDJPY単独", ["USDJPY"], 0.5), ("USDJPY単独", ["USDJPY"], 1.0),
              ("EUR/JPY/GBP", ["EURUSD", "USDJPY", "GBPUSD"], 0.5),
              ("EUR/JPY/GBP", ["EURUSD", "USDJPY", "GBPUSD"], 1.0)]
    for label, pairs, sm in combos:
        r_x = mom_daily_returns(pairs, spread_mult=sm)
        label = f"{label} spread×{sm}"
        idx = r_c.index.intersection(r_x.index)
        a, b = r_c.reindex(idx).fillna(0.0), r_x.reindex(idx).fillna(0.0)
        b = b * (a.std() / b.std())                     # リスクパリティ(volを揃える)
        cr = float(np.corrcoef(a.values, b.values)[0, 1])
        print(f"=== {label} (半スプレッド, 日次, vol正規化) corr={cr:+.3f} ===")
        print(f"{'w':>5} {'empCAGR':>8} {'Δemp':>7} {'robCAGR':>8} {'Δrob':>7} {'p95@empL':>9} {'偽装?':>6}")
        for w in (0.0, 0.1, 0.2, 0.3):
            rp = a if w == 0 else (1 - w) * a + w * b
            Le, cge = lever_to_dd(rp, 0.20, robust=False)
            Lr, cgr = lever_to_dd(rp, 0.20, robust=True)
            p95e = p95_at_lever(rp, Le)
            disguise = "YES" if p95e > p95_c + 0.005 else "no"
            print(f"{w:>5.1f} {cge:>8.3f} {cge-cagr_c_e:>+7.3f} {cgr:>8.3f} {cgr-cagr_c_r:>+7.3f} {p95e:>9.3f} {disguise:>6}")
        print()
    print("判定: Δrob>0(robust較正で純増)& 偽装=no(p95悪化なし)なら本物のCAGR底上げ補完。")


if __name__ == "__main__":
    main()
