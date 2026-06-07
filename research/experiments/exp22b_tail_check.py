"""イテレーション22b: 22の核心矛盾を詰める。
  - 最良CAGR構成(gate=0.05 w=0.25)は CAGR +26.5%だが boot95 -31.6%(基準-28.7%より悪化)=テール悪化。
  - 唯一テール改善した構成(gate=0.05 w=1.0)は boot95 -23.3%・empDD -13.2%だが CAGR +21.8%(基準+0.2pp)。
  - F: IS較正→OOS素検証で統合 +27.7% < champion単独 +30.5%(overlayがOOSでドラッグ)。

詰める点:
  1. robust較正(boot p95=20%に縛る)で比較。empirical 20%同士の比較は「テールを犠牲にCAGRを買った」だけかも。
     robust 20%なら同一テール基準で CAGR を比べられる → overlay が本当に純増か。
  2. boot95を n_boot=3000 で安定化、複数seedで頑健性。
  3. champion単独 robust基準(+14.3%)と統合 robust を比較。
実行: uv run python exp22b_tail_check.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import mm_lab as mm

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)

BEST = ("adx_trend", {"fast": 30, "slow": 100, "adx_period": 14, "adx_th": 20}, "short")


def build_gated(overlay_pool):
    pool_c = mm.build_pool()
    closes = mm.load_closes()
    from mm_production import _fz
    pc = pool_c.copy(); pc["src"] = "champ"
    po = overlay_pool.copy(); po["src"] = "ovl"
    both = pd.concat([pc, po], ignore_index=True).sort_values("entry").reset_index(drop=True)
    fbar = float(np.mean([_fz(z) for z in pool_c["z_entry"].to_numpy()])) or 1.0
    instr = both["instr"].to_numpy(); ret = both["ret"].to_numpy(); bh = both["bars_held"].to_numpy()
    src = both["src"].to_numpy()
    keysrc = {}
    for i in range(len(both)):
        keysrc[(instr[i], round(float(ret[i]), 12), int(bh[i]))] = src[i]
    return both, closes, fbar, keysrc


def make_factory(fbar, keysrc, w, gate):
    from mm_production import _fz
    def make_sizing(k):
        base = k / 8
        def sizing(ctx):
            s = keysrc.get((ctx["instr"], round(float(ctx["ret"]), 12), int(ctx["bars_held"])), "champ")
            if s == "champ":
                return ctx["equity_real"] * base * (_fz(ctx["z"]) / fbar)
            if gate is not None and ctx["dd_mtm"] >= -gate:
                return 0.0
            return ctx["equity_real"] * base * w
        return sizing
    return make_sizing


def stable_boot95(eqm, seeds=(0, 1, 2, 3, 4), n_boot=2000):
    vals = [mm.bootstrap_maxdd(eqm, n_boot=n_boot, seed=s)["p95"] for s in seeds]
    return float(np.mean(vals)), float(np.std(vals))


def main():
    name, params, side = BEST
    mod = __import__(f"strategies.{name}", fromlist=["x"])
    ovl = mm.build_pool_for(mod, params, tf="H4", side=side,
                            tag=f"{name}_{'_'.join(str(v) for v in params.values())}_{side}")
    both, closes, fbar, keysrc = build_gated(ovl)

    # --- champion単独: empirical & robust ---
    from mm_production import champion_sizing
    pool_c = mm.build_pool()
    mkc = champion_sizing(pool_c, max_pos=8)
    kE, emE, erE, iE = mm.calibrate(pool_c, closes, mkc, 0.20, 8)
    sE = mm.stats(emE, erE, iE)
    b95E, b95Es = stable_boot95(emE)
    kR, emR, erR, iR, p95R = mm.calibrate_robust(pool_c, closes, mkc, 0.20, 8, n_boot=800)
    sR = mm.stats(emR, erR, iR)
    print("=== champion単独 基準 ===")
    print(f"  empirical20%: k={kE:.2f} CAGR {sE['cagr']:+.1%} empDD {sE['maxdd_mtm']:+.1%} "
          f"Sharpe {sE['sharpe']:.2f} boot95 {b95E:+.1%}±{b95Es:.1%} +年 {sE['pos_year_rate']:.0%}")
    print(f"  robust20%(p95縛り): k={kR:.2f} CAGR {sR['cagr']:+.1%} empDD {sR['maxdd_mtm']:+.1%} "
          f"Sharpe {sR['sharpe']:.2f} +年 {sR['pos_year_rate']:.0%}\n")

    # --- 統合: 主要構成を empirical & robust 両方で ---
    cfgs = [(0.05, 0.25), (0.05, 0.5), (0.05, 1.0), (0.10, 0.5), (0.10, 1.0)]
    print("=== 統合: empirical20% と robust20% の両基準で(同一テール基準の純増判定)===")
    print(f"{'gate':>6}{'w':>5} | {'empCAGR':>8}{'empboot95':>11}{'empShrp':>8} | "
          f"{'robCAGR':>8}{'robk':>7}{'robShrp':>8}")
    print("-" * 72)
    for gate, w in cfgs:
        make = make_factory(fbar, keysrc, w, gate)
        kE2, em2, er2, i2 = mm.calibrate(both, closes, make, 0.20, 8)
        s2 = mm.stats(em2, er2, i2)
        b952, b952s = stable_boot95(em2)
        kR2, emr2, err2, ir2, p95r2 = mm.calibrate_robust(both, closes, make, 0.20, 8, n_boot=800)
        sr2 = mm.stats(emr2, err2, ir2)
        print(f"{gate:>6.2f}{w:>5.2f} | {s2['cagr']:>+8.1%}{b952:>+10.1%}±{s2['sharpe']:>7.2f} | "
              f"{sr2['cagr']:>+8.1%}{kR2:>7.2f}{sr2['sharpe']:>8.2f}")

    print(f"\n  champion単独 robust CAGR = {sR['cagr']:+.1%}(これを超えれば「同一テール基準で純増」)")


if __name__ == "__main__":
    main()
