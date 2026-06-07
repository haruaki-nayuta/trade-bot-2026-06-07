"""イテレーション21d: DD連動ゲート付きオーバーレイの最終頑健性(IS較正→OOS素検証)。

exp21c の発見: オーバーレイを「champion が dd_mtm < -gate のドローダウン中だけ」発火させると、
平時の保険料を払わずに失血窓クッションが効き、統合の Sharpe/tail/CAGR が改善。
本実験: 上位ゲート構成を IS(<2022)で k 較正 → OOS(>=2022)で素のままシミュ、で過剰最適化を暴く。
ゲート無し(常時)との対比も出す。

実行: uv run python exp21d_gate_robust.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import mm_lab as mm

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)

OOS_START = "2022-01-01"

CANDIDATES = [
    ("adx_trend", {"fast": 20, "slow": 50, "adx_period": 14, "adx_th": 20}, "both", 0.05, 1.0),
    ("adx_trend", {"fast": 30, "slow": 100, "adx_period": 14, "adx_th": 20}, "short", 0.05, 1.0),
    ("adx_trend", {"fast": 30, "slow": 100, "adx_period": 14, "adx_th": 20}, "both", 0.05, 1.0),
    ("ma_cross", {"fast": 30, "slow": 100}, "both", 0.05, 1.0),
]


def _build_both(overlay_pool):
    pool_c = mm.build_pool()
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
    return both, fbar, keysrc


def make_sizing_factory(fbar, keysrc, w, gate, max_pos):
    from mm_production import _fz
    def make_sizing(k):
        base = k / max_pos
        def sizing(ctx):
            s = keysrc.get((ctx["instr"], round(float(ctx["ret"]), 12), int(ctx["bars_held"])), "champ")
            if s == "champ":
                return ctx["equity_real"] * base * (_fz(ctx["z"]) / fbar)
            if ctx["dd_mtm"] < -gate:
                return ctx["equity_real"] * base * w
            return 0.0
        return sizing
    return make_sizing


def full_and_oos(both, closes, fbar, keysrc, w, gate, max_pos=8, target_dd=0.20):
    make = make_sizing_factory(fbar, keysrc, w, gate, max_pos)
    # フル期間
    k, eqm, eqr, info = mm.calibrate(both, closes, make, target_dd=target_dd, max_pos=max_pos)
    s = mm.stats(eqm, eqr, info)
    bs = mm.bootstrap_maxdd(eqm, n_boot=800)
    full = {"k": k, "cagr": s["cagr"], "maxdd": s["maxdd_mtm"], "sharpe": s["sharpe"],
            "boot95": bs["p95"], "posyr": s["pos_year_rate"]}
    # IS較正 → OOS素検証
    is_both = both[both["entry"] < OOS_START].reset_index(drop=True)
    oos_both = both[both["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]; oos_cl = closes[closes.index >= OOS_START]
    k_is, *_ = mm.calibrate(is_both, is_cl, make, target_dd=target_dd, max_pos=max_pos)
    eqm_o, eqr_o, info_o = mm.simulate(oos_both, oos_cl, make(k_is), max_pos=max_pos)
    so = mm.stats(eqm_o, eqr_o, info_o)
    oos = {"k_is": k_is, "cagr": so["cagr"], "maxdd": so["maxdd_mtm"],
           "sharpe": so["sharpe"], "posyr": so["pos_year_rate"]}
    return full, oos


def main():
    closes = mm.load_closes()
    # champion単独のIS較正→OOS素検証(基準)
    pool_c = mm.build_pool()
    from mm_production import champion_sizing
    mk = champion_sizing(pool_c, max_pos=8)
    is_c = pool_c[pool_c["entry"] < OOS_START].reset_index(drop=True)
    oos_c = pool_c[pool_c["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]; oos_cl = closes[closes.index >= OOS_START]
    kf, em, er, inf = mm.calibrate(pool_c, closes, mk, 0.20, 8)
    sf = mm.stats(em, er, inf); bsf = mm.bootstrap_maxdd(em, n_boot=800)
    k_is_c, *_ = mm.calibrate(is_c, is_cl, mk, 0.20, 8)
    emo, ero, info_o = mm.simulate(oos_c, oos_cl, mk(k_is_c), max_pos=8)
    sc_o = mm.stats(emo, ero, info_o)
    print("=== 基準: champion単独 ===")
    print(f"  FULL: CAGR {sf['cagr']:+.1%} maxDD {sf['maxdd_mtm']:+.1%} Sharpe {sf['sharpe']:.2f} "
          f"boot95 {bsf['p95']:+.1%} +年 {sf['pos_year_rate']:.0%}")
    print(f"  OOS(IS較正→素): CAGR {sc_o['cagr']:+.1%} maxDD {sc_o['maxdd_mtm']:+.1%} "
          f"Sharpe {sc_o['sharpe']:.2f} +年 {sc_o['pos_year_rate']:.0%}\n")

    print("=== ゲート付きオーバーレイ統合: FULL と OOS(過剰最適化チェック) ===")
    for name, params, side, gate, w in CANDIDATES:
        mod = __import__(f"strategies.{name}", fromlist=["x"])
        ovl = mm.build_pool_for(mod, params, tf="H4", side=side,
                                tag=f"{name}_{'_'.join(str(v) for v in params.values())}_{side}")
        both, fbar, keysrc = _build_both(ovl)
        full, oos = full_and_oos(both, closes, fbar, keysrc, w, gate)
        pstr = "_".join(f"{k}{v}" for k, v in params.items() if k in ("fast", "slow", "adx_th"))
        print(f"\n{name} {pstr} {side} | gate={gate} w={w}")
        print(f"  FULL: k={full['k']:.2f} CAGR {full['cagr']:+.1%} maxDD {full['maxdd']:+.1%} "
              f"Sharpe {full['sharpe']:.2f} boot95 {full['boot95']:+.1%} +年 {full['posyr']:.0%}")
        print(f"  OOS(IS較正k={oos['k_is']:.2f}→素): CAGR {oos['cagr']:+.1%} maxDD {oos['maxdd']:+.1%} "
              f"Sharpe {oos['sharpe']:.2f} +年 {oos['posyr']:.0%}")


if __name__ == "__main__":
    main()
