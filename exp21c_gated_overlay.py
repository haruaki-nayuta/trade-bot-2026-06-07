"""イテレーション21c: DD連動ゲート付きオーバーレイ — 「平時の保険料」を払わない統合。

exp21b の発見: ベスト失血窓ヘッジ(adx_trend/ma_cross)は失血窓で確かに稼ぐ(6/7年プラス,
2022除外でもプラス)が、**固定比率で常時投入すると平時83%の月の保険料が CAGR を潰す**
(統合 CAGR が +21.6% を大きく下回り、boot_p95 も悪化=テールが増える)。

仮説: オーバーレイを「チャンピオンが既にドローダウン中(失血レジーム)のときだけ」発火させれば、
平時の保険料を払わずに失血窓のクッションだけ得られる → 統合 DD が下がり k 余地が増えて CAGR 純増?

実装: simulate を自前ループで拡張せず、integrated_dd_test の make_sizing を流用しつつ
「オーバーレイのエントリー時点で champion MtM が dd_mtm < -gate ならフル、そうでなければ 0」
にする gated sizing を作って calibrate。champion 側は乖離連動zで常時。

実行: uv run python exp21c_gated_overlay.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import bleed_lab as bl
import mm_lab as mm

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)


CANDIDATES = [
    ("adx_trend", {"fast": 30, "slow": 100, "adx_period": 14, "adx_th": 20}, "short"),
    ("adx_trend", {"fast": 20, "slow": 50, "adx_period": 14, "adx_th": 20}, "both"),
    ("adx_trend", {"fast": 30, "slow": 100, "adx_period": 14, "adx_th": 20}, "both"),
    ("ma_cross", {"fast": 30, "slow": 100}, "both"),
]


def integrated_gated(overlay_pool, overlay_weight=1.0, gate=0.0, max_pos=8, target_dd=0.20):
    """champion + overlay を1口座統合。overlay は ctx['dd_mtm'] < -gate のときだけ発火(平時は 0)。

    gate=0.0 で「水面下なら発火」。gate=0.05 で「-5%超のドローダウン中だけ」。
    """
    pool_c = mm.build_pool()
    closes = mm.load_closes()
    from mm_production import champion_sizing, _fz  # noqa: F401
    from mm_production import champion_sizing as _cs  # noqa: F401
    pc = pool_c.copy(); pc["src"] = "champ"
    po = overlay_pool.copy(); po["src"] = "ovl"
    both = pd.concat([pc, po], ignore_index=True).sort_values("entry").reset_index(drop=True)

    from mm_production import _fz as fz
    fbar = float(np.mean([fz(z) for z in pool_c["z_entry"].to_numpy()])) or 1.0
    src = both["src"].to_numpy()
    instr = both["instr"].to_numpy(); ret = both["ret"].to_numpy(); bh = both["bars_held"].to_numpy()
    keysrc = {}
    for i in range(len(both)):
        keysrc[(instr[i], round(float(ret[i]), 12), int(bh[i]))] = src[i]

    def make_sizing(k):
        base = k / max_pos
        def sizing(ctx):
            s = keysrc.get((ctx["instr"], round(float(ctx["ret"]), 12), int(ctx["bars_held"])), "champ")
            if s == "champ":
                return ctx["equity_real"] * base * (fz(ctx["z"]) / fbar)
            # overlay: DD連動ゲート。水面下(-gate超)のときだけ発火。
            if ctx["dd_mtm"] < -gate:
                return ctx["equity_real"] * base * overlay_weight
            return 0.0
        return sizing

    k, eqm, eqr, info = mm.calibrate(both, closes, make_sizing, target_dd=target_dd, max_pos=max_pos)
    s = mm.stats(eqm, eqr, info)
    bs = mm.bootstrap_maxdd(eqm, n_boot=800)
    return {"k": k, "cagr": s["cagr"], "maxdd_mtm": s["maxdd_mtm"], "sharpe": s["sharpe"],
            "pos_year_rate": s["pos_year_rate"], "boot_p95": bs["p95"], "worst_year": s["worst_year"]}


def main():
    print("基準: champion単独 CAGR +21.6% / Sharpe 1.21 / 100%プラス年 / boot_p95 -28.7%\n")
    print("=== DD連動ゲート付きオーバーレイ(平時は発火せず=保険料ゼロ) ===")
    print(f"{'候補':<40}{'gate':>6}{'w':>5}{'k':>7}{'CAGR':>9}{'maxDD':>8}{'Shrp':>7}{'boot95':>9}{'+年':>6}")
    print("-" * 97)
    for name, params, side in CANDIDATES:
        mod = __import__(f"strategies.{name}", fromlist=["x"])
        ovl = mm.build_pool_for(mod, params, tf="H4", side=side,
                                tag=f"{name}_{'_'.join(str(v) for v in params.values())}_{side}")
        if ovl.empty:
            continue
        pstr = "_".join(f"{k}{v}" for k, v in params.items() if k in ("fast", "slow", "adx_th"))
        label = f"{name} {pstr} {side}"
        for gate in [0.0, 0.05, 0.10]:
            for w in [0.5, 1.0, 1.5]:
                r = integrated_gated(ovl, overlay_weight=w, gate=gate, max_pos=8)
                flag = "  <<" if r["cagr"] > 0.216 else ""
                print(f"{label:<40}{gate:>6.2f}{w:>5.1f}{r['k']:>7.2f}{r['cagr']:>+9.1%}"
                      f"{r['maxdd_mtm']:>+8.1%}{r['sharpe']:>7.2f}{r['boot_p95']:>+9.1%}"
                      f"{r['pos_year_rate']:>5.0%}{flag}")
        print()


if __name__ == "__main__":
    main()
