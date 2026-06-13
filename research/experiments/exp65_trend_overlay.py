"""exp65: トレンドフォロー・オーバーレイ統合(現チャンピオン d1+P4.0+mp8 で再検証)。

web リサーチの第1推奨=「平均回帰にトレンドフォローを混ぜる(crisis alpha・凸性・無相関)」。
これは唯一「正しい凸性」を持つ分散候補: MR変種(D1/xsec/statarb)は全て持続トレンドで一緒に沈む
([[29]])が、トレンドフォローは**チャンピオンの失血窓(持続USDトレンド)でこそ稼ぐ**はず。

reports/10 は旧チャンピオン(P4.0/d1 前)+共通スロット+月次統合で「net黒字にならない」と棄却。
本実験は現チャンピオン+独立スロット simulate_books+失血窓スクリーン(新プロトコル[[29]])で再検証。

手順(新プロトコル: 統合前にスクリーン):
  ① 各トレンド書(tsmom H4/D1, breakout H4)の単独 Sharpe と **失血窓十分位での平均リターン**。
     失血窓でプラスでなければ「正しい凸性」を持たない=統合しても無駄(即閉鎖)。
  ② スクリーン通過(失血窓プラス)した書だけ、独立スロットで weight α スイープ・robust p95=20% 較正。
     +10%(robust≥+20%)に届くか、または reports/10 同様に保険料(平時ドラッグ)がDD便益を超えるか。
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))

import mm_lab as mm  # noqa: E402
from mm_production import Z0, P, CLIP_LO, CLIP_HI, build_pool_d1  # noqa: E402
from exp60_mtf_riskbudget import simulate_books  # noqa: E402
import strategies.tsmom as tsmom  # noqa: E402
import strategies.breakout_trend as brk  # noqa: E402
from fxlab import universe as uni  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)


def _fz(z):
    return float(np.clip((z / Z0) ** P, CLIP_LO, CLIP_HI)) if np.isfinite(z) else 1.0


def tag(pool, book):
    p = pool.copy(); p["book"] = book; return p


def champion_eq_monthly(pool_h4, closes):
    fbar = float(np.mean([_fz(z) for z in pool_h4["z_entry"]]))
    sz = lambda c: c["equity_real"] * (1 / 8) * (_fz(c["z"]) / fbar)
    eqm, _, _ = mm.simulate(pool_h4, closes, sz, max_pos=8)
    return eqm.resample("ME").last().pct_change().dropna()


def trend_book_eq_monthly(pool_tr, closes, cap=6):
    eqm, _, _ = simulate_books(tag(pool_tr, "TR"), closes,
                               lambda c: c["equity_real"] * (1 / cap), caps={"TR": cap})
    return eqm.resample("ME").last().pct_change().dropna(), eqm


def stats_simple(eqm):
    years = (eqm.index[-1] - eqm.index[0]).days / 365.25
    cagr = (eqm.iloc[-1] / eqm.iloc[0]) ** (1 / years) - 1 if eqm.iloc[-1] > 0 else -1
    r = eqm.pct_change().dropna()
    sh = r.mean() / r.std() * np.sqrt(6 * 252) if r.std() > 0 else float("nan")
    return cagr, sh


def make_mixed(pool, alpha, caps):
    zz = pool.loc[pool["book"] == "H4", "z_entry"].to_numpy()
    fbar = float(np.mean([_fz(z) for z in zz]))

    def make(m):
        def sizing(ctx):
            bk = ctx["book"]; cap = caps.get(bk, 8)
            if bk == "H4":
                return ctx["equity_real"] * (m / cap) * (_fz(ctx["z"]) / fbar)
            return ctx["equity_real"] * (m / cap) * alpha
        return sizing
    return make


def calib(pool, closes, make, caps, target=0.20, n_boot=400, lo=0.02, hi=12.0, iters=18, seed=0):
    def p95(m):
        eqm, _, _ = simulate_books(pool, closes, make(m), caps=caps)
        return abs(mm.bootstrap_maxdd(eqm, n_boot=n_boot, seed=seed)["p95"])
    if p95(hi) <= target:
        eqm, eqr, info = simulate_books(pool, closes, make(hi), caps=caps)
        return hi, eqm, eqr, info
    for _ in range(iters):
        mid = (lo + hi) / 2
        if p95(mid) > target:
            hi = mid
        else:
            lo = mid
    eqm, eqr, info = simulate_books(pool, closes, make(lo), caps=caps)
    return lo, eqm, eqr, info


def main():
    print("=" * 72)
    print("  exp65: トレンドフォロー・オーバーレイ統合(現チャンピオンで再検証)")
    print("=" * 72)
    uni.register_cross_spreads(3.0)
    pool_h4 = tag(build_pool_d1(tf="H4"), "H4")
    closes = mm.load_closes(tf="H4")
    mh4 = champion_eq_monthly(pool_h4, closes)
    thr = mh4.quantile(0.10)
    print(f"  champion {len(pool_h4)} trades / 失血窓閾値(月次下位十分位) {thr:+.2%}")

    # トレンド書の候補プール
    print("\n[①] トレンド書スクリーン(単独 Sharpe + 失血窓十分位の平均リターン)")
    candidates = {}
    specs = [
        ("tsmom_H4_lb100", tsmom, dict(tsmom.PARAMS), "H4"),
        ("tsmom_H4_lb200", tsmom, {"lookback": 200, "band": 0.0}, "H4"),
        ("tsmom_D1_lb60", tsmom, {"lookback": 60, "band": 0.0}, "D1"),
        ("breakout_H4", brk, dict(brk.PARAMS), "H4"),
    ]
    print(f"  {'book':>16} {'tf':>4} {'trades':>7} {'CAGR':>8} {'Sharpe':>7} {'bleed窓平均':>11} {'通常月平均':>10}")
    for name, mod, params, tf in specs:
        pool_tr = mm.build_pool_for(mod, params, tf=tf, tag=name, side="both")
        if pool_tr.empty:
            print(f"  {name:>16} {tf:>4}   (空)")
            continue
        mtr, eqfull = trend_book_eq_monthly(pool_tr, closes)
        cagr, sh = stats_simple(eqfull)
        j = pd.concat([mh4.rename("h4"), mtr.rename("tr")], axis=1).dropna()
        bleed = j[j["h4"] <= thr]["tr"].mean()
        normal = j[j["h4"] > thr]["tr"].mean()
        print(f"  {name:>16} {tf:>4} {len(pool_tr):>7} {cagr:>+7.2%} {sh:>7.2f} {bleed:>+10.3%} {normal:>+9.3%}")
        candidates[name] = (pool_tr, bleed)

    # ② 失血窓プラスの書を統合
    print("\n[②] 失血窓プラスの書を独立スロット統合(robust p95=20%, weight α スイープ)")
    base = None
    for name, (pool_tr, bleed) in candidates.items():
        if bleed <= 0:
            print(f"  -- {name}: 失血窓 {bleed:+.3%} ≤ 0 = 正しい凸性なし → 統合スキップ")
            continue
        print(f"\n  == {name}(失血窓 {bleed:+.3%})統合 ==")
        pool_mix = pd.concat([pool_h4, tag(pool_tr, "TR")], ignore_index=True).sort_values("entry").reset_index(drop=True)
        CAPS = {"H4": 8, "TR": 6}
        print(f"     {'α':>5} {'CAGR':>9} {'p95':>8} {'empDD':>8} {'posY':>6} {'worstY':>8}")
        for alpha in [0.0, 0.25, 0.5, 1.0, 2.0]:
            make = make_mixed(pool_mix, alpha, CAPS)
            m, eqm, eqr, info = calib(pool_mix, closes, make, CAPS, n_boot=400)
            s = mm.stats(eqm, eqr, info, tf="H4")
            bs = mm.bootstrap_maxdd(eqm, n_boot=1200, seed=0)
            if alpha == 0.0:
                base = s["cagr"]
            print(f"     {alpha:>5.2f} {s['cagr']:>+8.2%} {bs['p95']:>+7.1%} {s['maxdd_mtm']:>+7.1%} "
                  f"{s['pos_year_rate']:>5.0%} {s['worst_year']:>+7.1%}"
                  + (f"  ({(s['cagr']/base-1)*100:+.1f}% 相対)" if base else ""))

    print("\n" + "=" * 72)
    print("  判定: いずれかの書×α で robust ≥ +20%(+10%相対)→ 採用候補(要敵対検証)。")
    print("        全て base 未満なら reports/10 を現チャンピオンで再確認=トレンド分散も閉鎖。")
    print("=" * 72)


if __name__ == "__main__":
    main()
