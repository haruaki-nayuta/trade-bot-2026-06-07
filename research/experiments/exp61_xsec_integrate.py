"""exp61: クロスセクション平均回帰(xsec)をチャンピオンに統合 — 最後の FX 分散候補の確定判定。

背景: xsec_meanrev は「19銘柄を vol正規化リターンで順位付け→最も負けロング/最も勝ちショート」の
ポートフォリオ・ランキング戦略。チャンピオン(per-instrument 平均回帰)とは機構が異なり、月次相関
0.135(≒無相関)。docstring 上は「真価は分散」とあるが、単独 PF1.10・常時8脚で資本を食う弱いエッジ。

exp60 で MTF(D1)は「弱いブックは独立サイジングしても DD予算を食うだけ・失血窓で稼がない」で死亡。
xsec も同じ枠組み(独立スロット+weight αスイープ+失血窓テスト)で robust p95=20% 較正にかけ、
+10%(robust≥+20%)に届くかを確定させる。届かなければ「FX 分散で +10% は不能」を airtight にする。

xsec のサイジングは FLAT(等脚ランキングなので z-power 非適用)。champion は本番 z-power。
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
import xsec_meanrev as xs  # noqa: E402
from fxlab import config, universe as uni  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)


def _fz(z):
    return float(np.clip((z / Z0) ** P, CLIP_LO, CLIP_HI)) if np.isfinite(z) else 1.0


def build_xsec_pool(tf="H4", lookback=9, hold=24, max_legs=4, cross_spread=3.0):
    """xsec のトレードを mm プール形式(instr,entry,exit,dir,entry_price,ret,bars_held,z_entry)で生成。"""
    uni.register_cross_spreads(cross_spread)
    close = xs.universe_close(tf)
    names = list(close.columns)
    mom = close.pct_change(lookback)
    vol = close.pct_change().rolling(50).std()
    mp = close.mean()
    hs = {p: config.spread_pips(p) * config.pip_size(p) / 2.0 / mp[p] for p in names}
    recs = []
    for t in range(max(lookback, 50) + 1, len(close) - hold, hold):
        score = mom.iloc[t] / vol.iloc[t]
        if score.isna().any():
            continue
        score = score - score.mean()
        s = score.sort_values()
        longs = s[s < 0].index[:max_legs]
        shorts = s[s > 0].index[-max_legs:]
        ts_in, ts_out = close.index[t], close.index[t + hold]
        for p in longs:
            ret = (close.iloc[t + hold][p] / close.iloc[t][p] - 1.0) - 2 * hs[p]
            recs.append((p, ts_in, ts_out, 1, close.iloc[t][p], ret, hold))
        for p in shorts:
            ret = -(close.iloc[t + hold][p] / close.iloc[t][p] - 1.0) - 2 * hs[p]
            recs.append((p, ts_in, ts_out, -1, close.iloc[t][p], ret, hold))
    pool = pd.DataFrame(recs, columns=["instr", "entry", "exit", "dir", "entry_price",
                                       "ret", "bars_held"])
    pool["z_entry"] = np.nan  # flat サイジング(z 非使用)
    pool["book"] = "XSEC"
    return pool.sort_values("entry").reset_index(drop=True)


def tag(pool, book):
    p = pool.copy(); p["book"] = book; return p


def make_mixed_sizing(pool, alpha, caps):
    """H4=z-power(本番), XSEC=flat。alpha は XSEC ブックの相対倍率。make(m)→sizing。"""
    zz = pool.loc[pool["book"] == "H4", "z_entry"].to_numpy()
    fbar_h4 = float(np.mean([_fz(z) for z in zz])) or 1.0

    def make(m):
        def sizing(ctx):
            bk = ctx["book"]; cap = caps.get(bk, 8)
            if bk == "H4":
                return ctx["equity_real"] * (m / cap) * (_fz(ctx["z"]) / fbar_h4)
            return ctx["equity_real"] * (m / cap) * alpha  # XSEC flat
        return sizing
    return make


def calibrate_robust_books(pool, closes, make, caps, target=0.20, n_boot=400,
                           lo=0.02, hi=12.0, iters=18, seed=0):
    def p95_of(m):
        eqm, _, _ = simulate_books(pool, closes, make(m), caps=caps)
        return abs(mm.bootstrap_maxdd(eqm, n_boot=n_boot, seed=seed)["p95"])
    if p95_of(hi) <= target:
        eqm, eqr, info = simulate_books(pool, closes, make(hi), caps=caps)
        return hi, eqm, eqr, info
    for _ in range(iters):
        mid = (lo + hi) / 2
        if p95_of(mid) > target:
            hi = mid
        else:
            lo = mid
    eqm, eqr, info = simulate_books(pool, closes, make(lo), caps=caps)
    return lo, eqm, eqr, info


def main():
    print("=" * 72)
    print("  exp61: xsec(クロスセクション平均回帰)統合 — 確定判定")
    print("=" * 72)
    pool_h4 = tag(build_pool_d1(tf="H4"), "H4")
    pool_xs = build_xsec_pool(tf="H4")
    closes = mm.load_closes(tf="H4")
    print(f"  H4 {len(pool_h4)} trades / XSEC {len(pool_xs)} trades / grid {len(closes)}")
    CAPS = {"H4": 8, "XSEC": 8}

    pool_mix = pd.concat([pool_h4, pool_xs], ignore_index=True).sort_values("entry").reset_index(drop=True)
    print("\n[②] XSEC weight α スイープ(独立スロット H4=8/XSEC=8, robust p95=20%, seed0)")
    print(f"     {'α':>5} {'m':>6} {'CAGR':>9} {'p95':>8} {'empDD':>8} {'posY':>6} {'worstY':>8} {'Sharpe':>7} {'taken':>7}")
    results = []
    for alpha in [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]:
        make = make_mixed_sizing(pool_mix, alpha, CAPS)
        m, eqm, eqr, info = calibrate_robust_books(pool_mix, closes, make, CAPS, n_boot=400)
        s = mm.stats(eqm, eqr, info, tf="H4")
        bs = mm.bootstrap_maxdd(eqm, n_boot=1500, seed=0)
        results.append((alpha, s["cagr"]))
        print(f"     {alpha:>5.2f} {m:>6.2f} {s['cagr']:>+8.2%} {bs['p95']:>+7.1%} "
              f"{s['maxdd_mtm']:>+7.1%} {s['pos_year_rate']:>5.0%} {s['worst_year']:>+7.1%} "
              f"{s['sharpe']:>7.2f} {s['n_taken']:>7.0f}")
    base = [c for a, c in results if a == 0.0][0]
    best_a, best_c = max(results, key=lambda x: x[1])
    print(f"\n     ベースライン(α=0, 純H4) = {base:+.2%}")
    print(f"     最良 α={best_a:.2f} → {best_c:+.2%}  ({best_c-base:+.2f}pp, {(best_c/base-1)*100:+.1f}% 相対)")

    # 失血窓テスト
    print("\n[③] 失血窓テスト: H4 最悪十分位月における XSEC の平均月次リターン")
    fbar_h4 = float(np.mean([_fz(z) for z in pool_h4["z_entry"]])) or 1.0
    s_h4 = mm.simulate(pool_h4, closes, lambda c: c["equity_real"]*(1/8)*(_fz(c["z"])/fbar_h4), max_pos=8)
    mh4 = s_h4[0].resample("ME").last().pct_change().dropna()
    # XSEC equity (flat, 8 slots)
    exm, _, _ = simulate_books(pool_xs, closes, lambda c: c["equity_real"]*(1/8), caps={"XSEC": 8})
    mxs = exm.resample("ME").last().pct_change().dropna()
    j = pd.concat([mh4.rename("h4"), mxs.rename("xs")], axis=1).dropna()
    thr = j["h4"].quantile(0.10)
    bleed, normal = j[j["h4"] <= thr], j[j["h4"] > thr]
    print(f"     corr(H4,XSEC) 月次 = {j['h4'].corr(j['xs']):+.3f}")
    print(f"     H4 最悪十分位月(n={len(bleed)}): XSEC 平均 {bleed['xs'].mean():+.3%} (H4 {bleed['h4'].mean():+.3%})")
    print(f"     その他の月(n={len(normal)}):       XSEC 平均 {normal['xs'].mean():+.3%}")

    print("\n" + "=" * 72)
    print("  判定: 最良 α robust ≥ +20.0%(+10%相対)かつ p95 非悪化 → 次段。さもなくば FX分散を閉鎖。")
    print("=" * 72)


if __name__ == "__main__":
    main()
