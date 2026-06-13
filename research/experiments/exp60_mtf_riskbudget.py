"""exp60: リスク予算配分のマルチ時間足アンサンブル — MTF レバーの確定判定。

exp59(スカウト)の発見:
  ・H4 単独 robust +18.18%(Sharpe 1.37)/ D1 単独 robust +1.34%(Sharpe 0.25, 弱い)/ H1 死亡。
  ・corr(H4, D1) = -0.025 ≈ ゼロ(分散の素地は理想的)。
  ・しかし素朴統合(共通 k・共通 max_pos)は +8.9% に半減 → D1 の高い1トレード MtM ボラが
    口座全体の k を 6.08→2.83 に引き下げ、優秀な H4 を兵糧攻めにしていた(サイジング欠陥)。

本実験は欠陥を正した「正しい分散テスト」:
  ① 各ブックを独立にサイジング(D1 を weight α で縮小)+ 独立スロット(H4=8, D1=8 別枠)。
  ② グローバル倍率 m を二分探索して統合 p95 DD=20% に較正 → CAGR(α) 曲線。
     α=0 で純 H4(+18.18%)を回収するはず(サニティ)。最良 α で +2pp(+10%相対)超えるか。
  ③ 失血窓条件付き(reports/10 流): H4 の最悪十分位月に D1 が稼ぐか(テール便益の機構確認)。
  ④ 診断: D1 を d0(遅延なし)や短い window でも測り、D1 の弱さがパラメータか本質かを切り分け
     (採用候補ではなく機構理解のため)。

判定: 最良 α で robust ≥ +20.0%(+10%相対)かつ p95 非悪化なら次段(敵対検証)へ。
さもなくば MTF-同一エッジを恒久閉鎖し、reports/19 の「同一リスク契約での天井」を再確認。
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

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)


def _fz(z):
    return float(np.clip((z / Z0) ** P, CLIP_LO, CLIP_HI)) if np.isfinite(z) else 1.0


def simulate_books(pool, closes, sizing, *, init=10_000.0, caps=None, vol_win=120):
    """ブック別スロット上限を持つバー駆動シミュレータ(mm.simulate の拡張)。

    pool は 'book' 列を持つ。caps = {book: max_pos}。sizing(ctx) は ctx['book'] を見てよい。
    """
    caps = caps or {}
    grid = closes.index
    col_of = {c: i for i, c in enumerate(closes.columns)}
    carr = closes.to_numpy()
    n = len(grid)

    gi = grid.to_numpy()
    entry_pos = np.clip(np.searchsorted(gi, pool["entry"].to_numpy(), side="left"), 0, n - 1)
    exit_pos = np.clip(np.searchsorted(gi, pool["exit"].to_numpy(), side="left"), 0, n - 1)

    by_entry = {}
    for ti in range(len(pool)):
        by_entry.setdefault(int(entry_pos[ti]), []).append(ti)

    instr_arr = pool["instr"].to_numpy()
    book_arr = pool["book"].to_numpy()
    dir_arr = pool["dir"].to_numpy().astype(float)
    eprice_arr = pool["entry_price"].to_numpy()
    ret_arr = pool["ret"].to_numpy()
    z_arr = pool["z_entry"].to_numpy()
    bars_arr = pool["bars_held"].to_numpy()

    equity = init
    peak_mtm = init
    open_pos = []
    eq_mtm = np.empty(n)
    eq_real = np.empty(n)
    conc = []
    skipped = 0

    for b in range(n):
        if open_pos:
            still = []
            for p_ in open_pos:
                if p_["exit_pos"] <= b:
                    equity += p_["alloc"] * p_["ret"]
                else:
                    still.append(p_)
            open_pos = still

        unreal = 0.0
        for p_ in open_pos:
            px = carr[b, p_["col"]]
            unreal += p_["alloc"] * p_["dir"] * (px / p_["eprice"] - 1.0)
        mtm = equity + unreal
        eq_mtm[b] = mtm
        eq_real[b] = equity
        peak_mtm = max(peak_mtm, mtm)
        dd_mtm = mtm / peak_mtm - 1.0

        if b in by_entry:
            # ブック別 open 数
            n_open_book = {}
            for p_ in open_pos:
                n_open_book[p_["book"]] = n_open_book.get(p_["book"], 0) + 1
            for ti in by_entry[b]:
                bk = book_arr[ti]
                cap = caps.get(bk, 8)
                if n_open_book.get(bk, 0) >= cap:
                    skipped += 1
                    continue
                ctx = {"equity_real": equity, "equity_mtm": mtm, "peak_mtm": peak_mtm,
                       "dd_mtm": dd_mtm, "z": float(z_arr[ti]), "book": bk,
                       "instr": instr_arr[ti], "ret": float(ret_arr[ti]),
                       "bars_held": int(bars_arr[ti])}
                alloc = float(sizing(ctx))
                if alloc <= 0:
                    skipped += 1
                    continue
                open_pos.append({"col": col_of[instr_arr[ti]], "dir": dir_arr[ti],
                                 "eprice": eprice_arr[ti], "alloc": alloc,
                                 "exit_pos": int(exit_pos[ti]), "ret": float(ret_arr[ti]),
                                 "book": bk})
                n_open_book[bk] = n_open_book.get(bk, 0) + 1
                conc.append(len(open_pos))

    eq_mtm = pd.Series(eq_mtm, index=grid)
    eq_real = pd.Series(eq_real, index=grid)
    info = {"final": equity, "skipped": skipped, "n_taken": len(conc),
            "max_conc": max(conc) if conc else 0}
    return eq_mtm, eq_real, info


def make_book_sizing(pool, alpha, caps):
    """ブック別サイジング。各ブック内で f(z)/fbar_book を正規化、D1 は alpha で縮小。
    返り値 make(m): グローバル倍率 m を掛けるサイジング関数。"""
    fbar = {}
    for bk in pool["book"].unique():
        zz = pool.loc[pool["book"] == bk, "z_entry"].to_numpy()
        fbar[bk] = float(np.mean([_fz(z) for z in zz])) or 1.0

    def make(m):
        def sizing(ctx):
            bk = ctx["book"]
            cap = caps.get(bk, 8)
            scale = alpha if bk == "D1" else 1.0
            return ctx["equity_real"] * (m / cap) * (_fz(ctx["z"]) / fbar[bk]) * scale
        return sizing
    return make


def calibrate_books_robust(pool, closes, make, caps, target=0.20, n_boot=400,
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


def tag(pool, book):
    p = pool.copy()
    p["book"] = book
    return p


def main():
    print("=" * 72)
    print("  exp60: リスク予算配分 MTF アンサンブル — 確定判定")
    print("=" * 72)

    pool_h4 = tag(build_pool_d1(tf="H4"), "H4")
    pool_d1 = tag(build_pool_d1(tf="D1"), "D1")
    closes = mm.load_closes(tf="H4")  # 統合グリッド(細かい方)
    print(f"  H4 {len(pool_h4)} trades / D1 {len(pool_d1)} trades / grid {len(closes)}")

    CAPS = {"H4": 8, "D1": 8}

    # --- ② α スイープ(独立サイジング+独立スロット, robust p95=20%) ---
    print("\n[②] D1 weight α スイープ(独立スロット H4=8/D1=8, robust p95=20%, seed0)")
    print(f"     {'α':>5} {'m':>6} {'CAGR':>9} {'p95':>8} {'empDD':>8} {'posY':>6} {'worstY':>8} {'Sharpe':>7} {'taken':>7}")
    pool_mix = pd.concat([pool_h4, pool_d1], ignore_index=True).sort_values("entry").reset_index(drop=True)
    results = []
    for alpha in [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]:
        make = make_book_sizing(pool_mix, alpha, CAPS)
        m, eqm, eqr, info = calibrate_books_robust(pool_mix, closes, make, CAPS, n_boot=400)
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

    # --- ③ 失血窓条件付き: H4 の最悪十分位月に D1 は稼ぐか ---
    print("\n[③] 失血窓テスト: H4 の月次MtMリターン下位十分位の月における D1 の平均月次リターン")
    mk_h4 = make_book_sizing(pool_h4.assign(book="H4"), 0.0, {"H4": 8})  # alpha無関係
    # H4 単独 MtM 月次
    s_h4 = mm.simulate(pool_h4, mm.load_closes(tf="H4"), lambda c: c["equity_real"]*(1/8)*(_fz(c["z"])/np.mean([_fz(z) for z in pool_h4["z_entry"]])), max_pos=8)
    mh4 = s_h4[0].resample("ME").last().pct_change().dropna()
    s_d1 = mm.simulate(pool_d1, mm.load_closes(tf="D1"), lambda c: c["equity_real"]*(1/8)*(_fz(c["z"])/np.mean([_fz(z) for z in pool_d1["z_entry"]])), max_pos=8)
    md1 = s_d1[0].resample("ME").last().pct_change().dropna()
    j = pd.concat([mh4.rename("h4"), md1.rename("d1")], axis=1).dropna()
    thr = j["h4"].quantile(0.10)
    bleed = j[j["h4"] <= thr]
    normal = j[j["h4"] > thr]
    print(f"     H4 最悪十分位月(閾値 {thr:+.2%}, n={len(bleed)}): D1 平均 {bleed['d1'].mean():+.3%} (H4 平均 {bleed['h4'].mean():+.3%})")
    print(f"     その他の月(n={len(normal)}):                    D1 平均 {normal['d1'].mean():+.3%}")
    print(f"     → D1 が失血窓で明確にプラスなら、Sharpe 算術以上のテール便益が出るはず")

    print("\n" + "=" * 72)
    print("  判定基準: 最良 α が robust ≥ +20.0%(+10%相対)かつ p95 非悪化 → 次段。さもなくば閉鎖。")
    print("=" * 72)


if __name__ == "__main__":
    main()
