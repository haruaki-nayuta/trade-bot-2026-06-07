"""改善A: USDJPY H1 モメンタムを lookback アンサンブル化したときの two-book robust 増分。

exp60_twobook_robust.py と完全に同じ手順(champion 側は不変、book B のモメンタムだけ差し替え)で、
baseline(USDJPY lb24 単一, w=0.2)に対する各アンサンブル構成・各 w の robCAGR を測る。

book B 候補:
  - lb24-single   (= baseline, strategies.tsmom lookback=24 band=0)
  - ens_12_24     (lookbacks=[12,24])
  - ens_12_24_48  (lookbacks=[12,24,48])
  - ens_24_48_72  (lookbacks=[24,48,72])

champion daily 系列は1度だけ作って全候補で共有(同一 framework)。
各 book B を独立に p95=20% 較正(H1 fixed-fractional, max_pos=1)→ 日次化 →
w で合成 → 合成系列を日次ブロックブートストラップで p95=20% に再レバ → robCAGR。
baseline(lb24, w=0.2)の robCAGR を基準に、各候補の各 w の Δpp を出す。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import mm_lab as mm
import mm_production as mp
from fxlab import universe as uni
from strategies import tsmom
from strategies import usdjpy_momentum_ensemble as ens

TRADING_DAYS = 252
JPY = ["USDJPY"]
W_GRID = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


# ---- 日次ブロックブートストラップ p95 maxDD(exp60 と同一) ----
def daily_block_bootstrap_p95(daily_ret, n_boot=2000, block=21, seed=0):
    r = np.asarray(daily_ret, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < block * 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block, size=(n_boot, n_blocks))
    dds = np.empty(n_boot)
    for i in range(n_boot):
        idx = (starts[i][:, None] + np.arange(block)).ravel()[:n]
        path = np.cumprod(1.0 + r[idx])
        peak = np.maximum.accumulate(path)
        dds[i] = (path / peak - 1.0).min()
    return float(np.percentile(dds, 5))


def lever_to_p95(daily_ret, target=0.20, n_boot=2000, block=21, seed=0,
                 lo=0.05, hi=20.0, iters=30):
    def p95_at(L):
        return abs(daily_block_bootstrap_p95(daily_ret * L, n_boot=n_boot, block=block, seed=seed))
    if p95_at(hi) <= target:
        return hi, p95_at(hi)
    if p95_at(lo) > target:
        return lo, p95_at(lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if p95_at(mid) > target:
            hi = mid
        else:
            lo = mid
    return lo, p95_at(lo)


def cagr_of(daily_ret, index):
    path = np.cumprod(1.0 + np.nan_to_num(daily_ret))
    years = (index[-1] - index[0]).days / 365.25
    final = path[-1]
    return (final ** (1 / years) - 1) if final > 0 else -1.0


def worst_year(daily_ret, index):
    eq = pd.Series(np.cumprod(1.0 + np.nan_to_num(daily_ret)), index=index)
    yearly = eq.groupby(eq.index.year).last()
    yr = yearly.pct_change()
    yr.iloc[0] = yearly.iloc[0] - 1.0
    return float(yr.min())


def to_daily_ret(eqm):
    d = eqm.resample("1D").last().dropna()
    return d.pct_change().dropna()


def empirical_maxdd(daily_ret, L):
    path = np.cumprod(1.0 + np.nan_to_num(daily_ret) * L)
    peak = np.maximum.accumulate(path)
    return float((path / peak - 1.0).min())


def build_book_daily(strategy_mod, params, tag):
    """book B(USDJPY H1)を pool 化 → fixed-fractional を p95=20% に較正 → 日次リターン化。

    返り値: (rj_daily, info_dict)。
    """
    pool_j = mm.build_pool_for(strategy_mod, params, tf="H1",
                               instruments=JPY, tag=tag, side="both", cache=False)
    closes_j = pd.DataFrame({"USDJPY": uni.instrument_close("USDJPY", "H1")}).sort_index().ffill()
    n_long = int((pool_j["dir"] > 0).sum())
    n_short = int((pool_j["dir"] < 0).sum())

    def mk_j(k):
        return lambda ctx: ctx["equity_real"] * k
    k_j, eqm_j, eqr_j, info_j, p95_j_cal = mm.calibrate_robust(
        pool_j, closes_j, mk_j, target_dd=0.20, max_pos=1, n_boot=800)
    dd_j = abs(float((eqm_j / eqm_j.cummax() - 1.0).min()))
    cagr_j = (eqm_j.iloc[-1] / 10000.0) ** (1 / ((eqm_j.index[-1] - eqm_j.index[0]).days / 365.25)) - 1
    rj = to_daily_ret(eqm_j)
    info = dict(n_trades=len(pool_j), n_long=n_long, n_short=n_short,
                k=k_j, cagr_h1=cagr_j, maxdd_h1=-dd_j, p95_cal=p95_j_cal)
    return rj, info


def main():
    print("=== exp61: lookback アンサンブル two-book robust (改善A) ===\n")

    # ---------- BOOK A: champion d1 (H4) — exp60 と同一、1度だけ ----------
    pool_c = mp.build_pool_d1()
    closes_c = mm.load_closes()
    mk_c = mp.champion_sizing(pool_c, max_pos=8)
    k_c, eqm_c, eqr_c, info_c, p95_c_cal = mm.calibrate_robust(
        pool_c, closes_c, mk_c, target_dd=0.20, max_pos=8, n_boot=800)
    s_c = mm.stats(eqm_c, eqr_c, info_c)
    print(f"[A champion d1] k={k_c:.2f} CAGR={s_c['cagr']:+.2%} "
          f"maxDD_mtm={s_c['maxdd_mtm']:+.1%} trades={len(pool_c)}")
    rc_full = to_daily_ret(eqm_c)

    # ---------- book B 候補 ----------
    books = {
        "lb24_single": (tsmom, {"lookback": 24, "band": 0.0}, "exp61_tsmom_lb24"),
        "ens_12_24": (ens, {"lookbacks": (12, 24), "band": 0.0}, "exp61_ens_12_24"),
        "ens_12_24_48": (ens, {"lookbacks": (12, 24, 48), "band": 0.0}, "exp61_ens_12_24_48"),
        "ens_24_48_72": (ens, {"lookbacks": (24, 48, 72), "band": 0.0}, "exp61_ens_24_48_72"),
    }

    book_daily = {}
    book_info = {}
    for name, (mod, params, tag) in books.items():
        rj, info = build_book_daily(mod, params, tag)
        book_daily[name] = rj
        book_info[name] = info
        print(f"[B {name:13s}] trades={info['n_trades']:4d} "
              f"L={info['k']:.2f} CAGR_h1={info['cagr_h1']:+.2%} "
              f"maxDD_h1={info['maxdd_h1']:+.1%} p95_cal={info['p95_cal']:+.1%} "
              f"(long={info['n_long']}/short={info['n_short']})")

    # ---------- champion 単独 robCAGR(共通グリッドは lb24 と合わせる: 全 book で同じ日次 union) ----
    # 各 book で共通日を取り、champion を再較正(book ごとにグリッドが少し違いうるため book 内で完結)
    summary = {}
    print("\n=== 各 book × w → 日次 p95=20% 再較正 robCAGR ===")
    for name, rj in book_daily.items():
        common = rc_full.index.intersection(rj.index)
        rc = rc_full.reindex(common).fillna(0.0)
        rjc = rj.reindex(common).fillna(0.0)
        corr = float(np.corrcoef(rc.values, rjc.values)[0, 1])

        # champion 単独 baseline(この book のグリッド上で)
        L_c, p95_c_d = lever_to_p95(rc.values, target=0.20)
        champ_robCAGR = cagr_of(rc.values * L_c, common)
        emaxdd_champ = empirical_maxdd(rc.values, L_c)

        rows = []
        for w in W_GRID:
            blend = (1 - w) * rc.values + w * rjc.values
            L, p95 = lever_to_p95(blend, target=0.20)
            levered = blend * L
            cagr = cagr_of(levered, common)
            wy = worst_year(levered, common)
            emaxdd = empirical_maxdd(blend, L)
            drob_vs_champ = (cagr - champ_robCAGR) * 100
            p95_worse = p95 > abs(p95_c_d) + 0.003
            rows.append(dict(w=w, L=round(L, 4), p95=round(p95, 5), robCAGR=round(cagr, 5),
                             d_vs_champ_pp=round(drob_vs_champ, 3),
                             worst_yr=round(wy, 5), emaxdd=round(emaxdd, 5),
                             p95_worse_vs_champ=p95_worse))
        summary[name] = dict(corr=round(corr, 4), champ_robCAGR=round(champ_robCAGR, 5),
                             champ_p95=round(p95_c_d, 5), champ_emaxdd=round(emaxdd_champ, 5),
                             n_days=len(common), rows=rows)

        print(f"\n--- book={name}  corr={corr:+.3f}  champ_robCAGR={champ_robCAGR:+.2%} "
              f"champ_emaxdd={emaxdd_champ:+.1%} ---")
        print(f"  {'w':>5} {'L':>7} {'p95':>8} {'robCAGR':>9} {'Δvschamp':>9} "
              f"{'worst_yr':>9} {'emaxDD':>8} {'p95worse':>9}")
        for r in rows:
            print(f"  {r['w']:>5.2f} {r['L']:>7.3f} {r['p95']:>+8.1%} {r['robCAGR']:>+9.2%} "
                  f"{r['d_vs_champ_pp']:>+9.2f} {r['worst_yr']:>+9.1%} {r['emaxdd']:>+8.1%} "
                  f"{str(r['p95_worse_vs_champ']):>9}")

    # ---------- baseline = lb24_single @ w=0.20 ----------
    def get_row(book, w):
        for r in summary[book]["rows"]:
            if abs(r["w"] - w) < 1e-9:
                return r
        return None

    base_row = get_row("lb24_single", 0.20)
    baseline_robCAGR = base_row["robCAGR"]
    print(f"\n[BASELINE] lb24_single w=0.20 robCAGR={baseline_robCAGR:+.2%} "
          f"emaxDD={base_row['emaxdd']:+.1%} p95={base_row['p95']:+.1%}")

    # 各アンサンブル構成の best w(robCAGR 最大, w>0)と baseline 比較
    print("\n=== アンサンブル各構成 best(vs baseline lb24 w=0.20) ===")
    ens_results = {}
    for name in ["ens_12_24", "ens_12_24_48", "ens_24_48_72"]:
        cand = [r for r in summary[name]["rows"] if r["w"] > 0]
        best = max(cand, key=lambda r: r["robCAGR"])
        delta_vs_base = (best["robCAGR"] - baseline_robCAGR) * 100
        # baseline と同じ w=0.20 での比較も
        same_w = get_row(name, 0.20)
        delta_samew = (same_w["robCAGR"] - baseline_robCAGR) * 100
        # plateau: w=0.20 近傍 (0.15,0.20,0.25) で baseline 比の符号が一致するか
        signs = []
        for w in [0.15, 0.20, 0.25]:
            rr = get_row(name, w)
            signs.append(np.sign(rr["robCAGR"] - baseline_robCAGR))
        plateau = len(set(signs)) == 1
        ens_results[name] = dict(best_w=best["w"], best_robCAGR=best["robCAGR"],
                                 delta_best_vs_base_pp=round(delta_vs_base, 3),
                                 samew_robCAGR=same_w["robCAGR"],
                                 delta_samew_vs_base_pp=round(delta_samew, 3),
                                 best_p95=best["p95"], best_emaxdd=best["emaxdd"],
                                 best_p95_worse=best["p95_worse_vs_champ"],
                                 plateau_sign=plateau, near_w_signs=[int(s) for s in signs])
        print(f"  {name:13s} best_w={best['w']:.2f} robCAGR={best['robCAGR']:+.2%} "
              f"Δbest_vs_base={delta_vs_base:+.2f}pp | @w0.20 Δ={delta_samew:+.2f}pp "
              f"emaxDD={best['emaxdd']:+.1%} p95worse={best['p95_worse_vs_champ']} "
              f"plateau={plateau} signs={[int(s) for s in signs]}")

    print("\n=== SUMMARY_JSON ===")
    print(json.dumps({
        "champion_book_k": round(float(k_c), 4),
        "baseline_lb24_w020_robCAGR": baseline_robCAGR,
        "baseline_emaxdd": base_row["emaxdd"],
        "book_info": {k: {kk: (round(vv, 5) if isinstance(vv, float) else vv)
                          for kk, vv in v.items()} for k, v in book_info.items()},
        "ensemble_results": ens_results,
        "per_book_summary": summary,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
