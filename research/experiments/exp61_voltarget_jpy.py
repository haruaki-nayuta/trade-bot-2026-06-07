"""改善C: vol-target サイジングを USDJPY モメンタムスリーブに適用 → two-book robust 増分。

exp60 を踏襲。champion 側はそのまま(calibrate_robust p95=20%)。
**モメンタム book だけ** 固定比率 → vol-target に差し替えて apples-to-apples で増分を測る。

vol-target サイジング:
  各トレードのエントリー時に USDJPY の直近 vol_win 本 H1 リターン std を計算し、
  weight_i = clip(target_vol / vol_at_entry_i, 0, clip)。
  weight は平均 1.0 に正規化(=配分の"形"だけを変え、全体の大きさは k で動く)。
  alloc = equity_real * k * weight_i。calibrate_robust(p95=20%) で k を縛る。

USDJPY tsmom lb24 は単一銘柄・max_pos=1・完全非重複(ドテン)=エントリーは pool 順に逐次処理される。
→ サイジング関数を「pool 順に weight を消費する」stateful 関数にすれば simulate の ctx に
   entry timestamp が無くても各トレードに正しい weight を当てられる(別途逐次シミュで検証済み)。

baseline = exp60 の固定比率 lb24 を w=0.20 で混ぜたときの robCAGR。差(pp)で報告。
vol_win {30,60,120}, clip {2,3,5} で plateau を確認。

注意(task): xsec では vol-target がエッジを削った前科。モメンタムで逆効果でないか正直に判定。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import mm_lab as mm
import mm_production as mp
from fxlab import universe as uni
from strategies import tsmom


# ---- 日次ブロックブートストラップ p95 (exp60 と同一) ----------------------
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


def daily_maxdd(daily_ret):
    eq = np.cumprod(1.0 + np.nan_to_num(daily_ret))
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def build_jpy_book():
    JPY = ["USDJPY"]
    pool = mm.build_pool_for(tsmom, {"lookback": 24, "band": 0.0}, tf="H1",
                             instruments=JPY, tag="tsmom_usdjpy_lb24", side="both",
                             cache=False)
    closes = pd.DataFrame({"USDJPY": uni.instrument_close("USDJPY", "H1")}).sort_index().ffill()
    return pool, closes


def compute_weights(pool, vol_win, clip):
    """各トレードに vol-target weight を割当(平均1.0正規化)。先読み回避で vol は .shift(1)。"""
    close = uni.instrument_close("USDJPY", "H1").sort_index()
    ret = close.pct_change()
    vol = ret.rolling(vol_win).std().shift(1)        # 直前バーまでで確定 = look-ahead 回避
    ev = vol.reindex(pool["entry"]).to_numpy()
    med = float(np.nanmedian(ev))
    ev_f = np.where(np.isfinite(ev) & (ev > 0), ev, med)
    target_vol = med
    raw = np.clip(target_vol / ev_f, 0.0, clip)
    return raw / float(np.mean(raw))                  # 平均 1.0


def make_weighted_sizing_factory(weights):
    """pool 順に weight を消費する stateful サイジング工場。
    USDJPY tsmom は max_pos=1・非重複なので新規エントリーは pool 順に逐次発生 = weights を順に消費。
    各 calibrate 試行で counter をリセットするため、factory が呼ばれるたびに新しい closure を返す。"""
    def factory(k):
        state = {"i": 0}
        def sz(ctx):
            i = state["i"]
            w = weights[i] if i < len(weights) else 1.0
            state["i"] = i + 1
            return ctx["equity_real"] * k * float(w)
        return sz
    return factory


def verify_sequential(pool, weights, k=0.5):
    """検証: stateful simulate と直接逐次複利が一致するか(非重複・max_pos=1 前提)。"""
    closes = pd.DataFrame({"USDJPY": uni.instrument_close("USDJPY", "H1")}).sort_index().ffill()
    fac = make_weighted_sizing_factory(weights)
    eqm, eqr, info = mm.simulate(pool, closes, fac(k), max_pos=1, init=10000.0)
    # 直接逐次: 各トレードで equity *= (1 + k*w_i*ret_i)
    eq = 10000.0
    for i in range(len(pool)):
        eq *= (1.0 + k * float(weights[i]) * float(pool["ret"].iloc[i]))
    return float(eqr.iloc[-1]), eq, info["skipped"]


def main():
    print("=== exp61: vol-target サイジング(USDJPY モメンタムスリーブ)===\n")

    # ---------- BOOK A: champion d1 (exp60 と完全同一) ----------
    pool_c = mp.build_pool_d1()
    closes_c = mm.load_closes()
    mk_c = mp.champion_sizing(pool_c, max_pos=8)
    k_c, eqm_c, eqr_c, info_c, p95_c_cal = mm.calibrate_robust(
        pool_c, closes_c, mk_c, target_dd=0.20, max_pos=8, n_boot=800)
    rc = to_daily_ret(eqm_c)
    print(f"[A champion] k={k_c:.2f} trades={len(pool_c)}")

    # ---------- BOOK B: USDJPY lb24 tsmom ----------
    pool_j, closes_j = build_jpy_book()
    print(f"[B tsmom] pool trades={len(pool_j)} long={int((pool_j['dir']>0).sum())} "
          f"short={int((pool_j['dir']<0).sum())}")

    # === baseline: 固定比率(exp60 BOOK B と同一サイジング) ===
    def mk_j_fixed(k):
        return lambda ctx: ctx["equity_real"] * k
    k_jf, eqm_jf, eqr_jf, info_jf, p95_jf = mm.calibrate_robust(
        pool_j, closes_j, mk_j_fixed, target_dd=0.20, max_pos=1, n_boot=800)
    rj_fixed = to_daily_ret(eqm_jf)
    print(f"[B fixed] k={k_jf:.3f} soloDD={abs(float((eqm_jf/eqm_jf.cummax()-1).min())):.1%}")

    # ---- 検証: stateful weighted simulate == 逐次複利 (weights=全1 で固定比率と一致するはず) ----
    ones = np.ones(len(pool_j))
    seq_a, seq_b, sk = verify_sequential(pool_j, ones, k=k_jf)
    print(f"[verify] weighted(ones) eqr={seq_a:.1f} vs 逐次={seq_b:.1f} "
          f"diff={abs(seq_a-seq_b):.4f} skipped={sk}")

    # 日次共通グリッド
    common = rc.index.intersection(rj_fixed.index)
    rc_c = rc.reindex(common).fillna(0.0).values
    rjf_c = rj_fixed.reindex(common).fillna(0.0).values

    L_c, p95_c_d = lever_to_p95(rc_c, target=0.20)
    champ_rob = cagr_of(rc_c * L_c, common)

    W_BASE = 0.20
    blend_base = (1 - W_BASE) * rc_c + W_BASE * rjf_c
    L_base, p95_base = lever_to_p95(blend_base, target=0.20)
    baseline_rob = cagr_of(blend_base * L_base, common)
    base_corr = float(np.corrcoef(rc_c, rjf_c)[0, 1])
    base_empDD = daily_maxdd(blend_base * L_base)
    print(f"\n[基準] champion単独 robCAGR={champ_rob:+.2%} (p95={p95_c_d:+.1%})")
    print(f"[基準] baseline(+固定比率 w0.20) robCAGR={baseline_rob:+.2%} "
          f"p95={p95_base:+.1%} corr={base_corr:+.3f} empDD={base_empDD:+.1%}\n")

    # ---------- vol-target 変種 ----------
    CLIPS = [2.0, 3.0, 5.0]
    VOL_WINS = [30, 60, 120]
    print("=== vol-target 変種(各 win×clip を p95=20% 較正 → w0.20 合成 → 日次再較正)===")
    print(f"  {'win':>4} {'clip':>5} {'k':>6} {'soloDD':>7} | "
          f"{'L':>6} {'p95':>7} {'robCAGR':>9} {'Δbase':>7} {'empDD':>7} {'corr':>7} {'wstY':>7}")
    rows = []
    for vol_win in VOL_WINS:
        for clip in CLIPS:
            weights = compute_weights(pool_j, vol_win, clip)
            fac = make_weighted_sizing_factory(weights)
            k_v, eqm_v, eqr_v, info_v, p95_v = mm.calibrate_robust(
                pool_j, closes_j, fac, target_dd=0.20, max_pos=1, n_boot=800)
            solo_dd = abs(float((eqm_v / eqm_v.cummax() - 1).min()))
            rj_v = to_daily_ret(eqm_v).reindex(common).fillna(0.0).values
            blend = (1 - W_BASE) * rc_c + W_BASE * rj_v
            L, p95 = lever_to_p95(blend, target=0.20)
            levered = blend * L
            rob = cagr_of(levered, common)
            d_base = (rob - baseline_rob) * 100
            empDD = daily_maxdd(levered)
            corr = float(np.corrcoef(rc_c, rj_v)[0, 1])
            wy = worst_year(levered, common)
            print(f"  {vol_win:>4} {clip:>5.1f} {k_v:>6.3f} {solo_dd:>7.1%} | "
                  f"{L:>6.3f} {p95:>+7.1%} {rob:>+9.2%} {d_base:>+7.2f} {empDD:>+7.1%} "
                  f"{corr:>+7.3f} {wy:>+7.1%}")
            rows.append(dict(vol_win=vol_win, clip=clip, k=k_v, solo_dd=solo_dd,
                             L=L, p95=p95, robCAGR=rob, d_base_pp=d_base,
                             empDD=empDD, corr=corr, worst_yr=wy))

    best = max(rows, key=lambda r: r["robCAGR"])
    # p95 偽装判定: 合成 p95 が baseline 比悪化 or empDD が baseline 比悪化
    p95_worse = best["p95"] > p95_base + 0.003
    empdd_worse = abs(best["empDD"]) > abs(base_empDD) + 0.003
    # plateau: 全変種で Δbase の符号が一致しているか
    signs = [np.sign(r["d_base_pp"]) for r in rows if abs(r["d_base_pp"]) > 0.05]
    plateau = len(set(signs)) <= 1 if signs else True

    print(f"\n[BEST win={best['vol_win']} clip={best['clip']}] robCAGR={best['robCAGR']:+.2%} "
          f"Δbase={best['d_base_pp']:+.2f}pp p95={best['p95']:+.1%}(base {p95_base:+.1%}) "
          f"empDD={best['empDD']:+.1%}(base {base_empDD:+.1%})")
    print(f"  p95_worse={p95_worse} empDD_worse={empdd_worse} plateau={plateau}")

    print("\n=== SUMMARY_JSON ===")
    print(json.dumps({
        "champ_robCAGR": round(champ_rob, 5),
        "baseline_robCAGR": round(baseline_rob, 5),
        "baseline_p95": round(p95_base, 5),
        "baseline_empDD": round(base_empDD, 5),
        "best_win": best["vol_win"], "best_clip": best["clip"],
        "best_robCAGR": round(best["robCAGR"], 5),
        "best_d_base_pp": round(best["d_base_pp"], 3),
        "best_p95": round(best["p95"], 5),
        "best_empDD": round(best["empDD"], 5),
        "p95_worse": bool(p95_worse), "empDD_worse": bool(empdd_worse),
        "plateau": bool(plateau),
        "all": [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}
                for r in rows],
    }, indent=2))


if __name__ == "__main__":
    main()
