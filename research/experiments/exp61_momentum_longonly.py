"""改善D: モメンタムbook を long-only 化 + band 調整 して two-book robust 寄与を測る。

基盤 = exp60_twobook_robust.py(champion 側は完全に同一=apples-to-apples)。
モメンタムbook だけを差し替えて、baseline(both, lb24, band0, w0.2)との増分を測る。

検証内容:
  (A) モメンタムbook 単独の NET Sharpe(H1): both vs long-only × band∈{0,0.001,0.002,0.003}
      long脚/short脚の分離 Sharpe も出す(short が死に脚かの再確認)。
  (B) two-book robust 合成(日次ブロックブートストラップ p95=20% 再較正):
      各 momentum config を champion と w で合成し robCAGR / ΔCAGR / p95悪化 / 最悪年 を算出。
      baseline(both lb24 band0 w0.2)の robCAGR を自分で再計算し、それとの delta_pp を報告。
  (C) long-only のドリフト依存度: long-only モメンタム日次リターンが
      USD/JPY の上昇局面に乗っているだけか。
      - USDJPY buy&hold 日次リターンとの相関
      - USDJPY 上昇日 / 下落日 でのモメンタムbook平均日次リターン(上昇日だけで稼いでいないか)
      - 年次 PnL と USDJPY 年次方向の符号一致率
  (D) plateau: band 近傍で ΔCAGR の符号が維持されるか。

NET(通常スプレッド)で評価。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import mm_lab as mm
import mm_production as mp
from fxlab import metrics, run
from fxlab import universe as uni
from strategies import tsmom


# ---- 日次ブロックブートストラップで p95 maxDD を出す(日次粒度) ----------
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


def build_momentum_book(lookback, band, side):
    """USDJPY H1 tsmom の二口座用 pool + eqm(p95=20% 較正済み daily ret)を返す。"""
    JPY = ["USDJPY"]
    tag = f"tsmom_usdjpy_lb{lookback}_b{int(band*1000)}_{side}"
    pool_j = mm.build_pool_for(tsmom, {"lookback": lookback, "band": band}, tf="H1",
                               instruments=JPY, tag=tag, side=side, cache=False)
    closes_j = pd.DataFrame({"USDJPY": uni.instrument_close("USDJPY", "H1")}).sort_index().ffill()

    def mk_j(k):
        return lambda ctx: ctx["equity_real"] * k
    k_j, eqm_j, eqr_j, info_j, p95_j = mm.calibrate_robust(
        pool_j, closes_j, mk_j, target_dd=0.20, max_pos=1, n_boot=800)
    n_long = int((pool_j["dir"] > 0).sum())
    n_short = int((pool_j["dir"] < 0).sum())
    return dict(pool=pool_j, eqm=eqm_j, p95=p95_j, k=k_j,
                n_long=n_long, n_short=n_short, n=len(pool_j))


def net_sharpe(lookback, band, side):
    """USDJPY H1 tsmom の NET Sharpe(vectorbt 直接, 固定キャッシュ)。"""
    data = uni.instrument_data("USDJPY", "H1")
    pf = run("USDJPY", "H1", tsmom.generate_signals,
             {"lookback": lookback, "band": band}, data=data,
             size_mode="value", side=side)
    m = metrics(pf)
    return float(m["sharpe"].iloc[0]), float(m["num_trades"].iloc[0]), \
        float(m["total_return"].iloc[0]), float(m["profit_factor"].iloc[0])


def main():
    print("=== exp61: モメンタムbook long-only化 + band調整 (改善D) ===\n")

    # ============ champion book(exp60 と同一) ============
    pool_c = mp.build_pool_d1()
    closes_c = mm.load_closes()
    mk_c = mp.champion_sizing(pool_c, max_pos=8)
    k_c, eqm_c, eqr_c, info_c, p95_c_cal = mm.calibrate_robust(
        pool_c, closes_c, mk_c, target_dd=0.20, max_pos=8, n_boot=800)
    s_c = mm.stats(eqm_c, eqr_c, info_c)
    print(f"[champion d1] k={k_c:.2f} CAGR={s_c['cagr']:+.2%} "
          f"maxDD_mtm={s_c['maxdd_mtm']:+.1%} trades={len(pool_c)}")
    rc_full = to_daily_ret(eqm_c)

    # ============ (A) momentum book NET Sharpe: both vs long-only × band ============
    print("\n=== (A) momentum NET Sharpe (USDJPY H1, value-sizing) ===")
    print(f"  {'side':>5} {'band':>6} {'sharpe':>8} {'PF':>6} {'trades':>7} {'tot_ret':>9}")
    bands = [0.0, 0.001, 0.002, 0.003]
    sharpe_tbl = {}
    for side in ["both", "long", "short"]:
        for band in bands:
            sh, nt, tr, pf = net_sharpe(24, band, side)
            sharpe_tbl[(side, band)] = dict(sharpe=sh, num_trades=nt, tot_ret=tr, pf=pf)
            print(f"  {side:>5} {band:>6.3f} {sh:>+8.3f} {pf:>6.2f} {nt:>7.0f} {tr:>+9.1%}")

    # ============ baseline + variant momentum books ============
    # baseline = both, lb24, band0 (★現チャンピオン構成)
    configs = [("both", 0.0)]  # baseline first
    for band in bands:
        configs.append(("long", band))

    books = {}
    print("\n=== momentum book robust 較正 (p95=20%, H1) ===")
    print(f"  {'side':>5} {'band':>6} {'n':>5} {'long':>5} {'short':>6} {'k':>6}")
    for side, band in configs:
        b = build_momentum_book(24, band, side)
        books[(side, band)] = b
        print(f"  {side:>5} {band:>6.3f} {b['n']:>5} {b['n_long']:>5} {b['n_short']:>6} {b['k']:>6.2f}")

    # ============ 共通日次グリッド(全 book で同一になるよう champion と各 book の交差) ============
    # champion 単独 robCAGR は exp60 同様、共通グリッド上で再較正して基準に。
    # baseline robCAGR の再計算には baseline(both) book を使う。

    def blend_robust(book, w_list, label, champ_robCAGR_ref=None):
        rj = to_daily_ret(book["eqm"])
        common = rc_full.index.intersection(rj.index)
        rc = rc_full.reindex(common).fillna(0.0)
        rjj = rj.reindex(common).fillna(0.0)
        corr = float(np.corrcoef(rc.values, rjj.values)[0, 1])
        # champion 単独 再較正(この共通グリッド上で)
        L_c, p95_c_d = lever_to_p95(rc.values, target=0.20)
        champ_rob = cagr_of(rc.values * L_c, common)
        out = []
        for w in w_list:
            blend = (1 - w) * rc.values + w * rjj.values
            L, p95 = lever_to_p95(blend, target=0.20)
            levered = blend * L
            cagr = cagr_of(levered, common)
            wy = worst_year(levered, common)
            p95_worse = p95 > abs(p95_c_d) + 0.003
            out.append(dict(w=w, L=L, p95=p95, robCAGR=cagr,
                            drob_vs_champ=(cagr - champ_rob) * 100,
                            worst_yr=wy, p95_worse=p95_worse))
        return dict(label=label, corr=corr, champ_rob=champ_rob, p95_c=p95_c_d,
                    common=common, rj=rjj, rc=rc, rows=out)

    w_list = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    # baseline (both lb24 band0)
    base = blend_robust(books[("both", 0.0)], w_list, "baseline_both_lb24_b0")
    print(f"\n=== (baseline) both lb24 band0 — champ_rob={base['champ_rob']:+.2%} "
          f"corr={base['corr']:+.3f} ===")
    print(f"  {'w':>5} {'L':>6} {'p95':>7} {'robCAGR':>9} {'Δvs_champ':>10} {'worst_yr':>9} {'p95worse':>9}")
    for r in base["rows"]:
        print(f"  {r['w']:>5.2f} {r['L']:>6.2f} {r['p95']:>+7.1%} {r['robCAGR']:>+9.2%} "
              f"{r['drob_vs_champ']:>+10.2f} {r['worst_yr']:>+9.1%} {str(r['p95_worse']):>9}")
    # baseline robCAGR @ w=0.2
    base_w02 = next(r for r in base["rows"] if abs(r["w"] - 0.20) < 1e-9)
    baseline_robCAGR = base_w02["robCAGR"]
    print(f"\n  >>> baseline robCAGR @ w=0.20 = {baseline_robCAGR:+.4%}")

    # long-only variants
    variant_results = {}
    for band in bands:
        bk = books[("long", band)]
        res = blend_robust(bk, w_list, f"long_lb24_b{int(band*1000)}")
        variant_results[band] = res
        print(f"\n=== (variant) LONG-ONLY lb24 band{band:.3f} — corr={res['corr']:+.3f} ===")
        print(f"  {'w':>5} {'L':>6} {'p95':>7} {'robCAGR':>9} {'Δvs_base_pp':>11} {'worst_yr':>9} {'p95worse':>9}")
        for r in res["rows"]:
            dvb = (r["robCAGR"] - baseline_robCAGR) * 100
            r["drob_vs_baseline"] = dvb
            print(f"  {r['w']:>5.2f} {r['L']:>6.2f} {r['p95']:>+7.1%} {r['robCAGR']:>+9.2%} "
                  f"{dvb:>+11.2f} {r['worst_yr']:>+9.1%} {str(r['p95_worse']):>9}")

    # ============ (C) ドリフト依存度: long-only band0.001 (実弾推奨パラメータ) ============
    print("\n=== (C) long-only ドリフト依存度 (USD/JPY 上昇局面に乗っているだけか) ===")
    usdjpy_close = uni.instrument_close("USDJPY", "H1").sort_index().ffill()
    usdjpy_daily = usdjpy_close.resample("1D").last().dropna()
    bh_ret = usdjpy_daily.pct_change().dropna()  # buy&hold 日次

    drift_report = {}
    for band in [0.0, 0.001]:
        rj = to_daily_ret(books[("long", band)]["eqm"])
        common = rj.index.intersection(bh_ret.index)
        m_d = rj.reindex(common).fillna(0.0)
        bh_d = bh_ret.reindex(common).fillna(0.0)
        corr_bh = float(np.corrcoef(m_d.values, bh_d.values)[0, 1])
        up = bh_d > 0
        dn = bh_d < 0
        mean_up = float(m_d[up].mean())
        mean_dn = float(m_d[dn].mean())
        # 年次符号一致
        mom_yr = pd.Series(np.cumprod(1.0 + m_d.values), index=common).groupby(common.year).last().pct_change()
        bh_yr = pd.Series(np.cumprod(1.0 + bh_d.values), index=common).groupby(common.year).last().pct_change()
        both = pd.concat([mom_yr, bh_yr], axis=1).dropna()
        sign_match = float((np.sign(both.iloc[:, 0]) == np.sign(both.iloc[:, 1])).mean())
        yr_corr = float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1]) if len(both) > 2 else float("nan")
        drift_report[band] = dict(corr_bh=corr_bh, mean_up=mean_up, mean_dn=mean_dn,
                                  sign_match=sign_match, yr_corr=yr_corr)
        print(f"  band{band:.3f}: corr_vs_BH={corr_bh:+.3f}  上昇日平均={mean_up:+.5f} "
              f"下落日平均={mean_dn:+.5f}  年次符号一致={sign_match:.2f} 年次corr={yr_corr:+.3f}")

    # ============ BEST variant + plateau 判定 ============
    # 各 long band の内点最大 robCAGR(w>0)を取り、baseline比 delta_pp
    best_per_band = {}
    for band in bands:
        rows = [r for r in variant_results[band]["rows"] if r["w"] > 0]
        bestrow = max(rows, key=lambda r: r["robCAGR"])
        best_per_band[band] = bestrow
    # 全体ベスト(long-only 中)
    overall_best_band = max(bands, key=lambda b: best_per_band[b]["robCAGR"])
    bb = best_per_band[overall_best_band]
    best_robCAGR = bb["robCAGR"]
    best_delta_pp = (best_robCAGR - baseline_robCAGR) * 100

    # plateau: best band の隣接 band で同符号(delta_vs_baseline)か
    band_deltas = {b: (best_per_band[b]["robCAGR"] - baseline_robCAGR) * 100 for b in bands}
    signs = [np.sign(band_deltas[b]) for b in bands]
    plateau = len(set(signs)) == 1  # 全 band で同符号 = 高原(符号が安定)

    # p95悪化(レバ偽装): best variant の best w で p95_worse か
    p95_worsens = bool(bb["p95_worse"])

    print("\n=== BEST long-only variant ===")
    print(f"  band={overall_best_band:.3f} w={bb['w']:.2f} robCAGR={best_robCAGR:+.3%} "
          f"Δvs_baseline={best_delta_pp:+.2f}pp p95={bb['p95']:+.1%} "
          f"worst_yr={bb['worst_yr']:+.1%} p95_worsens={p95_worsens}")
    print(f"  band別 Δvs_baseline(pp, 内点最大): "
          f"{ {b: round(band_deltas[b],2) for b in bands} }  plateau={plateau}")

    print("\n=== SUMMARY_JSON ===")
    print(json.dumps({
        "baseline_robCAGR": round(baseline_robCAGR, 5),
        "best_long_only_band": overall_best_band,
        "best_long_only_w": bb["w"],
        "best_long_only_robCAGR": round(best_robCAGR, 5),
        "delta_vs_baseline_pp": round(best_delta_pp, 3),
        "p95_worsens": p95_worsens,
        "plateau": bool(plateau),
        "band_deltas_pp": {str(b): round(band_deltas[b], 3) for b in bands},
        "netSharpe_both_b0": round(sharpe_tbl[("both", 0.0)]["sharpe"], 4),
        "netSharpe_long_by_band": {str(b): round(sharpe_tbl[("long", b)]["sharpe"], 4) for b in bands},
        "netSharpe_short_by_band": {str(b): round(sharpe_tbl[("short", b)]["sharpe"], 4) for b in bands},
        "drift_corr_bh_long_b001": round(drift_report[0.001]["corr_bh"], 4),
        "drift_meanup_long_b001": round(drift_report[0.001]["mean_up"], 6),
        "drift_meandn_long_b001": round(drift_report[0.001]["mean_dn"], 6),
        "drift_yr_signmatch_long_b001": round(drift_report[0.001]["sign_match"], 3),
        "drift_yr_corr_long_b001": round(drift_report[0.001]["yr_corr"], 4),
    }, indent=2))


if __name__ == "__main__":
    main()
