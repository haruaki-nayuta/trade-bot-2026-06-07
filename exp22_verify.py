"""敵対的検証: キャリー overlay がチャンピオンv2を"本当に"底上げするか反証する。

検証対象 (exp22_altpremia_bleed.py の勝者):
  family = キャリー(高金利ロング/低金利ショート, 内蔵近似金利)
  best   = hold=42本(≈7日), k=2(片側2脚=常時4脚), both, tf=H4
  失血窓平均 = 355.6 (k=2) / 420.6 (k=5)  IS+196.7/OOS+530.3 (k=2)

懐疑的に: 称賛でなく反証。
 1. ヘッジ頑健性: (a) bleed閾値 -3/-5/-8% (b) IS/OOS (c) パラメータ近傍 (d) 2022除外/2022以外の窓
 2. 統合DDテスト(核心): champion + carry overlay を1口座でDD=20%較正 → CAGR が champ単独(+21.6%)を上回るか
    weight {0.25,0.5,1.0,1.5,2.0} × max_pos {8,12,16} スイープ。weight=0 sanity も。
 3. 正直な総括: overlayのドラッグ vs 失血窓のDD低減。verdict。

実行: uv run python exp22_verify.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import bleed_lab as bl
import mm_lab as mm
from fxlab import carry, config
from fxlab import universe as uni

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)

NOTIONAL = 10_000.0


# ---------- carry overlay: trades(月次PnL用) + pool(統合DD用) ----------
def universe_close(tf="H4", exclude=("AUDJPY",)):
    names = [x for x in uni.universe(crosses=True) if x not in set(exclude)]
    return pd.DataFrame({n: uni.instrument_close(n, tf) for n in names}).dropna()


def _hs(names, mp):
    return {p: config.spread_pips(p) * config.pip_size(p) / 2.0 / mp[p] for p in names}


def carry_records(close, *, hold, k):
    """1脚=1レコード。価格PnL + 保有ぶんキャリー受払 − 往復spread。先読みなし(t->t+hold)。"""
    names = list(close.columns)
    mp = close.mean()
    hs = _hs(names, mp)
    recs = []
    for t in range(0, len(close) - hold, hold):
        ts_e = close.index[t]
        year = ts_e.year
        car = pd.Series({p: carry.carry_annual(p, year) for p in names}).sort_values()
        lows = car.index[:k]
        highs = car.index[-k:]
        ts_x = close.index[t + hold]
        days = (ts_x - ts_e).total_seconds() / 86400.0
        for p in highs:
            fwd = close[p].iloc[t + hold] / close[p].iloc[t] - 1.0
            cpnl = (carry.carry_annual(p, year) / 100.0) * (days / 365.0)
            ret = (fwd - 2 * hs[p]) + cpnl
            recs.append((p, ts_e, ts_x, 1, close[p].iloc[t], ret, hold))
        for p in lows:
            fwd = close[p].iloc[t + hold] / close[p].iloc[t] - 1.0
            cpnl = -(carry.carry_annual(p, year) / 100.0) * (days / 365.0)
            ret = (-fwd - 2 * hs[p]) + cpnl
            recs.append((p, ts_e, ts_x, -1, close[p].iloc[t], ret, hold))
    df = pd.DataFrame(recs, columns=["instr", "entry", "exit", "dir", "entry_price", "ret", "bars_held"])
    return df.sort_values("entry").reset_index(drop=True)


def carry_pool(close, *, hold, k):
    df = carry_records(close, hold=hold, k=k)
    df = df.copy()
    df["z_entry"] = 1.0
    df["vol_entry"] = 0.01
    return df


def monthly_pnl(records):
    m = pd.DatetimeIndex(records["exit"]).to_period("M")
    return records.assign(m=m, pnl=records["ret"] * NOTIONAL).groupby("m")["pnl"].sum()


# ---------- 1. ヘッジ頑健性 ----------
def hedge_robustness(close, eqm, *, hold, k):
    print("=" * 100)
    print(f"1. ヘッジ頑健性 (carry hold={hold} k={k})")
    print("=" * 100)
    rec = carry_records(close, hold=hold, k=k)
    mser = monthly_pnl(rec)

    # (a) bleed閾値スイープ
    print("\n(a) bleed閾値スイープ — in_bleed が閾値に頑健か:")
    print(f"  {'thresh':>7} {'#bleed':>7} {'in_bleed':>10} {'normal':>9} {'edge':>9} {'IS':>9} {'OOS':>9} {'win':>6}")
    rows_a = []
    for th in (0.03, 0.05, 0.08):
        mask, _ = bl.bleed_mask_monthly(eqm, thresh=th)
        sc = bl.conditional_score(mser, mask)
        rows_a.append((th, sc))
        print(f"  {th:>7.0%} {int(mask.sum()):>7} {sc['mean_in_bleed']:>+10.1f} "
              f"{sc['mean_normal']:>+9.1f} {sc['hedge_edge']:>+9.1f} "
              f"{sc['mean_in_bleed_IS']:>+9.1f} {sc['mean_in_bleed_OOS']:>+9.1f} "
              f"{sc['winrate_in_bleed']:>6.0%}")
    a_ok = all(sc["mean_in_bleed"] > 0 for _, sc in rows_a)
    a_isoos = all(sc["mean_in_bleed_IS"] > 0 and sc["mean_in_bleed_OOS"] > 0 for _, sc in rows_a)
    print(f"  -> 全閾値で in_bleed>0: {a_ok} ; 全閾値で IS&OOS両プラス: {a_isoos}")

    # (b) パラメータ近傍 (-5% mask固定)
    mask5, _ = bl.bleed_mask_monthly(eqm, thresh=0.05)
    print("\n(b) パラメータ近傍 — hold/k を動かして in_bleed・IS・OOS が持続するか (-5% mask):")
    print(f"  {'hold':>5} {'k':>3} {'in_bleed':>10} {'IS':>9} {'OOS':>9} {'persist(IS&OOS>0)':>18}")
    rows_b = []
    for hh in (30, 36, 42, 48):
        for kk in (1, 2, 3):
            r = carry_records(close, hold=hh, k=kk)
            sc = bl.conditional_score(monthly_pnl(r), mask5)
            persist = sc["mean_in_bleed_IS"] > 0 and sc["mean_in_bleed_OOS"] > 0
            rows_b.append((hh, kk, sc, persist))
            print(f"  {hh:>5} {kk:>3} {sc['mean_in_bleed']:>+10.1f} "
                  f"{sc['mean_in_bleed_IS']:>+9.1f} {sc['mean_in_bleed_OOS']:>+9.1f} {str(persist):>18}")
    b_frac_inb = np.mean([sc["mean_in_bleed"] > 0 for *_, sc, _ in [(0,0,sc,p) for *_,sc,p in rows_b]]) if rows_b else 0
    b_frac_inb = np.mean([sc["mean_in_bleed"] > 0 for (_, _, sc, _) in rows_b])
    b_frac_persist = np.mean([p for (_, _, _, p) in rows_b])
    print(f"  -> 近傍 in_bleed>0 割合: {b_frac_inb:.0%} ; IS&OOS両プラス割合: {b_frac_persist:.0%}")

    # (c) 2022除外 / 各年クラスタ別
    print("\n(c) 失血窓を年で分解 — 2022一発か, 他年でも稼ぐか (-5% mask):")
    s = mser.reindex(mask5.index).fillna(0.0)
    inb = s[mask5.values]
    inb_years = pd.Series(inb.values, index=[p.year for p in inb.index])
    by_year = inb_years.groupby(level=0).agg(["sum", "mean", "count"])
    print(by_year.round(1).to_string())
    excl2022 = inb_years[inb_years.index != 2022]
    print(f"\n  失血窓 total(全): {inb.sum():+.0f}  mean: {inb.mean():+.1f}")
    print(f"  失血窓 total(2022除く): {excl2022.sum():+.0f}  mean: {excl2022.mean():+.1f}  (n={len(excl2022)})")
    # 2022を除いても窓平均がプラスか, かつ最大の貢献年が2022単独でないか
    yrs_pos = (by_year["sum"] > 0)
    c_ex2022_ok = excl2022.mean() > 0
    c_breadth = int(yrs_pos.sum())
    top_year_share = by_year["sum"].max() / by_year["sum"][by_year["sum"] > 0].sum() if (by_year["sum"] > 0).any() else 1.0
    print(f"  -> 2022除外でも窓平均>0: {c_ex2022_ok} ; プラス貢献の年数: {c_breadth}/{len(by_year)} ; "
          f"最大年シェア: {top_year_share:.0%}")

    robust = a_ok and a_isoos and (b_frac_persist >= 0.5) and c_ex2022_ok
    print(f"\n  ===> ヘッジ頑健 (全条件): {robust}")
    return {"a_ok": a_ok, "a_isoos": a_isoos, "b_frac_inb": b_frac_inb,
            "b_frac_persist": b_frac_persist, "c_ex2022_ok": c_ex2022_ok,
            "c_breadth": c_breadth, "top_year_share": top_year_share, "robust": robust}


# ---------- 2. 統合DDテスト ----------
def integrated_sweep(close, *, hold, k):
    print("\n" + "=" * 100)
    print(f"2. 統合DDテスト (champion + carry overlay hold={hold} k={k})")
    print("=" * 100)

    # sanity: weight=0 で champion 単独が再現するか (overlay pool を渡すが weight=0)
    ovl = carry_pool(close, hold=hold, k=k)
    print(f"\noverlay pool: {len(ovl)} legs, standalone(ret*10k)={ (ovl['ret']*NOTIONAL).sum():+.0f}")

    print("\nsanity: weight=0 (overlay資本ゼロ) → champion単独に一致すべき (基準 CAGR+21.6/Sh1.21/p95-28.5):")
    for mp in (8,):
        r = bl.integrated_dd_test(ovl, overlay_weight=0.0, max_pos=mp)
        print(f"  max_pos={mp} w=0.00: CAGR={r['cagr']:+.1%} Sharpe={r['sharpe']:.2f} "
              f"DDmtm={r['maxdd_mtm']:+.1%} p95={r['boot_p95']:+.1%} pos_year={r['pos_year_rate']:.0%} k={r['k']:.2f}")

    print("\nweight × max_pos スイープ (CAGR が champ単独 +21.6% を上回るか):")
    print(f"  {'max_pos':>7} {'weight':>7} {'k':>6} {'CAGR':>8} {'Sharpe':>7} {'DDmtm':>8} {'p95':>8} {'pos_yr':>7} {'worst_yr':>9}")
    results = []
    for max_pos in (8, 12, 16):
        for w in (0.25, 0.5, 1.0, 1.5, 2.0):
            r = bl.integrated_dd_test(ovl, overlay_weight=w, max_pos=max_pos)
            r["max_pos"] = max_pos
            r["weight"] = w
            results.append(r)
            print(f"  {max_pos:>7} {w:>7.2f} {r['k']:>6.2f} {r['cagr']:>+8.1%} {r['sharpe']:>7.2f} "
                  f"{r['maxdd_mtm']:>+8.1%} {r['boot_p95']:>+8.1%} {r['pos_year_rate']:>7.0%} {r['worst_year']:>+9.1%}")
    df = pd.DataFrame(results)
    return df, ovl


def champion_alone_baseline():
    """weight=0 とは別に、純チャンピオン(overlay無し)も直接測る(二重確認)。"""
    pool_c = mm.build_pool()
    closes = mm.load_closes()
    from mm_production import champion_sizing
    mk = champion_sizing(pool_c, max_pos=8)
    k, eqm, eqr, info = mm.calibrate(pool_c, closes, mk, target_dd=0.20, max_pos=8)
    s = mm.stats(eqm, eqr, info)
    bs = mm.bootstrap_maxdd(eqm, n_boot=800)
    return {"cagr": s["cagr"], "sharpe": s["sharpe"], "maxdd_mtm": s["maxdd_mtm"],
            "boot_p95": bs["p95"], "pos_year_rate": s["pos_year_rate"], "k": k}


def main():
    uni.register_cross_spreads(3.0)
    close = universe_close("H4")
    eqm, eqr, pool, closes = bl.champion_mtm()
    mask5, _ = bl.bleed_mask_monthly(eqm, thresh=0.05)
    print(f"universe={close.shape[1]} bars={len(close)}  champ bleed(-5%): {int(mask5.sum())}/{len(mask5)} 月\n")

    print("=== champion 単独ベースライン(直接測定, mm_lab)===")
    base = champion_alone_baseline()
    print(f"  CAGR={base['cagr']:+.1%} Sharpe={base['sharpe']:.2f} DDmtm={base['maxdd_mtm']:+.1%} "
          f"p95={base['boot_p95']:+.1%} pos_year={base['pos_year_rate']:.0%} k={base['k']:.2f}\n")

    HOLD, K = 42, 2  # 候補のbest統合overlay
    rob = hedge_robustness(close, eqm, hold=HOLD, k=K)
    df, ovl = integrated_sweep(close, hold=HOLD, k=K)

    # 最良統合(p95 を 20% 近傍に保ちつつ CAGR 最大)
    print("\n" + "=" * 100)
    print("3. 正直な総括")
    print("=" * 100)
    base_cagr = base["cagr"]
    best = df.sort_values("cagr", ascending=False).iloc[0]
    # テールが基準より悪化していないものの中で最良も探す
    tail_ok = df[df["boot_p95"] >= base["boot_p95"]]  # p95 が基準以上(=テール改善 or 同等)
    best_tail = tail_ok.sort_values("cagr", ascending=False).iloc[0] if len(tail_ok) else None
    print(f"champ単独: CAGR={base_cagr:+.1%} Sharpe={base['sharpe']:.2f} p95={base['boot_p95']:+.1%}")
    print(f"統合 最良CAGR: max_pos={int(best['max_pos'])} w={best['weight']:.2f} -> "
          f"CAGR={best['cagr']:+.1%} Sharpe={best['sharpe']:.2f} p95={best['boot_p95']:+.1%} "
          f"pos_year={best['pos_year_rate']:.0%}")
    if best_tail is not None:
        print(f"統合 テール非悪化下の最良: max_pos={int(best_tail['max_pos'])} w={best_tail['weight']:.2f} -> "
              f"CAGR={best_tail['cagr']:+.1%} Sharpe={best_tail['sharpe']:.2f} p95={best_tail['boot_p95']:+.1%}")
    beats = best["cagr"] > base_cagr
    print(f"\n  integrated_cagr({best['cagr']:+.1%}) > champ単独({base_cagr:+.1%}) ? {beats}")
    print(f"  CAGR差: {(best['cagr']-base_cagr)*100:+.1f}pp")

    return {"base": base, "rob": rob, "sweep": df, "best": best, "best_tail": best_tail}


if __name__ == "__main__":
    main()
