"""イテレーション22(敵対的検証): ADXトレンドshort H4 を「失血窓ヘッジ + DD連動ゲート統合」として
独立に反証する。称賛でなく反証。

検証対象: adx_trend fast=30 slow=100 adx_period=14 adx_th=20 side=short tf=H4
          + DD連動ゲート(champion dd_mtm<-gate のときだけ overlay 発火)

攻撃ポイント:
  A. keysrc 衝突バグ点検: integrated統合は (instr, round(ret,12), bars_held) で champ/ovl を識別。
     champ と ovl のトレードがこのキーで衝突すると sizing が誤帰属する → 統合結果の信頼性が崩れる。
  B. sanity: weight=0(overlay無効) で champion単独 +21.6%/Sharpe1.21/boot95≈-28.5% が再現するか。
  C. ヘッジ頑健性: bleed閾値 {-3%,-5%,-8%} × IS/OOS × 2022除外窓 で mean_in_bleed がプラスか。
  D. パラメータ近傍: adx_th {15,20,25} / fast-slow {20-50,30-100,20-100} で hedge_edge がプラス持続か。
  E. 統合DDテスト(核心): gated統合で weight {0.25,0.5,1.0,1.5,2.0}、gate {0.0,0.05,0.10} をスイープし
     champion+overlay@20%DD の最良 CAGR > 21.6% か。Sharpe(1.21)/boot95(-28.5%)/プラス年も併記。
     非ゲート(常時)も対照で出す。

実行: uv run python exp22_adversarial_adx.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import bleed_lab as bl
import mm_lab as mm

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 50)

OOS_START = "2022-01-01"
BEST = ("adx_trend", {"fast": 30, "slow": 100, "adx_period": 14, "adx_th": 20}, "short")


# ============================================================ A. keysrc衝突点検
def check_keysrc_collision(overlay_pool):
    pool_c = mm.build_pool()
    pc = pool_c.copy(); pc["src"] = "champ"
    po = overlay_pool.copy(); po["src"] = "ovl"
    both = pd.concat([pc, po], ignore_index=True)
    keys = list(zip(both["instr"], both["ret"].round(12), both["bars_held"].astype(int)))
    both = both.assign(_k=keys)
    # 同一キーに champ と ovl 両方が居る = 衝突して誤帰属しうる
    grp = both.groupby("_k")["src"].agg(lambda s: set(s))
    collide = grp[grp.apply(lambda st: len(st) > 1)]
    n_keys = both["_k"].nunique()
    # overlayトレードのうち、衝突キーに該当する数
    ovl_keys = set(zip(po["instr"], po["ret"].round(12), po["bars_held"].astype(int)))
    champ_keys = set(zip(pc["instr"], pc["ret"].round(12), pc["bars_held"].astype(int)))
    shared = ovl_keys & champ_keys
    ovl_collide = po.assign(_k=list(zip(po["instr"], po["ret"].round(12), po["bars_held"].astype(int))))
    n_ovl_in_shared = ovl_collide["_k"].isin(shared).sum()
    print("=== A. keysrc衝突点検(誤帰属リスク) ===")
    print(f"  champ={len(pc)} ovl={len(po)} 合計={len(both)} / ユニークキー={n_keys}")
    print(f"  champ⋂ovl 衝突キー数={len(shared)} / overlayトレードの衝突該当={n_ovl_in_shared} "
          f"({n_ovl_in_shared/len(po):.1%})")
    if len(shared) == 0:
        print("  → 衝突なし。keysrc識別は健全。\n")
    else:
        print("  → 衝突あり! 衝突overlayトレードは champ扱いされ(乖離連動z)、ゲートが効かない可能性。\n")
    return len(shared), n_ovl_in_shared


# ============================================================ B/C. ヘッジ頑健性
def champion_bleed_masks():
    eqm, eqr, pool, closes = bl.champion_mtm()
    masks = {}
    for thr in (0.03, 0.05, 0.08):
        m, dd = bl.bleed_mask_monthly(eqm, thresh=thr)
        masks[thr] = m
    return masks, eqm


def hedge_robustness(overlay_name, params, side, masks):
    cand = bl.strategy_monthly_pnl(overlay_name, params=params, side=side, tf="H4")
    rows = []
    for thr, mask in masks.items():
        sc = bl.conditional_score(cand, mask, oos_start="2022-01")
        # 2022除外窓
        no22 = mask.copy()
        no22[(no22.index >= pd.Period("2022-01", "M")) & (no22.index <= pd.Period("2022-12", "M"))] = False
        s2 = cand.reindex(no22.index).fillna(0.0)
        mean_no22 = float(s2[no22.values].mean()) if no22.sum() else float("nan")
        rows.append({
            "thr": thr, "n_bleed": sc["n_bleed_months"],
            "mean_in_bleed": sc["mean_in_bleed"], "mean_normal": sc["mean_normal"],
            "hedge_edge": sc["hedge_edge"], "winrate": sc["winrate_in_bleed"],
            "IS": sc["mean_in_bleed_IS"], "OOS": sc["mean_in_bleed_OOS"],
            "no2022": mean_no22, "total_in_bleed": sc["total_in_bleed"],
            "total_all": sc["total_all"],
        })
    return pd.DataFrame(rows)


def per_year_bleed_contribution(overlay_name, params, side, mask):
    """失血窓を年別に分け、各年の窓内合計PnLを出す(2022一発でないかの直接確認)。"""
    cand = bl.strategy_monthly_pnl(overlay_name, params=params, side=side, tf="H4")
    s = cand.reindex(mask.index).fillna(0.0)
    inb = s[mask.values]
    by_year = inb.groupby([p.year for p in inb.index]).sum()
    cnt = pd.Series(1, index=inb.index).groupby([p.year for p in inb.index]).sum()
    return pd.DataFrame({"bleed_months": cnt, "total_pnl": by_year, "mean_pnl": by_year / cnt})


# ============================================================ E. 統合DDテスト
def build_gated_integration(overlay_pool):
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


def integrated(both, closes, fbar, keysrc, w, gate, max_pos=8, target_dd=0.20, n_boot=800):
    from mm_production import _fz

    def make_sizing(k):
        base = k / max_pos
        def sizing(ctx):
            s = keysrc.get((ctx["instr"], round(float(ctx["ret"]), 12), int(ctx["bars_held"])), "champ")
            if s == "champ":
                return ctx["equity_real"] * base * (_fz(ctx["z"]) / fbar)
            if gate is None:  # 非ゲート(常時)
                return ctx["equity_real"] * base * w
            if ctx["dd_mtm"] < -gate:
                return ctx["equity_real"] * base * w
            return 0.0
        return sizing

    k, eqm, eqr, info = mm.calibrate(both, closes, make_sizing, target_dd=target_dd, max_pos=max_pos)
    s = mm.stats(eqm, eqr, info)
    bs = mm.bootstrap_maxdd(eqm, n_boot=n_boot)
    return {"k": k, "cagr": s["cagr"], "maxdd": s["maxdd_mtm"], "sharpe": s["sharpe"],
            "boot95": bs["p95"], "posyr": s["pos_year_rate"], "worst": s["worst_year"]}


def champion_alone_baseline():
    """sanity: champion単独(overlayプール空) → +21.6%/Sharpe1.21/boot95を再現するか。"""
    pool_c = mm.build_pool()
    closes = mm.load_closes()
    from mm_production import champion_sizing
    mk = champion_sizing(pool_c, max_pos=8)
    k, eqm, eqr, info = mm.calibrate(pool_c, closes, mk, target_dd=0.20, max_pos=8)
    s = mm.stats(eqm, eqr, info)
    bs = mm.bootstrap_maxdd(eqm, n_boot=800)
    return {"k": k, "cagr": s["cagr"], "maxdd": s["maxdd_mtm"], "sharpe": s["sharpe"],
            "boot95": bs["p95"], "posyr": s["pos_year_rate"], "worst": s["worst_year"]}


def main():
    name, params, side = BEST
    mod = __import__(f"strategies.{name}", fromlist=["x"])
    ovl = mm.build_pool_for(mod, params, tf="H4", side=side,
                            tag=f"{name}_{'_'.join(str(v) for v in params.values())}_{side}")
    print(f"対象: {name} {params} side={side}  overlay={len(ovl)}トレード\n")

    # A. 衝突点検
    check_keysrc_collision(ovl)

    # B. sanity baseline
    print("=== B. sanity: champion単独 (overlay weight=0相当) ===")
    base = champion_alone_baseline()
    print(f"  k={base['k']:.2f} CAGR {base['cagr']:+.2%} maxDD {base['maxdd']:+.1%} "
          f"Sharpe {base['sharpe']:.2f} boot95 {base['boot95']:+.1%} +年 {base['posyr']:.0%} "
          f"最悪年 {base['worst']:+.1%}")
    print("  基準値: CAGR +21.6% / Sharpe 1.21 / boot95 -28.5% / 100%プラス年\n")

    # C. ヘッジ頑健性(bleed閾値 × IS/OOS × 2022除外)
    masks, eqm = champion_bleed_masks()
    print("=== C. ヘッジ頑健性: bleed閾値スイープ × IS/OOS × 2022除外(adx_th=20 short) ===")
    rob = hedge_robustness(name, params, side, masks)
    print(rob.round(1).to_string(index=False))
    print("  (mean_in_bleed/IS/OOS/no2022 が全てプラスなら持続。一つでも負なら hedge_robust 疑義)\n")

    print("=== C2. 失血窓の年別貢献(thr=-5%, 2022一発でないか)===")
    yr = per_year_bleed_contribution(name, params, side, masks[0.05])
    print(yr.round(1).to_string())
    pos_years = (yr["total_pnl"] > 0).sum()
    print(f"  失血窓のある年 {len(yr)}年中 {pos_years}年で窓内合計プラス\n")

    # D. パラメータ近傍
    print("=== D. パラメータ近傍の hedge_edge(thr=-5%, side=short) ===")
    neigh = [
        {"fast": 30, "slow": 100, "adx_period": 14, "adx_th": 15},
        {"fast": 30, "slow": 100, "adx_period": 14, "adx_th": 20},
        {"fast": 30, "slow": 100, "adx_period": 14, "adx_th": 25},
        {"fast": 20, "slow": 50, "adx_period": 14, "adx_th": 20},
        {"fast": 20, "slow": 100, "adx_period": 14, "adx_th": 20},
        {"fast": 10, "slow": 50, "adx_period": 14, "adx_th": 20},
    ]
    nrows = []
    for p in neigh:
        cand = bl.strategy_monthly_pnl(name, params=p, side=side, tf="H4")
        sc = bl.conditional_score(cand, masks[0.05], oos_start="2022-01")
        nrows.append({"cfg": f"f{p['fast']}/s{p['slow']}/th{p['adx_th']}",
                      "mean_in_bleed": sc["mean_in_bleed"], "mean_normal": sc["mean_normal"],
                      "hedge_edge": sc["hedge_edge"], "IS": sc["mean_in_bleed_IS"],
                      "OOS": sc["mean_in_bleed_OOS"], "total_all": sc["total_all"]})
    nd = pd.DataFrame(nrows)
    print(nd.round(1).to_string(index=False))
    pos_edge = (nd["hedge_edge"] > 0).sum()
    print(f"  近傍 {len(nd)}構成中 {pos_edge}構成で hedge_edge>0\n")

    # E. 統合DDテスト(核心)
    both, closes, fbar, keysrc = build_gated_integration(ovl)
    print("=== E. 統合DDテスト(champion+overlay@20%DD較正) ===")
    print(f"基準(champion単独): CAGR {base['cagr']:+.1%} Sharpe {base['sharpe']:.2f} "
          f"boot95 {base['boot95']:+.1%} +年 {base['posyr']:.0%}")
    print(f"{'mode':<14}{'gate':>6}{'w':>5}{'k':>7}{'CAGR':>9}{'maxDD':>8}{'Shrp':>7}"
          f"{'boot95':>9}{'+年':>6}{'worst':>8}")
    print("-" * 92)
    best_cagr = -9; best_cfg = None
    # gated
    for gate in [0.0, 0.05, 0.10]:
        for w in [0.25, 0.5, 1.0, 1.5, 2.0]:
            r = integrated(both, closes, fbar, keysrc, w, gate)
            flag = "  <<" if r["cagr"] > base["cagr"] else ""
            print(f"{'gated':<14}{gate:>6.2f}{w:>5.2f}{r['k']:>7.2f}{r['cagr']:>+9.1%}"
                  f"{r['maxdd']:>+8.1%}{r['sharpe']:>7.2f}{r['boot95']:>+9.1%}"
                  f"{r['posyr']:>5.0%}{r['worst']:>+8.1%}{flag}")
            if r["cagr"] > best_cagr:
                best_cagr = r["cagr"]; best_cfg = ("gated", gate, w, r)
        print()
    # 非ゲート(常時)対照
    print("--- 対照: 非ゲート(常時投入) ---")
    for w in [0.25, 0.5, 1.0]:
        r = integrated(both, closes, fbar, keysrc, w, None)
        flag = "  <<" if r["cagr"] > base["cagr"] else ""
        print(f"{'always':<14}{'-':>6}{w:>5.2f}{r['k']:>7.2f}{r['cagr']:>+9.1%}"
              f"{r['maxdd']:>+8.1%}{r['sharpe']:>7.2f}{r['boot95']:>+9.1%}"
              f"{r['posyr']:>5.0%}{r['worst']:>+8.1%}{flag}")

    print(f"\n=== 最良統合: {best_cfg[0]} gate={best_cfg[1]} w={best_cfg[2]} ===")
    r = best_cfg[3]
    print(f"  CAGR {r['cagr']:+.2%} (基準 {base['cagr']:+.2%}, 差 {r['cagr']-base['cagr']:+.2%}pp) "
          f"Sharpe {r['sharpe']:.2f} (基準 {base['sharpe']:.2f}) "
          f"boot95 {r['boot95']:+.1%} (基準 {base['boot95']:+.1%}) +年 {r['posyr']:.0%}")

    # F. 最良構成の IS較正→OOS素検証(過剰最適化チェック)
    print("\n=== F. 最良構成の IS較正→OOS素検証 ===")
    g, w = best_cfg[1], best_cfg[2]
    from mm_production import _fz
    def make_sizing(k):
        basek = k / 8
        def sizing(ctx):
            s = keysrc.get((ctx["instr"], round(float(ctx["ret"]), 12), int(ctx["bars_held"])), "champ")
            if s == "champ":
                return ctx["equity_real"] * basek * (_fz(ctx["z"]) / fbar)
            if ctx["dd_mtm"] < -g:
                return ctx["equity_real"] * basek * w
            return 0.0
        return sizing
    is_both = both[both["entry"] < OOS_START].reset_index(drop=True)
    oos_both = both[both["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]; oos_cl = closes[closes.index >= OOS_START]
    k_is, *_ = mm.calibrate(is_both, is_cl, make_sizing, target_dd=0.20, max_pos=8)
    eqm_o, eqr_o, info_o = mm.simulate(oos_both, oos_cl, make_sizing(k_is), max_pos=8)
    so = mm.stats(eqm_o, eqr_o, info_o)
    # champion単独のOOS
    from mm_production import champion_sizing
    mkc = champion_sizing(mm.build_pool(), max_pos=8)
    is_c = mm.build_pool()[mm.build_pool()["entry"] < OOS_START].reset_index(drop=True)
    oos_c = mm.build_pool()[mm.build_pool()["entry"] >= OOS_START].reset_index(drop=True)
    k_isc, *_ = mm.calibrate(is_c, is_cl, mkc, target_dd=0.20, max_pos=8)
    emo, ero, infoo = mm.simulate(oos_c, oos_cl, mkc(k_isc), max_pos=8)
    sco = mm.stats(emo, ero, infoo)
    print(f"  統合 OOS(IS較正k={k_is:.2f}→素): CAGR {so['cagr']:+.1%} maxDD {so['maxdd_mtm']:+.1%} "
          f"Sharpe {so['sharpe']:.2f} +年 {so['pos_year_rate']:.0%}")
    print(f"  champion単独 OOS(IS較正k={k_isc:.2f}→素): CAGR {sco['cagr']:+.1%} "
          f"maxDD {sco['maxdd_mtm']:+.1%} Sharpe {sco['sharpe']:.2f} +年 {sco['pos_year_rate']:.0%}")


if __name__ == "__main__":
    main()
