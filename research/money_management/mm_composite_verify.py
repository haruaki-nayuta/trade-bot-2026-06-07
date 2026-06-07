"""敵対的検証: mm_composite.py の主張(3レバー乗算合成, CAGR=34.4%, k=9.86, max_pos=8)を反証する。

独立に make_sizing を再実装し(per-inst Kelly × f(z)=(z/z0)^p × max_pos), さらに既存 factory と
クロスチェックした上で以下を実測:
  1. 再現: 20%DD較正で CAGR/MtM DD を独立に再現できるか
  2. 理論DDのブロック感度: block in {21,63,126,252} で p95 範囲
  3. 逆方向IS/OOS: OOS(2022-)で較正→IS(2016-2021)で素のMtM最大DD
  4. 高原性: best_config の shape を ± でずらして CAGR が崩れないか
  5. 単年DD: 年ごとの MtM DD が全て 20% 以内か
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import mm_lab as mm

BEST = dict(max_pos=8, shrink=0.5, min_trades=15, kelly_clip=4.0,
            z0=2.2, p=2.0, z_lo=0.3, z_hi=3.0)


# --- 独立再実装: per-instrument logopt Kelly 重み(平均1正規化) -----------
def kelly_logopt(r, fmax=60.0, n=4000):
    if len(r) < 5:
        return 0.0
    fs = np.linspace(0.0, fmax, n)
    best_f, best_g = 0.0, -1e18
    for f in fs:
        x = 1.0 + f * r
        if (x <= 0).any():
            break
        g = np.mean(np.log(x))
        if g > best_g:
            best_g, best_f = g, f
    return float(best_f)


def per_inst_weights(pool, min_trades=15, shrink=0.5):
    glob = kelly_logopt(pool["ret"].to_numpy())
    w = {}
    for nm, g in pool.groupby("instr"):
        r = g["ret"].to_numpy()
        if len(r) < min_trades:
            fi = glob
        else:
            fi = shrink * kelly_logopt(r) + (1 - shrink) * glob
        w[nm] = fi
    arr = np.array(list(w.values()))
    mean_w = arr[arr > 0].mean() if (arr > 0).any() else 1.0
    return {k: (v / mean_w if mean_w > 0 else 1.0) for k, v in w.items()}


def make_composite_indep(pool, *, max_pos=8, shrink=0.5, min_trades=15, kelly_clip=4.0,
                         z0=2.2, p=2.0, z_lo=0.3, z_hi=3.0):
    """独立実装の3レバー乗算合成 make_sizing(k)。重複/相殺の検証のためゼロから書く。"""
    weights = per_inst_weights(pool, min_trades=min_trades, shrink=shrink)

    def fz(z):
        return float(np.clip((z / z0) ** p, z_lo, z_hi))

    zvals = pool["z_entry"].to_numpy()
    fbar = float(np.mean([fz(z) if np.isfinite(z) else 1.0 for z in zvals]))
    if fbar <= 0:
        fbar = 1.0

    def make_sizing(k):
        base = k / max_pos

        def sizing(ctx):
            mult = base
            wi = min(weights.get(ctx["instr"], 1.0), kelly_clip)
            mult *= wi
            z = ctx["z"]
            f = fz(z) if np.isfinite(z) else 1.0
            mult *= f / fbar
            return ctx["equity_real"] * mult
        return sizing

    return make_sizing


def max_dd(eq):
    return float((eq / eq.cummax() - 1.0).min())


def yearly_mtm_dd(eqm):
    out = {}
    for yr, sub in eqm.groupby(eqm.index.year):
        out[int(yr)] = float((sub / sub.cummax() - 1.0).min())
    return out


def main():
    pool = mm.build_pool()
    closes = mm.load_closes()
    print(f"プール {len(pool)} / グリッド {len(closes)} / 期間 {closes.index[0].date()}->{closes.index[-1].date()}\n")

    # ===== 0. ベースライン参照(固定比率 max_pos=6)を独立に再較正 =====
    from mm_maxpos import make_sizing_factory as ff
    rb = mm.evaluate_method("baseline_ff_mp6", pool, closes, ff(6),
                            target_dd=0.20, max_pos=6, n_boot=1500)
    print("=== 0. ベースライン固定比率 mp6 (独立再較正, n_boot=1500) ===")
    print(f"  k={rb['k']:.2f} CAGR={rb['cagr']:+.2%} DD={rb['maxdd_mtm']:+.2%} Sh={rb['sharpe']:.2f} "
          f"+yr={rb['pos_year_rate']:.0%} p95={rb['boot_p95']:+.2%} p99={rb['boot_p99']:+.2%} "
          f"OOS_CAGR={rb.get('oos_cagr',float('nan')):+.2%} OOS_DD={rb.get('oos_maxdd_mtm',float('nan')):+.2%}")
    print()

    # ===== 1. 再現(独立実装) + 既存factoryクロスチェック =====
    print("=== 1. 再現 (独立実装, 20%DD較正, max_pos=8) ===")
    mk = make_composite_indep(pool, **BEST)
    k, eqm, eqr, info = mm.calibrate(pool, closes, mk, target_dd=0.20, max_pos=BEST["max_pos"])
    s = mm.stats(eqm, eqr, info)
    print(f"  [INDEP] k={k:.3f}  CAGR={s['cagr']:+.2%}  MtM DD={s['maxdd_mtm']:+.2%}  "
          f"real DD={s['maxdd_real']:+.2%}  Sharpe={s['sharpe']:.2f}  +yr={s['pos_year_rate']:.0%}  "
          f"worst_yr={s['worst_year']:+.2%}  max_conc={info['max_conc']}  skipped={info['skipped']}")

    from mm_composite import make_composite as orig_factory
    mk_orig = orig_factory(pool, max_pos=BEST["max_pos"], use_kelly=True, use_z=True,
                           shrink=BEST["shrink"], min_trades=BEST["min_trades"],
                           kelly_clip=BEST["kelly_clip"], z0=BEST["z0"], p=BEST["p"],
                           z_lo=BEST["z_lo"], z_hi=BEST["z_hi"])
    k2, eqm2, eqr2, info2 = mm.calibrate(pool, closes, mk_orig, target_dd=0.20, max_pos=BEST["max_pos"])
    s2 = mm.stats(eqm2, eqr2, info2)
    print(f"  [ORIG ] k={k2:.3f}  CAGR={s2['cagr']:+.2%}  MtM DD={s2['maxdd_mtm']:+.2%}  "
          f"(独立実装と一致すべき)")
    # full eval (boot p95/p99 + 内蔵 IS->OOS) on independent impl
    rc = mm.evaluate_method("composite_best_indep", pool, closes, mk,
                            target_dd=0.20, max_pos=BEST["max_pos"], n_boot=1500)
    print(f"  boot p95={rc['boot_p95']:+.2%} p99={rc['boot_p99']:+.2%} worst={rc['boot_worst']:+.2%}")
    print(f"  内蔵 IS->OOS: k_is={rc.get('k_is',float('nan')):.2f} OOS_CAGR={rc.get('oos_cagr',float('nan')):+.2%} "
          f"OOS_DD={rc.get('oos_maxdd_mtm',float('nan')):+.2%} OOS_+yr={rc.get('oos_pos_year',float('nan')):.0%}")
    print()

    # ===== 5. 単年DD =====
    print("=== 5. 単年(暦年内)MtM DD(較正済 k) ===")
    ydd = yearly_mtm_dd(eqm)
    worst_year_dd = min(ydd.values())
    for yr in sorted(ydd):
        flag = "  <<< 20%超!" if ydd[yr] < -0.20 else ""
        print(f"  {yr}: {ydd[yr]:+.2%}{flag}")
    print(f"  >> 最悪単年DD = {worst_year_dd:+.2%}")
    print()

    # ===== 2. 理論DD ブロック感度 =====
    print("=== 2. 理論DD ブロック感度 (bootstrap p95, n_boot=2000) ===")
    boot_p95 = {}
    for blk in [21, 63, 126, 252]:
        bs = mm.bootstrap_maxdd(eqm, n_boot=2000, block=blk, seed=0)
        boot_p95[blk] = bs["p95"]
        print(f"  block={blk:>3}: p50={bs['p50']:+.1%}  p95={bs['p95']:+.1%}  "
              f"p99={bs['p99']:+.1%}  worst={bs['worst']:+.1%}")
    eqm_b, _, _ = mm.simulate(pool, closes, ff(6)(rb["k"]), max_pos=6)
    print("  --- ベースライン参照(同条件) ---")
    base_p95 = {}
    for blk in [21, 63, 126, 252]:
        bs = mm.bootstrap_maxdd(eqm_b, n_boot=2000, block=blk, seed=0)
        base_p95[blk] = bs["p95"]
        print(f"  block={blk:>3}: p50={bs['p50']:+.1%}  p95={bs['p95']:+.1%}  p99={bs['p99']:+.1%}")
    print(f"  >> composite p95範囲: [{min(boot_p95.values()):+.1%}, {max(boot_p95.values()):+.1%}]  "
          f"/ baseline p95範囲: [{min(base_p95.values()):+.1%}, {max(base_p95.values()):+.1%}]")
    print()

    # ===== 3. 逆方向IS/OOS: OOSで較正→ISで素検証 =====
    print("=== 3. 逆方向IS/OOS: OOS(2022-)で較正 → IS(2016-2021)で素のMtM DD ===")
    oos_start = "2022-01-01"
    is_pool = pool[pool["entry"] < oos_start].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= oos_start].reset_index(drop=True)
    is_closes = closes[closes.index < oos_start]
    oos_closes = closes[closes.index >= oos_start]
    # OOSプールで重み再推定して較正(真にOOSだけで決める)
    mk_oos = make_composite_indep(oos_pool, **BEST)
    k_oos, eqm_oc, eqr_oc, info_oc = mm.calibrate(oos_pool, oos_closes, mk_oos,
                                                   target_dd=0.20, max_pos=BEST["max_pos"])
    s_oos = mm.stats(eqm_oc, eqr_oc, info_oc)
    eqm_is, eqr_is, info_is = mm.simulate(is_pool, is_closes, mk_oos(k_oos), max_pos=BEST["max_pos"])
    s_is = mm.stats(eqm_is, eqr_is, info_is)
    oos_to_is_maxdd = s_is["maxdd_mtm"]
    print(f"  OOS較正: k={k_oos:.3f}  OOS自身CAGR={s_oos['cagr']:+.2%}  OOS自身DD={s_oos['maxdd_mtm']:+.2%}")
    print(f"  → IS素検証: IS CAGR={s_is['cagr']:+.2%}  IS MtM DD={oos_to_is_maxdd:+.2%}  +yr={s_is['pos_year_rate']:.0%}")
    print(f"  (20%を大きく超えれば較正の過剰最適化)")
    print()

    # ===== 4. 高原性: shape を±でずらす =====
    print("=== 4. 高原性: best_config 近傍スイープ (各20%DD較正, n_boot=400) ===")
    neighbors = [
        ("base", {}),
        ("p=1.5", dict(p=1.5)), ("p=2.5", dict(p=2.5)),
        ("z0=2.0", dict(z0=2.0)), ("z0=2.4", dict(z0=2.4)),
        ("shrink=0.3", dict(shrink=0.3)), ("shrink=0.7", dict(shrink=0.7)),
        ("kelly_clip=3.0", dict(kelly_clip=3.0)), ("kelly_clip=6.0", dict(kelly_clip=6.0)),
        ("z_hi=4.0", dict(z_hi=4.0)),
        ("max_pos=6", dict(max_pos=6)), ("max_pos=10", dict(max_pos=10)), ("max_pos=12", dict(max_pos=12)),
    ]
    center_cagr = s["cagr"]
    shape_cagrs = []  # max_pos=8 を保つ shape 近傍だけ
    all_rows = []
    for label, override in neighbors:
        cfg = dict(BEST); cfg.update(override)
        mp = cfg["max_pos"]
        mkn = make_composite_indep(pool, **cfg)
        r = mm.evaluate_method(label, pool, closes, mkn, target_dd=0.20, max_pos=mp, n_boot=400)
        all_rows.append((label, r["cagr"], r["maxdd_mtm"], r["boot_p95"], r["pos_year_rate"],
                         r.get("oos_cagr", float("nan")), r.get("oos_maxdd_mtm", float("nan"))))
        if mp == 8:
            shape_cagrs.append(r["cagr"])
        print(f"  {label:<16} k={r['k']:>5.2f} CAGR={r['cagr']:>+7.2%} DD={r['maxdd_mtm']:>+6.1%} "
              f"p95={r['boot_p95']:>+6.1%} +yr={r['pos_year_rate']:>4.0%} | "
              f"OOS_CAGR={r.get('oos_cagr',float('nan')):>+7.2%} OOS_DD={r.get('oos_maxdd_mtm',float('nan')):>+6.1%}")
    plateau_ok = min(shape_cagrs) >= 0.85 * center_cagr
    print(f"  >> shape近傍(max_pos=8固定)CAGR範囲: [{min(shape_cagrs):+.2%}, {max(shape_cagrs):+.2%}]  "
          f"center={center_cagr:+.2%}  plateau_ok(>=85%)={plateau_ok}")
    print()

    # ===== サマリ =====
    print("===== サマリ =====")
    print(f"baseline   CAGR={rb['cagr']:+.2%}  p95={rb['boot_p95']:+.2%}")
    print(f"composite  CAGR={s['cagr']:+.2%}  k={k:.2f}  p95(block63)={rc['boot_p95']:+.2%}")
    print(f"CAGR差 = {(s['cagr']-rb['cagr'])*100:+.2f}pt")
    print(f"boot_p95 by block = " + "  ".join(f"{b}:{boot_p95[b]:+.1%}" for b in [21,63,126,252]))
    print(f"oos_to_is_maxdd = {oos_to_is_maxdd:+.2%}")
    print(f"worst_year_dd = {worst_year_dd:+.2%}")
    print(f"plateau_ok = {plateau_ok}")

    return dict(rb=rb, cagr=s["cagr"], maxdd=s["maxdd_mtm"], k=k, boot_p95=boot_p95,
                rc=rc, oos_to_is=oos_to_is_maxdd, worst_year=worst_year_dd,
                plateau=plateau_ok, pos_year=s["pos_year_rate"])


if __name__ == "__main__":
    main()
