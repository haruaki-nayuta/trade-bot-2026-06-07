"""敵対的検証: mm_composite のベスト構成(per-inst Kelly × z-power, max_pos=6)を独立再現・反証する。

検証項目:
  1. 再現: 20%DD較正で CAGR / MtM DD を独立に再現できるか。
  2. 理論DDのブロック感度: bootstrap_maxdd を block∈{21,63,126,252} で回し p95 範囲。
  3. 逆方向IS/OOS: OOS(2022-)で較正→IS(2016-2021)で素のMtM最大DD。
  4. 高原性: shape(p, z0, shrink, max_pos)を±でずらしてCAGRが崩れないか。
  5. 単年DD: 年ごとのMtM最大DD。
  6. ベースライン比較(固定比率 max_pos=6)。

独立実装: make_composite を自前で書き直す(mm_composite に依存しない)。weights は mm_kelly を流用。
数値はすべて自分で再計算。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mm_lab as mm
from mm_kelly import per_instrument_weights


# --- 独立に書き直した合成サイジング -----------------------------------------
def make_composite_indep(pool, *, max_pos=6, shrink=0.5, min_trades=15, kelly_clip=4.0,
                         z0=2.2, p=2.0, z_lo=0.3, z_hi=3.0):
    """per-inst Kelly × z-power の make_sizing(k)。各レバー平均1正規化 → 総建玉は k に線形。"""
    weights = per_instrument_weights(pool, min_trades=min_trades, shrink=shrink)

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
            wi = weights.get(ctx["instr"], 1.0)
            mult *= min(wi, kelly_clip)
            z = ctx["z"]
            f = fz(z) if np.isfinite(z) else 1.0
            mult *= f / fbar
            return ctx["equity_real"] * mult

        return sizing

    return make_sizing


def _max_dd(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1.0).min())


def yearly_mtm_dd(eqm: pd.Series) -> pd.Series:
    """各暦年内での MtM 最大DD(年内のピーク→ボトム)。"""
    out = {}
    for yr, grp in eqm.groupby(eqm.index.year):
        out[yr] = _max_dd(grp)
    return pd.Series(out)


def main():
    pool = mm.build_pool()
    closes = mm.load_closes()
    print(f"プール {len(pool)} トレード / グリッド {len(closes)} 本\n")

    # ========== 1. 再現(20%DD較正) ==========
    print("=" * 70)
    print("1. 再現: best config を独立に 20%DD較正")
    print("=" * 70)
    mk_best = make_composite_indep(pool, max_pos=6, shrink=0.5, min_trades=15,
                                   kelly_clip=4.0, z0=2.2, p=2.0, z_lo=0.3, z_hi=3.0)
    k, eqm, eqr, info = mm.calibrate(pool, closes, mk_best, target_dd=0.20, max_pos=6)
    s = mm.stats(eqm, eqr, info)
    print(f"  較正 k        = {k:.4f}   (主張 7.147)")
    print(f"  CAGR          = {s['cagr']:+.4f}  (主張 +30.92%)")
    print(f"  MtM 最大DD    = {s['maxdd_mtm']:+.4f}")
    print(f"  実現 最大DD   = {s['maxdd_real']:+.4f}")
    print(f"  Sharpe        = {s['sharpe']:.3f}")
    print(f"  プラス年率    = {s['pos_year_rate']:.0%} ({int(s['pos_year_rate']*s['n_years'])}/{s['n_years']})")
    print(f"  最悪年        = {s['worst_year']:+.2%}")
    print(f"  最大同時建玉  = {s['max_conc']}  /  見送り {s['skipped']}  /  建玉 {s['n_taken']}")

    # ========== ベースライン(固定比率 max_pos=6) ==========
    print("\n" + "=" * 70)
    print("参考: ベースライン固定比率 max_pos=6")
    print("=" * 70)
    from mm_maxpos import make_sizing_factory
    mk_base = make_sizing_factory(6)
    kb, eqmb, eqrb, infob = mm.calibrate(pool, closes, mk_base, target_dd=0.20, max_pos=6)
    sb = mm.stats(eqmb, eqrb, infob)
    print(f"  k={kb:.3f}  CAGR={sb['cagr']:+.2%}  MtM DD={sb['maxdd_mtm']:+.2%}  Sharpe={sb['sharpe']:.2f}  +yr={sb['pos_year_rate']:.0%}")

    # ========== 2. 理論DDのブロック感度 ==========
    print("\n" + "=" * 70)
    print("2. 理論DDブロック感度 (best config の eqm)")
    print("=" * 70)
    p95s = []
    for block in [21, 63, 126, 252]:
        bs = mm.bootstrap_maxdd(eqm, n_boot=1500, block=block)
        p95s.append(bs["p95"])
        print(f"  block={block:>3}  p50={bs['p50']:+.1%}  p95={bs['p95']:+.1%}  p99={bs['p99']:+.1%}  worst={bs['worst']:+.1%}")
    print(f"  p95 範囲: [{min(p95s):+.1%}, {max(p95s):+.1%}]")
    # ベースラインの p95 (block=63 標準)
    bsb = mm.bootstrap_maxdd(eqmb, n_boot=1500, block=63)
    print(f"  ベースライン p95(block63)={bsb['p95']:+.1%}  (基準 -28.1%)")

    # ========== 3. 逆方向 IS/OOS ==========
    print("\n" + "=" * 70)
    print("3. 逆方向 IS/OOS: OOS(2022-)で較正 → IS(2016-2021)で素検証")
    print("=" * 70)
    oos_start = "2022-01-01"
    is_pool = pool[pool["entry"] < oos_start].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= oos_start].reset_index(drop=True)
    is_closes = closes[closes.index < oos_start]
    oos_closes = closes[closes.index >= oos_start]
    print(f"  IS trades={len(is_pool)}  OOS trades={len(oos_pool)}")
    # 順方向(参考): IS較正→OOS素
    k_is_fwd, *_ = mm.calibrate(is_pool, is_closes, mk_best, target_dd=0.20, max_pos=6)
    eqm_fwd, eqr_fwd, info_fwd = mm.simulate(oos_pool, oos_closes, mk_best(k_is_fwd), max_pos=6)
    s_fwd = mm.stats(eqm_fwd, eqr_fwd, info_fwd)
    print(f"  [順] IS較正 k={k_is_fwd:.3f} → OOS素: CAGR={s_fwd['cagr']:+.2%}  MtM DD={s_fwd['maxdd_mtm']:+.2%}  +yr={s_fwd['pos_year_rate']:.0%}")
    # 逆方向: OOS較正→IS素
    k_oos, *_ = mm.calibrate(oos_pool, oos_closes, mk_best, target_dd=0.20, max_pos=6)
    eqm_rev, eqr_rev, info_rev = mm.simulate(is_pool, is_closes, mk_best(k_oos), max_pos=6)
    s_rev = mm.stats(eqm_rev, eqr_rev, info_rev)
    print(f"  [逆] OOS較正 k={k_oos:.3f} → IS素: CAGR={s_rev['cagr']:+.2%}  MtM DD={s_rev['maxdd_mtm']:+.2%}  +yr={s_rev['pos_year_rate']:.0%}")
    oos_to_is_maxdd = s_rev["maxdd_mtm"]

    # ========== 4. 高原性(shape ±スイープ) ==========
    print("\n" + "=" * 70)
    print("4. 高原性: best config 近傍を ±スイープ (n_boot=400, 20%DD較正)")
    print("=" * 70)
    base_cfg = dict(max_pos=6, shrink=0.5, min_trades=15, kelly_clip=4.0,
                    z0=2.2, p=2.0, z_lo=0.3, z_hi=3.0)
    neighbors = [
        ("best (p=2.0,z0=2.2,shrink=0.5,mp6)", {}),
        ("p=1.5", dict(p=1.5)),
        ("p=2.5", dict(p=2.5)),
        ("z0=2.0", dict(z0=2.0)),
        ("z0=2.4", dict(z0=2.4)),
        ("shrink=0.3", dict(shrink=0.3)),
        ("shrink=0.7", dict(shrink=0.7)),
        ("kelly_clip=3.0", dict(kelly_clip=3.0)),
        ("kelly_clip=6.0", dict(kelly_clip=6.0)),
        ("z_hi=4.0", dict(z_hi=4.0)),
        ("max_pos=5", dict(max_pos=5)),
        ("max_pos=8", dict(max_pos=8)),
    ]
    cagrs = []
    for label, override in neighbors:
        cfg = {**base_cfg, **override}
        mk = make_composite_indep(pool, **cfg)
        r = mm.evaluate_method(label, pool, closes, mk, target_dd=0.20,
                               max_pos=cfg["max_pos"], n_boot=400)
        cagrs.append((label, r["cagr"], r["maxdd_mtm"], r["boot_p95"], r.get("oos_cagr", np.nan), r.get("oos_maxdd_mtm", np.nan)))
        print(f"  {label:<38} CAGR={r['cagr']:>+7.2%}  DD={r['maxdd_mtm']:>+6.1%}  p95={r['boot_p95']:>+6.1%}  "
              f"OOS_CAGR={r.get('oos_cagr', float('nan')):>+7.2%}  OOS_DD={r.get('oos_maxdd_mtm', float('nan')):>+6.1%}")
    near_cagrs = [c for (lab, c, *_ ) in cagrs]
    print(f"\n  近傍CAGR範囲: [{min(near_cagrs):+.2%}, {max(near_cagrs):+.2%}]  spread={max(near_cagrs)-min(near_cagrs):.2%}")

    # ========== 5. 単年DD ==========
    print("\n" + "=" * 70)
    print("5. 単年MtM最大DD (best config, フル較正の eqm)")
    print("=" * 70)
    ydd = yearly_mtm_dd(eqm)
    for yr, dd in ydd.items():
        flag = "  <-- 20%超" if dd < -0.20 else ""
        print(f"  {yr}: {dd:+.2%}{flag}")
    worst_year_dd = float(ydd.min())
    print(f"  最悪単年DD = {worst_year_dd:+.2%}")

    # ========== 確定評価 n_boot=1500 ==========
    print("\n" + "=" * 70)
    print("確定: best config full evaluate_method (n_boot=1500)")
    print("=" * 70)
    final = mm.evaluate_method("composite_verify", pool, closes, mk_best,
                               target_dd=0.20, max_pos=6, n_boot=1500)
    for key in ["k", "cagr", "maxdd_mtm", "maxdd_real", "sharpe", "pos_year_rate",
                "worst_year", "boot_p95", "boot_p99", "k_is", "oos_cagr",
                "oos_maxdd_mtm", "oos_pos_year"]:
        v = final[key]
        print(f"  {key:>16s}: {v:+.4f}" if isinstance(v, float) else f"  {key:>16s}: {v}")

    # サマリ
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"reproduced_cagr      = {s['cagr']:.4f}")
    print(f"reproduced_maxdd_mtm = {s['maxdd_mtm']:.4f}")
    print(f"boot_p95_range       = [{min(p95s):.4f}, {max(p95s):.4f}]")
    print(f"oos_to_is_maxdd      = {oos_to_is_maxdd:.4f}")
    print(f"worst_year_dd        = {worst_year_dd:.4f}")
    print(f"near_cagr_range      = [{min(near_cagrs):.4f}, {max(near_cagrs):.4f}]")
    print(f"baseline_cagr        = {sb['cagr']:.4f}")


if __name__ == "__main__":
    main()
