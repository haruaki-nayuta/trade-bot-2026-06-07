"""複合資金管理: Phase1 勝者レバー(per-inst ケリー × 乖離連動サイズ × max_pos)の融合。

狙い: Phase1 で個別に 20%DD 較正下で CAGR が高かった3レバーを **1つのサイジング関数に乗算合成**し、
joint で再チューニングする。合成は個別ベストを上回るか? それとも効果が重複/相殺するか? を実測。

レバー(各係数は平均1へ正規化 → 総量は k で素直に線形制御):
  L1 per-instrument ケリー: w_instr (銘柄別 logopt f_i を glob へ shrink 縮小し平均1正規化, clip)
  L2 乖離連動サイズ:        f(z)/fbar, f(z)=(z/z0)^p (深い乖離=より良い反転に厚く), clip
  L3 同時建玉上限 max_pos:  base = k/max_pos。max_pos を上げると1玉を小さく分散 → 同DDでより高い k 許容。

合成: alloc = equity_real * (k/max_pos) * w_instr * (f(z)/fbar)
  - w_instr は平均1, f(z)/fbar も平均1 なので、満玉時の総建玉平均 ~ equity*k(銘柄/zミックス一定なら厳密)。
    → calibrate(k) の単調性を保つ。
  - recent_vol は使わない(ケリーも乖離も edge/odds ベース)。立上り期も weights/f は定義済なのでフォールバック不要。

実行: uv run python mm_composite.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mm_lab as mm
from mm_kelly import per_instrument_weights


# --- 乖離連動係数 f(z) -------------------------------------------------------
def _make_fz(z0=2.2, p=2.0, lo=0.3, hi=3.0):
    """f(z)=(z/z0)^p を [lo,hi] でクリップ。Phase1 ベスト: p=2.0, z0=2.2。"""
    def _f(z):
        return float(np.clip((z / z0) ** p, lo, hi))
    return _f


# --- 複合 make_sizing ファクトリ --------------------------------------------
def make_composite(pool, *, max_pos=6,
                   use_kelly=True, shrink=0.5, min_trades=15, kelly_clip=4.0,
                   use_z=True, z0=2.2, p=2.0, z_lo=0.3, z_hi=3.0):
    """3レバー乗算合成の make_sizing(k) を返す。

    各レバーは ON/OFF 可。w_instr / (f/fbar) はそれぞれ平均1正規化済 → 総建玉は k に線形。
    """
    # L1: per-instrument ケリー重み(平均1)
    if use_kelly:
        weights = per_instrument_weights(pool, min_trades=min_trades, shrink=shrink)
    else:
        weights = {}

    # L2: 乖離連動係数 f(z) を fbar(プール平均)で正規化
    if use_z:
        fz = _make_fz(z0=z0, p=p, lo=z_lo, hi=z_hi)
        zvals = pool["z_entry"].to_numpy()
        fbar = float(np.mean([fz(z) if np.isfinite(z) else 1.0 for z in zvals]))
        if fbar <= 0:
            fbar = 1.0
    else:
        fz, fbar = None, 1.0

    def make_sizing(k):
        base = k / max_pos

        def sizing(ctx):
            mult = base
            if use_kelly:
                wi = weights.get(ctx["instr"], 1.0)
                mult *= min(wi, kelly_clip)
            if use_z:
                z = ctx["z"]
                f = fz(z) if np.isfinite(z) else 1.0
                mult *= f / fbar
            return ctx["equity_real"] * mult

        return sizing

    return make_sizing


def _row(name, r):
    return (f"{name:<34} k={r['k']:>5.2f} CAGR={r['cagr']:>+7.2%} "
            f"DD={r['maxdd_mtm']:>+6.1%} real={r['maxdd_real']:>+6.1%} "
            f"Sh={r['sharpe']:>4.2f} +yr={r['pos_year_rate']:>4.0%} "
            f"p95={r['boot_p95']:>+6.1%} p99={r['boot_p99']:>+6.1%} | "
            f"OOS CAGR={r.get('oos_cagr', float('nan')):>+7.2%} "
            f"DD={r.get('oos_maxdd_mtm', float('nan')):>+6.1%} "
            f"+yr={r.get('oos_pos_year', float('nan')):>4.0%}")


def main():
    pool = mm.build_pool()
    closes = mm.load_closes()
    print(f"プール {len(pool)} トレード / グリッド {len(closes)} 本\n")

    # ベースライン参照(固定比率, max_pos=6)
    print("=== 参照: ベースライン固定比率 max_pos=6 ===")
    base_mk = mm.fixed_fractional  # placeholder; use factory below
    from mm_maxpos import make_sizing_factory as ff_factory
    rb = mm.evaluate_method("baseline_ff_mp6", pool, closes, ff_factory(6),
                            target_dd=0.20, max_pos=6, n_boot=400)
    print(_row("baseline fixed mp6", rb))
    print()

    # --- 個別レバー(再確認, 同一プールで) ---
    print("=== 個別レバー(再確認) max_pos=6 ===")
    r_k = mm.evaluate_method("kelly_only", pool, closes,
                             make_composite(pool, max_pos=6, use_kelly=True, use_z=False),
                             target_dd=0.20, max_pos=6, n_boot=400)
    print(_row("L1 kelly only mp6", r_k))
    r_z = mm.evaluate_method("z_only", pool, closes,
                             make_composite(pool, max_pos=6, use_kelly=False, use_z=True),
                             target_dd=0.20, max_pos=6, n_boot=400)
    print(_row("L2 z only mp6", r_z))
    # L3 individual best (fixed-fractional, max_pos=12)
    r_mp = mm.evaluate_method("maxpos12", pool, closes, ff_factory(12),
                              target_dd=0.20, max_pos=12, n_boot=400)
    print(_row("L3 maxpos12 (fixed)", r_mp))
    print()

    # --- 2レバー合成 ---
    print("=== 2レバー合成 ===")
    r_kz = mm.evaluate_method("kelly+z mp6", pool, closes,
                              make_composite(pool, max_pos=6, use_kelly=True, use_z=True),
                              target_dd=0.20, max_pos=6, n_boot=400)
    print(_row("L1+L2 kelly*z mp6", r_kz))

    r_kmp = mm.evaluate_method("kelly mp12", pool, closes,
                               make_composite(pool, max_pos=12, use_kelly=True, use_z=False),
                               target_dd=0.20, max_pos=12, n_boot=400)
    print(_row("L1+L3 kelly mp12", r_kmp))

    r_zmp = mm.evaluate_method("z mp12", pool, closes,
                               make_composite(pool, max_pos=12, use_kelly=False, use_z=True),
                               target_dd=0.20, max_pos=12, n_boot=400)
    print(_row("L2+L3 z mp12", r_zmp))
    print()

    # --- 3レバー合成: max_pos スイープ ---
    print("=== 3レバー合成 kelly*z, max_pos スイープ ===")
    results = {}
    for mp in [6, 8, 10, 12, 14]:
        mk = make_composite(pool, max_pos=mp, use_kelly=True, use_z=True)
        r = mm.evaluate_method(f"composite_mp{mp}", pool, closes, mk,
                               target_dd=0.20, max_pos=mp, n_boot=400)
        results[mp] = r
        print(_row(f"L1+L2+L3 kelly*z mp{mp}", r))
    print()

    # --- 3レバー合成: shape スイープ (best max_pos 固定) ---
    best_mp = max(results, key=lambda m: results[m]["cagr"])
    print(f"=== shape スイープ (max_pos={best_mp} 固定) ===")
    shape_grid = [
        dict(p=1.5, z0=2.2, shrink=0.5),
        dict(p=2.0, z0=2.0, shrink=0.5),
        dict(p=2.0, z0=2.4, shrink=0.5),
        dict(p=2.5, z0=2.2, shrink=0.5),
        dict(p=2.0, z0=2.2, shrink=0.3),
        dict(p=2.0, z0=2.2, shrink=0.7),
        dict(p=2.0, z0=2.2, z_hi=4.0),
    ]
    shape_results = {}
    for cfg in shape_grid:
        mk = make_composite(pool, max_pos=best_mp, use_kelly=True, use_z=True, **cfg)
        r = mm.evaluate_method(f"comp_{cfg}", pool, closes, mk,
                               target_dd=0.20, max_pos=best_mp, n_boot=400)
        key = ",".join(f"{k}={v}" for k, v in cfg.items())
        shape_results[key] = r
        print(_row(f"  {key}", r))
    print()

    # --- 最終確定: 全候補から CAGR 最大 → n_boot=1500 ---
    candidates = {f"mp{m}_default": results[m] for m in results}
    candidates.update(shape_results)
    best_key = max(candidates, key=lambda kk: candidates[kk]["cagr"])
    print(f"=== 暫定ベスト構成: {best_key} (CAGR={candidates[best_key]['cagr']:+.2%}) ===")

    # 再構成して n_boot=1500 で確定
    if best_key.startswith("mp"):
        bmp = int(best_key.split("_")[0][2:])
        final_mk = make_composite(pool, max_pos=bmp, use_kelly=True, use_z=True)
        final_max_pos = bmp
        best_cfg_str = f"max_pos={bmp},p=2.0,z0=2.2,shrink=0.5(default shape)"
    else:
        cfg = dict(x.split("=") for x in best_key.split(","))
        kw = {}
        for k2, v2 in cfg.items():
            kw[k2] = float(v2)
        final_mk = make_composite(pool, max_pos=best_mp, use_kelly=True, use_z=True, **kw)
        final_max_pos = best_mp
        best_cfg_str = f"max_pos={best_mp}," + best_key

    print(f"\n=== 確定評価 (n_boot=1500): {best_cfg_str} ===")
    final = mm.evaluate_method("composite_final", pool, closes, final_mk,
                               target_dd=0.20, max_pos=final_max_pos, n_boot=1500)
    for key in ["method", "k", "cagr", "total_return", "maxdd_mtm", "maxdd_real",
                "sharpe", "sortino", "pos_year_rate", "worst_year",
                "boot_p95", "boot_p99", "boot_worst", "max_conc", "n_taken", "skipped",
                "k_is", "oos_cagr", "oos_maxdd_mtm", "oos_pos_year", "oos_sharpe"]:
        if key in final:
            v = final[key]
            print(f"  {key:>16s}: {v:+.4f}" if isinstance(v, float) else f"  {key:>16s}: {v}")

    return results, shape_results, final, best_cfg_str


if __name__ == "__main__":
    main()
