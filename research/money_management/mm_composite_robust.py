"""複合資金管理の過剰最適化チェック: ベスト近傍の高原性 + OOS 健全性。

mm_composite の暫定ベスト(max_pos=8, kelly*z)周辺を密にスイープし、
shrink/p/z0/max_pos の近傍で CAGR が崩れないか(高原 vs ナイフエッジ)を判定する。
特に shrink を上げる方向(per-inst 推定を信じる=過剰最適化リスク大)で単調増加なら危険信号。
"""

from __future__ import annotations

import numpy as np

import mm_lab as mm
from mm_composite import make_composite, _row


def main():
    pool = mm.build_pool()
    closes = mm.load_closes()

    print("=== shrink 近傍(max_pos=8, p=2.0, z0=2.2)— 過剰最適化方向の感度 ===")
    for shrink in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        mk = make_composite(pool, max_pos=8, use_kelly=True, use_z=True, shrink=shrink)
        r = mm.evaluate_method(f"shrink{shrink}", pool, closes, mk,
                               target_dd=0.20, max_pos=8, n_boot=400)
        print(_row(f"shrink={shrink}", r))
    print()

    print("=== p 近傍(max_pos=8, z0=2.2, shrink=0.5)===")
    for p in [1.0, 1.5, 2.0, 2.5, 3.0]:
        mk = make_composite(pool, max_pos=8, use_kelly=True, use_z=True, p=p, shrink=0.5)
        r = mm.evaluate_method(f"p{p}", pool, closes, mk,
                               target_dd=0.20, max_pos=8, n_boot=400)
        print(_row(f"p={p}", r))
    print()

    print("=== max_pos 近傍(p=2.0, z0=2.2, shrink=0.5)===")
    for mp in [6, 7, 8, 9, 10]:
        mk = make_composite(pool, max_pos=mp, use_kelly=True, use_z=True, shrink=0.5)
        r = mm.evaluate_method(f"mp{mp}", pool, closes, mk,
                               target_dd=0.20, max_pos=mp, n_boot=400)
        print(_row(f"max_pos={mp}", r))
    print()

    # 保守的な「高原中央」構成 vs 攻めた「エッジ」構成 を n_boot=1500 で確定比較
    print("=== 確定比較 (n_boot=1500) ===")
    configs = {
        "central(mp8,p2,z2.2,shr0.5)": dict(max_pos=8, p=2.0, z0=2.2, shrink=0.5),
        "edge(mp8,p2,z2.2,shr0.7)":    dict(max_pos=8, p=2.0, z0=2.2, shrink=0.7),
    }
    for name, cfg in configs.items():
        mp = cfg.pop("max_pos")
        mk = make_composite(pool, max_pos=mp, use_kelly=True, use_z=True, **cfg)
        r = mm.evaluate_method(name, pool, closes, mk, target_dd=0.20, max_pos=mp, n_boot=1500)
        print(_row(name, r))
        print(f"   detail: total_ret={r['total_return']:+.1%} sortino={r['sortino']:.2f} "
              f"worst_yr={r['worst_year']:+.1%} boot_worst={r['boot_worst']:+.1%} "
              f"OOS_sharpe={r.get('oos_sharpe', float('nan')):.2f} k_is={r.get('k_is', float('nan')):.2f}")


if __name__ == "__main__":
    main()
