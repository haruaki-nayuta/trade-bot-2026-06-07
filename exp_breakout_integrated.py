"""最終判定: 最良 breakout_trend overlay を champion に統合し DD=20% 較正 → CAGR が +21.6% を超えるか。

bleed sweep の勝者: breakout_trend H4 short e80/x20/t200 (mean_in_bleed=140.9, IS+178/OOS+100, 持続)。
比較基準: champion 単独 CAGR +21.6% / Sharpe 1.21 / 100%プラス年 / 理論DD p95 -28.5%。
overlay_weight を振って「保険のドラッグを払ってなお純増」する重みがあるかを実測。

実行: uv run python exp_breakout_integrated.py
"""

from __future__ import annotations

import importlib

import pandas as pd

import bleed_lab as bl
import mm_lab as mm

pd.set_option("display.width", 200)


def main():
    bt = importlib.import_module("strategies.breakout_trend")

    # 上位候補(持続ヘッジ・短期側に強い)
    candidates = [
        ("breakout_trend", {"entry": 80, "exit": 20, "trend": 200}, "short", "bt_e80x20t200_S"),
        ("breakout_trend", {"entry": 55, "exit": 20, "trend": 100}, "short", "bt_e55x20t100_S"),
        ("breakout_trend", {"entry": 80, "exit": 20, "trend": 200}, "both", "bt_e80x20t200_B"),
    ]

    # champion 単独基準(同インフラで再算出)
    pool_c = mm.build_pool()
    closes = mm.load_closes()
    from mm_production import champion_sizing
    mk = champion_sizing(pool_c, max_pos=8)
    k0, eqm0, eqr0, info0 = mm.calibrate(pool_c, closes, mk, target_dd=0.20, max_pos=8)
    s0 = mm.stats(eqm0, eqr0, info0)
    bs0 = mm.bootstrap_maxdd(eqm0, n_boot=800)
    print(f"=== champion 単独基準 ===")
    print(f"  k={k0:.3f} CAGR={s0['cagr']:+.1%} Sharpe={s0['sharpe']:.2f} "
          f"maxDD={s0['maxdd_mtm']:.1%} p95={bs0['p95']:.1%} プラス年={s0['pos_year_rate']:.0%}\n")

    for strat, params, side, tag in candidates:
        mod = importlib.import_module(f"strategies.{strat}")
        overlay = mm.build_pool_for(mod, params, side=side, tag=tag)
        print(f"=== overlay: {tag} ({strat} {params} {side}) / {len(overlay)} trades ===")
        for w in [0.25, 0.5, 1.0, 1.5, 2.0]:
            r = bl.integrated_dd_test(overlay, overlay_weight=w, max_pos=8)
            flag = "  <== beats champ" if r["cagr"] > s0["cagr"] else ""
            print(f"  w={w:>4.2f}  k={r['k']:.3f}  CAGR={r['cagr']:+.1%}  "
                  f"Sharpe={r['sharpe']:.2f}  maxDD={r['maxdd_mtm']:.1%}  "
                  f"p95={r['boot_p95']:.1%}  プラス年={r['pos_year_rate']:.0%}  "
                  f"worstYr={r['worst_year']:+.1%}{flag}")
        print()


if __name__ == "__main__":
    main()


def fine():
    """最良 overlay の低weight細粒度: champion +21.6% を超える点が本当に無いか確定。"""
    import importlib
    bt = importlib.import_module("strategies.breakout_trend")
    overlay = mm.build_pool_for(bt, {"entry": 80, "exit": 20, "trend": 200}, side="short",
                                tag="bt_e80x20t200_S")
    print("=== 最良 overlay 低weight細粒度 (champ基準 +21.6%) ===")
    for w in [0.05, 0.10, 0.15, 0.20]:
        r = bl.integrated_dd_test(overlay, overlay_weight=w, max_pos=8)
        print(f"  w={w:>4.2f}  k={r['k']:.3f}  CAGR={r['cagr']:+.1%}  Sharpe={r['sharpe']:.2f}  "
              f"maxDD={r['maxdd_mtm']:.1%}  p95={r['boot_p95']:.1%}  プラス年={r['pos_year_rate']:.0%}")
