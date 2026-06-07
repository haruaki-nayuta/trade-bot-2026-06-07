"""トップ候補(bb_breakout short 等)を champion に統合し、DD=20%較正でCAGRが改善するか。

比較基準: champion単独 CAGR +21.6% / Sharpe 1.21 / 100%プラス年 / DD p95 -28.5%。
"""

from __future__ import annotations

import warnings

import bleed_lab as bl
import mm_lab as mm
from strategies import bb_breakout, squeeze_breakout

warnings.filterwarnings("ignore")


CANDS = [
    ("bb", bb_breakout, {"period": 60, "mult": 1.5}, "short", "bb_p60m15_short"),
    ("bb", bb_breakout, {"period": 60, "mult": 2.0}, "short", "bb_p60m20_short"),
    ("sq", squeeze_breakout, {"period": 40, "mult": 2.5, "squeeze": 100}, "short", "sq_p40m25s100_short"),
    ("bb", bb_breakout, {"period": 60, "mult": 1.5}, "both", "bb_p60m15_both"),
]


def main():
    print("=== champion 単独 baseline ===")
    pool_c = mm.build_pool()
    closes = mm.load_closes()
    from mm_production import champion_sizing
    mk = champion_sizing(pool_c, max_pos=8)
    k0, eqm0, eqr0, info0 = mm.calibrate(pool_c, closes, mk, target_dd=0.20, max_pos=8)
    s0 = mm.stats(eqm0, eqr0, info0)
    bs0 = mm.bootstrap_maxdd(eqm0, n_boot=800)
    print(f"  CAGR {s0['cagr']:+.1%}  Sharpe {s0['sharpe']:.2f}  maxDD {s0['maxdd_mtm']:.1%} "
          f"posYear {s0['pos_year_rate']:.0%}  boot_p95 {bs0['p95']:.1%}  k={k0:.2f}\n")

    for fam, mod, params, side, tag in CANDS:
        pool_o = mm.build_pool_for(mod, params, tf="H4", side=side, tag=tag)
        print(f"--- overlay {tag} ({side}, {len(pool_o)} trades) ---")
        for w in [0.5, 1.0, 1.5, 2.0]:
            try:
                r = bl.integrated_dd_test(pool_o, overlay_weight=w, max_pos=8)
            except Exception as e:  # noqa: BLE001
                print(f"  w={w}: ERR {e}")
                continue
            flag = "  <== beats baseline" if r["cagr"] > s0["cagr"] else ""
            print(f"  w={w:.1f}: CAGR {r['cagr']:+.1%}  Sharpe {r['sharpe']:.2f}  "
                  f"maxDD {r['maxdd_mtm']:.1%}  posYear {r['pos_year_rate']:.0%}  "
                  f"boot_p95 {r['boot_p95']:.1%}  k={r['k']:.2f}{flag}")
        print()


if __name__ == "__main__":
    main()
