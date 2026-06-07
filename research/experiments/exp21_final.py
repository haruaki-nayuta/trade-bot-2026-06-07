"""exp21 最終判定: ベスト持続ヘッジ構成(tsmom H4 lb=200 band=0.005 side=long)を
champion+overlay で統合し DD=20% 較正 → CAGR が champion 単独(+21.6%)を上回るか。
"""
from __future__ import annotations
import importlib, warnings
import bleed_lab as bl
import mm_lab as mm

warnings.simplefilter("ignore")

mod = importlib.import_module("strategies.tsmom")
params = {"lookback": 200, "band": 0.005}
side = "long"; tf = "H4"
ovl = mm.build_pool_for(mod, params, tf=tf, side=side, tag="tsmom_lb200_b0005_long")
print(f"overlay pool: {len(ovl)} trades  (tsmom {tf} lb=200 band=0.005 {side})")

for w in [0.25, 0.5, 1.0, 1.5]:
    r = bl.integrated_dd_test(ovl, overlay_weight=w)
    print(f"weight={w:>4}: CAGR={r['cagr']:+.1%}  maxdd={r['maxdd_mtm']:+.1%}  "
          f"sharpe={r['sharpe']:.2f}  boot_p95={r['boot_p95']:+.1%}  "
          f"pos_year={r['pos_year_rate']:.0%}  worst_year={r['worst_year']:+.1%}")
print("比較: champion単独 CAGR +21.6% / Sharpe 1.21 / 100%プラス年 / boot_p95 -28.5%")
