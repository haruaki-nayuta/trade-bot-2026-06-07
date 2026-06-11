"""exp33: ベースライン再現 — champion v2 + z-power, mp8/mp11/mp12 を empirical/robust 両較正で固定。

reports/09/10/11 の物差しをこの worktree で再現する:
  empirical 20%: mp8 +21.62% / mp11 +23.83%
  robust(p95=20%, n_boot=600, seed0): mp11 +15.21% / mp12 +15.3%

以後の全候補はこの数値との「ペアシード」比較で判定する。
実行: PYTHONPATH=. uv run python research/experiments/exp33_baseline_repro.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))

import mm_lab as mm  # noqa: E402
from mm_production import champion_sizing  # noqa: E402

OUT = ROOT / "research" / "outputs" / "exp33_baseline.json"


def main() -> int:
    pool = mm.build_pool()
    closes = mm.load_closes()
    print(f"pool {len(pool)} trades / grid {len(closes)} bars")

    res = {}
    for mp in [8, 11, 12]:
        mk = champion_sizing(pool, max_pos=mp)
        k, eqm, eqr, info = mm.calibrate(pool, closes, mk, target_dd=0.20, max_pos=mp)
        s = mm.stats(eqm, eqr, info)
        bs = mm.bootstrap_maxdd(eqm, n_boot=1500)
        res[f"mp{mp}_emp"] = {"k": k, "cagr": s["cagr"], "dd": s["maxdd_mtm"],
                              "p95": bs["p95"], "sharpe": s["sharpe"],
                              "pos_year": s["pos_year_rate"], "worst_year": s["worst_year"]}
        print(f"mp{mp} empirical: k={k:.2f} CAGR={s['cagr']:+.2%} DD={s['maxdd_mtm']:+.1%} "
              f"p95={bs['p95']:+.1%} sharpe={s['sharpe']:.2f}")

    for mp in [11, 12]:
        mk = champion_sizing(pool, max_pos=mp)
        kr, eqm, eqr, info, p95 = mm.calibrate_robust(pool, closes, mk, target_dd=0.20,
                                                      max_pos=mp, n_boot=600)
        s = mm.stats(eqm, eqr, info)
        bs = mm.bootstrap_maxdd(eqm, n_boot=1500)
        res[f"mp{mp}_rob"] = {"k": kr, "cagr": s["cagr"], "dd": s["maxdd_mtm"],
                              "p95_cal": p95, "p95_1500": bs["p95"], "sharpe": s["sharpe"],
                              "pos_year": s["pos_year_rate"], "worst_year": s["worst_year"]}
        print(f"mp{mp} robust:    k={kr:.2f} CAGR={s['cagr']:+.2%} DD={s['maxdd_mtm']:+.1%} "
              f"p95(cal600)={p95:+.1%} p95(1500)={bs['p95']:+.1%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"saved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
