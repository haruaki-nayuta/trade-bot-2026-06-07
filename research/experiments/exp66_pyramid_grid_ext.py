"""exp66: ピラミッド用量グリッドの下方延長(θ=0.1%/0.2%)のアーカイブ — 敵対検証の手続き指摘対応。

exp65 の用量グリッドは θ∈{0.3,0.5,0.75,1.0}% で、IS-argmax がグリッド端(0.3%)に落ちた。
reports/19 プロトコル#2(argmax が端なら測定として延長)に従い 0.1/0.2% を測定したが、
当初インライン実行で保存していなかった(敵対検証 wf_ff66ef67 の指摘)。本ファイルが正式アーカイブ。

実行: PYTHONPATH=. uv run python research/experiments/exp66_pyramid_grid_ext.py
出力: research/outputs/exp66_result.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

import mm_lab as mm  # noqa: E402
from mm_production import build_pool_d1  # noqa: E402
from exp47_entry_delay import reconstruct  # noqa: E402
from exp65_pyramid_protocol import SEEDS, build_addons, full_eval  # noqa: E402
from fxlab import universe as uni  # noqa: E402

OUT_DIR = ROOT / "research" / "outputs"
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1().copy().reset_index(drop=True)
    closes = mm.load_closes()
    rc = reconstruct(pool)
    print("=== exp66: ピラミッド用量グリッド下方延長 θ∈{0.1%, 0.2%} ===")

    base = full_eval("base", pool, closes)
    out = {"base": {"rob_mean": base["rob_cagr_mean"],
                    "rob": {str(s): base["rob"][s]["cagr"] for s in SEEDS},
                    "emp_k": base["emp_k"], "emp_cagr": base["emp_cagr"],
                    "emp_p95": base["emp_p95"], "is_rob": base["is_rob_cagr"],
                    "oos_rob": base["oos_rob_cagr"], "oos_emp": base["oos_emp_cagr"]}}
    print(f"base rob_mean {base['rob_cagr_mean']:+.2%}  [{time.time()-t0:.0f}s]")

    for th in (0.001, 0.002):
        ad = build_addons(pool, rc, th)
        r_ = ad["ret"]
        is_m = ad["entry"] < OOS_START
        yr = ad.groupby(ad["exit"].dt.year)["ret"].sum()
        by = int(yr.idxmax())
        aug = pd.concat([pool, ad], ignore_index=True).sort_values("entry").reset_index(drop=True)
        r = full_eval(f"pyr{th:.3%}", aug, closes)
        per = {s: r["rob"][s]["cagr"] - base["rob"][s]["cagr"] for s in SEEDS}
        g3 = (r["oos_rob_cagr"] > base["oos_rob_cagr"]) and \
             (r["oos_emp_cagr"] > base["oos_emp_cagr"])
        out[f"{th}"] = {
            "pool": {"n": len(ad), "sum": float(r_.sum()), "mean_bps": float(r_.mean() * 1e4),
                     "win": float((r_ > 0).mean()), "is": float(r_[is_m].sum()),
                     "oos": float(r_[~is_m].sum()), "best_year": by,
                     "keep_excl_best": float(yr.drop(by).sum() / r_.sum())},
            "rob_mean": r["rob_cagr_mean"], "gain_pp": (r["rob_cagr_mean"] - base["rob_cagr_mean"]) * 100,
            "rob": {str(s): r["rob"][s]["cagr"] for s in SEEDS},
            "per_seed_pp": {str(s): v * 100 for s, v in per.items()},
            "all_seeds_pos": bool(all(v > 0 for v in per.values())),
            "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"], "emp_p95": r["emp_p95"],
            "is_rob": r["is_rob_cagr"], "oos_rob": r["oos_rob_cagr"],
            "oos_emp": r["oos_emp_cagr"], "g3_raw_both": bool(g3),
        }
        print(f"θ={th:.3%}: n={len(ad)} rob_mean {r['rob_cagr_mean']:+.2%} "
              f"(gain {out[str(th)]['gain_pp']:+.2f}pp) emp_k {r['emp_k']:.2f} "
              f"IS_rob {r['is_rob_cagr']:+.2%} OOS_rob {r['oos_rob_cagr']:+.2%} "
              f"OOS_emp {r['oos_emp_cagr']:+.2%} G3raw={'+' if g3 else 'x'}  [{time.time()-t0:.0f}s]")

    # 拡張グリッド全体の IS-argmax(exp65 の保存値と合成)
    e65 = json.loads((OUT_DIR / "exp65_result.json").read_text())
    is_rob_all = {"base": base["is_rob_cagr"],
                  "0.001": out["0.001"]["is_rob"], "0.002": out["0.002"]["is_rob"],
                  **{k: v["is_rob_cagr"] for k, v in e65["results"].items() if k != "base"}}
    argmax = max(is_rob_all, key=is_rob_all.get)
    out["is_argmax_extended"] = {"values": is_rob_all, "argmax": argmax}
    print(f"\n拡張グリッド IS-argmax(rob): {argmax}  "
          f"({'端=下端。延長してもISは過剰用量側を選び続ける(罠用量はフル期間で負ける)' if argmax=='0.001' else '内部点'})")

    (OUT_DIR / "exp66_result.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"saved -> {OUT_DIR / 'exp66_result.json'}\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
