"""exp28: 分割エントリー(scale-in)— 鉄の三角形の未踏領域③。

深押しストリーム(entry_z=2.5, 他は完全固定)は単体でプールPF1.91(チャンピオン1.71超)・
OOS 2.15・最悪年≈0。その64%はチャンピオン保有中に発生=「z=2.0で1玉目、z=2.5で2玉目」の
事前計画2トランシェ建玉と等価(エントリー条件・コストは正確に計上済み)。

注意: これは固定予算・回数上限つきの計画的トランシェであり、損失追撃で建玉を増やす
マーチンゲール/ナンピン(禁止事項)ではない。総リスクは DD=20% 較正と建玉枠で拘束される。

検証: ens_lab の2ストリーム統合で CAGR@DD20 を実測。
  基準: チャンピオン単独 mp8 +21.6% / mp11 +23.8%
  比較: champ+deep の枠/ウェイト掃引、deep 単独、champ_mp11 と同一総枠での置換構成。

実行: PYTHONPATH=. uv run python research/experiments/exp28_scalein.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))

import mm_lab as mm  # noqa: E402
import ens_lab as ens  # noqa: E402
import strategies.confluence_meanrev_v2 as v2  # noqa: E402

pd.set_option("display.width", 240)

OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
COLS = ["instr", "entry", "exit", "dir", "entry_price", "ret", "z_entry", "stream", "w"]


def evaluate(pool, closes, budgets, label, target_dd=0.20, n_boot=800):
    fbars = ens.stream_fbars(pool)
    k, eqm, eqr, info = ens.calibrate_streams(pool, closes, budgets, fbars=fbars,
                                              target_dd=target_dd)
    s = mm.stats(eqm, eqr, info)
    bs = mm.bootstrap_maxdd(eqm, n_boot=n_boot)
    is_pool = pool[pool["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]
    oos_cl = closes[closes.index >= OOS_START]
    k_is, *_ = ens.calibrate_streams(is_pool, is_cl, budgets, fbars=fbars, target_dd=target_dd)
    eqo, ero, io = ens.simulate_streams(oos_pool, oos_cl, k_is, budgets, fbars=fbars)
    so = mm.stats(eqo, ero, io)
    print(f"  {label:36s} k={k:5.2f} CAGR={s['cagr']:+7.2%} DD={s['maxdd_mtm']:+6.1%} "
          f"Sh={s['sharpe']:4.2f} boot95={bs['p95']:+6.1%} +年={s['pos_year_rate']:3.0%} "
          f"worst={s['worst_year']:+5.1%} skip={s['skipped']:3d} | "
          f"OOS CAGR={so['cagr']:+7.2%} DD={so['maxdd_mtm']:+6.1%} +年={so['pos_year_rate']:3.0%}")
    return {"label": label, "cagr": s["cagr"], "boot95": bs["p95"], "oos_cagr": so["cagr"]}


def main() -> int:
    closes = mm.load_closes()
    champ = mm.build_pool().copy()
    champ["stream"] = "champ"; champ["w"] = 1.0
    p = dict(v2.PARAMS); p["entry_z"] = 2.5
    deep = mm.build_pool_for(v2, p, tf="H4", tag="v2_z250").copy()
    deep["stream"] = "deep"; deep["w"] = 1.0

    print("=== 基準 ===")
    evaluate(champ[COLS], closes, {"champ": 8}, "champ only mp8")
    evaluate(champ[COLS], closes, {"champ": 11}, "champ only mp11")

    print("\n=== deep 単独(高品質プールの DD 効率を直接測る) ===")
    for mp in [4, 6, 8, 11]:
        evaluate(deep[COLS], closes, {"deep": mp}, f"deep only mp{mp}")

    print("\n=== champ + deep 統合 ===")
    results = []
    for w_deep in [0.5, 1.0]:
        d = deep.copy(); d["w"] = w_deep
        pool = pd.concat([champ[COLS], d[COLS]], ignore_index=True).sort_values("entry").reset_index(drop=True)
        for bc, bd in [(8, 3), (8, 4), (11, 3), (11, 4), (9, 5)]:
            r = evaluate(pool, closes, {"champ": bc, "deep": bd},
                         f"champ:{bc} deep:{bd} w_deep={w_deep}")
            results.append(r)

    best = max(results, key=lambda r: r["cagr"])
    print(f"\nベスト: {best['label']}  CAGR={best['cagr']:+.2%} (OOS {best['oos_cagr']:+.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
