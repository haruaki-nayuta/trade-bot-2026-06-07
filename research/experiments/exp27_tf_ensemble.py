"""exp27: 時間足アンサンブル H4+D1 — チャンピオン族を「時間軸方向」に分散して DD 予算を稼ぐ。

exp25/D1チェックの発見: 同一エッジ構造を D1 スケールで回すと **単体で黒字**
(プールPF1.55, IS1.46/OOS1.66, 26trades/年)かつ **H4 ストリームと月次相関 ≈ 0**。
レポート10で失敗した補完オーバーレイは「単体赤字の保険」だったが、これは「単体黒字の同族」
= 統合すれば同じ DD=20% でレバ k を上げられる可能性がある(分散の無料ランチの正当版)。

プロトコル: ens_lab.simulate_streams(ストリーム別建玉枠・z-powerサイジング)を H4 グリッドで回し、
empirical MtM DD=20% に較正して CAGR を比較。基準 = H4単独(mp8: +21.6% / mp11: +23.8%)。
IS較正→OOS素検証・ブートストラップ理論DDつき。

実行: PYTHONPATH=. uv run python research/experiments/exp27_tf_ensemble.py
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


def build_streams(w_d1: float) -> pd.DataFrame:
    pool_h4 = mm.build_pool().copy()
    pool_h4["stream"] = "h4"
    pool_h4["w"] = 1.0

    pool_d1 = mm.build_pool_for(v2, dict(v2.PARAMS), tf="D1", tag="v2_d1scale").copy()
    # D1 ラベル d 00:00 の約定は d+1 00:00 = H4 グリッドでは d 20:00 バーの終値
    pool_d1["entry"] = pool_d1["entry"] + pd.Timedelta(hours=20)
    pool_d1["exit"] = pool_d1["exit"] + pd.Timedelta(hours=20)
    pool_d1["stream"] = "d1"
    pool_d1["w"] = w_d1

    cols = ["instr", "entry", "exit", "dir", "entry_price", "ret", "z_entry", "stream", "w"]
    both = pd.concat([pool_h4[cols], pool_d1[cols]], ignore_index=True)
    return both.sort_values("entry").reset_index(drop=True)


def evaluate(pool, closes, budgets, label, target_dd=0.20, n_boot=800):
    fbars = ens.stream_fbars(pool)
    k, eqm, eqr, info = ens.calibrate_streams(pool, closes, budgets, fbars=fbars,
                                              target_dd=target_dd)
    s = mm.stats(eqm, eqr, info)
    bs = mm.bootstrap_maxdd(eqm, n_boot=n_boot)
    # IS較正→OOS素検証
    is_pool = pool[pool["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]
    oos_cl = closes[closes.index >= OOS_START]
    k_is, *_ = ens.calibrate_streams(is_pool, is_cl, budgets, fbars=fbars, target_dd=target_dd)
    eqo, ero, io = ens.simulate_streams(oos_pool, oos_cl, k_is, budgets, fbars=fbars)
    so = mm.stats(eqo, ero, io)
    print(f"  {label:34s} k={k:5.2f} CAGR={s['cagr']:+7.2%} DD={s['maxdd_mtm']:+6.1%} "
          f"Sh={s['sharpe']:4.2f} boot95={bs['p95']:+6.1%} +年={s['pos_year_rate']:3.0%} "
          f"worst={s['worst_year']:+5.1%} skip={s['skipped']:3d} | "
          f"OOS k_is={k_is:.2f} CAGR={so['cagr']:+7.2%} DD={so['maxdd_mtm']:+6.1%} +年={so['pos_year_rate']:3.0%}")
    return {"label": label, "k": k, "cagr": s["cagr"], "boot95": bs["p95"],
            "oos_cagr": so["cagr"], "oos_dd": so["maxdd_mtm"], "worst": s["worst_year"]}


def main() -> int:
    closes = mm.load_closes()

    print("=== 基準: H4 単独(プロトコル整合確認: mp8 ≈ +21.6% / mp11 ≈ +23.8%) ===")
    for mp in [8, 11]:
        p = build_streams(1.0)
        p = p[p["stream"] == "h4"].reset_index(drop=True)
        evaluate(p, closes, {"h4": mp}, f"H4 only mp{mp}")

    print("\n=== D1 単独 ===")
    pd1 = build_streams(1.0)
    pd1 = pd1[pd1["stream"] == "d1"].reset_index(drop=True)
    for mp in [3, 4, 6]:
        evaluate(pd1, closes, {"d1": mp}, f"D1 only mp{mp}")

    print("\n=== アンサンブル H4+D1 ===")
    results = []
    for w_d1 in [0.5, 1.0, 1.5]:
        pool = build_streams(w_d1)
        for b_h4, b_d1 in [(8, 2), (8, 4), (11, 3), (11, 4), (11, 6)]:
            r = evaluate(pool, closes, {"h4": b_h4, "d1": b_d1},
                         f"h4:{b_h4} d1:{b_d1} w_d1={w_d1}")
            results.append(r)

    best = max(results, key=lambda r: r["cagr"])
    print(f"\nベスト: {best['label']}  CAGR={best['cagr']:+.2%} (OOS {best['oos_cagr']:+.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
