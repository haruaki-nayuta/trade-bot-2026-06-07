"""exp69: robust較正(p95 DD=20%)の資産曲線の「体感」統計 — 負に振動していないか?(ユーザー問)

「p95=20%」はテール較正(ブートストラップで20回に1回級の別歴史が踏む最大DD)であって、
実現パスの常時挙動ではない。本実験は本番構成(d1+P4.0+mp8, k=rob 5シード平均)の
MtM 資産曲線について以下を実測する:
  1. 実現DD: 最大・エピソード数(-5%/-10%超え)・水面下時間シェア・最長水面下期間
  2. 初期資金割れ: 実現パスで初期資金を下回った期間 / ブートストラップでの
     P(初期割れ) を運用開始後 6m/1y/2y 地平で推定(=「最悪のタイミングで始めた人」のリスク)
  3. 月次・年次の振動: マイナス月の頻度・最悪月・年次(全year既知プラス)
  4. 回復算術: -20% テール被弾時の回復所要(CAGR18.6%前提)

実行: PYTHONPATH=. uv run python research/experiments/exp69_path_experience.py
出力: research/outputs/exp69_result.json
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

import mm_lab as mm  # noqa: E402
from mm_production import build_pool_d1, champion_sizing  # noqa: E402
from tail_protocol import boot_dd, cagr_of  # noqa: E402
from fxlab import universe as uni  # noqa: E402

OUT_DIR = ROOT / "research" / "outputs"
K_ROB_M5 = 6.205   # exp56/57 base_d1 の rob seeds0-4 平均 k(検証済み)
BARS_1Y = 6 * 252


def episodes_below(dd: np.ndarray, th: float) -> int:
    below = dd <= th
    return int(np.sum(below[1:] & ~below[:-1]) + (1 if below[0] else 0))


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1()
    closes = mm.load_closes()
    mk = champion_sizing(pool, max_pos=8)
    eqm, eqr, info = mm.simulate(pool, closes, mk(K_ROB_M5), max_pos=8)
    eq = eqm / eqm.iloc[0]
    dd = (eq / eq.cummax() - 1.0).to_numpy()
    idx = eq.index
    print(f"=== exp69: 本番較正(k={K_ROB_M5}) パス体感統計 ===")
    print(f"CAGR {cagr_of(eqm):+.2%}  最大DD(実現) {dd.min():+.1%}")

    # 1. 水面下時間とエピソード
    res = {"k": K_ROB_M5, "cagr": cagr_of(eqm), "max_dd_realized": float(dd.min())}
    for th in (-0.01, -0.05, -0.10):
        res[f"time_below_{int(-th*100)}pct"] = float(np.mean(dd <= th))
        res[f"episodes_below_{int(-th*100)}pct"] = episodes_below(dd, th)
    # 最長水面下(ピーク→新高値)
    peak_run, longest, cur_start = eq.iloc[0], 0, idx[0]
    longest_span = (idx[0], idx[0])
    cm = eq.cummax()
    is_uw = (eq < cm).to_numpy()
    start = None
    for i in range(len(eq)):
        if is_uw[i] and start is None:
            start = i
        elif not is_uw[i] and start is not None:
            d = (idx[i] - idx[start]).days
            if d > longest:
                longest, longest_span = d, (idx[start], idx[i])
            start = None
    if start is not None:
        d = (idx[-1] - idx[start]).days
        if d > longest:
            longest, longest_span = d, (idx[start], idx[-1])
    res["longest_underwater_days"] = longest
    res["longest_underwater_span"] = [str(longest_span[0].date()), str(longest_span[1].date())]

    # 2. 初期資金割れ(実現パス)
    below_init = eq < 1.0
    res["time_below_initial"] = float(below_init.mean())
    res["min_vs_initial"] = float(eq.min() - 1.0)
    res["min_vs_initial_date"] = str(eq.idxmin().date())
    res["last_below_initial"] = str(idx[np.where(below_init)[0][-1]].date()) if below_init.any() else None

    # 3. 月次
    mret = eq.resample("ME").last().pct_change().dropna()
    res["neg_month_share"] = float((mret < 0).mean())
    res["worst_month"] = float(mret.min())
    res["best_month"] = float(mret.max())

    # 4. ブートストラップ地平リスク(block=63)
    r = eq.pct_change().dropna().to_numpy()
    rng = np.random.default_rng(0)
    block, n_boot = 63, 4000
    out_h = {}
    for label, horizon in (("6m", BARS_1Y // 2), ("1y", BARS_1Y), ("2y", BARS_1Y * 2)):
        n_blocks = int(np.ceil(horizon / block))
        starts = rng.integers(0, len(r) - block, size=(n_boot, n_blocks))
        end_below = mins10 = mins20 = 0
        for i in range(n_boot):
            seq = r[(starts[i][:, None] + np.arange(block)).ravel()[:horizon]]
            path = np.cumprod(1.0 + seq)
            end_below += path[-1] < 1.0
            mins10 += path.min() < 0.90
            mins20 += path.min() < 0.80
        out_h[label] = {"p_end_below_initial": end_below / n_boot,
                        "p_touch_minus10": mins10 / n_boot,
                        "p_touch_minus20": mins20 / n_boot}
        print(f"  地平{label}: P(終了時に初期割れ)={end_below/n_boot:.1%}  "
              f"P(途中で-10%タッチ)={mins10/n_boot:.1%}  P(-20%タッチ)={mins20/n_boot:.1%}")
    res["bootstrap_horizons"] = out_h
    bs = boot_dd(eqm, n_boot=2000, seed=0)
    res["boot_p95"] = bs["p95"]
    res["boot_p50"] = bs["p50"]

    print(f"\n水面下時間: >1% {res['time_below_1pct']:.0%} / >5% {res['time_below_5pct']:.0%} / "
          f">10% {res['time_below_10pct']:.0%}")
    print(f"DDエピソード: -5%超 {res['episodes_below_5pct']}回 / -10%超 {res['episodes_below_10pct']}回 (11年)")
    print(f"最長水面下: {longest}日 ({res['longest_underwater_span'][0]} → {res['longest_underwater_span'][1]})")
    print(f"初期資金割れ: 時間シェア {res['time_below_initial']:.1%} / 最小 {res['min_vs_initial']:+.1%} "
          f"({res['min_vs_initial_date']}) / 最後に割った日 {res['last_below_initial']}")
    print(f"月次: マイナス月 {res['neg_month_share']:.0%} / 最悪月 {res['worst_month']:+.1%} / 最良月 {res['best_month']:+.1%}")
    print(f"ブートDD: p50 {bs['p50']:+.1%} / p95 {bs['p95']:+.1%}")

    (OUT_DIR / "exp69_result.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\nsaved -> {OUT_DIR / 'exp69_result.json'}\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
