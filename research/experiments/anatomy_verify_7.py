"""anatomy_verify_7: 敵対検証 — エッジ源泉の分解主張を独立再計算する.

主張 (mechanism):
  - 完全プラセボ (時刻乱択 + 方向50/50): mean +0.1bps, sd 2.7bps, max 8.8bps
  - 方向のみ実際・時刻乱択 (B): mean -0.5bps, max 6.5bps
  - 実測 gross +17.9bps → タイミング寄与 = 実際 - B = +18.4bps = グロスの102.8%
  - 方向寄与 B - A = -0.6bps (sd内 = ゼロ)

独立再計算: プールparquetと価格系列だけから自前で再構築する。
実行: uv run python -m research.experiments.anatomy_verify_7
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import load
from fxlab import universe as uni

POOL = "results/mm_pool_v2_H4_19.parquet"
N_RESAMPLE = 300
SEED = 42
MAJORS = {"EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"}


def close_series(instr: str) -> pd.Series:
    if instr in MAJORS:
        return load(instr, "H4")["close"]
    return uni.instrument_close(instr, "H4")


def main() -> None:
    uni.register_cross_spreads(3.0)
    pool = pd.read_parquet(POOL)

    # --- ベースライン検算 -------------------------------------------------
    n = len(pool)
    print(f"baseline: n={n} sum(ret)={pool['ret'].sum():+.4f} "
          f"mean={pool['ret'].mean()*1e4:+.2f}bps win={(pool['ret'] > 0).mean():.3f}")

    # --- 銘柄ごとの close 配列とトレード位置 -------------------------------
    rng = np.random.default_rng(SEED)
    actual_gross = np.full(n, np.nan)

    # per-instrument 構造体
    instr_info = {}
    for instr, g in pool.groupby("instr"):
        c = close_series(instr)
        idx = c.index
        ent_pos = idx.get_indexer(pd.DatetimeIndex(g["entry"]))
        ext_pos = idx.get_indexer(pd.DatetimeIndex(g["exit"]))
        if (ent_pos < 0).any() or (ext_pos < 0).any():
            raise RuntimeError(f"{instr}: timestamp not aligned to H4 index")
        carr = c.to_numpy(float)
        d = g["dir"].to_numpy(float)
        gross = d * (carr[ext_pos] / carr[ent_pos] - 1.0)
        actual_gross[g.index.to_numpy()] = gross
        instr_info[instr] = dict(
            close=carr,
            lo=int(ent_pos.min()), hi=int(ent_pos.max()),
            dirs=d,
            held=g["bars_held"].to_numpy(int),
            pos_diff=(ext_pos - ent_pos),
        )

    assert not np.isnan(actual_gross).any()
    mean_actual = actual_gross.mean() * 1e4
    # bars_held とインデックス位置差の整合チェック (保有の再現に使う分布の妥当性)
    diff_match = np.concatenate(
        [v["pos_diff"] - v["held"] for v in instr_info.values()])
    print(f"actual gross mean = {mean_actual:+.2f}bps  "
          f"(net mean {pool['ret'].mean()*1e4:+.2f}bps, "
          f"implied cost {mean_actual - pool['ret'].mean()*1e4:.2f}bps)")
    print(f"bars_held vs index-pos-diff: exact match ratio = "
          f"{(diff_match == 0).mean():.3f} (median diff {np.median(diff_match):.1f})")

    # --- プラセボ A: 時刻乱択 + 方向50/50 + 保有ブートストラップ ------------
    # --- プラセボ B: dir/bars_held は実トレードのまま、時刻のみ乱択 ----------
    sums_a = np.zeros(N_RESAMPLE)
    sums_b = np.zeros(N_RESAMPLE)
    for instr, v in instr_info.items():
        carr, lo, hi = v["close"], v["lo"], v["hi"]
        ni = len(v["dirs"])
        last = len(carr) - 1

        # A
        ent = rng.integers(lo, hi + 1, size=(N_RESAMPLE, ni))
        held = rng.choice(v["held"], size=(N_RESAMPLE, ni), replace=True)
        ext = np.minimum(ent + held, last)
        dirs = rng.choice([-1.0, 1.0], size=(N_RESAMPLE, ni))
        sums_a += (dirs * (carr[ext] / carr[ent] - 1.0)).sum(axis=1)

        # B
        ent = rng.integers(lo, hi + 1, size=(N_RESAMPLE, ni))
        ext = np.minimum(ent + v["held"][None, :], last)
        sums_b += (v["dirs"][None, :] * (carr[ext] / carr[ent] - 1.0)).sum(axis=1)

    means_a = sums_a / n * 1e4
    means_b = sums_b / n * 1e4

    def stats(x, name):
        print(f"{name}: mean={x.mean():+.2f}bps sd={x.std(ddof=1):.2f}bps "
              f"min={x.min():+.2f} max={x.max():+.2f}")

    stats(means_a, "placebo A (time random + dir 50/50)")
    stats(means_b, "placebo B (dir/held real, time random)")

    pct_a = (means_a < mean_actual).mean() * 100
    z_a = (mean_actual - means_a.mean()) / means_a.std(ddof=1)
    dir_contrib = means_b.mean() - means_a.mean()
    timing_contrib = mean_actual - means_b.mean()
    print(f"actual vs A: percentile={pct_a:.1f}  z={z_a:.1f}sigma")
    print(f"direction contribution (B-A) = {dir_contrib:+.2f}bps "
          f"(B sd {means_b.std(ddof=1):.2f}bps)")
    print(f"timing contribution (actual-B) = {timing_contrib:+.2f}bps "
          f"= {timing_contrib / mean_actual * 100:.1f}% of gross")


if __name__ == "__main__":
    main()
