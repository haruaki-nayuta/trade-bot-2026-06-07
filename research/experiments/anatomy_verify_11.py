"""敵対検証: 「損切り・時間ストップ全滅の物理」主張の独立再計算。

主張(出典: micro / anatomy):
  1. 建玉中の残り期待値 E[final - current] はどの時点 k でも +15〜+20bps、
     含み損側(path[k]<0)に限れば +15〜+24bps。
     深い含み損×長期保有セル(k=24本 & 走行≤-2%)では +78.6bps、回復率(残り>0)78.9%。
  2. E[回復 | MAE -x 初到達点から]:
     x=0.5%: +27.1bps(回復>0率71.7%) / 1%: +35.9(72.5%) / 2%: +53.6(72.9%) / 3%: +107.5(75.7%)
  3. 反実仮想(close基準の楽観近似):
     SL-0.5% → 総益 +1.909→+0.549(-71.2%) / SL-1% → +0.927(-51.4%) / SL-3% → +1.511(-20.8%)
     時間ストップ 12本 → +0.720(-62.3%) / 24本 → +1.268(-33.5%) / 72本 → -0.2%

方法(主張の記載に合わせ、コードは独立に新規実装):
  - path[k] = dir * (close[entry_bar + k] / entry_price - 1)。k=0 はエントリーバー終値。
  - E[残り] は bars_held > k のトレードで path[-1] - path[k]。
  - SL 反実仮想 = しきい値初到達バーの close で決済(楽観近似)。
    実出口コスト(= path[-1] - ret ≈ 半スプレッド)を引いた調整版も併記。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import universe as uni

POOL = "results/mm_pool_v2_H4_19.parquet"
BPS = 1e4


def build_paths(pool: pd.DataFrame) -> list[np.ndarray]:
    closes = {instr: uni.instrument_close(instr, "H4") for instr in pool["instr"].unique()}
    paths: list[np.ndarray] = []
    bad_align = 0
    for row in pool.itertuples():
        s = closes[row.instr]
        i0 = s.index.get_loc(row.entry)
        i1 = s.index.get_loc(row.exit)
        if (i1 - i0) != row.bars_held:
            bad_align += 1
        arr = s.to_numpy()[i0 : i1 + 1]
        paths.append(row.dir * (arr / row.entry_price - 1.0))
    print(f"[align] bars mismatch: {bad_align}/{len(pool)}")
    return paths


def main() -> None:
    uni.register_cross_spreads(3.0)
    pool = pd.read_parquet(POOL)

    # ---- 0. ベースライン検算 ----
    total = pool["ret"].sum()
    pf = pool.loc[pool.ret > 0, "ret"].sum() / -pool.loc[pool.ret < 0, "ret"].sum()
    print(
        f"[baseline] n={len(pool)} sum={total:+.4f} mean={pool.ret.mean()*BPS:+.1f}bps "
        f"wr={(pool.ret>0).mean():.3f} PF={pf:.3f} med_hold={pool.bars_held.median():.0f}"
    )

    paths = build_paths(pool)
    rets = pool["ret"].to_numpy()
    finals = np.array([p[-1] for p in paths])
    exit_cost = finals - rets  # 実出口コスト(半スプレッド相当)
    print(
        f"[sanity] path[-1] vs ret: mean diff={exit_cost.mean()*BPS:+.2f}bps "
        f"median={np.median(exit_cost)*BPS:+.2f}bps max|.|={np.abs(exit_cost).max()*BPS:.2f}bps"
    )

    # ---- 1. E[final - current] @ k ----
    print("\n== E[path[-1]-path[k]] (bps) : 全体 / 含み損側 ==")
    print(f"{'k':>4} {'n_all':>6} {'E_all':>8} {'n_loss':>6} {'E_loss':>8}")
    ks = [1, 2, 4, 6, 8, 12, 16, 20, 24, 30, 36, 48, 60, 72, 96]
    e_all_rng, e_loss_rng = [], []
    for k in ks:
        rem, rem_loss = [], []
        for p in paths:
            if len(p) - 1 > k:  # bars_held > k
                r = p[-1] - p[k]
                rem.append(r)
                if p[k] < 0:
                    rem_loss.append(r)
        if len(rem) >= 30:
            e_all = np.mean(rem) * BPS
            e_all_rng.append(e_all)
        else:
            e_all = float("nan")
        if len(rem_loss) >= 30:
            e_loss = np.mean(rem_loss) * BPS
            e_loss_rng.append(e_loss)
        else:
            e_loss = float("nan")
        print(f"{k:>4} {len(rem):>6} {e_all:>8.1f} {len(rem_loss):>6} {e_loss:>8.1f}")
    print(f"range all (n>=30): [{min(e_all_rng):+.1f}, {max(e_all_rng):+.1f}] bps  (claim +15..+20)")
    print(f"range loss(n>=30): [{min(e_loss_rng):+.1f}, {max(e_loss_rng):+.1f}] bps  (claim +15..+24)")

    # 複合セル: k=24 & path[24] <= -2%
    cell = [p[-1] - p[24] for p in paths if len(p) - 1 > 24 and p[24] <= -0.02]
    cell = np.array(cell)
    print(
        f"\n[cell k=24 & run<=-2%] n={len(cell)} E={cell.mean()*BPS:+.1f}bps "
        f"recov>0率={(cell>0).mean():.3f}  (claim +78.6bps / 78.9%)"
    )

    # ---- 2. E[回復 | MAE -x 初到達] ----
    print("\n== MAE -x 初到達点からの回復 (close基準) ==")
    print(f"{'x%':>5} {'n':>5} {'E_recov_bps':>12} {'recov>0率':>9}   claim")
    claims = {0.5: (27.1, 0.717), 1.0: (35.9, 0.725), 2.0: (53.6, 0.729), 3.0: (107.5, 0.757)}
    for x in [0.5, 1.0, 2.0, 3.0]:
        rec = []
        thr = -x / 100
        for p in paths:
            idx = np.argmax(p <= thr) if (p <= thr).any() else -1
            if idx >= 0:
                rec.append(p[-1] - p[idx])
        rec = np.array(rec)
        c = claims[x]
        print(
            f"{x:>5.1f} {len(rec):>5} {rec.mean()*BPS:>12.1f} {(rec>0).mean():>9.3f}"
            f"   (+{c[0]}bps / {c[1]:.1%})"
        )

    # ---- 3. 反実仮想: SL / 時間ストップ (close基準楽観近似) ----
    print("\n== 反実仮想 total sum(ret) : baseline=%.4f ==" % total)
    print(f"{'variant':>12} {'raw':>8} {'raw%':>7} {'adj(出口コスト込)':>10} {'adj%':>7}   claim")
    sl_claims = {0.5: 0.549, 1.0: 0.927, 3.0: 1.511}
    for x in [0.5, 1.0, 3.0]:
        thr = -x / 100
        raw = adj = 0.0
        n_stop = 0
        for p, r, ec in zip(paths, rets, exit_cost):
            hit = (p <= thr).any()
            if hit:
                k = int(np.argmax(p <= thr))
                raw += p[k]
                adj += p[k] - ec
                n_stop += 1
            else:
                raw += r
                adj += r
        print(
            f"{'SL-'+str(x)+'%':>12} {raw:>8.4f} {raw/total-1:>7.1%} {adj:>10.4f} "
            f"{adj/total-1:>7.1%}   ({sl_claims[x]:+.3f}) stopped={n_stop}"
        )
    ts_claims = {12: 0.720, 24: 1.268, 72: None}
    for k in [12, 24, 72]:
        raw = adj = 0.0
        n_stop = 0
        for p, r, ec in zip(paths, rets, exit_cost):
            if len(p) - 1 > k:
                raw += p[k]
                adj += p[k] - ec
                n_stop += 1
            else:
                raw += r
                adj += r
        cl = ts_claims[k]
        cl_s = f"({cl:+.3f})" if cl is not None else "(-0.2%)"
        print(
            f"{'TS-'+str(k)+'本':>12} {raw:>8.4f} {raw/total-1:>7.1%} {adj:>10.4f} "
            f"{adj/total-1:>7.1%}   {cl_s} stopped={n_stop}"
        )


if __name__ == "__main__":
    main()
