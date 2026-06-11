"""anatomy_mechanism.py — チャンピオン confluence_meanrev_v2 は「何を収穫して」儲けているのか。

トレードプール results/mm_pool_v2_H4_19.parquet を素材に、
  Q1 グロスエッジとコストの食い分・方向反転の純益
  Q2 タイミングプラセボ（ランダムエントリー分布との比較、方向/タイミング寄与の分離）
  Q3 出口の寄与（実z出口 vs 固定k本出口）
  Q4 エントリー後の平均累積グロスリターン経路（1〜60本）
  Q5 ペイオフ形状（保険売り型かどうか）
を全て実測する。実行: uv run python -m research.experiments.anatomy_mechanism
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from fxlab import config, universe as uni

SEED = 42
N_RESAMPLE = 300
FIXED_KS = [6, 12, 18, 24, 36, 48]
PATH_MAX = 60
POOL_PATH = config.RESULTS_DIR / "mm_pool_v2_H4_19.parquet"

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def bps(x: float) -> str:
    return f"{x * 1e4:+.1f}bps"


def main() -> None:
    uni.register_cross_spreads(3.0)
    pool = pd.read_parquet(POOL_PATH)
    n = len(pool)
    total_ret = pool["ret"].sum()
    print(f"=== pool: n={n} sum(ret)={total_ret:+.4f} 平均={bps(pool['ret'].mean())}/trade "
          f"勝率={(pool['ret'] > 0).mean():.3f} ===\n")

    # ---- close 配列とインデックス位置の解決 -------------------------------
    closes: dict[str, np.ndarray] = {}
    idx_of: dict[str, pd.DatetimeIndex] = {}
    for nm in pool["instr"].unique():
        s = uni.instrument_close(nm, "H4")
        closes[nm] = s.to_numpy(float)
        idx_of[nm] = s.index

    e_pos = np.empty(n, dtype=int)
    x_pos = np.empty(n, dtype=int)
    for nm, g in pool.groupby("instr"):
        ii = idx_of[nm]
        ep = ii.get_indexer(pd.DatetimeIndex(g["entry"]))
        xp = ii.get_indexer(pd.DatetimeIndex(g["exit"]))
        assert (ep >= 0).all() and (xp >= 0).all(), f"timestamp mismatch in {nm}"
        e_pos[g.index] = ep
        x_pos[g.index] = xp
    bars_chk = (x_pos - e_pos == pool["bars_held"].to_numpy()).mean()
    print(f"[整合] exit_pos-entry_pos == bars_held 一致率: {bars_chk:.4f}")

    d = pool["dir"].to_numpy(float)
    ret = pool["ret"].to_numpy(float)
    ec = np.array([closes[nm][i] for nm, i in zip(pool["instr"], e_pos)])
    xc = np.array([closes[nm][i] for nm, i in zip(pool["instr"], x_pos)])

    # ---- gross / cost の再構成と検算 --------------------------------------
    gross = d * (xc / ec - 1.0)
    cost = gross - ret
    print(f"[検算] sum(gross)-sum(cost) = {gross.sum() - cost.sum():+.4f}  (== sum(ret) {total_ret:+.4f})")
    print(f"[cost分布] mean={bps(cost.mean())} min={bps(cost.min())} p1={bps(np.percentile(cost, 1))} "
          f"max={bps(cost.max())}  負のcost比率={(cost < -1e-12).mean():.4f}")
    maj = ~pool["instr"].isin(uni.CROSS_DEFS).to_numpy()
    print(f"[cost] メジャー平均={bps(cost[maj].mean())} (n={maj.sum()}) / クロス平均={bps(cost[~maj].mean())} (n={(~maj).sum()})\n")

    # ======================= Q1 グロスエッジ ===============================
    print("=" * 70)
    print("Q1. グロスエッジの大きさ")
    sg, sc = gross.sum(), cost.sum()
    print(f"  sum(gross)={sg:+.4f} ({bps(gross.mean())}/trade) / sum(cost)={sc:+.4f} ({bps(cost.mean())}/trade)")
    print(f"  → コストがグロスを食う割合: {sc / sg * 100:.1f}%  (純益はグロスの {total_ret / sg * 100:.1f}%)")
    rev = -gross - cost  # 方向反転（同コスト）
    print(f"  方向反転の純益: sum={rev.sum():+.4f} ({bps(rev.mean())}/trade) 勝率={(rev > 0).mean():.3f}")
    print(f"  方向の情報量(対称性): 順方向グロス {bps(gross.mean())} vs 反転グロス {bps(-gross.mean())} "
          f"→ 方向スプレッド {bps(2 * gross.mean())}/trade\n")

    # ======================= Q2 タイミングプラセボ ==========================
    print("=" * 70)
    print(f"Q2. タイミングプラセボ（{N_RESAMPLE}リサンプル, seed={SEED}）")
    rng = np.random.default_rng(SEED)

    groups = []
    for nm, g in pool.groupby("instr"):
        c = closes[nm]
        lo = int(e_pos[g.index].min())
        hi = int(e_pos[g.index].max())
        holds_actual = pool.loc[g.index, "bars_held"].to_numpy(int)
        dirs_actual = pool.loc[g.index, "dir"].to_numpy(float)
        groups.append((nm, c, lo, hi, holds_actual, dirs_actual, len(g)))

    placebo_full = np.empty(N_RESAMPLE)   # タイミング乱択 + 方向50/50
    placebo_dir = np.empty(N_RESAMPLE)    # タイミング乱択 + 方向は実トレードと同じ
    for r in range(N_RESAMPLE):
        acc_f, acc_d, cnt = 0.0, 0.0, 0
        for nm, c, lo, hi, holds_act, dirs_act, ni in groups:
            last = len(c) - 1
            # full: 保有本数は実分布からサンプル、方向は50/50
            h_f = rng.choice(holds_act, size=ni, replace=True)
            ent_f = rng.integers(lo, np.minimum(hi, last - h_f) + 1)
            dir_f = rng.choice([-1.0, 1.0], size=ni)
            acc_f += np.sum(dir_f * (c[ent_f + h_f] / c[ent_f] - 1.0))
            # dir-kept: 各トレードの方向・保有本数は実際のまま、エントリー時刻だけ乱択
            ent_d = rng.integers(lo, np.minimum(hi, last - holds_act) + 1)
            acc_d += np.sum(dirs_act * (c[ent_d + holds_act] / c[ent_d] - 1.0))
            cnt += ni
        placebo_full[r] = acc_f / cnt
        placebo_dir[r] = acc_d / cnt

    gm = gross.mean()
    pct_full = (placebo_full < gm).mean() * 100
    pct_dir = (placebo_dir < gm).mean() * 100
    print(f"  実際の平均gross = {bps(gm)}")
    print(f"  [A] 完全プラセボ(時刻乱択+方向50/50): mean={bps(placebo_full.mean())} "
          f"sd={bps(placebo_full.std())} p95={bps(np.percentile(placebo_full, 95))} max={bps(placebo_full.max())}")
    print(f"      → 実際の平均grossは {pct_full:.1f} パーセンタイル")
    print(f"  [B] 方向のみ実際・時刻乱択: mean={bps(placebo_dir.mean())} "
          f"sd={bps(placebo_dir.std())} p95={bps(np.percentile(placebo_dir, 95))} max={bps(placebo_dir.max())}")
    print(f"      → 実際の平均grossは {pct_dir:.1f} パーセンタイル")
    print(f"  寄与の分離: 方向(ドリフト捕捉) = B−A = {bps(placebo_dir.mean() - placebo_full.mean())}/trade")
    print(f"               タイミング(状態選別) = 実際−B = {bps(gm - placebo_dir.mean())}/trade "
          f"(グロスの {(gm - placebo_dir.mean()) / gm * 100:.1f}%)\n")

    # ======================= Q3 出口の寄与 =================================
    print("=" * 70)
    print("Q3. 実z出口 vs 固定k本出口（エントリー・コスト同一、exitのみ置換）")
    print(f"  実際(z出口): sum(ret)={total_ret:+.4f}  平均={bps(ret.mean())} 保有中央値={int(pool['bars_held'].median())}本")
    rows = []
    for k in FIXED_KS:
        xs = np.empty(n)
        clipped = 0
        for j, (nm, ei) in enumerate(zip(pool["instr"], e_pos)):
            c = closes[nm]
            t = ei + k
            if t > len(c) - 1:
                t = len(c) - 1
                clipped += 1
            xs[j] = c[t]
        g_k = d * (xs / ec - 1.0)
        net_k = g_k - cost
        rows.append(dict(k=k, sum_net=net_k.sum(), mean_bps=net_k.mean() * 1e4,
                         win=(net_k > 0).mean(), vs_actual_pct=net_k.sum() / total_ret * 100,
                         clipped=clipped))
    t3 = pd.DataFrame(rows)
    print(t3.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()

    # ======================= Q4 累積グロス経路 =============================
    print("=" * 70)
    print("Q4. エントリー後の平均累積グロスリターン経路（1〜60本, 保有継続の反実仮想）")
    path = np.full((n, PATH_MAX + 1), np.nan)
    for j, (nm, ei) in enumerate(zip(pool["instr"], e_pos)):
        c = closes[nm]
        tmax = min(PATH_MAX, len(c) - 1 - ei)
        seg = c[ei: ei + tmax + 1]
        path[j, : tmax + 1] = d[j] * (seg / seg[0] - 1.0)
    win = ret > 0
    lng = d > 0
    sub = {"全体": np.ones(n, bool), "勝ち": win, "負け": ~win, "ロング": lng, "ショート": ~lng}
    ks_show = [1, 2, 3, 6, 12, 18, 24, 36, 48, 60]
    tbl = pd.DataFrame({lab: [np.nanmean(path[m, k]) * 1e4 for k in ks_show] for lab, m in sub.items()},
                       index=[f"{k}本" for k in ks_show])
    print(tbl.to_string(float_format=lambda v: f"{v:+.1f}"))
    print(f"  (単位bps。実際の平均gross={bps(gm)}=100%として)")
    for k in [6, 12, 18]:
        print(f"  最初の{k:>2}本で確定: {np.nanmean(path[:, k]) / gm * 100:.1f}%")
    print()

    # ======================= Q5 ペイオフ形状 ===============================
    print("=" * 70)
    print("Q5. ペイオフ形状（ret, 純額）")
    r_ = pd.Series(ret)
    print(f"  median={bps(r_.median())} mean={bps(r_.mean())} skew={stats.skew(ret):.2f} "
          f"kurtosis(excess)={stats.kurtosis(ret):.2f}")
    print(f"  p1={bps(np.percentile(ret, 1))} p0.1={bps(np.percentile(ret, 0.1))} "
          f"min={bps(ret.min())} max={bps(ret.max())}")
    med_win = np.median(ret[win])
    med_loss = np.median(ret[~win])
    worst10 = np.sort(ret)[: max(1, n // 10)]
    print(f"  勝ちの典型幅 median(win)={bps(med_win)} / 負け median(loss)={bps(med_loss)} "
          f"/ ワースト10%平均={bps(worst10.mean())}")
    print(f"  負けテール比: |ワースト10%平均| / median(win) = {abs(worst10.mean()) / med_win:.1f}x, "
          f"|p1|/median(win) = {abs(np.percentile(ret, 1)) / med_win:.1f}x")
    gl = ret[win].sum() / -ret[~win].sum()
    print(f"  PF={gl:.3f} 総損失に占めるワースト10%比率={-worst10.sum() / -ret[~win].sum() * 100:.1f}%")
    print(f"  保有本数: 勝ち中央値={int(pool.loc[win, 'bars_held'].median())}本 "
          f"負け中央値={int(pool.loc[~win, 'bars_held'].median())}本")

    # z-power 加重（本番ウェイト）での照合
    f = np.clip((pool["z_entry"].to_numpy() / 2.2) ** 4.0, 0.3, 3.0)
    ret_w = ret * f / f.mean()
    print(f"  [参考] z-power加重: sum={ret_w.sum():+.4f} 平均={bps(ret_w.mean())} skew={stats.skew(ret_w):.2f}")


if __name__ == "__main__":
    main()
