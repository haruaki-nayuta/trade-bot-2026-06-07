"""anatomy_cost.py — チャンピオン手法 confluence_meanrev_v2 のコスト・実装現実監査。

バックテスト純益 +1.9086 のうち、現実口座で消える分を実測で定量化する。
  Q1: スプレッドコスト（総コスト/グロス比、銘柄別、×1.5/×2.0シナリオ）
  Q2: スワップ/ファンディング（carry_annual による事後会計）
  Q3: 約定感度（エントリー/イグジット1バー遅延）
  Q4: 週末ギャップ（持ち越しトレードのギャップ寄与・ワースト10）
  Q5: 楽観/中立/悲観シナリオでの「残る割合」

実行: uv run python -m research.experiments.anatomy_cost
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import universe as uni
from fxlab.carry import carry_annual

POOL = "results/mm_pool_v2_H4_19.parquet"
BASE_NET = 1.9086  # ベースライン総純益（検算対象）

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")


def sec(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    uni.register_cross_spreads(3.0)
    pool = pd.read_parquet(POOL).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 価格再構成: 各トレードの entry/exit バーの close と前後バー
    # ------------------------------------------------------------------
    closes: dict[str, pd.Series] = {}
    for instr in sorted(pool["instr"].unique()):
        closes[instr] = uni.instrument_close(instr, "H4")

    n = len(pool)
    entry_close = np.full(n, np.nan)
    exit_close = np.full(n, np.nan)
    entry_next_close = np.full(n, np.nan)   # エントリー1バー遅延の約定価格
    exit_next_close = np.full(n, np.nan)    # イグジット1バー遅延の約定価格
    entry_skip = np.zeros(n, dtype=bool)    # 遅延エントリーがexitバー以降になる
    exit_clip = np.zeros(n, dtype=bool)     # 遅延イグジットがデータ末尾を超える

    for instr, g in pool.groupby("instr"):
        s = closes[instr]
        idx_e = s.index.get_indexer(g["entry"])
        idx_x = s.index.get_indexer(g["exit"])
        assert (idx_e >= 0).all() and (idx_x >= 0).all(), f"{instr}: timestamp miss"
        rows = g.index.to_numpy()
        entry_close[rows] = s.to_numpy()[idx_e]
        exit_close[rows] = s.to_numpy()[idx_x]
        ie1 = np.minimum(idx_e + 1, len(s) - 1)
        ix1 = np.minimum(idx_x + 1, len(s) - 1)
        entry_next_close[rows] = s.to_numpy()[ie1]
        exit_next_close[rows] = s.to_numpy()[ix1]
        entry_skip[rows] = (idx_e + 1) >= idx_x   # 遅延後の保有期間が消滅/逆転
        exit_clip[rows] = (idx_x + 1) > (len(s) - 1)

    d = pool["dir"].to_numpy().astype(float)
    ret = pool["ret"].to_numpy()
    gross = d * (exit_close / entry_close - 1.0)
    cost = gross - ret  # per-trade 往復コスト（リターン単位）

    sec("0. 検算: gross/cost 再構成の整合性")
    print(f"n={n}  sum(ret)={ret.sum():+.4f}  (ベースライン {BASE_NET:+.4f})")
    print(f"sum(gross)={gross.sum():+.4f}  sum(cost)={cost.sum():+.4f}")
    print(f"検算: sum(gross)-sum(cost)={gross.sum()-cost.sum():+.4f} == sum(ret) ✓"
          if abs((gross.sum() - cost.sum()) - ret.sum()) < 1e-9 else "検算 NG!")
    print(f"cost>0 のトレード比率: {(cost > 0).mean()*100:.1f}%  "
          f"cost中央値={np.median(cost)*1e4:.2f}bps  cost平均={cost.mean()*1e4:.2f}bps")
    # entry_price(スリッページ込み)との整合: dir=+1なら entry_price > entry_close
    ep = pool["entry_price"].to_numpy()
    half_sp = d * (ep / entry_close - 1.0)  # 半スプレッド(リターン単位)のはず
    print(f"entry_price由来の片道スリッページ中央値={np.median(half_sp)*1e4:.2f}bps "
          f"(往復={2*np.median(half_sp)*1e4:.2f}bps; cost中央値と整合するはず)")

    df = pool.copy()
    df["gross"] = gross
    df["cost"] = cost
    df["entry_close"] = entry_close
    df["exit_close"] = exit_close
    df["year"] = df["exit"].dt.year          # 決済年集計（ベースライン規約）
    df["days_held"] = (df["exit"] - df["entry"]).dt.total_seconds() / 86400.0

    # ------------------------------------------------------------------
    # Q1. スプレッドコスト
    # ------------------------------------------------------------------
    sec("Q1. 総コスト / グロス利益、銘柄別、スプレッド×1.5/×2.0")
    gross_sum, cost_sum = df["gross"].sum(), df["cost"].sum()
    print(f"グロス利益 sum(gross) = {gross_sum:+.4f}")
    print(f"総コスト   sum(cost)  = {cost_sum:+.4f}  (グロスの {cost_sum/gross_sum*100:.1f}%)")
    print(f"純益       sum(ret)   = {df['ret'].sum():+.4f}")
    print(f"総純益(+{BASE_NET})に対する総コスト比 = {cost_sum/BASE_NET*100:.1f}%")

    tbl = df.groupby("instr").agg(
        n=("ret", "size"), gross=("gross", "sum"), cost=("cost", "sum"), net=("ret", "sum"),
        cost_bps_med=("cost", lambda x: np.median(x) * 1e4),
    )
    tbl["cost/gross%"] = tbl["cost"] / tbl["gross"].where(tbl["gross"] > 0) * 100
    tbl = tbl.sort_values("net", ascending=False)
    print("\n[銘柄別] (cost_bps_med=往復コスト中央値bps, cost/gross%=グロスが正の銘柄のみ)")
    print(tbl.to_string(float_format=lambda x: f"{x:+.4f}" if abs(x) < 10 else f"{x:.1f}"))

    # スプレッドシナリオ: コストはスプレッドに線形 → net(k) = gross - k*cost
    print("\n[スプレッドシナリオ] net(k) = gross - k×cost")
    yearly = {}
    for k in (1.0, 1.5, 2.0):
        net_k = df["gross"] - k * df["cost"]
        ys = net_k.groupby(df["year"]).sum()
        yearly[k] = ys
        n_neg = int((ys < 0).sum())
        print(f"  ×{k:.1f}: 総純益={net_k.sum():+.4f} (ベースライン比 {net_k.sum()/BASE_NET*100:.1f}%) "
              f" マイナス年={n_neg} {list(ys.index[ys < 0]) if n_neg else ''}")
    ytab = pd.DataFrame({f"×{k}": v for k, v in yearly.items()})
    print("\n[年次純益(決済年)]")
    print(ytab.to_string(float_format=lambda x: f"{x:+.4f}"))

    # ------------------------------------------------------------------
    # Q2. スワップ/ファンディング（事後会計）
    # ------------------------------------------------------------------
    sec("Q2. キャリー(スワップ)会計: carry = dir × carry_annual(instr, entry年)/100 × 暦日/365")
    ca = np.array([carry_annual(r.instr, r.entry.year) for r in df.itertuples()])
    df["carry_ann"] = ca
    df["carry"] = d * (ca / 100.0) * (df["days_held"] / 365.0)
    csum = df["carry"].sum()
    print(f"(a) キャリー合計 = {csum:+.4f}  (純益+{BASE_NET}の {csum/BASE_NET*100:+.1f}%)")
    neg_share = (d * ca < 0).mean()
    zero_share = (d * ca == 0).mean()
    print(f"(b) 負キャリー方向保有のトレード比率 = {neg_share*100:.1f}% (ゼロ {zero_share*100:.1f}%)")
    df["era"] = np.where(df["entry"].dt.year >= 2022, "2022+", "2016-21")
    era = df.groupby("era").agg(
        n=("carry", "size"), carry_sum=("carry", "sum"),
        carry_bps_mean=("carry", lambda x: x.mean() * 1e4),
        neg_dir_share=("carry", lambda x: (x < 0).mean()),
        days_mean=("days_held", "mean"),
    )
    print("\n(c) 時代別:")
    print(era.to_string(float_format=lambda x: f"{x:+.4f}"))
    yr_carry = df.groupby(df["entry"].dt.year)["carry"].sum()
    print("\n  年別キャリー合計(エントリー年):")
    print("  " + "  ".join(f"{y}:{v*1e4:+.0f}bp" for y, v in yr_carry.items()))
    # (d) スワップマークアップ: 保有中、方向によらず片側 0.5%/1.0%年率のドラッグ
    for mk in (0.5, 1.0):
        drag = (mk / 100.0) * (df["days_held"] / 365.0)
        print(f"(d) マークアップ片側年率{mk}% → 追加ドラッグ = -{drag.sum():.4f} "
              f"(純益の {drag.sum()/BASE_NET*100:.1f}%)")

    # ------------------------------------------------------------------
    # Q3. 約定感度（1バー遅延）
    # ------------------------------------------------------------------
    sec("Q3. 約定感度: エントリー/イグジットを1バー遅延(次バーclose約定)")
    # エントリー遅延: 次バーcloseで約定。コスト(スプレッド)は同一とみなす。
    gross_ed = d * (exit_close / entry_next_close - 1.0)
    net_ed = gross_ed - cost
    print(f"[エントリー1バー遅延] 総純益 {df['ret'].sum():+.4f} → {net_ed.sum():+.4f} "
          f"(差 {net_ed.sum()-df['ret'].sum():+.4f} = 純益の {(net_ed.sum()-df['ret'].sum())/BASE_NET*100:+.1f}%)")
    print(f"  平均 {(net_ed.mean()-df['ret'].mean())*1e4:+.2f}bps/trade  "
          f"遅延でexitバー以降になるトレード: {entry_skip.sum()}件 (機械的にclose比で計算)")
    ys_ed = pd.Series(net_ed).groupby(df["year"]).sum()
    print(f"  マイナス年: {int((ys_ed<0).sum())} {list(ys_ed.index[ys_ed<0]) if (ys_ed<0).any() else ''}")

    # イグジット遅延: 次バーcloseで決済
    gross_xd = d * (exit_next_close / entry_close - 1.0)
    net_xd = gross_xd - cost
    print(f"[イグジット1バー遅延] 総純益 → {net_xd.sum():+.4f} "
          f"(差 {net_xd.sum()-df['ret'].sum():+.4f} = 純益の {(net_xd.sum()-df['ret'].sum())/BASE_NET*100:+.1f}%)"
          f"  末尾クリップ {exit_clip.sum()}件")
    gross_bd = d * (exit_next_close / entry_next_close - 1.0)
    net_bd = gross_bd - cost
    print(f"[両側1バー遅延]     総純益 → {net_bd.sum():+.4f} "
          f"(差 {net_bd.sum()-df['ret'].sum():+.4f} = 純益の {(net_bd.sum()-df['ret'].sum())/BASE_NET*100:+.1f}%)")

    # ------------------------------------------------------------------
    # Q4. 週末ギャップ
    # ------------------------------------------------------------------
    sec("Q4. 週末ギャップ: 持ち越しトレードのギャップ寄与")
    # 各銘柄の H4 index で、連続バー間隔 > 8h を「週末/長期休場ギャップ」とみなす
    gap_rows = []
    for instr, g in pool.groupby("instr"):
        s = closes[instr]
        sidx = s.index
        vals = s.to_numpy()
        dt_gap = np.diff(sidx.values).astype("timedelta64[s]").astype(float) / 3600.0
        gap_pos = np.where(dt_gap > 8.0)[0]  # i: bar i close → bar i+1 close が休場跨ぎ
        if len(gap_pos) == 0:
            continue
        gap_before_t = sidx.values[gap_pos]      # 金曜最終バーのlabel
        gap_after_t = sidx.values[gap_pos + 1]   # 週明け最初バーのlabel
        gap_move = vals[gap_pos + 1] - vals[gap_pos]
        for r in g.itertuples():
            # 保有中 (entryバーclose後 〜 exitバーclose) に跨いだギャップ
            m = (gap_before_t >= np.datetime64(r.entry.tz_localize(None))) & (
                gap_after_t <= np.datetime64(r.exit.tz_localize(None)))
            if not m.any():
                continue
            ec = entry_close[r.Index]
            for j in np.where(m)[0]:
                contrib = r.dir * gap_move[j] / ec
                gap_rows.append({
                    "row": r.Index, "instr": instr, "dir": r.dir,
                    "fri": pd.Timestamp(gap_before_t[j]), "mon": pd.Timestamp(gap_after_t[j]),
                    "gap_hours": (gap_after_t[j] - gap_before_t[j]) / np.timedelta64(1, "h"),
                    "contrib": contrib,
                })
    gaps = pd.DataFrame(gap_rows)
    held_weekend = gaps["row"].nunique() if len(gaps) else 0
    print(f"週末(休場>8h)持ち越しトレード: {held_weekend}/{n} 件 ({held_weekend/n*100:.1f}%) "
          f" ギャップ跨ぎ延べ {len(gaps)} 回")
    if len(gaps):
        print(f"ギャップ寄与合計 = {gaps['contrib'].sum():+.4f} (純益+{BASE_NET}の {gaps['contrib'].sum()/BASE_NET*100:+.1f}%)")
        print(f"  有利方向 {len(gaps[gaps.contrib>0])}回 +{gaps[gaps.contrib>0].contrib.sum():.4f} / "
              f"不利方向 {len(gaps[gaps.contrib<0])}回 {gaps[gaps.contrib<0].contrib.sum():+.4f}")
        worst = gaps.nsmallest(10, "contrib").copy()
        worst["contrib_bps"] = worst["contrib"] * 1e4
        print("\n[不利方向ワースト10ギャップ] (contrib_bps=1単位リターンbps)")
        print(worst[["instr", "dir", "fri", "mon", "gap_hours", "contrib_bps"]]
              .to_string(index=False, float_format=lambda x: f"{x:+.1f}"))
        # ギャップ寄与が大きいトレード単位の集計
        per_trade = gaps.groupby("row")["contrib"].sum()
        print(f"\nトレード単位ギャップ寄与: 中央値={per_trade.median()*1e4:+.1f}bps "
              f"最悪={per_trade.min()*1e4:+.1f}bps ({pool.loc[per_trade.idxmin(),'instr']})")

    # ------------------------------------------------------------------
    # Q5. シナリオ統合
    # ------------------------------------------------------------------
    sec("Q5. 現実口座に残る割合: 楽観/中立/悲観")
    carry = df["carry"]
    drag05 = (0.5 / 100.0) * (df["days_held"] / 365.0)
    drag10 = (1.0 / 100.0) * (df["days_held"] / 365.0)
    # 約定ストレスは「両側1バー遅延」(エントリー遅延単体は実測で+に出るため悲観に使わない)
    both_delay_cost = pd.Series(net_bd - df["ret"].to_numpy())

    scenarios = {
        "楽観 (spread×1.0, swap=carry-0.5%mk, 遅延なし)":
            df["gross"] - 1.0 * df["cost"] + carry - drag05,
        "中立 (spread×1.5, swap=carry-1.0%mk, 両側遅延影響の半分)":
            df["gross"] - 1.5 * df["cost"] + carry - drag10 + 0.5 * both_delay_cost,
        "悲観 (spread×2.0, swap=carry-1.0%mk, 両側1バー全遅延)":
            df["gross"] - 2.0 * df["cost"] + carry - drag10 + 1.0 * both_delay_cost,
    }
    for name, net_s in scenarios.items():
        ys = net_s.groupby(df["year"]).sum()
        neg = list(ys.index[ys < 0])
        print(f"{name}\n  総純益 = {net_s.sum():+.4f}  残存率 = {net_s.sum()/BASE_NET*100:.1f}% "
              f" マイナス年={len(neg)} {neg if neg else ''}")

    print("\n[シナリオ年次表]")
    stab = pd.DataFrame({k.split(' ')[0]: v.groupby(df['year']).sum() for k, v in scenarios.items()})
    stab["base"] = df.groupby("year")["ret"].sum()
    print(stab.to_string(float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
