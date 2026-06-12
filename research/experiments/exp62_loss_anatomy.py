"""exp62: 損失テールの解剖(d1 プール) — エッジケースで損失が膨らむ機構の特定。

ユーザーの問い: ボラは考慮しているか / 損失が膨らむエッジケースを探り、
損失トレードの特徴を重点観察せよ。

前提(既知): reports/08 = ワースト10%が総損失の72.5%・塩漬け(corr(保有,損益)=-0.85)。
reports/15 = 失血署名は中庸ボラ×持続トレンド。本実験は **現行 d1 プール**で
(a) 損失集中度の更新 (b) 「塩漬けの緩慢ブリード vs 単発ギャップ」の寄与分解
(c) エントリー後ボラ膨張(入口は静穏フィルタ済みだが事後の爆発は無防備)
(d) 週末ギャップ・既知イベント(フラッシュクラッシュ/侵攻/COVID)との重なり
(e) ワースト20個票 + 勝者vs敗者の特徴量比較(exp60 の因果特徴量15列を再利用)
を測る。**分析のみ・採用判断なし**(出口・閾値・エントリーフィルタ層は閉鎖済み)。

実行: PYTHONPATH=. uv run python research/experiments/exp62_loss_anatomy.py
出力: research/outputs/exp62_trades.csv(全トレード+パス統計) / exp62_summary.json
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

from mm_production import build_pool_d1  # noqa: E402
from exp60_wf_ml_veto import build_features  # noqa: E402
from fxlab import universe as uni  # noqa: E402

OUT_DIR = ROOT / "research" / "outputs"

# 既知のマクロイベント窓(UTC) — トレード窓と重なるかのタグ付け
EVENTS = {
    "GBP_flash_2016": ("2016-10-06", "2016-10-08"),
    "JPY_flash_2019": ("2019-01-02", "2019-01-04"),
    "COVID_2020": ("2020-03-01", "2020-03-31"),
    "UKR_invasion_2022": ("2022-02-24", "2022-03-11"),
    "JPY_intervention_2022": ("2022-09-22", "2022-10-24"),
    "SVB_2023": ("2023-03-09", "2023-03-20"),
    "JPY_carry_unwind_2024": ("2024-08-01", "2024-08-09"),
}


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def path_stats(pool: pd.DataFrame) -> pd.DataFrame:
    """各トレードの H4 終値パス統計(エントリー close 基準・方向調整済み)。"""
    n = len(pool)
    out = {c: np.full(n, np.nan) for c in
           ["mae", "mfe", "t_mae_bars", "t_mfe_bars", "worst_bar", "worst_bar_t",
            "max_gap_h", "n_weekends", "vol_expand", "ret_from_mae",
            "bleed_share", "slowz_exit_dir", "ever_pos_05"]}
    for instr, g in pool.groupby("instr"):
        d = uni.instrument_data(instr, "H4")
        close = d["close"]
        vol20 = close.pct_change().rolling(20).std()
        zs = (close - close.rolling(250).mean()) / close.rolling(250).std()
        idx = close.index
        pos_of = pd.Series(np.arange(len(idx)), index=idx)
        e_pos = pos_of.reindex(g["entry"]).to_numpy()
        x_pos = pos_of.reindex(g["exit"]).to_numpy()
        carr = close.to_numpy()
        tarr = idx.to_numpy()
        for row, (ti, e, x) in enumerate(zip(g.index.to_numpy(), e_pos, x_pos)):
            if not (np.isfinite(e) and np.isfinite(x)):
                continue
            e, x = int(e), int(x)
            dirv = float(pool.at[ti, "dir"])
            ec = carr[e]
            seg = carr[e:x + 1]
            path = dirv * (seg / ec - 1.0)
            steps = np.diff(path)
            out["mae"][ti] = path.min()
            out["mfe"][ti] = path.max()
            out["t_mae_bars"][ti] = int(np.argmin(path))
            out["t_mfe_bars"][ti] = int(np.argmax(path))
            if len(steps):
                wb = int(np.argmin(steps))
                out["worst_bar"][ti] = steps[wb]
                out["worst_bar_t"][ti] = wb + 1
                neg = steps[steps < 0].sum()
                out["bleed_share"][ti] = (steps.min() / neg) if neg < 0 else np.nan
            gaps = np.diff(tarr[e:x + 1]).astype("timedelta64[m]").astype(float) / 60.0
            out["max_gap_h"][ti] = gaps.max() if len(gaps) else 0.0
            out["n_weekends"][ti] = int((gaps > 8).sum()) if len(gaps) else 0
            v_in = vol20.iloc[e]
            v_max = vol20.iloc[e:x + 1].max()
            out["vol_expand"][ti] = v_max / v_in if v_in and np.isfinite(v_in) else np.nan
            out["ret_from_mae"][ti] = path[-1] - path.min()
            out["ever_pos_05"][ti] = float(path.max() >= 0.005)
            out["slowz_exit_dir"][ti] = -dirv * zs.iloc[x]
    return pd.DataFrame(out, index=pool.index)


def event_tag(pool: pd.DataFrame) -> pd.Series:
    tags = pd.Series("", index=pool.index)
    for name, (a, b) in EVENTS.items():
        a, b = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1)
        hit = (pool["entry"] < b) & (pool["exit"] >= a)
        tags[hit] = tags[hit].where(tags[hit] == "", tags[hit] + "+") + name
    return tags


def bucket_table(df, col, edges, labels=None):
    cats = pd.cut(df[col], edges, labels=labels)
    g = df.groupby(cats, observed=False)["ret"]
    t = pd.DataFrame({"n": g.size(), "mean_bps": g.mean() * 1e4, "sum": g.sum(),
                      "win": g.apply(lambda s: (s > 0).mean()),
                      "loss_sum": g.apply(lambda s: s[s < 0].sum())})
    return t


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1().copy()
    print(f"=== exp62: 損失テール解剖 (d1 pool n={len(pool)}) ===")
    feats = build_features(pool)
    ps = path_stats(pool)
    df = pd.concat([pool, feats.add_prefix("f_"), ps], axis=1)
    df["ev"] = event_tag(pool)
    df["year"] = df["exit"].dt.year

    # --- 1. 損失集中度 -------------------------------------------------------
    sec("1. 損失の集中度(プール段)")
    r = df["ret"]
    losses = r[r < 0].sort_values()
    tot_loss = losses.sum()
    print(f"勝率 {(r>0).mean():.1%}  総益 {r[r>0].sum():+.3f}  総損 {tot_loss:+.3f}  "
          f"純益 {r.sum():+.3f}")
    for q in (0.01, 0.05, 0.10):
        k = max(1, int(len(df) * q))
        sh = losses.iloc[:k].sum() / tot_loss
        print(f"  ワースト{q:.0%}({k}件) = 総損失の {sh:.1%}  "
          f"(最小 {losses.iloc[0]:+.3%} 〜 {losses.iloc[k-1]:+.3%})")
    wl = df.nsmallest(121, "ret")  # ワースト10%
    print("\nワースト10%の構成: 年次 " +
          " ".join(f"{int(y)}:{c}" for y, c in wl["year"].value_counts().sort_index().items()))
    print("  銘柄上位: " + "  ".join(f"{i}:{c}" for i, c in
                                  wl["instr"].value_counts().head(6).items()))
    print(f"  方向: Long {(wl['dir']>0).sum()} / Short {(wl['dir']<0).sum()}"
          f"  (全体: {(df['dir']>0).mean():.0%} Long)")
    print(f"  保有: median {wl['bars_held'].median():.0f}本 vs 全体 {df['bars_held'].median():.0f}本"
          f" / corr(bars_held, ret) = {df['bars_held'].corr(df['ret']):.2f}")

    # --- 2. 塩漬けブリード vs 単発ギャップ ----------------------------------
    sec("2. 損失の機構分解: 緩慢ブリード vs 単発ギャップ")
    wl_all = df[df["ret"] < 0]
    print(f"敗者の bleed_share(最悪1本/総下落)中央値: {wl_all['bleed_share'].median():.2f} "
          f"(1.0=単発ギャップが全て / 小=多数バーの累積)")
    print(f"ワースト10%のそれ: {wl['bleed_share'].median():.2f}")
    print(f"ワースト10%の worst_bar(1本の最大逆行)中央値: {wl['worst_bar'].median():+.3%} "
          f"vs 最終損失中央値 {wl['ret'].median():+.3%}")
    print(f"ワースト10%で『一度も+0.5%に届かず』: {(wl['ever_pos_05']==0).mean():.0%}")
    print(f"MAE→exit の戻し(ret_from_mae) ワースト10%中央値: {wl['ret_from_mae'].median():+.3%}")

    # --- 3. ボラの事後爆発 ----------------------------------------------------
    sec("3. エントリー後ボラ膨張(入口は静穏フィルタ済・事後は無防備)")
    tb = bucket_table(df.dropna(subset=["vol_expand"]), "vol_expand",
                      [0, 1.25, 1.75, 2.5, 4.0, np.inf],
                      ["<1.25x", "1.25-1.75x", "1.75-2.5x", "2.5-4x", ">4x"])
    print(tb.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\ncorr(vol_expand, ret) = {df['vol_expand'].corr(df['ret']):.2f} / "
          f"corr(vol_expand, bars_held) = {df['vol_expand'].corr(df['bars_held']):.2f}")

    # --- 4. 週末ギャップ・イベント曝露 --------------------------------------
    sec("4. 週末ギャップ曝露と既知イベント")
    tb2 = bucket_table(df, "n_weekends", [-0.5, 0.5, 1.5, 3.5, np.inf],
                       ["0", "1", "2-3", "4+"])
    print("跨いだ週末数別:")
    print(tb2.to_string(float_format=lambda x: f"{x:.3f}"))
    ev_rows = []
    for name in EVENTS:
        sub = df[df["ev"].str.contains(name, regex=False)]
        if len(sub) == 0:
            continue
        ev_rows.append({"event": name, "n": len(sub), "sum": sub["ret"].sum(),
                        "mean_bps": sub["ret"].mean() * 1e4, "win": (sub["ret"] > 0).mean(),
                        "worst": sub["ret"].min()})
    evt = pd.DataFrame(ev_rows)
    print("\nイベント窓に重なったトレード:")
    print(evt.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    no_ev = df[df["ev"] == ""]
    print(f"イベント外: n={len(no_ev)} sum={no_ev['ret'].sum():+.3f} "
          f"mean={no_ev['ret'].mean()*1e4:+.1f}bps")

    # --- 5. 勝者 vs 敗者(ワースト10%)の特徴量 -------------------------------
    sec("5. 特徴量比較(ワースト10% vs 勝者 vs 全体)")
    feat_cols = [c for c in df.columns if c.startswith("f_")] + \
        ["mae", "mfe", "t_mae_bars", "vol_expand", "n_weekends", "bars_held",
         "worst_bar", "slowz_exit_dir"]
    winners = df[df["ret"] > 0]
    comp = pd.DataFrame({
        "worst10%": df.nsmallest(121, "ret")[feat_cols].mean(),
        "losers_all": df[df["ret"] < 0][feat_cols].mean(),
        "winners": winners[feat_cols].mean(),
        "all": df[feat_cols].mean(),
    })
    comp["w10_vs_win"] = comp["worst10%"] / comp["winners"].replace(0, np.nan)
    print(comp.to_string(float_format=lambda x: f"{x:.3f}"))

    # --- 6. ワースト20 個票 ---------------------------------------------------
    sec("6. ワースト20 個票")
    w20 = df.nsmallest(20, "ret")
    show = w20[["instr", "dir", "entry", "ret", "bars_held", "mae", "mfe",
                "worst_bar", "vol_expand", "n_weekends", "f_er40", "f_slow_z_dir",
                "ev"]].copy()
    show["entry"] = show["entry"].dt.strftime("%Y-%m-%d %H:%M")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # --- 保存 -----------------------------------------------------------------
    df.drop(columns=["ev"]).assign(ev=df["ev"]).to_csv(OUT_DIR / "exp62_trades.csv", index=False)
    summary = {
        "loss_concentration": {f"worst_{int(q*100)}pct": float(
            losses.iloc[:max(1, int(len(df) * q))].sum() / tot_loss) for q in (0.01, 0.05, 0.10)},
        "bleed_share_median_worst10": float(wl["bleed_share"].median()),
        "vol_expand_table": tb.to_dict(),
        "weekend_table": tb2.to_dict(),
        "events": ev_rows,
    }
    (OUT_DIR / "exp62_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nsaved -> {OUT_DIR / 'exp62_trades.csv'} / exp62_summary.json")
    print(f"総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
