"""勝ち/負けトレードの特徴ビジュアライズ用データ抽出(reports/25 の可視化付録)。

d1 プール(mm_pool_v2d1_H4_19.parquet)から、勝ち/負けの特徴量を多角的に集計して
research/outputs/winloss_viz_data.json に吐く。HTML(win_loss_anatomy.html)が埋め込み利用。

切り口: 基礎統計 / リターン分布 / 保有期間 / 集中度(Lorenz+Gini) / 銘柄別 / 年別 /
方向別 / エントリー時刻(UTC)別 / 曜日別 / z_entry深度別 / vol五分位別 / 保有vs損益散布 /
hot-hand / 早期シグネチャ・MFE捕捉・決済後ドリフト(exp70_result.json から転載)。

実行: cd リポジトリ直下 && PYTHONPATH=. uv run python research/outputs/make_winloss_viz_data.py
注意: 記述統計のみ(検定なし)。セル分割の多重比較に注意(reports/25 §6)。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))

OUT = ROOT / "research" / "outputs"


def bucket_stats(df: pd.DataFrame, key) -> list[dict]:
    rows = []
    for name, g in df.groupby(key, observed=True):
        rows.append({
            "label": str(name),
            "n": int(len(g)),
            "win_rate": float((g["ret"] > 0).mean()),
            "mean_bps": float(g["ret"].mean() * 1e4),
            "sum_ret": float(g["ret"].sum()),
        })
    return rows


def lorenz(vals: np.ndarray) -> list[list[float]]:
    """絶対値の昇順累積シェア曲線 [share_of_count, share_of_total] (0-1)。"""
    v = np.sort(np.abs(vals))
    cum = np.cumsum(v) / v.sum()
    n = len(v)
    pts = [[0.0, 0.0]]
    for i in range(0, n, max(1, n // 100)):
        pts.append([(i + 1) / n, float(cum[i])])
    pts.append([1.0, 1.0])
    return pts


def gini(vals: np.ndarray) -> float:
    v = np.sort(np.abs(vals))
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def main() -> int:
    from mm_production import build_pool_d1

    pool = build_pool_d1()
    assert len(pool) == 1207 and abs(pool["ret"].sum() - 1.9622) < 0.01, "プール検算不一致"
    df = pool.copy()
    df["win"] = df["ret"] > 0
    df["bps"] = df["ret"] * 1e4
    df["entry"] = pd.to_datetime(df["entry"], utc=True)
    df["year"] = df["entry"].dt.year
    df["hour"] = df["entry"].dt.hour
    df["dow"] = df["entry"].dt.dayofweek  # 0=月
    wins, losses = df[df["win"]], df[~df["win"]]

    res: dict = {"n": len(df)}

    # 基礎
    res["basic"] = {
        "win_rate": float(df["win"].mean()),
        "n_win": int(df["win"].sum()),
        "n_loss": int((~df["win"]).sum()),
        "mean_win_bps": float(wins["bps"].mean()),
        "mean_loss_bps": float(losses["bps"].mean()),
        "median_win_bps": float(wins["bps"].median()),
        "median_loss_bps": float(losses["bps"].median()),
        "expectancy_bps": float(df["bps"].mean()),
        "payoff": float(wins["bps"].mean() / -losses["bps"].mean()),
        "gross_profit": float(wins["ret"].sum()),
        "gross_loss": float(losses["ret"].sum()),
        "hold_med_win": float(wins["bars_held"].median()),
        "hold_med_loss": float(losses["bars_held"].median()),
        "hold_p90_win": float(wins["bars_held"].quantile(0.9)),
        "hold_p90_loss": float(losses["bars_held"].quantile(0.9)),
    }

    # リターン分布(bps, 10bps刻み, -300..+300 にクリップ)
    bins = np.arange(-300, 310, 10)
    res["ret_hist"] = {
        "bins": bins[:-1].tolist(),
        "win": np.histogram(wins["bps"].clip(-299, 299), bins=bins)[0].tolist(),
        "loss": np.histogram(losses["bps"].clip(-299, 299), bins=bins)[0].tolist(),
    }

    # 保有期間分布(4本=16h刻み 0..100)
    hbins = np.arange(0, 105, 4)
    res["hold_hist"] = {
        "bins": hbins[:-1].tolist(),
        "win": np.histogram(wins["bars_held"].clip(0, 104), bins=hbins)[0].tolist(),
        "loss": np.histogram(losses["bars_held"].clip(0, 104), bins=hbins)[0].tolist(),
    }

    # Lorenz / Gini
    res["lorenz"] = {
        "win": lorenz(wins["ret"].to_numpy()), "loss": lorenz(losses["ret"].to_numpy()),
        "gini_win": gini(wins["ret"].to_numpy()), "gini_loss": gini(losses["ret"].to_numpy()),
    }

    # 各種バケット
    res["by_instr"] = sorted(bucket_stats(df, "instr"), key=lambda r: r["mean_bps"])
    res["by_year"] = bucket_stats(df, "year")
    res["by_dir"] = bucket_stats(df.assign(d=np.where(df["dir"] > 0, "ロング", "ショート")), "d")
    res["by_hour"] = bucket_stats(df, "hour")
    res["by_dow"] = bucket_stats(df.assign(w=df["dow"].map({0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 6: "日"})), "w")

    # z_entry 深度バケット / vol 五分位
    df["zb"] = pd.cut(df["z_entry"], [0, 2.0, 2.25, 2.5, 2.75, 3.0, 10],
                      labels=["<2.0", "2.0-2.25", "2.25-2.5", "2.5-2.75", "2.75-3.0", ">3.0"])
    res["by_zdepth"] = bucket_stats(df, "zb")
    df["vq"] = pd.qcut(df["vol_entry"], 5, labels=["Q1(低)", "Q2", "Q3", "Q4", "Q5(高)"])
    res["by_volq"] = bucket_stats(df, "vq")

    # 散布図(全点。保有 vs bps)
    res["scatter"] = [[int(b), round(float(r), 1)] for b, r in zip(df["bars_held"], df["bps"])]

    # hot-hand(同一銘柄・直前トレード)
    prev_win = df.sort_values("entry").groupby("instr")["win"].shift(1)
    m = prev_win.notna()
    res["hothand"] = {
        "after_win": {"n": int((prev_win[m] == True).sum()),  # noqa: E712
                      "win_rate": float(df.loc[m & (prev_win == True), "win"].mean()),  # noqa: E712
                      "mean_bps": float(df.loc[m & (prev_win == True), "bps"].mean())},  # noqa: E712
        "after_loss": {"n": int((prev_win[m] == False).sum()),  # noqa: E712
                       "win_rate": float(df.loc[m & (prev_win == False), "win"].mean()),  # noqa: E712
                       "mean_bps": float(df.loc[m & (prev_win == False), "bps"].mean())},  # noqa: E712
        "base": float(df["win"].mean()),
    }

    # 累積純益(プール等加重 cumsum, 月次)
    ts = df.set_index("entry")["ret"].sort_index().cumsum().resample("ME").last().ffill()
    res["cum_pool"] = {"t": [str(d.date()) for d in ts.index], "v": ts.round(4).tolist()}

    # exp70 から転載(早期シグネチャ / MFE / 決済後)
    e70 = json.loads((OUT / "exp70_result.json").read_text())
    res["exp70"] = {
        "pos_at_k": e70["B"]["pos_at_k"], "ever_mfe": e70["B"]["ever_mfe"],
        "capture": e70["C"]["winners"], "losers_mfe": e70["C"]["losers"],
        "post_exit": e70["D"]["post_exit"],
        "losers_never_in_profit": e70["B"]["losers_never_in_profit_share"],
    }

    (OUT / "winloss_viz_data.json").write_text(json.dumps(res, ensure_ascii=False))
    print(f"saved -> {OUT / 'winloss_viz_data.json'}  ({len(json.dumps(res))//1024}KB)")
    # 発見チェック用に主要カットを表示
    for k in ["by_dir", "by_dow", "by_hour", "by_zdepth", "by_volq"]:
        print(f"\n--- {k} ---")
        for r in res[k]:
            print(f"  {r['label']:>10}  n={r['n']:4d}  勝率{r['win_rate']:.1%}  平均{r['mean_bps']:+6.1f}bps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
