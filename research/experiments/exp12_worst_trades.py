"""チャンピオン confluence_meanrev のワーストトレードを全ユニバース横断で解剖する診断。

目的: 損切り無し設計の「テールリスク(限界3)」が本当にワーストの正体か検証し、
改善の打ち手(時間ストップ / Zブローアウト / カタストロフ損切り)の効き目を実測する。

実行: uv run python exp12_worst_trades.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import universe as uni
from fxlab.trades import trade_table
from strategies.confluence_meanrev import generate_signals

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)

TF = "H4"
PARAMS = {"window": 50, "entry_z": 2.0, "exit_z": 0.5, "rsi_p": 14, "rsi_low": 35,
          "rsi_high": 65, "vol_win": 100, "vol_pct": 0.70, "slow_win": 250, "slow_z": 1.75}

uni.register_cross_spreads(3.0)
INSTRUMENTS = [x for x in uni.universe(crosses=True) if x != "AUDJPY"]


def all_trades(params, **run_kw) -> pd.DataFrame:
    """全対象のトレードを1枚に結合(列に instrument 追加)。size_mode=value で本番準拠。"""
    from fxlab.backtest import run
    frames = []
    for name in INSTRUMENTS:
        data = uni.instrument_data(name, TF)
        pf = run(name, TF, generate_signals, params, data=data,
                 size_mode="value", **run_kw)
        tt = trade_table(pf, data)
        if tt.empty:
            continue
        tt["instrument"] = name
        frames.append(tt)
    df = pd.concat(frames, ignore_index=True)
    df["year"] = pd.DatetimeIndex(df["exit"]).year
    return df


def agg_stats(df: pd.DataFrame) -> dict:
    pnl = df["pnl"]
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    # 年次ポート(対象等加重・決済年合算): 毎年プラスか
    yr = df.groupby("year")["pnl"].sum()
    return {
        "trades": len(df),
        "total_pnl": float(pnl.sum()),
        "PF": gp / gl if gl else float("inf"),
        "win_rate": float((pnl > 0).mean()),
        "worst_trade_%": float(df["return_pct"].min()),
        "worst_trade_$": float(pnl.min()),
        "mean_bars_held": float(df["bars_held"].mean()),
        "yrs_positive": f"{int((yr > 0).sum())}/{len(yr)}",
        "min_year_pnl": float(yr.min()),
    }


def main():
    base = all_trades(PARAMS)
    print(f"=== ベースライン(損切り無し)全{len(INSTRUMENTS)}対象 / {TF} ===")
    print(pd.Series(agg_stats(base)).to_string(), "\n")

    # --- ワースト20トレードの素性 ---
    worst = base.sort_values("return_pct").head(20)
    cols = ["instrument", "dir", "entry", "return_pct", "pnl", "bars_held", "hours"]
    w = worst[cols].copy()
    w["entry"] = pd.to_datetime(w["entry"]).dt.strftime("%Y-%m-%d")
    w["return_pct"] = w["return_pct"].round(2)
    w["pnl"] = w["pnl"].round(0)
    print("=== 💀 ワースト20トレード ===")
    print(w.to_string(index=False), "\n")

    # --- 保有期間と損益の関係(テールが長期保有か) ---
    q = base["bars_held"].quantile([0.5, 0.9, 0.95, 0.99])
    print("=== 保有バー数(bars_held)の分位 ===")
    print(q.round(0).to_string(), "\n")

    base = base.copy()
    base["held_bucket"] = pd.cut(base["bars_held"],
                                 bins=[-1, 5, 15, 30, 60, 120, 10_000],
                                 labels=["0-5", "6-15", "16-30", "31-60", "61-120", "120+"])
    g = base.groupby("held_bucket", observed=True).agg(
        n=("pnl", "size"),
        avg_ret_pct=("return_pct", "mean"),
        total_pnl=("pnl", "sum"),
        win_rate=("pnl", lambda s: (s > 0).mean()),
    ).round(3)
    print("=== 保有期間バケット別の成績(テールの正体) ===")
    print(g.to_string(), "\n")

    # ワースト10%トレードの平均保有 vs 全体
    cut = base["return_pct"].quantile(0.10)
    tail = base[base["return_pct"] <= cut]
    print(f"ワースト10%トレード平均保有 {tail['bars_held'].mean():.1f}本 / "
          f"全体平均 {base['bars_held'].mean():.1f}本")
    print(f"ワースト10%が総損失に占める割合: "
          f"{-tail['pnl'][tail['pnl']<0].sum() / -base['pnl'][base['pnl']<0].sum():.1%}\n")

    print(f"corr(bars_held, return_pct) = {base['bars_held'].corr(base['return_pct']):.3f}")


if __name__ == "__main__":
    main()
