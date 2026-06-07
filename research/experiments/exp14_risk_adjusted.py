"""出口改善を「リスク調整後(Sharpe / 最大DD)」で再評価する。

exp13 で総利益はどの出口改善でも増えないと判明。だが平均回帰では損切りが
リターンを削るのは既知。真に問うべきは「テールを切ると risk-adjusted は改善するか」。
ポート(19対象等加重・各SIZE固定)の月次損益から Sharpe と最大DD を比較する。

実行: uv run python exp14_risk_adjusted.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import config
from exp13_exit_improve import engine_universe, sim_universe, SIZE, INSTRUMENTS

pd.set_option("display.width", 200)


def curve_metrics(df, exit_col):
    """トレード群 → 月次ポート損益 → Sharpe・最大DD。"""
    ts = pd.DatetimeIndex(df[exit_col])
    monthly = df.assign(m=ts.to_period("M")).groupby("m")["pnl"].sum()
    monthly = monthly.asfreq("M", fill_value=0.0)
    sharpe = float(monthly.mean() / monthly.std() * np.sqrt(12)) if monthly.std() else np.nan
    equity = monthly.cumsum()
    # 名目資本 = 全対象の建玉合計(19×SIZE)を初期資本とみなして DD を%化
    cap = SIZE * len(INSTRUMENTS)
    eq = cap + equity
    peak = eq.cummax()
    maxdd = float(((eq - peak) / peak).min())
    pnl = df["pnl"]
    gp = float(pnl[pnl > 0].sum()); gl = float(-pnl[pnl < 0].sum())
    ret_on_cap = float(equity.iloc[-1] / cap)            # 11年累積(対名目資本)
    return {
        "total_pnl": round(float(pnl.sum())),
        "ret/cap_11y": f"{ret_on_cap:.1%}",
        "PF": round(gp / gl, 3) if gl else np.inf,
        "Sharpe(m)": round(sharpe, 2),
        "maxDD": f"{maxdd:.1%}",
        "ret/DD": round(ret_on_cap / -maxdd, 2) if maxdd else np.nan,
        "worst%": round(float(df["ret"].min()), 2),
    }


def main():
    variants = {
        "基準: 無ストップ":        engine_universe(),
        "A. sl=3%(純正損切)":     engine_universe(sl_stop=0.03),
        "B. 時間 max_bars=50":     sim_universe(max_bars=50),
        "B. 時間 max_bars=40":     sim_universe(max_bars=40),
        "B. 時間 max_bars=30":     sim_universe(max_bars=30),
    }
    rows = {}
    for label, df in variants.items():
        col = "exit" if "exit" in df.columns else "exit_ts"
        rows[label] = curve_metrics(df, col)
    out = pd.DataFrame(rows).T
    print("=== リスク調整後の出口比較(ポート19対象・月次)===\n")
    print(out.to_string())
    out.to_csv(config.RESULTS_DIR / "exp14_risk_adjusted.csv")
    print(f"\n保存: {config.RESULTS_DIR}/exp14_risk_adjusted.csv")


if __name__ == "__main__":
    main()
