"""イテレーション5 実験: クロス通貨ペアを合成して分散を拡張(「毎年プラス」狙い)。

メジャー7ペアだけだと USD 主導で悪い年(2020,2024…)が共通。レンジ性が強く回帰しやすい
クロス(EURGBP, AUDNZD, EURCHF 等)を majors から合成し、レジームのタイミングが異なる
対象で分散すれば、ポートフォリオの年次マイナスを埋められる可能性がある。

confluence_meanrev は close ベースなので、クロスは close を合成して評価(OHLC は close で代用)。
クロスのスプレッドは広めに見積もる(config を一時上書き)。  実行: uv run python exp05.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import config
from fxlab import yearly as ylib
from fxlab.data import available_pairs, load
from strategies.confluence_meanrev import generate_signals as cmr, PARAMS

pd.set_option("display.width", 200)
TF = "H4"

# クロス定義: name -> (式)。USD建てメジャーから合成。
def build_crosses():
    c = {p: load(p, TF)["close"] for p in available_pairs()}
    df = pd.DataFrame(c).dropna()
    cross = {
        "EURGBP": df["EURUSD"] / df["GBPUSD"],
        "EURCHF": df["EURUSD"] * df["USDCHF"],
        "EURAUD": df["EURUSD"] / df["AUDUSD"],
        "EURCAD": df["EURUSD"] * df["USDCAD"],
        "GBPAUD": df["GBPUSD"] / df["AUDUSD"],
        "GBPCHF": df["GBPUSD"] * df["USDCHF"],
        "AUDNZD": df["AUDUSD"] / df["NZDUSD"],
        "AUDCAD": df["AUDUSD"] * df["USDCAD"],
        "NZDCAD": df["NZDUSD"] * df["USDCAD"],
        "AUDCHF": df["AUDUSD"] * df["USDCHF"],
        "EURJPY": df["EURUSD"] * df["USDJPY"],
        "GBPJPY": df["GBPUSD"] * df["USDJPY"],
        "AUDJPY": df["AUDUSD"] * df["USDJPY"],
    }
    return cross


def synth_ohlc(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 1.0}, index=close.index)


def yearly_for_series(name, close):
    return ylib.yearly(name, TF, cmr, PARAMS, data=synth_ohlc(close), size_mode="value")


def portfolio_from(tables: dict):
    accum = {}
    for tbl in tables.values():
        for year, r in tbl.iterrows():
            a = accum.setdefault(int(year), [0.0, 0.0, 0.0, 0.0])
            a[0] += r["gross_profit"]; a[1] += r["gross_loss"]; a[2] += r["pnl"]; a[3] += r["trades"]
    rows = {}
    for year, (gp, gl, pnl, tr) in sorted(accum.items()):
        rows[year] = {"trades": int(tr), "profit_factor": round(gp/gl, 2) if gl > 0 else np.inf,
                      "pnl": round(pnl, 0), "positive": pnl > 0}
    return pd.DataFrame(rows).T


def report(label, tables):
    port = portfolio_from(tables)
    pf = port["profit_factor"].replace(np.inf, np.nan)
    print(f"\n=== {label} ({len(tables)}対象) ===")
    print(port.to_string())
    print(f"  → プラス年率 {(port['pnl']>0).mean():.0%} / PF中央 {pf.median():.2f} / PF最小 {pf.min():.2f} / 年取引 {int(port['trades'].mean())}")


def main():
    # クロスのスプレッドを広めに設定(往復コストを現実的に)
    for nm in ["EURGBP","EURCHF","EURAUD","EURCAD","GBPAUD","GBPCHF","AUDNZD","AUDCAD",
               "NZDCAD","AUDCHF","EURJPY","GBPJPY","AUDJPY"]:
        config.SPREADS_PIPS[nm] = 2.0

    majors = {p: ylib.yearly(p, TF, cmr, PARAMS, size_mode="value") for p in available_pairs()}
    crosses = {nm: yearly_for_series(nm, cl) for nm, cl in build_crosses().items()}

    # 個別クロスの通算成績
    print("=== 個別クロスの通算 (PF / total_pnl) ===")
    for nm, tbl in crosses.items():
        pf = ylib.profit_factor  # noqa
        gp = tbl["gross_profit"].sum(); gl = tbl["gross_loss"].sum()
        print(f"  {nm}: PF={gp/gl:.2f}  pnl={tbl['pnl'].sum():+.0f}  pos_yr={ (tbl['pnl']>0).mean():.0%}")

    report("メジャー7のみ", majors)
    report("クロス13のみ", crosses)
    report("メジャー+クロス 全20", {**majors, **crosses})


if __name__ == "__main__":
    main()
