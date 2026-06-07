"""イテレーション16: コインテグレーション・ペアトレード(マーケットニュートラルなスプレッド平均回帰)。

構造的に異なる最後の型。2つの経済連動メジャーの log スプレッドを z-score 化し、
乖離で「割安側ロング・割高側ショート」、平均回帰で手仕舞い。方向性(ドル全面高/安)に
依存しないため、高PFと毎年プラスの両立が理論的に最も期待できる。スプレッド中心は
ローリング窓で追従(緩やかなヘッジ水準変化に対応, 先読みなし)。

ペアは a-priori 経済連動(Oceania/欧州/資源)で選定。各脚にスプレッドコストを往復計上。
実行: uv run python exp16.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import config
from fxlab.data import load

pd.set_option("display.width", 200)
TF = "H4"


def leg_close(sym):
    """USD建てメジャー or 反転(JPY/CHF/CAD は 1/USDxxx)で各通貨のUSD建てcloseを返す。"""
    direct = {"EUR": "EURUSD", "GBP": "GBPUSD", "AUD": "AUDUSD", "NZD": "NZDUSD"}
    inv = {"JPY": "USDJPY", "CHF": "USDCHF", "CAD": "USDCAD"}
    if sym in direct:
        return load(direct[sym], TF)["close"], config.spread_pips(direct[sym]) * config.pip_size(direct[sym])
    s = load(inv[sym], TF)["close"]
    return 1.0 / s, config.spread_pips(inv[sym]) * config.pip_size(inv[sym]) / s.mean()


# a-priori 経済連動ペア(両脚ともUSD建てに正規化して同方向相関)
PAIRS = [("AUD", "NZD"), ("EUR", "GBP"), ("EUR", "CHF"), ("AUD", "CAD"),
         ("NZD", "CAD"), ("GBP", "CHF"), ("AUD", "CHF")]


def pair_trades(a, b, win=250, entry=2.0, exit=0.5):
    ca, sa = leg_close(a)
    cb, sb = leg_close(b)
    df = pd.concat([ca, cb], axis=1).dropna()
    df.columns = ["A", "B"]
    spread = np.log(df["B"]) - np.log(df["A"])             # log スプレッド
    m = spread.rolling(win).mean(); sd = spread.rolling(win).std()
    z = (spread - m) / sd
    # コスト(往復・2脚): 価格比で近似
    cost = (sa / ca.mean()) + (sb / cb.mean())             # 片道2脚ぶん ≈ 往復片側
    rt_cost = 2 * cost
    rb = df["B"].pct_change(); ra = df["A"].pct_change()

    trades = []
    pos = 0; entry_i = None
    zv = z.values; idx = df.index
    for i in range(win + 1, len(df)):
        if pos == 0:
            if zv[i] < -entry:
                pos = 1; entry_i = i          # スプレッド割安→ B ロング / A ショート
            elif zv[i] > entry:
                pos = -1; entry_i = i         # スプレッド割高→ B ショート / A ロング
        elif pos == 1 and zv[i] > -exit:
            ret = (df["B"].iloc[i]/df["B"].iloc[entry_i]-1) - (df["A"].iloc[i]/df["A"].iloc[entry_i]-1) - rt_cost
            trades.append((idx[i].year, ret)); pos = 0
        elif pos == -1 and zv[i] < exit:
            ret = (df["A"].iloc[i]/df["A"].iloc[entry_i]-1) - (df["B"].iloc[i]/df["B"].iloc[entry_i]-1) - rt_cost
            trades.append((idx[i].year, ret)); pos = 0
    return trades


def main():
    print(f"ペアトレード(マーケットニュートラル) {TF}  ペア: {PAIRS}\n")
    for entry, exit in [(2.0, 0.5), (1.5, 0.3), (2.5, 0.5)]:
        alltr = []
        for a, b in PAIRS:
            alltr += pair_trades(a, b, entry=entry, exit=exit)
        df = pd.DataFrame(alltr, columns=["year", "ret"])
        rows = {}
        for y, g in df.groupby("year"):
            gp = g["ret"][g["ret"] > 0].sum(); gl = -g["ret"][g["ret"] < 0].sum()
            rows[int(y)] = {"trades": len(g), "pf": gp/gl if gl > 0 else np.inf, "ret%": g["ret"].sum()*100}
        t = pd.DataFrame(rows).T
        pf = t["pf"].replace(np.inf, np.nan)
        print(f"=== entry={entry} exit={exit} ===")
        print(t.round(2).to_string())
        print(f"  → 毎年プラス {(t['ret%']>0).mean():.0%} / PF中央 {pf.median():.2f} / "
              f"PF最小 {pf.min():.2f} / 年取引 {int(t['trades'].mean())} / 通算 {df['ret'].sum()*100:.0f}%\n")


if __name__ == "__main__":
    main()
