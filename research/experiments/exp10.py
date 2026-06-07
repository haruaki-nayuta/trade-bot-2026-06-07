"""イテレーション10: ユニバース全面拡張(全28対象)× 高選別で PF2.0 を狙う。

仮説: 対象を 7メジャー+全21クロス=28 に広げれば、エントリーを厳選して 1対象あたり PF を
上げても、本数が増える分ポート合算の取引数 100+ を維持できる。分散拡大は非カーブフィット。
8通貨(USD,EUR,GBP,AUD,NZD,JPY,CHF,CAD)から作れる非USDクロス全21本を合成して評価する。

目標: 毎年プラス100% かつ 年100取引以上 を満たしつつ、ポート集計 PF を最大化(2.0 到達可否)。
実行: uv run python exp10.py
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from fxlab import config
from fxlab import yearly as ylib
from fxlab.data import load
from strategies.confluence_meanrev import generate_signals as g

pd.set_option("display.width", 200)
TF = "H4"

# 各非USD通貨の「USD建て」close を作る(JPY/CHF/CAD は逆数)
def xusd_series():
    return {
        "EUR": load("EURUSD", TF)["close"],
        "GBP": load("GBPUSD", TF)["close"],
        "AUD": load("AUDUSD", TF)["close"],
        "NZD": load("NZDUSD", TF)["close"],
        "JPY": 1.0 / load("USDJPY", TF)["close"],
        "CHF": 1.0 / load("USDCHF", TF)["close"],
        "CAD": 1.0 / load("USDCAD", TF)["close"],
    }


def build_universe():
    """7メジャー(実close) + 全21クロス(合成close)。name->close。"""
    majors = {p: load(p, TF)["close"] for p in
              ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]}
    xu = xusd_series()
    curs = ["EUR", "GBP", "AUD", "NZD", "JPY", "CHF", "CAD"]
    crosses = {}
    for a, b in itertools.combinations(curs, 2):
        df = pd.concat([xu[a], xu[b]], axis=1).dropna()
        crosses[a + b] = df.iloc[:, 0] / df.iloc[:, 1]
    return majors, crosses


def synth(close):
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                         "volume": 1.0}, index=close.index)


def portfolio(tables):
    accum = {}
    for t in tables.values():
        for yr, r in t.iterrows():
            a = accum.setdefault(int(yr), [0.0, 0.0, 0.0, 0.0, 0, 0])
            a[0] += r["gross_profit"]; a[1] += r["gross_loss"]; a[2] += r["pnl"]
            a[3] += r["trades"]; a[4] += int(r["pnl"] > 0); a[5] += 1
    rows = {yr: {"trades": int(v[3]), "pf": v[0]/v[1] if v[1] > 0 else np.inf,
                 "pnl": v[2], "pos": f"{v[4]}/{v[5]}"} for yr, v in sorted(accum.items())}
    return pd.DataFrame(rows).T


def eval_cfg(majors, crosses, exclude, **p):
    base = dict(window=50, entry_z=2.0, exit_z=0.5, rsi_p=14, rsi_low=35, rsi_high=65,
                vol_win=100, vol_pct=0.70, slow_win=250, slow_z=1.75)
    base.update(p)
    tabs = {}
    for nm, cl in {**majors, **crosses}.items():
        if nm in exclude:
            continue
        data = load(nm, TF) if nm in majors else synth(cl)
        y = ylib.yearly(nm, TF, g, base, data=data, size_mode="value")
        if not y.empty:
            tabs[nm] = y
    port = portfolio(tabs)
    pf = port["pf"].replace(np.inf, np.nan)
    return (port["pnl"] > 0).mean(), pf.median(), pf.min(), int(port["trades"].mean()), len(tabs), port


def main():
    majors, crosses = build_universe()
    # 全クロスのスプレッドを厳しめ 3pips に
    for nm in crosses:
        config.SPREADS_PIPS[nm] = 3.0
    # トレンド性が強く平均回帰に不適な対象(キャリー系)は除外候補
    exclude = {"AUDJPY", "NZDJPY", "CADJPY", "AUDCHF"}

    print(f"全{7+len(crosses)}対象(メジャー7+クロス{len(crosses)})  除外: {exclude}\n")
    print(f"{'cfg':28} {'対象':>4} {'毎年+':>6} {'PF中央':>7} {'PF最小':>7} {'年取引':>7}")
    grid = [
        dict(),
        dict(slow_z=2.0),
        dict(entry_z=2.25, slow_z=2.0),
        dict(entry_z=2.5, slow_z=2.0),
        dict(entry_z=2.25, slow_z=1.75),
        dict(entry_z=2.5, slow_z=2.25),
    ]
    best = None
    for p in grid:
        pos, med, mn, tr, n, port = eval_cfg(majors, crosses, exclude, **p)
        label = ",".join(f"{k}={v}" for k, v in p.items()) or "(default)"
        flag = "★" if (pos == 1.0 and tr >= 100 and med >= 2.0) else (" ✓100%" if pos == 1.0 and tr >= 100 else "")
        print(f"{label:28} {n:>4} {pos:>6.0%} {med:>7.2f} {mn:>7.2f} {tr:>7d}{flag}")
        if pos == 1.0 and tr >= 100 and (best is None or med > best[1]):
            best = (label, med, mn, tr, port)
    if best:
        print(f"\n=== 100%プラス年&年100取引 で最高PF: {best[0]} (PF中央 {best[1]:.2f}) ===")
        b = best[4].copy(); b["pf"] = b["pf"].replace(np.inf, np.nan).round(2); b["pnl"] = b["pnl"].round(0)
        print(b.to_string())


if __name__ == "__main__":
    main()
