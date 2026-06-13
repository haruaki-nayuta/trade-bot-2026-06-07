"""セッション報告 +0.5 の最有力解釈 = 「モメンタム符号 × 次バーリターン」の
連続ポジション Sharpe (経路: position = sign(past_lookback_return), 毎バー保有)を
7ペアで測り、UTC20-23の保有を除外して除染する。

これはポートフォリオエンジン(ドテン/コスト)を通さない純粋な符号エッジ測定で、
tsmom系の "GROSS Sharpe ~+0.5" に最も近い。除染後も7ペア平均で正かを見る。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import load

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
CONFIGS = [("H1", 24), ("M30", 48), ("M15", 96)]
BAN_HOURS = {20, 21, 22, 23}


def bars_per_year(tf: str) -> float:
    # FX ~24h, ~252 trading days, ~5/7 calendar
    per_day = {"H1": 24, "M30": 48, "M15": 96}[tf]
    return per_day * 252 * (5 / 7) * (7 / 5)  # 約 per_day*252... 単純に年率係数


def annualization(tf: str) -> float:
    per_day = {"H1": 24, "M30": 48, "M15": 96}[tf]
    # おおむね年間営業日260, 24h市場
    return np.sqrt(per_day * 260)


def signed_sharpe(data: pd.DataFrame, lookback: int, tf: str, ban: bool):
    close = data["close"]
    ret = close.pct_change()
    mom = close / close.shift(lookback) - 1.0
    pos = np.sign(mom).shift(1)  # 確定したモメンタム符号で次バーを保有(先読みなし)
    strat = pos * ret
    if ban:
        hours = data.index.hour
        mask = pd.Series([h in BAN_HOURS for h in hours], index=data.index)
        strat = strat[~mask]
    strat = strat.dropna()
    mu = strat.mean()
    sd = strat.std()
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(mu / sd * annualization(tf))


def main():
    rows = []
    for tf, lb in CONFIGS:
        for pair in PAIRS:
            data = load(pair, tf)
            raw = signed_sharpe(data, lb, tf, ban=False)
            clean = signed_sharpe(data, lb, tf, ban=True)
            rows.append({"tf": tf, "lb": lb, "pair": pair, "raw": raw, "clean": clean})
            print(f"{tf} lb{lb} {pair}: raw={raw:+.4f} clean={clean:+.4f}")

    df = pd.DataFrame(rows)
    print("\n===== signed-return Sharpe per-config (7-pair mean) =====")
    for tf, lb in CONFIGS:
        sub = df[(df.tf == tf) & (df.lb == lb)]
        print(
            f"{tf} lb{lb}: raw_mean={sub.raw.mean():+.4f} clean_mean={sub.clean.mean():+.4f} "
            f"(raw_pos {int((sub.raw>0).sum())}/7, clean_pos {int((sub.clean>0).sum())}/7)"
        )


if __name__ == "__main__":
    main()
