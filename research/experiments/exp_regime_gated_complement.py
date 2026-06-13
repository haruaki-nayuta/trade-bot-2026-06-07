"""通貨内トレンドの最後のsteelman: causal(先読みなし)ER-regime-gated 順張り。

Workflowの regime 診断: 順張りは高ER年でnet正・低ER年で大負け(breakout pnl-ER相関0.665)。
だがそれは事後の年分類。本実験は「リアルタイムのラグ付きERゲート(ER(40).shift(1)>=th の時だけ建玉)」で
その net正部分を捕まえられるか=実トレード可能な regime-gated trend が単体黒字になるか実測。

- breakout_trend と tsmom の entries を causal ER ゲートでフィルタ(exitは不変)。
- 7メジャー D1、gross/net、plateau(th × lookback)、IS/OOS。
- 単体黒字かつ頑健なら、2スリーブ robust でチャンピオンCAGRを上げるかも測る(Sharpe壁の確認)。
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from fxlab import load, run, metrics
import fxlab.config as C

MAJORS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
ER_DEFAULT = {k: v for k, v in C.SPREADS_PIPS.items()}
COMM = C.COMMISSION_FRACTION


def causal_er(close: pd.Series, w: int = 40) -> pd.Series:
    direction = (close - close.shift(w)).abs()
    vol = close.diff().abs().rolling(w).sum()
    return (direction / vol).replace([np.inf, -np.inf], np.nan)


def gated_breakout(data, entry=40, exit=20, trend=200, er_win=40, er_th=0.20):
    high, low, close = data["high"], data["low"], data["close"]
    upper = high.rolling(entry).max().shift()
    lower = low.rolling(entry).min().shift()
    exit_lower = low.rolling(exit).min().shift()
    exit_upper = high.rolling(exit).max().shift()
    sma = close.rolling(trend).mean()
    er = causal_er(close, er_win).shift(1)          # ★ 先読みなしのレジームゲート
    trending = er >= er_th
    long_entries = (close > upper) & (close > sma) & trending
    short_entries = (close < lower) & (close < sma) & trending
    long_exits = close < exit_lower
    short_exits = close > exit_upper
    return (long_entries.fillna(False), long_exits.fillna(False),
            short_entries.fillna(False), short_exits.fillna(False))


def gated_tsmom(data, lookback=100, er_win=40, er_th=0.20):
    close = data["close"]
    mom = close / close.shift(lookback) - 1.0
    er = causal_er(close, er_win).shift(1)
    trending = er >= er_th
    long_state = (mom > 0) & trending
    short_state = (mom < 0) & trending
    le = long_state & ~long_state.shift(fill_value=False)
    se = short_state & ~short_state.shift(fill_value=False)
    return le.fillna(False), se.fillna(False), se.fillna(False), le.fillna(False)


def run_set(sigfn, params, gross=False, period=None):
    if gross:
        C.SPREADS_PIPS = {k: 0.0 for k in ER_DEFAULT}
        C.COMMISSION_FRACTION = 0.0
    else:
        C.SPREADS_PIPS = dict(ER_DEFAULT)
        C.COMMISSION_FRACTION = COMM
    rows = []
    for p in MAJORS:
        data = load(p, "D1")
        if period:
            data = data.loc[period[0]:period[1]]
        def _f(x):
            try:
                return float(x)
            except Exception:  # noqa: BLE001
                return float(np.asarray(x).ravel()[0])
        try:
            pf = run(p, "D1", sigfn, params, data=data, size_mode="value", side="both")
            m = metrics(pf)
            rows.append((p, _f(m["total_return"]), _f(m["sharpe"]), _f(m["num_trades"]), _f(m["profit_factor"])))
        except Exception as e:  # noqa: BLE001
            rows.append((p, np.nan, np.nan, 0, np.nan))
    df = pd.DataFrame(rows, columns=["pair", "tr", "sharpe", "ntr", "pf"])
    return df


def summary(df):
    return dict(tr=float(df["tr"].mean()), sharpe=float(df["sharpe"].mean()),
                pos=int((df["tr"] > 0).sum()), ntr=int(df["ntr"].mean()))


def main():
    C.SPREADS_PIPS = dict(ER_DEFAULT)
    print("=== causal ER-gated 順張り(7メジャー D1)===\n")
    for name, sigfn, base in [("gated_breakout", gated_breakout, dict(entry=40, exit=20, trend=200)),
                              ("gated_tsmom", gated_tsmom, dict(lookback=100))]:
        print(f"--- {name} ---")
        print(f"{'er_th':>6} {'GROSS_tr':>9} {'GROSS_Sh':>9} {'NET_tr':>8} {'NET_Sh':>8} {'pos':>4} {'ntr':>5}")
        for th in (0.0, 0.15, 0.20, 0.25, 0.30):
            g = summary(run_set(sigfn, {**base, "er_th": th}, gross=True))
            n = summary(run_set(sigfn, {**base, "er_th": th}, gross=False))
            print(f"{th:>6.2f} {g['tr']:>9.3f} {g['sharpe']:>9.3f} {n['tr']:>8.3f} {n['sharpe']:>8.3f} {n['pos']:>4} {n['ntr']:>5}")
        # IS/OOS at a representative th=0.20
        is_ = summary(run_set(sigfn, {**base, "er_th": 0.20}, gross=False, period=("2016-01-01", "2021-12-31")))
        oos = summary(run_set(sigfn, {**base, "er_th": 0.20}, gross=False, period=("2022-01-01", "2026-12-31")))
        print(f"  th=0.20  IS net_tr={is_['tr']:.3f}(pos{is_['pos']}) / OOS net_tr={oos['tr']:.3f}(pos{oos['pos']})\n")
    print("判定: GROSS_Sh>0 が広いer_thで成立 & NET黒字ペア>=5 & IS/OOS両プラス なら通貨内regime-gated順張りは実在。")
    print("      GROSSが依然負なら=高ERゲートでも生エッジは正に届かず=通貨内トレンドの最後の石も閉じる。")
    C.SPREADS_PIPS = dict(ER_DEFAULT)  # restore


if __name__ == "__main__":
    main()
