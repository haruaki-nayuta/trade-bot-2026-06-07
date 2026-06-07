"""ベスト/ワーストのトレード抽出 + 各トレード直前の値動き(コンテキスト)分析。

「なぜ勝てた/負けたか」をパターン研究するための材料を取り出す。
  * trade_table(pf)        : 全トレードを整形(entry/exit, dir, pnl, return, 保有期間)
  * pre_features(...)       : エントリー直前 N 本から特徴量(モメンタム/ボラ/RSI/ATR/レンジ位置)
  * trade_context(...)      : 直前 N 本 + 建玉中 のOHLCV(phase列付き。値動きそのもの)
  * analyze(pf, data, ...)  : ベスト/ワースト n 件 + 特徴量 + 値動きを一括抽出
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt

from . import config


# --- 指標(全データに対して一度だけ計算) -----------------------------
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    return vbt.RSI.run(close, period).rsi


def _atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift()
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean() / c * 100


# --- トレード一覧 --------------------------------------------------------
def trade_table(pf: "vbt.Portfolio", data: pd.DataFrame) -> pd.DataFrame:
    """records_readable を分析しやすい形に整形。"""
    tr = pf.trades.records_readable
    out = pd.DataFrame(
        {
            "entry": tr["Entry Timestamp"],
            "exit": tr["Exit Timestamp"],
            "dir": tr["Direction"],
            "entry_price": tr["Avg Entry Price"],
            "exit_price": tr["Avg Exit Price"],
            "pnl": tr["PnL"],
            "return_pct": tr["Return"] * 100,
        }
    )
    idx = data.index
    out["bars_held"] = idx.get_indexer(out["exit"]) - idx.get_indexer(out["entry"])
    out["hours"] = (out["exit"] - out["entry"]).dt.total_seconds() / 3600
    return out


# --- トレード直前の特徴量 -----------------------------------------------
def pre_features(
    data: pd.DataFrame, entry_ts, lookback: int, rsi: pd.Series, atr_pct: pd.Series
) -> dict:
    """エントリー直前 lookback 本(エントリーバーは含めない)の値動きを数値化。"""
    idx = data.index
    pos = idx.get_loc(entry_ts)
    start = max(0, pos - lookback)
    win = data.iloc[start:pos]
    ce = float(data["close"].iloc[pos])  # エントリーバーの終値

    if len(win) >= 2:
        c = win["close"]
        rets = c.pct_change().dropna()
        x = np.arange(len(c))
        slope = float(np.polyfit(x, c.values / c.values[0], 1)[0] * 100)  # %/本
        feat = {
            "pre_ret_%": round((c.iloc[-1] / c.iloc[0] - 1) * 100, 3),   # 直前の累積変化
            "pre_vol_%": round(float(rets.std() * 100), 4),             # 1本ボラ
            "pre_trend_%/bar": round(slope, 5),                        # 傾き
            "up_bar_ratio": round(float((rets > 0).mean()), 3),         # 陽線率
            "dist_from_high_%": round((ce / win["high"].max() - 1) * 100, 3),
            "dist_from_low_%": round((ce / win["low"].min() - 1) * 100, 3),
        }
    else:
        feat = {k: np.nan for k in
                ("pre_ret_%", "pre_vol_%", "pre_trend_%/bar", "up_bar_ratio",
                 "dist_from_high_%", "dist_from_low_%")}

    feat["rsi_at_entry"] = round(float(rsi.iloc[pos]), 1) if pd.notna(rsi.iloc[pos]) else np.nan
    feat["atr_at_entry_%"] = round(float(atr_pct.iloc[pos]), 4) if pd.notna(atr_pct.iloc[pos]) else np.nan
    return feat


# --- 値動きそのもの(直前＋建玉中のOHLCV) -----------------------------
def trade_context(data: pd.DataFrame, entry_ts, exit_ts, lookback: int) -> pd.DataFrame:
    """直前 lookback 本(phase=pre)＋ 建玉中(phase=trade)の OHLCV を返す。"""
    idx = data.index
    e = idx.get_loc(entry_ts)
    x = idx.get_loc(exit_ts)
    start = max(0, e - lookback)
    seg = data.iloc[start : x + 1].copy()
    seg["phase"] = ["pre"] * (e - start) + ["trade"] * (x + 1 - e)
    return seg


# --- 一括抽出 ------------------------------------------------------------
def analyze(
    pf: "vbt.Portfolio",
    data: pd.DataFrame,
    *,
    n: int = 5,
    lookback: int = 50,
    by: str = "return_pct",
) -> dict:
    """ベスト/ワースト n 件のトレードと、その直前の値動き・特徴量を抽出。

    戻り値 dict:
      all      : 全トレード + 特徴量(DataFrame)
      best/worst: 上位/下位 n 件(DataFrame)
      contexts : {tag: 値動きOHLCV(DataFrame)}  tag 例 'best1','worst1'
    """
    rsi = _rsi(data["close"])
    atr = _atr_pct(data)

    tbl = trade_table(pf, data)
    feats = pd.DataFrame([pre_features(data, t, lookback, rsi, atr) for t in tbl["entry"]])
    full = pd.concat([tbl.reset_index(drop=True), feats], axis=1)

    ranked = full.sort_values(by, ascending=False)
    best = ranked.head(n).reset_index(drop=True)
    worst = ranked.tail(n).iloc[::-1].reset_index(drop=True)

    contexts = {}
    for tag, frame in (("best", best), ("worst", worst)):
        for i, row in frame.iterrows():
            contexts[f"{tag}{i+1}"] = trade_context(data, row["entry"], row["exit"], lookback)

    return {"all": full, "best": best, "worst": worst, "contexts": contexts,
            "n": n, "lookback": lookback, "by": by}
