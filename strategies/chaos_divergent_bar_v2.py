"""ビル・ウィリアムズ ダイバージェントバー v2 — グリッド端の外側を探索する版。

chaos_divergent_bar と同一ロジック。直前の evaluate で最適パラメータが
lookback=10 / window=5 / gate=1 とすべてグリッド上限に張り付いたため、
PARAM_GRID をその方向(外側)へ拡張しただけの変種。主戦場は H4
(唯一 Sharpe がプラスだった時間足)で再評価する。

強気ダイバージェントバー = 直近 lookback 本の最安値を付け、かつバーの上半分で引け、
かつアリゲーターの口の外(下)で出現したバー。その高値を window 本以内に上抜けたら買い。
弱気は対称。手仕舞いは終値が teeth を逆方向にクロス。

gate=0: バーがアリゲーター3線すべての外側(書籍に忠実) / gate=1: lips の外側のみ(緩め)。
"""

from __future__ import annotations

import pandas as pd

PARAMS = {"lookback": 10, "window": 5, "gate": 1}
PARAM_GRID = {"lookback": [10, 15, 20, 30], "window": [5, 8, 12], "gate": [0, 1]}


def _smma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def generate_signals(data: pd.DataFrame, lookback: int = 10, window: int = 5, gate: int = 1):
    high, low, close = data["high"], data["low"], data["close"]
    median = (high + low) / 2.0

    jaw = _smma(median, 13).shift(8)
    teeth = _smma(median, 8).shift(5)
    lips = _smma(median, 5).shift(3)
    below = (low < lips) if gate else (low < lips) & (low < teeth) & (low < jaw)
    above = (high > lips) if gate else (high > lips) & (high > teeth) & (high > jaw)

    mid = (high + low) / 2.0
    bull_db = (low == low.rolling(lookback).min()) & (close > mid) & below
    bear_db = (high == high.rolling(lookback).max()) & (close < mid) & above

    # ダイバージェントバーの高値/安値を window 本だけ有効なトリガー水準として保持
    buy_level = high.where(bull_db).ffill(limit=window)
    sell_level = low.where(bear_db).ffill(limit=window)

    long_entries = (close > buy_level) & (close.shift() <= buy_level.shift())
    short_entries = (close < sell_level) & (close.shift() >= sell_level.shift())

    long_exits = (close < teeth) & (close.shift() >= teeth.shift())
    short_exits = (close > teeth) & (close.shift() <= teeth.shift())

    return (
        long_entries.fillna(False),
        long_exits.fillna(False),
        short_entries.fillna(False),
        short_exits.fillna(False),
    )
