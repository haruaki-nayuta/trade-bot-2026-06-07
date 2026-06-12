"""ビル・ウィリアムズ『Trading Chaos』コアシステム: アリゲーター×フラクタル・ブレイクアウト。

- アリゲーター: SMMA(中値) jaw=13(+8シフト) / teeth=8(+5) / lips=5(+3)。
  チャート上のシフトは「過去に計算した値を現在に表示」なので shift(n) で再現(先読みなし)。
- フラクタル: 5本パターンの中央バー。確定には右側2本が必要なので shift(2) で確定後にのみ参照。
- エントリー(第3の賢者): 直近の確定上フラクタルを終値が上抜け、かつフラクタルが teeth より上。
- イグジット: 終値がアリゲーターの線(exit_line)を逆方向にクロス。ショートは対称。

scale はアリゲーター周期/シフトとフラクタル有効性をまとめて伸縮(1=書籍の 13/8/5)。
exit_line: 0=lips, 1=teeth, 2=jaw。align=1 で「口が開いている」(lips>teeth>jaw)時のみ入る。
"""

from __future__ import annotations

import pandas as pd

PARAMS = {"scale": 1.0, "exit_line": 1, "align": 1}
PARAM_GRID = {"scale": [1.0, 2.0, 3.0, 4.0, 5.0], "exit_line": [0, 1, 2], "align": [0, 1]}


def _smma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def generate_signals(data: pd.DataFrame, scale: float = 1.0, exit_line: int = 1, align: int = 1):
    high, low, close = data["high"], data["low"], data["close"]
    median = (high + low) / 2.0

    jaw_n, teeth_n, lips_n = (max(2, round(13 * scale)), max(2, round(8 * scale)), max(2, round(5 * scale)))
    jaw_s, teeth_s, lips_s = (max(1, round(8 * scale)), max(1, round(5 * scale)), max(1, round(3 * scale)))
    jaw = _smma(median, jaw_n).shift(jaw_s)
    teeth = _smma(median, teeth_n).shift(teeth_s)
    lips = _smma(median, lips_n).shift(lips_s)
    lines = {0: lips, 1: teeth, 2: jaw}
    ex = lines[int(exit_line)]

    # 確定フラクタル(中央バーは2本前。右側2本の確定を待つ=先読みなし)
    h2, l2 = high.shift(2), low.shift(2)
    up_fr = (h2 > high.shift(4)) & (h2 > high.shift(3)) & (h2 > high.shift(1)) & (h2 > high)
    dn_fr = (l2 < low.shift(4)) & (l2 < low.shift(3)) & (l2 < low.shift(1)) & (l2 < low)
    up_level = h2.where(up_fr).ffill()
    dn_level = l2.where(dn_fr).ffill()

    # フラクタルの有効性: 口の外側(買いは teeth より上)のみ
    valid_up = up_level > teeth
    valid_dn = dn_level < teeth

    long_entries = (close > up_level) & (close.shift() <= up_level.shift()) & valid_up
    short_entries = (close < dn_level) & (close.shift() >= dn_level.shift()) & valid_dn

    if align:
        long_entries &= (lips > teeth) & (teeth > jaw)
        short_entries &= (lips < teeth) & (teeth < jaw)

    long_exits = (close < ex) & (close.shift() >= ex.shift())
    short_exits = (close > ex) & (close.shift() <= ex.shift())

    return (
        long_entries.fillna(False),
        long_exits.fillna(False),
        short_entries.fillna(False),
        short_exits.fillna(False),
    )
