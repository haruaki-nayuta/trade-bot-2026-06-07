"""金(XAUUSD)向けドンチャン・ブレイクアウト+シャンデリア(ATRトレーリング)出口。

発想:
  * 金は 2016-2026 で $1272 → $4198 の長期強気であり、時系列モメンタムが
    機能することが既知(素の tsmom D1 lb60: PF2.0 級)。tsmom はポジションが
    切れにくいのに対し、本戦略は「N本極値ブレイクで入り、ATRトレールで出る」
    切れるトレンドフォローにして取引数を確保する(タートルズ+シャンデリア)。
  * エントリー: 終値が直前 N 本の高値を上抜け→ロング / 直前 N 本の安値を
    下抜け→ショート。
  * 出口: シャンデリア・イグジットのステートレス近似。
    ロング: 終値 < rolling高値極値(trail_n本) - ATR(atr_period)×k で手仕舞い。
    ショート: 終値 > rolling安値極値(trail_n本) + ATR×k で手仕舞い。
    trail_n=0 はエントリー窓 N と同じ窓を使う(保有期間 < N 本なら
    「エントリー以降の極値」とほぼ一致する近似)。trail_n=22 は古典的
    シャンデリアの窓。

先読み(look-ahead)なしの根拠:
  * rolling 極値はすべて .shift() で自バーを除外(直前バーまでの確定値のみ)。
  * ATR は確定バーの OHLC のみから計算され、判定は確定バーの終値で行う。
  * 未確定バーの値は一切使わない。約定は vectorbt がシグナルバー終値
    (スリッページ込み)で行う。
"""

from __future__ import annotations

import pandas as pd
import vectorbt as vbt

# 採用値(IS 70% sweep で決定。trades/yr>=24 制約を満たす中で IS sharpe 最大)
# 参考: 長期 n=240/k=4 のロング専用は PF2.6/+65% と強いが年5取引で制約外
PARAMS = {"n": 60, "k": 3.0, "trail_n": 22, "atr_period": 14}

# sweep 探索範囲(4×3×2 = 24 組合せ)
PARAM_GRID = {
    "n": [60, 120, 180, 240],
    "k": [2.0, 3.0, 4.0],
    "trail_n": [0, 22],
}


def generate_signals(
    data: pd.DataFrame,
    n: int = 60,
    k: float = 3.0,
    trail_n: int = 22,
    atr_period: int = 14,
):
    high, low, close = data["high"], data["low"], data["close"]
    tn = int(trail_n) if trail_n else int(n)

    # エントリー用ドンチャン(直前バーまでの極値。shift で先読み回避)
    upper = high.rolling(int(n)).max().shift()
    lower = low.rolling(int(n)).min().shift()

    # シャンデリア出口(トレール極値も shift で自バー除外)
    atr = vbt.ATR.run(high, low, close, int(atr_period)).atr
    trail_high = high.rolling(tn).max().shift()
    trail_low = low.rolling(tn).min().shift()
    long_stop = trail_high - atr * float(k)
    short_stop = trail_low + atr * float(k)

    long_entries = close > upper
    short_entries = close < lower
    long_exits = close < long_stop
    short_exits = close > short_stop
    return long_entries, long_exits, short_entries, short_exits
