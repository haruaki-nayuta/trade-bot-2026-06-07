"""USDJPY H1 モメンタムの lookback アンサンブル版 = [[usdjpy_momentum]] の精錬(reports/31)。

単一 lookback(=24)でなく複数 lookback の符号投票で状態を決め、平滑化で純Sharpe/安定性を上げる。
チャンピオンとの二口座 robust 合成(w=0.2)で、単一lb24(+1.26pp)から **+0.57pp 上積み**(=+1.83pp)。

検証(reports/31 改善A, 厳密 two-book robust + 5シード + 敵対検証):
  * **採用構成 = lookbacks=(12, 24)**。同一p95=20%較正でモメンタムbook単独CAGR +1.43%(lb24)→ +2.87%([12,24])に上昇、
    チャンピオンとの相関 −0.10 は不変(=両建て・負相関の分散性を保つ)。matched-w=0.20 で robust +0.57pp、
    5シードで +0.57〜+0.98pp(平均+0.81pp)全正、p95=20%維持・経験的maxDD −13.3%(縮小=レバ偽装でない)。
  * **限界(正直に)**: 「アンサンブル化が一律に効く」のではない。長いLBを足すと劣化([12,24,48]=−0.74pp、[24,48,72]=−0.27pp)
    =lookback-set空間は高原でなく、勝つのは短端 (12,24) ペアのみ=config特異。よって「lb24 を [12,24] に差し替える限定レバー」
    として採用し、増し盛り(w↑や本数追加)はしない。効果量は +0.5〜0.8pp で robust だが significance の縁。
  * long-only化(+0.73pp)はドリフトベータ化(USD/JPYと corr+0.73)で性質が変わるため非推奨。vol-target(+0.5pp)はU字感応で限界的。

シグナル(先読みなし・[[usdjpy_momentum]]/tsmom と同型のドテン構造):
  vote = Σ_L [ +1 if mom_L>band ; -1 if mom_L<-band ; else 0 ](mom_L=close/close.shift(L)-1)。
  vote>0 で net long / vote<0 で net short / 反転でドテン。
運用は [[usdjpy_momentum]] と同条件: USDJPY・H1・別ブック w=0.15〜0.25・long-only or ECN 推奨。
"""

from __future__ import annotations

import pandas as pd

# 採用構成=短端2本 (12,24)(reports/31: 唯一 robust に +0.57pp を出す config。長LB追加は劣化)。
PARAMS = {"lookbacks": (12, 24), "band": 0.0}
PARAM_GRID = {"lookbacks": [(12, 24), (12, 24, 48), (24, 48, 72)], "band": [0.0]}


def generate_signals(data: pd.DataFrame, lookbacks=(12, 24), band: float = 0.0):
    close = data["close"]
    vote = pd.Series(0, index=close.index, dtype="int64")
    for L in lookbacks:
        mom = close / close.shift(int(L)) - 1.0
        vote = vote + (mom > band).astype("int64") - (mom < -band).astype("int64")

    long_state = vote > 0
    short_state = vote < 0

    long_entries = long_state & ~long_state.shift(fill_value=False)
    short_entries = short_state & ~short_state.shift(fill_value=False)
    long_exits = short_entries                          # 反対転換でドテン
    short_exits = long_entries
    return long_entries, long_exits, short_entries, short_exits
