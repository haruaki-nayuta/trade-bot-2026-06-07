"""金(XAUUSD) H4 EMAクロス・ドテン+トレンド品質ゲート(ER OR ADX)。

発想:
  金は10年スパンの構造的強気(2016/6 $1272 → 2026/6 $4198)でトレンドフォロー適性が高いが、
  素のEMAクロス・ドテンはレンジ(チョップ)局面の往復ビンタで利益を吐き出す。そこで
  平均回帰チャンピオン(confluence_meanrev_v2)が「ER低=レンジでだけ入る」のに使っている
  Kaufman効率比を**逆向き**に使い、「トレンド品質が高いときだけ新規エントリー」する。

ルール:
  * レジーム: EMA(fast) > EMA(slow) → ロング地合い / 逆ならショート地合い(ドテン)。
  * 品質ゲート: ER(er_win) > er_min または ADX(adx_p) > adx_min のときだけ新規建玉。
      - ER = |Δn本| / Σ|Δ1本| ∈ [0,1]。高い=一直線のトレンド。
      - ゲートは**入口専用**。出口は常に反対クロス(ゲート消失や fast EMA 割れで切る変種は
        IS検証で全滅: gate-hold PF0.83 / fastEMAセグメント PF0.91 → 棄却)。
      - ゲートOFF中にクロスした場合は、同一レジーム継続中にゲートONになった時点で遅れて入る
        (reentry)。フリッカーによる多重建玉は from_signals が無視するので安全。
  * fast/slow は当初案 {20,30,50}×{100,150,200} だと取引数が構造的に 12-17回/年しか出ない
    (ゲートがレジームあたり最大1トレードに制限するため)。頻度制約(>=24回/年)を満たしつつ
    ISエッジが残る領域を IS のみで探索した結果、高速ドテン 10/25 + 複合ゲートを採用。
    近傍 (8-12)/(25-30) は滑らかにプラス=単一点の偶然ではない。

先読みなしの根拠:
  EMA・ER・ADX とも確定バーまでの値のみ(ewm/rolling/ta はすべて過去方向)。シフト不要の
  状態比較(EMA同士の大小)と「状態が新たに真になった」エッジ検出だけで構成。

IS(2016-06〜2023-06, 70%)実測の要点:
  * ゲート無し 10/25: Sharpe 0.252 / PF 1.063 / 60回/年 → ゲート付き: Sharpe 0.299 /
    PF 1.106 / DD -19%(無しは -31%)/ 34.6回/年。ゲートがDDと質を同時に改善。
  * ER単独(>0.25)は質最良(PF 1.150)だが 22.4回/年で頻度不足。ADX をORで足して頻度を確保。
  * グリッド端チェック済み: slow=15/20、er_min=0.20 はいずれも IS で明確に劣化
    → slow=25 / er_min=0.25 は実質的な内部最適(端の偶然ではない)。

最終成績(2016-2026, XAUUSD H4, size=full):
  * フル: +85.6% / PF 1.324 / Sharpe 0.60 / DD -19.4% / 33.4回/年
  * OOS(2023-06以降, 素): +57.0% / PF 1.715 / Sharpe 1.12 / 30.7回/年
  * ロング PF 1.76 / ショート PF 0.92(金の構造的強気。ショートは保険的ドラグだが
    外すと頻度制約割れ+ドテン構造が崩れるため維持)
  * 注意: 7 FXメジャーへ同一パラメータ適用は全敗 → 金専用(金のトレンド構造に依存)。
  * 推奨オーバーレイ: run(..., sl_stop=0.02, tp_stop=0.04)。IS と OOS の両方で独立に改善
    (IS Sharpe 0.30→0.44 / OOS 1.12→1.82, フル +124.7% / PF 1.45 / DD -14.9%)
    = 全期間チューニングではない。TSL2% は IS で悪化するため不採用。
"""

from __future__ import annotations

import pandas as pd

# 採用値(IS 2016-2023 のみで決定)
PARAMS = {"fast": 10, "slow": 25, "er_win": 40, "er_min": 0.25,
          "adx_p": 14, "adx_min": 25.0, "gate": "er_or_adx"}

# sweep 探索範囲(高原チェック用、18通り)
PARAM_GRID = {
    "fast": [8, 10, 12],
    "slow": [25, 30, 35],
    "er_min": [0.25, 0.30],
}


def _er(close: pd.Series, win: int) -> pd.Series:
    """Kaufman 効率比(過去 win 本)。確定値のみ使用。"""
    delta = (close - close.shift(win)).abs()
    vol = close.diff().abs().rolling(win).sum()
    return delta / vol


def _adx(data: pd.DataFrame, p: int) -> pd.Series:
    import ta
    return ta.trend.ADXIndicator(data["high"], data["low"], data["close"], window=p).adx()


def generate_signals(data: pd.DataFrame, fast: int = 10, slow: int = 25,
                     er_win: int = 40, er_min: float = 0.25,
                     adx_p: int = 14, adx_min: float = 25.0,
                     gate: str = "er_or_adx"):
    close = data["close"]
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    state_long = ema_f > ema_s
    cross_up = state_long & ~state_long.shift(fill_value=False)
    cross_dn = ~state_long & state_long.shift(fill_value=True)

    if gate == "none":
        g = pd.Series(True, index=close.index)
    elif gate == "er":
        g = _er(close, er_win) > er_min
    elif gate == "adx":
        g = _adx(data, adx_p) > adx_min
    elif gate == "er_or_adx":
        g = (_er(close, er_win) > er_min) | (_adx(data, adx_p) > adx_min)
    else:
        raise ValueError(f"unknown gate: {gate}")
    g = g.fillna(False)

    # 「レジーム継続中かつゲートON」が新たに成立した時点で建玉(クロス時にゲートOFFなら遅入り)
    ls = state_long & g
    ss = (~state_long) & g
    long_entries = ls & ~ls.shift(fill_value=False)
    short_entries = ss & ~ss.shift(fill_value=False)
    long_exits = cross_dn   # 出口は常に反対クロス(ドテン)
    short_exits = cross_up
    return long_entries, long_exits, short_entries, short_exits
