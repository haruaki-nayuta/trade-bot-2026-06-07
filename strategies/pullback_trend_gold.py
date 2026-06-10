"""金(XAUUSD) H4 トレンド中の押し目買い(トレンドゲート×押し目の再上抜けエントリー)。

発想:
  金は10年スパンの強い上昇トレンド(2016/6 $1272 → 2026/6 $4198)があり、
  素の tsmom(H4 lb360)でも PF1.79。ただしドテン型は取引数が少なく押しを拾えない。
  本戦略は「押し目の“底”ではなく“終わり”で入る」型のトレンドフォロー:

    * トレンドゲート: close > EMA(lb)(長期トレンドの向き)。
    * 押し目の確認: 直近10本の RSI(5) 最小値 < dip_rsi(実際に売られた局面があった)。
    * エントリー: ゲート中に close が EMA(ema_f) を下から上へ再クロス(押し目の終了)。
    * 手仕舞い: close が EMA(ema_f) を下抜け、またはゲート消滅。
  → 押し目“中”に入る型(RSI<15で買い・RSI>70で売り)は IS PF≒1.0 で棄却。
    再上抜け型は勝率 ~24% × 損小利大の正のスキューでトレンドレッグを丸ごと取る。

  ショート側は対称実装を残すが、金は IS 単独でショート PF0.71-0.89 と構造的に弱く、
  ロング専用(side='long' 相当)を採用。本ファイルはロング側のみ返す。

先読みなしの根拠:
  * EMA・RSI とも確定バーの終値のみで計算。
  * rolling(10).min() は自バーを含むが RSI(5) は確定値なので未来参照なし。
  * クロス判定は当該バーと前バーの確定値の比較のみ。

検証(10年 XAUUSD H4, IS70%/OOS30%, side=long):
  best = lb=240, ema_f=50, dip_rsi=30(IS sharpe 基準、近傍は滑らかな高原)
  IS:  +33.4% / PF1.58 / Sharpe0.69 / DD-10.5% / 27.9取引/年
  OOS: +73.1% / PF2.63 / Sharpe1.78 / DD-13.9% / 34.0取引/年(崩壊なし)
  全期間: +130.9% / PF2.11 / Sharpe1.11 / DD-13.9% / 29.7取引/年
  対称ショートを足すと全期間 PF1.49 / Sharpe0.81 に劣化(ショート単体 PF0.82)。
  7ペア横断では FX には移植不可(金の長期強気構造に依存)。時間足は H4 が最良。

敵対検証(2026-06-11 Workflow 3レンズ、reports/11 参照)で金トレンド候補中唯一の生存:
  * 過剰最適化: 拡張216組合せ全てSharpe>0の広い高原・グリッド端なし・WF 5/5プラス
    (fold別 +1.4/+16.8/+14.0/+17.2/+9.2%)。前向き期待のアンカーは Sharpe≈0.9-1.1。
  * コスト/実行: スプレッド2倍+手数料でも IS PF1.43/OOS PF2.52。1バー遅延で劣化なし
    (ISはむしろ改善=エントリータイミング依存なし)。risk1%サイジングで CAGR6.1%/DD-10.5%。
    週末ギャップは正のスキュー(ロングに有利、最悪-0.87%)。バー安値ベースDD監査も問題なし。
  * レジーム依存(残る正直な限界): 利益の~78%は2023-26の金暴騰期。暴騰前2016-2022単独
    でも +30.0%/PF1.56/Sharpe0.68 で正、金B&Hマイナスの2021-22にPF3.7/2.9の真のα。
    ただし「金の長期強気が崩れたら年率4%水準に低下」というマクロ前提込みで運用すること。
  * 出口オーバーレイの注意: TSL2%(full +156.5%)は全期間知識を使った選択でIS規律では
    選ばれない。IS-Sharpe規律の正直な選択は SL2%+TP4%(full +89.7%/PF1.79/DD-8.7%)で、
    リターン最大化なら素のまま(オーバーレイ無し)が正解。
"""

from __future__ import annotations

import pandas as pd
import vectorbt as vbt

# 単発検証で使うデフォルトパラメータ(IS 最適値)
PARAMS = {"lb": 240, "ema_f": 50, "dip_rsi": 30}

# sweep(パラメータ探索)で使う範囲: 3*3*3 = 27 組合せ
PARAM_GRID = {
    "lb": [180, 240, 360],      # トレンドゲート EMA(H4本数: 約30/40/60日)
    "ema_f": [40, 50, 65],      # 押し目判定の速い EMA
    "dip_rsi": [25, 30, 35],    # 押し目の深さ(直近10本の RSI(5) 最小 < dip_rsi)
}


def generate_signals(data: pd.DataFrame, lb: int = 240, ema_f: int = 50,
                     dip_rsi: float = 30):
    close = data["close"]

    # 長期トレンドゲート
    ema_slow = close.ewm(span=lb, adjust=False).mean()
    gate_up = (close > ema_slow).fillna(False)

    # 押し目の終わり = 速いEMAの再上抜け
    ema = close.ewm(span=ema_f, adjust=False).mean()
    above = (close > ema).fillna(False)
    cross_up = above & ~above.shift(fill_value=False)
    cross_dn = ~above & above.shift(fill_value=False)

    # 押し目の深さ確認(直近10本で短期RSIが実際に下がったか)
    rsi5 = vbt.RSI.run(close, 5).rsi
    dip_ok = (rsi5.rolling(10).min() < dip_rsi).fillna(False)

    long_entries = gate_up & cross_up & dip_ok
    long_exits = cross_dn | ~gate_up

    # ショートは金では構造的に弱い(IS PF0.71-0.89)ため無効(ロング専用)
    return long_entries, long_exits
