"""USDJPY 短期(H1)タイムシリーズ・モメンタム = チャンピオンを補う検証済みの相補スリーブ。

― 何のための戦略か ―――――――――――――――――――――――――――――――――――――
チャンピオン `confluence_meanrev_v2`(H4・平均回帰・無ストップ)は「中庸ボラ×持続USDトレンド」で
失血する(reports/15)。その失血窓を**順張りで稼ぐ**補完が理論上の理想だが、FXメジャーの順張りは
ほぼ全滅(reports/30: tsmom/breakout/adx は7ペア横断で GROSS Sharpe も負、CTA標準実装でも復活せず)。
**唯一の例外が USDJPY の短期(H1)~1日モメンタム**で、これは全ての敵対検証を生き残った
(reports/31)。本ファイルはその確定版=チャンピオンと二口座で回す相補スリーブ。

― 検証サマリ(reports/31, Workflow 2本=14エージェント)――――――――――――――――――――
  シグナル: 過去 lookback(=24本≈1日)リターンの符号で順張り、反転でドテン(=strategies/tsmom と同型)。
  * GROSS Sharpe: FULL +0.78 / IS(2016-2020) +0.34 / OOS(2021-2026) +1.23 = IS/OOS両期プラス。
    lookback{12,24,36,48}×band{0,.001,.002} の 21/21 セルで GROSS 正 = 単一セルのまぐれでない高原。
  * drift-beta 棄却(最重要): USD/JPY の +3.87%/年 上昇ドリフトを信号・評価の両方から完全除去しても
    Sharpe +0.662→+0.604(−9%のみ)。除去後 long+0.44/short+0.42 とほぼ対称・両側個別有意(t≈1.3-1.4)。
    年次PnLとbuy&hold方向の符号一致 6/11・corr+0.074 ≈ 0 = ドリフトに乗っているだけではない真の双方向モメンタム。
  * チャンピオンとの月次相関 −0.10〜−0.15(負)。失血窓で平均+106(平時+25)、2022(最悪失血年)で +1332 と
    **チャンピオンが轢かれるまさにその窓で稼ぐ**。
  * 厳密 two-book robust(mm_lab, 日次ブロックブートストラップ p95=20% 較正): champion 単独 robCAGR +20.6% に対し
    w(本スリーブ配分)0.15→+0.9pp / 0.20→+1.3pp / 0.25→+1.6pp / 0.40→+2.3pp(内点最大)。
    **レバ偽装でない決定的証拠**: 混ぜると経験的最大DDが −14.8%→−12.6% に「縮む」、same-tail署名
    (固定レバで混入)は empCAGR↓+p95↓=偽装署名の真逆。5シード+2.3〜2.9pp/ブロック長+2.2〜3.8pp/
    IS較正→OOS素検証(+30.5→+33.0%)全て純増。最悪年も w↑で +4.0%→+6.0% に改善。

― 正直な限界(=運用条件)――――――――――――――――――――――――――――――――――――
  1. **単一通貨集中**: net で生きるのは実質 USDJPY のみ(GBPJPY が弱く追随、他JPY/USDメジャーは負)。
     USDJPY を抜くと等加重ブックは Sharpe −0.76。n=1 通貨=10年単一標本の集中リスクは残る(構造的根拠=
     USD/JPY は金利差・JPYファンディング・BoJ政策で最もトレンドしやすいメジャー、だが要モニタ)。
  2. **執行タイミング感応**: エッジの約半分がバー内に集中。1バー遅延執行(H1終値→次バー始値約定)だと
     GROSS Sharpe 0.78→0.45 に半減。両建て×リテール0.7pip×1バー遅延だと NET ≒ 損益分岐。
     **→ 実弾化は (a) long-only スリーブ化(遅延×リテールでも +22.5%)か (b) ECN(≈0.2pip)前提**。
     損益分岐スプレッドは即時執行で 1.30pip、リテール0.7pip でも黒。年529取引(≒日次2件未満)で bot 可。
  3. **H1中心**: M30 は band 必須、H4 は lookback36/48 で崩れる=TF頑健ではない。H1 で運用する。

― 使い方 ――――――――――――――――――――――――――――――――――――――――――――
  対象 USDJPY・tf H1。チャンピオン本体には混ぜず**別ブック**として資本の 15〜25% 程度を配分し、
  二口座を robust(p95=20%)較正で合成する(基盤: research/experiments/exp60_twobook_robust.py)。
  実弾は long-only(side="long")または ECN スプレッドを推奨。band は 0.001〜0.002 でコスト耐性を上げる。
"""

from __future__ import annotations

import pandas as pd

# 検証済み本番パラメータ(USDJPY H1)。band はコスト耐性のため小さく入れる(reports/31)。
PARAMS = {"lookback": 24, "band": 0.001}
# 高原確認用(reports/31: lookback18-72 × band0-0.002 が net 正域)。
PARAM_GRID = {"lookback": [18, 24, 36, 48], "band": [0.0, 0.001, 0.002]}


def generate_signals(data: pd.DataFrame, lookback: int = 24, band: float = 0.001):
    """過去 lookback 本リターンの符号で順張り、反転でドテン(tsmom と同型・先読みなし)。

    band: 過去 lookback 本の累積リターン絶対値が band 超のときだけ建玉(微小モメンタムの
    頻繁ドテン=コスト負けを抑制)。USDJPY H1 で band 0.001〜0.002 がコスト後の高原。
    実弾でリテールスプレッド+約定遅延に晒すなら side="long" 推奨(short はコストで死に脚)。
    """
    close = data["close"]
    mom = close / close.shift(lookback) - 1.0          # 過去 lookback 本の累積リターン(確定値)

    long_state = mom > band
    short_state = mom < -band

    long_entries = long_state & ~long_state.shift(fill_value=False)
    short_entries = short_state & ~short_state.shift(fill_value=False)
    long_exits = short_entries                          # 反対転換でドテン
    short_exits = long_entries
    return long_entries, long_exits, short_entries, short_exits
