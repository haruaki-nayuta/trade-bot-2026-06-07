"""金(XAUUSD)向け 非対称タイムシリーズ・モメンタム(3クロック)。

発想:
  金は構造的な長期強気(2016: $1272 → 2026: $4198)で、対称な tsmom では
  ショート側が単体で負ける(PF 0.7 前後)。だがショート全廃は取引数を半減させ
  regime 依存(強気が続く前提)も強まる。そこで 3 本の時計(速い lb_fast / 中間
  lb_mid / 遅い lb_slow)でロングとショートを非対称に設計する。

  * ロング(入りやすく): 中間モメンタム mom_mid > band_long で保有(主経路)。
    さらに「長期強気(mom_slow > 0)中の中間調整」では、速いモメンタムの回復
    (mom_fast > band_long)でブリッジ・ロング(押し目のV字回復取り)。
  * ショート(厳しく): 速いモメンタムが強い下落(mom_fast < -band_short)かつ
    中間・長期がともに非ポジティブのときだけ。強気相場への逆張り売りを構造的に
    排除する(これが対称版ショート PF 0.73 → 1.0 超の源泉)。

  非対称化の 3 要素: (a) band 非対称(ロング 0 / ショート 0.02)、(b) ショートのみ
  追加確認(fast・mid・slow すべて非ポジティブ)、(c) lb 非対称(ロングは中長期、
  ショートのトリガーは短期)。

先読みなしの根拠:
  モメンタムは close / close.shift(lb) - 1 で、確定済みバーの終値のみ使用。
  state の遷移判定は shift(fill_value=False) による前バー比較のみ。
  rolling 極値・未確定バー参照は一切ない。

検証(XAUUSD H4, IS=前70%で最適化 → OOS素検証):
  IS +41.8% / PF 1.54 / Sharpe 0.55(25.4 tr/yr)
  OOS +91.7% / PF 3.06 / Sharpe 1.49
  full 10年 +159.4% / PF 2.09 / Sharpe 0.87 / DD -29.4% / 25.3 tr/yr
  素の tsmom lb360(+120% / PF 1.79 / Sharpe 0.69)を全指標で上回り、
  ショート側は PF 0.72(-18.8%)→ PF 1.11(+4.0%)に改善。金専用(他ペア不問)。

⚠ 敵対検証(2026-06-11, reports/11)で overfit レンズ fail。正直な但し書き:
  * 上の「素の tsmom lb360 を全指標で上回る」は対称版(ショート出血込み)との比較で
    ストローマン。公正な祖先=ロング専用 tsmom lb360 は full +170.8%/PF3.79/Sharpe0.96/
    DD-23.1%、OOS PF7.43 で、本戦略(5パラメータ)を両窓・全指標で支配する。
  * 新規部品は実測で負またはゼロ寄与: ブリッジロング単体 full PF0.60(-8.9%)、除去すると
    全指標改善(+181.7%/PF2.45)。ショート側は10年で +4.0%/PF1.11 のほぼ死荷重。
  * OOS PF3.06 は集中の産物(トップ1取引=OOS利益の49%、トップ5除外で PF0.58)。
  → 設計レベルの過剰適合(金βの誤帰属)。採用しない。記録として保持。
"""

from __future__ import annotations

import pandas as pd

# 採用値(XAUUSD H4。IS 最適化 → OOS 検証済み)
PARAMS = {
    "lb_fast": 120,
    "lb_mid": 360,
    "lb_slow": 480,
    "band_long": 0.0,
    "band_short": 0.02,
}

# 探索範囲(18 組合せ)
PARAM_GRID = {
    "lb_fast": [60, 120, 240],
    "lb_mid": [240, 360],
    "lb_slow": [480],
    "band_long": [0.0],
    "band_short": [0.01, 0.02, 0.05],
}


def generate_signals(
    data: pd.DataFrame,
    lb_fast: int = 120,
    lb_mid: int = 360,
    lb_slow: int = 480,
    band_long: float = 0.0,
    band_short: float = 0.02,
):
    close = data["close"]
    mom_f = close / close.shift(lb_fast) - 1.0   # 速い時計(ショートのトリガー/ブリッジ回復)
    mom_m = close / close.shift(lb_mid) - 1.0    # 中間時計(ロングの主経路)
    mom_s = close / close.shift(lb_slow) - 1.0   # 遅い時計(レジーム判定)

    # ロング: 中間強気、または「長期強気中の調整からの速い回復」(ブリッジ)
    long_state = (mom_m > band_long) | ((mom_s > 0) & (mom_f > band_long))

    # ショート: 速い強い下落 かつ 中間・長期ともに非ポジティブ(全クロック確認)
    short_state = (mom_f < -band_short) & (mom_m <= 0) & (mom_s <= 0) & ~long_state

    long_entries = long_state & ~long_state.shift(fill_value=False)
    long_exits = ~long_state & long_state.shift(fill_value=False)
    short_entries = short_state & ~short_state.shift(fill_value=False)
    short_exits = ~short_state & short_state.shift(fill_value=False)
    return long_entries, long_exits, short_entries, short_exits
