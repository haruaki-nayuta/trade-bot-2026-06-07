"""金(XAUUSD)向けタイムシリーズ・モメンタムの複数ルックバック合議(アンサンブル)。

発想:
  単一 lookback の tsmom(D1 lb40/60 で PF~2.0)は金で有効だが、lb の一点選択に
  脆く取引数も少ない(15-21回/年)。複数 lookback の「過去リターンの符号」を
  多数決させ、合議スコア = 賛成数 - 反対数 を使う:

    * スコア >= +k_in でロング、 <= -k_in_short でショート(エントリー閾値)
    * ロング保有中はスコアが min(k_out, k_in-1) 以下に落ちたら手仕舞い
      (ショートは対称)= 入りと出口を分けられるヒステリシス
    * k_out を深く(負に)するほど保有が粘り、浅くするほど回転が上がる

  採用構成は lb = 5/10/20/40/60(1週・2週・1ヶ月・2ヶ月・3ヶ月のカレンダー
  整合な全速アンサンブル)。過半数(スコア+1)でロング、過半数割れ(-1)で
  手仕舞い、4/5 が反対(-3)になって初めてショート、という非対称設計。
  金は長期強気でショート単体は負けやすい(単一 lb で PF0.72-0.86)ため、
  ショートだけ合議の確信度を要求する。単一 lb のドテンと違い「合議が割れたら
  flat」区間が挟まり、再エントリーで取引数を稼ぎつつレジーム判定が滑らかになる。

先読みなしの根拠:
  各 lookback のモメンタムは close / close.shift(lb) - 1 で、確定済みバーの
  終値のみ参照。スコアも同一バーまでの情報で閉じており未来値を一切使わない。
  ウォームアップ(最長 lb 未満)はスコアを 0 にしてノーシグナル扱い。

検証:
  uv run python evaluate.py tsmom_ens_gold --pair XAUUSD --tf D1

⚠ 敵対検証(2026-06-11, reports/11)による正直な但し書き:
  * 採用構成は full +180%/PF1.82/IS PF1.36/OOS PF2.81 を再現するが、lbセット固有の
    IS優位はセット選択フィットの署名(近傍セットは IS PF≈0.97-0.99、ランダム5本セット
    25個中25個が full/OOS PF>1)。実体は「金強気への希釈ロング+損益トントンのショート」。
  * 約定1バー(1日)遅延で IS PF 0.94 に崩壊(コスト・テールレンズ fail)。終値シグナル→
    翌バー始値約定なら IS PF1.34 で生存するため、日次クローズ直後の即時執行が必須条件。
  * リターン・Sharpe・DD とも同期間の金 B&H(+230%/0.89/-23.8%)に劣後。
  → 採用するなら pullback_trend_gold(H4)が全レンズで優位。本ファイルは記録として保持。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# lookback セット(evaluate.py の数値パラメータ制約に合わせ整数コードで指定)
LB_SETS = {
    1: (5, 10, 20, 40, 60),      # 採用: 全速・カレンダー整合(1w/2w/1M/2M/3M)
    2: (10, 20, 40, 60, 120),    # 中速(+半年)
    3: (5, 10, 20, 40, 80),      # 全速の近傍(robustness 用)
    4: (20, 40, 60, 120),        # 課題で提案された元セット(n=4)
}

# 採用値(IS 2016-2023 で sweep 探索、OOS 2023-2026 で素検証済み)
PARAMS = {"lbset": 1, "k_in": 1, "k_out": 1, "k_in_short": 3}

# sweep 探索範囲(k はスコアの整数閾値。k_out は内部で k_in-1 以下にクランプ)
PARAM_GRID = {
    "lbset": [1, 2, 3],
    "k_in": [1, 3],
    "k_out": [1, -1],
    "k_in_short": [1, 3, 5],
}


def generate_signals(
    data: pd.DataFrame,
    lbset: int | str = 1,
    k_in: int = 1,
    k_out: int = 1,
    k_in_short: int | None = 3,
):
    close = data["close"]
    # lbset は整数コード(LB_SETS)か "5-10-20" 形式の文字列を受け付ける
    if isinstance(lbset, str) and "-" in lbset:
        lb_list = [int(x) for x in lbset.split("-")]
    else:
        lb_list = list(LB_SETS[int(lbset)])

    # 合議スコア: 各 lookback の過去リターン符号(+1/-1)の合計
    score = None
    for lb in lb_list:
        vote = np.sign(close / close.shift(lb) - 1.0)
        score = vote if score is None else score + vote
    score = score.fillna(0.0)  # ウォームアップはノーシグナル

    # ショート側のエントリー閾値(未指定ならロングと対称)
    k_in_s = k_in if k_in_short is None else int(k_in_short)
    # 出口閾値はエントリー閾値より必ず下(同一バーで entry/exit が衝突しない)
    k_out_l = min(k_out, k_in - 1)
    k_out_s = min(k_out, k_in_s - 1)

    long_entries = score >= k_in
    long_exits = score <= k_out_l
    short_entries = score <= -k_in_s
    short_exits = score >= -k_out_s
    return long_entries, long_exits, short_entries, short_exits
