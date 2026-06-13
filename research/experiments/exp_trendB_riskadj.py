"""再構築B: リスク調整モメンタム + ヒステリシス (D1, 7 USDメジャー)。

シグナル = lookback日リターン / (日次vol * sqrt(lookback))
        = リスク調整トレンド強度 (年率vol正規化されたトレンドのt値的な量)。

ヒステリシス帯:
  |signal| > th        で建玉 (long if >0, short if <0)
  |signal| < th*0.5    でフラット解除 (whipsaw削減)
  その中間は現状維持 (state を持ち越す)

rebalance(粗い判定): rebalance本ごとにのみ state を更新 -> コスト/回転抑制。

state を vbt 用の entry/exit エッジに変換して両建てバックテスト。
size_mode="value" (固定キャッシュ・非複利) で7ペア横断比較。

先読みなし:
  - mom は確定リターン (close/close.shift(lb)-1, 自バー終値まで)
  - vol も確定 (日次リターンの rolling std, 自バーまで)
  これらは「自バー終値が確定した時点」の情報で、シグナルは次バーで約定 (vbtのfrom_signalsは
  当該バーのcloseで約定するが、シグナル自体は確定値のみ参照=look-aheadなし)。
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd

# repo root を import path に
sys.path.insert(0, "/Users/yutootsuka/Documents/economy/.claude/worktrees/hopeful-pike-e2e515")

from fxlab import load, run, metrics
import fxlab.config as C

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
TF = "D1"
IS_END = "2021-12-31"   # IS: 2016-2021, OOS: 2022-
SIZE_MODE = "value"


def generate_signals(data: pd.DataFrame, lookback: int = 252, th: float = 1.0,
                     rebalance: int = 5):
    """リスク調整モメンタム + ヒステリシス + 粗いrebalance。"""
    close = data["close"]
    n = len(close)

    # リスク調整トレンド強度 (確定値のみ)
    mom = close / close.shift(lookback) - 1.0
    daily_ret = close.pct_change()
    vol = daily_ret.rolling(lookback).std()
    denom = vol * np.sqrt(lookback)
    sig = mom / denom.replace(0.0, np.nan)
    sig = sig.values  # ndarray (NaN含む)

    exit_band = th * 0.5

    # state: +1 long, -1 short, 0 flat。ヒステリシス + rebalance本ごと判定
    state = np.zeros(n, dtype=np.int8)
    cur = 0
    last_rebal = -10**9
    for i in range(n):
        s = sig[i]
        if not np.isnan(s):
            # rebalance本ごとにのみ「状態遷移」を許可 (粗い判定でコスト抑制)
            if (i - last_rebal) >= rebalance:
                if cur == 0:
                    if s > th:
                        cur = 1; last_rebal = i
                    elif s < -th:
                        cur = -1; last_rebal = i
                elif cur == 1:
                    if s < exit_band:           # ロング解除帯
                        # 反対側に強ければドテン、そうでなければフラット
                        cur = -1 if s < -th else 0
                        last_rebal = i
                elif cur == -1:
                    if s > -exit_band:
                        cur = 1 if s > th else 0
                        last_rebal = i
        state[i] = cur

    st = pd.Series(state, index=close.index)
    long_state = st == 1
    short_state = st == -1

    # state -> エッジ (エントリー/エグジット)
    long_entries = long_state & ~long_state.shift(fill_value=False)
    long_exits = ~long_state & long_state.shift(fill_value=False)
    short_entries = short_state & ~short_state.shift(fill_value=False)
    short_exits = ~short_state & short_state.shift(fill_value=False)

    return long_entries, long_exits, short_entries, short_exits


def measure(pairs, params, gross: bool, data_slice=None):
    """7ペアを測定し DataFrame で返す。gross=True ならコスト0。"""
    if gross:
        C.SPREADS_PIPS = {k: 0.0 for k in C.SPREADS_PIPS}
        C.COMMISSION_FRACTION = 0.0
    rows = []
    for p in pairs:
        d = load(p, TF)
        if data_slice is not None:
            d = d.loc[data_slice[0]:data_slice[1]]
        pf = run(p, TF, generate_signals, params, data=d, size_mode=SIZE_MODE, side="both")
        m = metrics(pf).iloc[0].to_dict()
        m["pair"] = p
        rows.append(m)
    return pd.DataFrame(rows).set_index("pair")


def avg_block(df):
    return {
        "sharpe": float(df["sharpe"].mean()),
        "total_return": float(df["total_return"].mean()),
        "num_trades": float(df["num_trades"].mean()),
        "win_rate": float(df["win_rate"].mean()),
        "profit_factor": float(df["profit_factor"].replace([np.inf], np.nan).mean()),
        "pos_pairs": int((df["total_return"] > 0).sum()),
    }


if __name__ == "__main__":
    BASE = {"lookback": 252, "th": 1.0, "rebalance": 5}

    # ---- 1) フル期間 GROSS / NET (別プロセス的に config を切替え) ----
    # gross は subprocess 相当で測りたいが、同一プロセスで gross->net 順に測ると
    # net 測定時に SPREADS が0のまま。-> net を先に (元のconfigで)、gross を後で。
    print("=" * 70)
    print(f"BASE params: {BASE}  | size={SIZE_MODE} | {TF} | 7 majors")
    print("=" * 70)

    # NET (元のコスト)
    net_full = measure(PAIRS, BASE, gross=False)
    print("\n[NET full period]")
    print(net_full[["total_return", "sharpe", "num_trades", "win_rate", "profit_factor"]].round(3))
    net_avg = avg_block(net_full)
    print("NET avg:", {k: round(v, 4) for k, v in net_avg.items()})

    # GROSS (コスト0) — config を破壊するので最後に
    gross_full = measure(PAIRS, BASE, gross=True)
    print("\n[GROSS full period]")
    print(gross_full[["total_return", "sharpe", "num_trades"]].round(3))
    gross_avg = avg_block(gross_full)
    print("GROSS avg:", {k: round(v, 4) for k, v in gross_avg.items()})

    # cost drag
    drag = float(gross_full["total_return"].mean() - net_full["total_return"].mean())
    print(f"\ncost drag (gross-net total_return, 7pair avg): {drag:.4f}")

    print("\nRESULT_NET_AVG_SHARPE", round(net_avg["sharpe"], 4))
    print("RESULT_NET_AVG_TR", round(net_avg["total_return"], 4))
    print("RESULT_GROSS_AVG_SHARPE", round(gross_avg["sharpe"], 4))
    print("RESULT_GROSS_AVG_TR", round(gross_avg["total_return"], 4))
    print("RESULT_NET_POS_PAIRS", net_avg["pos_pairs"])
