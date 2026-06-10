"""敵対検証: xau_tsmom_lb60_D1_tslonly_tsl4 (exp_asym_exit.py の生き残り)。

対象構成: family=trend_asymmetric_exit, XAUUSD 単独, tsmom lb60 D1,
エントリーのみ(exits=allFalse), tsl_stop=0.04。
報告値: n=142, sum=0.6192, PF=1.733, IS=1.18, OOS=2.29。

検証項目(固定・全件報告):
  (1) repro      : 発見者と独立に書いたシグナル関数で再現(sum_ret ±20%)
  (2) delay1     : エントリーを 1 バー遅延(翌日終値で成行)。tsl は engine 側のまま
                   (trail レベルは過去 close の peak から構築 = 既に1バー前値ベース。
                    エントリーが成行(close)でありストップ注文ではないため、
                    「レベル1バー前固定」はエントリー遅延と同義)
  (3) cost1.5    : XAUUSD スプレッド $0.40 → $0.60(1.5倍)
  (4) ex2022     : 2022 年(エントリー基準)のトレードを除いた sum_ret
  (補) exit@close : tsl の約定を stoplimit(レベルそのもの)→ 当日終値に悲観化
                   (D1 ギャップでレベルを飛び越えた場合の楽観を潰す)
  (補) delay+cost : (2)+(3) 同時

実行: PYTHONPATH=. uv run python research/experiments/trend2/critic_xau_tslonly.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import vectorbt as vbt

ROOT = "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/research/lab")

import trend_lab as tl  # noqa: E402
from fxlab import config  # noqa: E402
from fxlab.backtest import _slippage_series  # noqa: E402
from fxlab.trades import trade_table  # noqa: E402

LB = 60
TSL = 0.04


# --- シグナル(発見者コードを見ずに tsmom 定義から独立に再実装) ----------
def signals_base(data: pd.DataFrame):
    close = data["close"]
    mom = close / close.shift(LB) - 1.0
    long_state = mom > 0.0
    short_state = mom < 0.0
    le = long_state & ~long_state.shift(fill_value=False)
    se = short_state & ~short_state.shift(fill_value=False)
    empty = pd.Series(False, index=data.index)
    return le, empty, se, empty


def signals_delay1(data: pd.DataFrame):
    le, lx, se, sx = signals_base(data)
    return le.shift(1, fill_value=False), lx, se.shift(1, fill_value=False), sx


def build_pool(gen, *, spread_mult: float = 1.0, stop_exit_price: str = "stoplimit"):
    """run() 相当を直接 from_signals で再現(stop_exit_price を差し替え可能に)。"""
    tl.register_spreads()
    config.SPREADS_PIPS["XAUUSD"] = config.SPREADS_PIPS["XAUUSD"] * spread_mult
    data = tl.load_tf("XAUUSD", "D1")
    le, lx, se, sx = gen(data)
    close = data["close"]
    pf = vbt.Portfolio.from_signals(
        close,
        entries=le, exits=lx, short_entries=se, short_exits=sx,
        slippage=_slippage_series("XAUUSD", close),
        fees=config.COMMISSION_FRACTION,
        init_cash=10_000,
        freq=config.TIMEFRAMES["D1"],
        size=10_000, size_type="value",
        sl_stop=TSL, sl_trail=True,
        stop_exit_price=stop_exit_price,
    )
    tt = trade_table(pf, data)
    pool = pd.DataFrame({
        "instr": "XAUUSD",
        "entry": tt["entry"].to_numpy(),
        "exit": tt["exit"].to_numpy(),
        "dir": np.where(tt["dir"].to_numpy() == "Long", 1, -1),
        "entry_price": tt["entry_price"].to_numpy(),
        "ret": tt["return_pct"].to_numpy() / 100.0,
        "bars_held": tt["bars_held"].to_numpy(),
    }).sort_values("entry").reset_index(drop=True)
    return pool, data


def report(name: str, pool: pd.DataFrame) -> dict:
    st = tl.pool_stats(pool)
    row = {"variant": name, **st}
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def main() -> None:
    # (1) 再現
    pool0, data = build_pool(signals_base)
    r0 = report("repro_base", pool0)

    # 先読み監査の補助情報: シグナルバーと約定バーの確認
    le, _, se, _ = signals_base(data)
    print(f"# data: {data.index[0]} .. {data.index[-1]}  rows={len(data)}")
    print(f"# signal bars: long={int(le.sum())} short={int(se.sum())}")
    ent = pd.DatetimeIndex(pool0["entry"])
    sig_idx = data.index[(le | se).to_numpy()]
    same_bar = ent.isin(sig_idx).mean()
    print(f"# entries filled on signal bar (market-on-close): {same_bar:.3f}")

    # (2) エントリー1バー遅延
    pool_d, _ = build_pool(signals_delay1)
    r_d = report("delay1", pool_d)

    # (3) コスト1.5倍
    pool_c, _ = build_pool(signals_base, spread_mult=1.5)
    r_c = report("cost1.5", pool_c)

    # (4) 2022年除外(エントリー基準・参考にexit基準も)
    ey = pd.DatetimeIndex(pool0["entry"]).year
    xy = pd.DatetimeIndex(pool0["exit"]).year
    ex2022_entry = float(pool0.loc[ey != 2022, "ret"].sum())
    ex2022_exit = float(pool0.loc[xy != 2022, "ret"].sum())
    sum2022 = float(pool0.loc[ey == 2022, "ret"].sum())
    print(f"# ex2022 (entry-year basis): {ex2022_entry:.4f}  "
          f"(exit-year basis: {ex2022_exit:.4f}; 2022 contribution: {sum2022:.4f})")

    yearly = pool0.groupby(pd.DatetimeIndex(pool0["exit"]).year)["ret"].agg(["sum", "count"])
    print("# yearly (exit-year):")
    print(yearly.round(4).to_string())

    # (補) tsl 約定を当日終値に悲観化
    pool_p, _ = build_pool(signals_base, stop_exit_price="close")
    report("exit_at_close(pessimistic)", pool_p)

    # (補) 遅延+コスト1.5倍 同時
    pool_dc, _ = build_pool(signals_delay1, spread_mult=1.5)
    report("delay1+cost1.5", pool_dc)

    # (補) 遅延+悲観約定+コスト1.5倍(全部盛り)
    pool_all, _ = build_pool(signals_delay1, spread_mult=1.5, stop_exit_price="close")
    report("delay1+cost1.5+exit_at_close", pool_all)

    out = {
        "repro_sum": r0["sum_ret"], "repro_n": r0["n"], "repro_pf": r0["pool_pf"],
        "repro_is_pf": r0["is_pf"], "repro_oos_pf": r0["oos_pf"],
        "delay1_oos_pf": r_d["oos_pf"], "delay1_sum": r_d["sum_ret"], "delay1_pf": r_d["pool_pf"],
        "cost15_oos_pf": r_c["oos_pf"], "cost15_sum": r_c["sum_ret"], "cost15_pf": r_c["pool_pf"],
        "ex2022_sum": round(ex2022_entry, 4),
    }
    print("RESULT " + json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
