"""critic_followup — 監査の残り定量3点。

 (a) vectorbt の value サイジングが現金制約で部分約定していないか(PnL = 10000*ret か)
 (b) キャリー(スワップ)無視バイアス: JPY ペアのロング保有日数×想定金利差で見積もり
 (c) XAUUSD ロングのポケットは「タイミング」か「ただのベータ」か:
     在場バーあたり bps vs 無条件の1バーあたりドリフト

実行: PYTHONPATH=. uv run python research/experiments/trend2/critic_followup.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c/research/lab")
import trend_lab as tl  # noqa: E402

from fxlab.backtest import run  # noqa: E402
from strategies.tsmom import generate_signals as tsmom_signals  # noqa: E402


def main() -> None:
    # (a) PnL ≒ 10000 * ret か(現金制約による部分約定の有無)
    data = tl.load_tf("EURUSD", "D1")
    tl.register_spreads()
    pf = run("EURUSD", "D1", tsmom_signals, {"lookback": 60, "band": 0.0},
             data=data, size_mode="value", side="both")
    tr = pf.trades.records_readable
    implied = tr["PnL"] / (tr["Size"] * tr["Avg Entry Price"])
    diff = (implied - tr["Return"]).abs().max()
    entry_value = (tr["Size"] * tr["Avg Entry Price"])
    print("(a) value サイジング検査 (EURUSD tsmom D1 lb60)")
    print(f"    Return と PnL/建玉価値 の最大乖離: {diff:.2e}")
    print(f"    建玉価値の min/median/max: {entry_value.min():.1f} / "
          f"{entry_value.median():.1f} / {entry_value.max():.1f} (目標10000)")
    n_partial = int((entry_value < 9999).sum())
    print(f"    10000 未満(現金制約で縮小された)トレード数: {n_partial}/{len(tr)}")

    # (b) キャリー無視の影響(JPYペア・ロング・2022以降)
    pool = tl.build_pool(tsmom_signals, {"lookback": 60, "band": 0.0},
                         tf="D1", side="both")
    jpy = pool[(pool["instr"].isin(["USDJPY", "EURJPY", "GBPJPY"]))
               & (pool["dir"] == 1)
               & (pool["entry"] >= pd.Timestamp("2022-01-01", tz="UTC"))].copy()
    jpy["days"] = (pd.to_datetime(jpy["exit"]) - pd.to_datetime(jpy["entry"])).dt.days
    carry_rate = 0.045  # 2022-2025 の対JPY短期金利差のラフな平均(米欧英 vs 日本)
    carry = jpy["days"].sum() * carry_rate / 365.0
    print("\n(b) キャリー無視バイアス (tsmom D1 lb60, JPYペアのロング, 2022+)")
    print(f"    対象トレード数: {len(jpy)}, 総保有日数: {int(jpy['days'].sum())}日")
    print(f"    価格リターン合計: {jpy['ret'].sum():+.4f}")
    print(f"    想定スワップ加算(金利差{carry_rate:.1%}/年): {carry:+.4f}"
          f" (≈ {carry / max(len(jpy), 1) * 1e4:.1f} bps/trade)")
    print(f"    → キャリー込み合計: {jpy['ret'].sum() + carry:+.4f}")

    # (c) XAUUSD ロングポケットの「タイミング寄与」検査
    print("\n(c) XAUUSD ロング: タイミングかベータか")
    g = tl.load_tf("XAUUSD", "D1")
    ret1 = g["close"].pct_change()
    uncond = float(ret1.mean() * 1e4)
    for fam, pl in (("tsmom_lb60_D1", pool),):
        sub = pl[(pl["instr"] == "XAUUSD") & (pl["dir"] == 1)]
        in_bars = float(sub["bars_held"].sum())
        per_bar = float(sub["ret"].sum() / in_bars * 1e4)
        total_bars = len(g)
        print(f"    [{fam}] long n={len(sub)} (n<100注意), 在場バー={int(in_bars)}"
              f" ({in_bars / total_bars:.1%} of {total_bars})")
        print(f"      在場バーあたり: {per_bar:+.2f} bps/bar vs 無条件ドリフト {uncond:+.2f} bps/bar")
        print(f"      → タイミング寄与 = {per_bar - uncond:+.2f} bps/bar")
        # ショート側も(逆風の大きさ)
        sh = pl[(pl["instr"] == "XAUUSD") & (pl["dir"] == -1)]
        print(f"      short n={len(sh)}, sum={sh['ret'].sum():+.4f}, "
              f"在場バーあたり {float(sh['ret'].sum() / max(sh['bars_held'].sum(), 1) * 1e4):+.2f} bps/bar")

    # 同じことを USDJPY ロングにも
    u = tl.load_tf("USDJPY", "D1")
    uncond_u = float(u["close"].pct_change().mean() * 1e4)
    sub = pool[(pool["instr"] == "USDJPY") & (pool["dir"] == 1)]
    per_bar = float(sub["ret"].sum() / sub["bars_held"].sum() * 1e4)
    print(f"    [tsmom USDJPY] long n={len(sub)} (n<100注意): 在場 {per_bar:+.2f} bps/bar"
          f" vs 無条件 {uncond_u:+.2f} bps/bar → タイミング寄与 {per_bar - uncond_u:+.2f}")


if __name__ == "__main__":
    main()
