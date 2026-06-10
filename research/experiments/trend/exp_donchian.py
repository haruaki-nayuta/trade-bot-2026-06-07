"""ドンチャン/チャネル・ブレイクアウト族の固定グリッド検証(trend_lab 共有基盤)。

- close ベースのドンチャン(既存 strategies/donchian_breakout.py は high/low 版だが、
  クロス12対象は close 複製 OHLC のため、close ベース版で全19対象に正しく適用する)。
- 先読み防止: rolling 極値は .shift(1) で「直前バーまで」の極値を使用。
- 固定グリッド(追加探索禁止):
    H4: entry {55,100,200} x exit {20,50}   (6構成)
    D1: entry {20,55,100} x exit {10,20}    (6構成)
    W1: entry {13,26}     x exit {4,8}      (4構成)
  side=both, 計16構成。

実行: PYTHONPATH=. uv run python research/experiments/trend/exp_donchian.py
"""

from __future__ import annotations

import json
import sys

import pandas as pd

sys.path.insert(0, "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c/research/lab")
import trend_lab as tl  # noqa: E402


def donchian_close(data: pd.DataFrame, entry_window: int = 55, exit_window: int = 20):
    """close ベースのドンチャン・ブレイクアウト(クロスにも正しく適用される)。

    エントリー: 終値が直前 entry_window 本の終値最高値を上抜け → ロング
                終値が直前 entry_window 本の終値最安値を下抜け → ショート
    手仕舞い : 終値が直前 exit_window 本の終値最安値割れ(ロング)/最高値抜け(ショート)
    """
    close = data["close"]
    upper = close.rolling(entry_window).max().shift(1)
    lower = close.rolling(entry_window).min().shift(1)
    exit_upper = close.rolling(exit_window).max().shift(1)
    exit_lower = close.rolling(exit_window).min().shift(1)

    long_entries = close > upper
    short_entries = close < lower
    long_exits = close < exit_lower
    short_exits = close > exit_upper
    return long_entries, long_exits, short_entries, short_exits


GRID = [
    ("H4", [(e, x) for e in (55, 100, 200) for x in (20, 50)]),
    ("D1", [(e, x) for e in (20, 55, 100) for x in (10, 20)]),
    ("W1", [(e, x) for e in (13, 26) for x in (4, 8)]),
]


def main() -> None:
    results = []
    for tf, combos in GRID:
        for entry_w, exit_w in combos:
            params = {"entry_window": entry_w, "exit_window": exit_w}
            pool = tl.build_pool(donchian_close, params, tf=tf, side="both")
            st = tl.pool_stats(pool)
            label = f"donch_e{entry_w}x{exit_w}_{tf}_both"
            row = {"label": label, "tf": tf, "side": "both",
                   "params": f"entry_window={entry_w},exit_window={exit_w}", **st}
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    out = pd.DataFrame(results)
    out_path = "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c/research/outputs/trend_donchian_grid.csv"
    out.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
