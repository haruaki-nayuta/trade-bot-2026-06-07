"""xs_momentum — クロスセクショナル・モメンタム(銘柄横断の相対モメンタム)。

設計:
  毎月初(D1 グリッド上の月替わり最初のバー)、過去 lb 日リターンで 19 対象を
  ランク付けし、上位 k をロング・下位 k をショート。1ヶ月保有して次の月初に
  終値で入替(同一銘柄が残留しても exit+re-entry として毎月コスト計上=保守的)。

先読み禁止:
  ランクに使うリターンは entry バーの「前日まで」で確定:
    mom_t = close[t-1] / close[t-1-lb] - 1   (close.shift(1) / close.shift(1+lb) - 1)
  エントリーは月初バーの終値、エグジットは次の月初バーの終値。

コスト:
  fxlab.config.spread_pips / pip_size を使用。tl.register_spreads() でクロスは
  3.0 pips に登録(メジャーは config 既定 0.6〜1.4 pips)。往復 1 フルスプレッド
  (fxlab 本体と同じ: 片道半スプレッド×2)を entry 価格比で控除。

注意:
  クロスは close 複製 OHLC だが、本検証は close のみ使用なので問題なし。

実行: リポジトリ直下で  PYTHONPATH=. uv run python research/experiments/trend/xs_momentum.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c/research/lab")

import numpy as np
import pandas as pd

import trend_lab as tl
from fxlab import config

# 固定グリッド(これ以外は回さない)
GRID_LB = [60, 120, 250]
GRID_K = [3, 5]

Z_ENTRY_FIXED = 2.2


def load_close_matrix(instruments: list[str]) -> pd.DataFrame:
    """全対象の D1 close を外部結合で1枚に(ffill で穴埋め、先頭 NaN はそのまま)。"""
    closes = {nm: tl.load_tf(nm, "D1")["close"] for nm in instruments}
    mat = pd.concat(closes, axis=1).sort_index()
    return mat.ffill()


def month_first_bars(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """D1 グリッド上の「月替わり最初のバー」。"""
    s = pd.Series(index=index, dtype="int8")
    grp = s.groupby([index.year, index.month])
    return pd.DatetimeIndex([g.index[0] for _, g in grp]).sort_values()


def build_xs_pool(lb: int, k: int, close: pd.DataFrame) -> pd.DataFrame:
    """月次リバランスのクロスセクショナル・モメンタムのトレードプールを構築。"""
    instruments = list(close.columns)
    # ランク用モメンタム: entry バーの前日までで確定(先読み禁止)
    mom = close.shift(1) / close.shift(1 + lb) - 1.0
    vol = close.pct_change().rolling(20).std()  # tl.build_pool と同じ定義

    # 往復コスト(価格比)はエントリー価格で割って算出するため、価格幅を前計算
    spread_price = {nm: config.spread_pips(nm) * config.pip_size(nm) for nm in instruments}

    rebal = month_first_bars(close.index)
    idx_pos = {ts: i for i, ts in enumerate(close.index)}

    rows = []
    for i, t in enumerate(rebal):
        m = mom.loc[t].dropna()
        # entry 価格が有効な銘柄のみ
        m = m[close.loc[t, m.index].notna()]
        if len(m) < 2 * k:
            continue  # ランク対象が足りない(lb 日分の履歴が無い初期)
        # エグジット = 次の月初バー(最終月はデータ末尾のバー)
        t_exit = rebal[i + 1] if i + 1 < len(rebal) else close.index[-1]
        if t_exit <= t:
            continue
        ranked = m.sort_values(ascending=False)
        longs = list(ranked.index[:k])
        shorts = list(ranked.index[-k:])
        for nm, d in [(nm, 1) for nm in longs] + [(nm, -1) for nm in shorts]:
            ep = close.at[t, nm]
            xp = close.at[t_exit, nm]
            if not (np.isfinite(ep) and np.isfinite(xp)):
                continue
            cost = spread_price[nm] / ep  # 往復1フルスプレッド(価格比)
            ret = d * (xp / ep - 1.0) - cost
            rows.append({
                "instr": nm,
                "entry": t,
                "exit": t_exit,
                "dir": d,
                "entry_price": float(ep),
                "ret": float(ret),
                "bars_held": idx_pos[t_exit] - idx_pos[t],
                "z_entry": Z_ENTRY_FIXED,
                "vol_entry": float(vol.at[t, nm]) if np.isfinite(vol.at[t, nm]) else np.nan,
            })
    pool = pd.DataFrame(rows, columns=["instr", "entry", "exit", "dir", "entry_price",
                                       "ret", "bars_held", "z_entry", "vol_entry"])
    return pool.sort_values("entry").reset_index(drop=True)


def main() -> None:
    tl.register_spreads()  # クロス 3.0 pips を config に登録
    instruments = tl.default_instruments()
    print(f"instruments ({len(instruments)}): {instruments}")
    print("spreads(pips):", {nm: config.spread_pips(nm) for nm in instruments})

    close = load_close_matrix(instruments)
    print(f"D1 grid: {close.index[0]} .. {close.index[-1]}  ({len(close)} bars)")

    results = []
    for lb in GRID_LB:
        for k in GRID_K:
            pool = build_xs_pool(lb, k, close)
            st = tl.pool_stats(pool)
            label = f"xsmom_lb{lb}_k{k}_D1_both"
            results.append((label, lb, k, st))
            print(f"\n=== {label} ===")
            print(st)
    print("\n--- summary ---")
    for label, lb, k, st in results:
        print(f"{label}: n={st['n']} pf={st['pool_pf']} is={st['is_pf']} oos={st['oos_pf']} "
              f"mean_bps={st['mean_bps']} yearly={st['yearly_pos']} worst={st['worst_year']}")


if __name__ == "__main__":
    main()
