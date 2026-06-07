"""年次(暦年)ごとの成績分解 — 目標の合否判定インフラ。

最終目標は「**どの年・どの通貨ペアでも年間プラス / PF 2.0 以上 / 年間 100 取引以上**」。
これを直接測るために、全期間で 1 回バックテストし、トレードを**決済年**でグループ化して
年ごとに集計する。利点:

  * 指標のウォームアップ消費は最初の 1 回だけ(=実運用に近い。年で切ると毎年頭が無効になる)。
  * 年跨ぎトレードも「決済した年」に正しく計上される。

PF を安定させるため、サイジングは既定で `value`(固定建玉=非複利)を使う。
複利(`full`)だと後年のトレードほど建玉が大きくなり、年次 PF が建玉スケールで歪むため。

主な関数:
  * yearly(pair, tf, gen, params)        : 1 ペアの年次成績テーブル
  * yearly_matrix(tf, gen, params, metric): 7 ペア × 年 のマトリクス(既定 profit_factor)
  * acceptance(tf, gen, params)          : 目標に対する合否サマリ(全ペア×全年)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import run
from .data import available_pairs, load
from .trades import trade_table

# 年次分析の既定サイジング(非複利=年ごとを公平比較できる)
_DEFAULT_SIZE = {"size_mode": "value", "size_value": None}


def profit_factor(pnl: pd.Series) -> float:
    """総利益 / 総損失。損失ゼロなら inf(利益あり)/ nan(取引なし)。"""
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    if gl == 0:
        return float("inf") if gp > 0 else float("nan")
    return gp / gl


def yearly(
    pair: str,
    tf: str,
    generate_signals,
    params: dict | None = None,
    *,
    data: pd.DataFrame | None = None,
    init_cash: float = 10_000,
    size_mode: str | None = None,
    size_value: float | None = None,
    **run_kw,
) -> pd.DataFrame:
    """1 ペアの年次成績。index=暦年, columns=[trades, win_rate, profit_factor, pnl, return_pct, avg_pnl]。

    決済年でグループ化して realized PnL を集計する。size_mode 未指定なら value(固定建玉)。
    data を渡すとその DataFrame を使う(テスト/期間スライス用)。
    """
    sz = dict(_DEFAULT_SIZE)
    if size_mode is not None:
        sz = {"size_mode": size_mode, "size_value": size_value}
    if data is None:
        data = load(pair, tf)
    pf = run(pair, tf, generate_signals, params, data=data, init_cash=init_cash, **sz, **run_kw)
    tt = trade_table(pf, data)
    if tt.empty:
        return pd.DataFrame(
            columns=["trades", "win_rate", "profit_factor", "pnl", "return_pct", "avg_pnl"]
        )
    tt = tt.assign(year=pd.DatetimeIndex(tt["exit"]).year)
    rows = {}
    for year, g in tt.groupby("year"):
        pnl = g["pnl"]
        rows[int(year)] = {
            "trades": int(len(g)),
            "win_rate": float((pnl > 0).mean()),
            "profit_factor": profit_factor(pnl),
            "pnl": float(pnl.sum()),
            "return_pct": float(pnl.sum() / init_cash * 100),
            "avg_pnl": float(pnl.mean()),
            "gross_profit": float(pnl[pnl > 0].sum()),
            "gross_loss": float(-pnl[pnl < 0].sum()),
        }
    return pd.DataFrame(rows).T


def portfolio_yearly(
    tf: str,
    generate_signals,
    params: dict | None = None,
    *,
    pairs: list[str] | None = None,
    init_cash: float = 10_000,
    **kw,
) -> pd.DataFrame:
    """7 ペアを等加重で合算したポートフォリオの年次成績。

    各ペアを固定建玉で回し、決済年で合算する。年100取引・毎年プラスは
    ポートフォリオ単位で達成する方が現実的なため、その判定に使う。
    index=暦年, columns=[trades, win_rate, profit_factor, pnl, return_pct, pairs_positive]。
    """
    pairs = pairs or available_pairs()
    per = {}
    for pair in pairs:
        try:
            per[pair] = yearly(pair, tf, generate_signals, params, init_cash=init_cash, **kw)
        except Exception:  # noqa: BLE001
            continue
    if not per:
        return pd.DataFrame()
    years = sorted({y for tbl in per.values() for y in tbl.index})
    rows = {}
    for year in years:
        gp = gl = pnl = trades = 0.0
        pos = tot = 0
        for tbl in per.values():
            if year in tbl.index:
                r = tbl.loc[year]
                gp += r["gross_profit"]; gl += r["gross_loss"]
                pnl += r["pnl"]; trades += r["trades"]
                tot += 1
                pos += int(r["pnl"] > 0)
        rows[int(year)] = {
            "trades": int(trades),
            "profit_factor": (gp / gl) if gl > 0 else float("inf"),
            "pnl": pnl,
            "return_pct": pnl / (init_cash * max(tot, 1)) * 100,  # ペア平均リターン%
            "pairs_positive": f"{pos}/{tot}",
        }
    return pd.DataFrame(rows).T


def yearly_matrix(
    tf: str,
    generate_signals,
    params: dict | None = None,
    *,
    metric: str = "profit_factor",
    pairs: list[str] | None = None,
    **kw,
) -> pd.DataFrame:
    """7 ペア × 年 のマトリクス(値 = metric)。index=pair, columns=year。"""
    pairs = pairs or available_pairs()
    rows = {}
    for pair in pairs:
        try:
            y = yearly(pair, tf, generate_signals, params, **kw)
        except Exception:  # noqa: BLE001
            continue
        if not y.empty and metric in y:
            rows[pair] = y[metric]
    if not rows:
        return pd.DataFrame()
    mat = pd.DataFrame(rows).T
    return mat.reindex(sorted(mat.columns), axis=1)


def acceptance(
    tf: str,
    generate_signals,
    params: dict | None = None,
    *,
    pf_target: float = 2.0,
    min_trades_per_year: int = 100,
    min_trades_for_check: int = 10,
    pairs: list[str] | None = None,
    **kw,
) -> dict:
    """目標に対する合否を全ペア×全年で判定。

    返り値 dict:
      per_pair   : ペアごとの要約 DataFrame(年数 / プラス年率 / 最小PF / 中央PF / 年平均取引数)
      cells      : (pair, year) フラット表(trades/pnl/profit_factor/positive)
      verdict    : 全体判定 dict(pass_positive / pass_pf / pass_trades / overall)

    判定方針(透明性重視):
      * 取引数が極端に少ない年(< min_trades_for_check)は「年プラス」判定から除外し partial として記録。
      * pass_positive : 上記を除く全 (pair, year) で pnl > 0。
      * pass_pf       : 同セルの profit_factor がすべて pf_target 以上。
      * pass_trades   : 各ペアの年平均取引数が min_trades_per_year 以上(全ペア)。
    """
    pairs = pairs or available_pairs()
    cells = []
    per_pair = {}
    for pair in pairs:
        try:
            y = yearly(pair, tf, generate_signals, params, **kw)
        except Exception:  # noqa: BLE001
            continue
        if y.empty:
            continue
        checkable = y[y["trades"] >= min_trades_for_check]
        for year, r in y.iterrows():
            cells.append({
                "pair": pair,
                "year": int(year),
                "trades": int(r["trades"]),
                "pnl": r["pnl"],
                "profit_factor": r["profit_factor"],
                "positive": bool(r["pnl"] > 0),
                "checked": bool(r["trades"] >= min_trades_for_check),
            })
        pos_years = checkable[checkable["pnl"] > 0]
        per_pair[pair] = {
            "years": int(len(y)),
            "checked_years": int(len(checkable)),
            "positive_year_rate": float((checkable["pnl"] > 0).mean()) if len(checkable) else float("nan"),
            "min_pf": float(checkable["profit_factor"].replace(np.inf, np.nan).min()) if len(checkable) else float("nan"),
            "median_pf": float(checkable["profit_factor"].replace(np.inf, np.nan).median()) if len(checkable) else float("nan"),
            "avg_trades_per_year": float(checkable["trades"].mean()) if len(checkable) else float("nan"),
            "total_pnl": float(y["pnl"].sum()),
        }

    cells_df = pd.DataFrame(cells)
    per_pair_df = pd.DataFrame(per_pair).T

    if cells_df.empty:
        verdict = {"pass_positive": False, "pass_pf": False, "pass_trades": False, "overall": False}
        return {"per_pair": per_pair_df, "cells": cells_df, "verdict": verdict}

    checked = cells_df[cells_df["checked"]]
    pass_positive = bool(checked["positive"].all()) if len(checked) else False
    pf_ok = checked["profit_factor"].replace(np.inf, np.nan).fillna(0) >= pf_target
    pass_pf = bool(pf_ok.all()) if len(checked) else False
    pass_trades = bool((per_pair_df["avg_trades_per_year"] >= min_trades_per_year).all()) if len(per_pair_df) else False
    verdict = {
        "pass_positive": pass_positive,
        "pass_pf": pass_pf,
        "pass_trades": pass_trades,
        "overall": pass_positive and pass_pf and pass_trades,
        "negative_cells": int((~checked["positive"]).sum()) if len(checked) else 0,
        "pf_target": pf_target,
        "min_pf_overall": float(checked["profit_factor"].replace(np.inf, np.nan).min()) if len(checked) else float("nan"),
        "frac_cells_pf_ok": float(pf_ok.mean()) if len(checked) else float("nan"),
    }
    return {"per_pair": per_pair_df, "cells": cells_df, "verdict": verdict}
