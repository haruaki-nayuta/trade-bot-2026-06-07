"""目標基準スクリーナ — strategies/ 全戦略を「目標(PF2.0・毎年プラス・年100取引)」で一括ランク。

leaderboard.py が OOS Sharpe 中心なのに対し、こちらは**最終目標そのもの**で評価する:
  * ポートフォリオ(7ペア等加重)の年次 PF と「プラス年率」
  * ペア×年セルのうち PF≥target を満たす割合 / 最小 PF
  * 年間取引数(ポートフォリオ合算 と ペア平均)

各戦略はデフォルト PARAMS を全ペア・全年に固定適用(=カーブフィットしない素の実力)。

  uv run python screen.py                       # 既定 H1。--tf で変更、--tfs で複数足
  uv run python screen.py --tfs H4 D1 --save
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import time

import numpy as np
import pandas as pd

import strategies
from fxlab import config
from fxlab import yearly as ylib

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)


def _discover() -> list[str]:
    return sorted(m.name for m in pkgutil.iter_modules(strategies.__path__)
                  if not m.name.startswith("_"))


def screen_one(name: str, tf: str, pf_target: float, min_trades: int, **kw) -> dict | None:
    mod = importlib.import_module(f"strategies.{name}")
    params = getattr(mod, "PARAMS", {})
    gen = mod.generate_signals
    try:
        port = ylib.portfolio_yearly(tf, gen, params, **kw)
        acc = ylib.acceptance(tf, gen, params, pf_target=pf_target,
                              min_trades_per_year=min_trades, **kw)
    except Exception as e:  # noqa: BLE001
        print(f"  {name}: 失敗 ({e})")
        return None
    if port.empty:
        return None
    pf_series = port["profit_factor"].replace(np.inf, np.nan)
    v = acc["verdict"]
    return {
        "strategy": name,
        "params": str(params),
        "port_pos_year%": round(float((port["pnl"] > 0).mean()) * 100, 0),
        "port_median_PF": round(float(pf_series.median()), 2),
        "port_min_PF": round(float(pf_series.min()), 2),
        "port_trades/yr": int(port["trades"].mean()),
        "cells_pos%": round((1 - v.get("negative_cells", 0) /
                             max(len(acc["cells"][acc["cells"]["checked"]]), 1)) * 100, 0),
        "cells_PFok%": round(v.get("frac_cells_pf_ok", float("nan")) * 100, 0),
        "min_cell_PF": round(v.get("min_pf_overall", float("nan")), 2),
        "overall": "✅" if v.get("overall") else "—",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="目標基準スクリーナ")
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--tfs", nargs="+", help="複数足を一括評価(例: --tfs H4 D1)")
    ap.add_argument("--size", default="value", choices=["full", "value", "amount", "risk"])
    ap.add_argument("--size-value", type=float)
    ap.add_argument("--pf-target", type=float, default=2.0)
    ap.add_argument("--min-trades", type=int, default=100)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    tfs = args.tfs or [args.tf]
    names = _discover()
    kw = {"size_mode": args.size, "size_value": args.size_value}
    print(f"対象戦略({len(names)}): {', '.join(names)}\n")

    for tf in tfs:
        t0 = time.time()
        print(f"################  時間足 {tf}  (size={args.size})  ################")
        rows = []
        for name in names:
            r = screen_one(name, tf, args.pf_target, args.min_trades, **kw)
            if r:
                rows.append(r)
                print(f"  ✓ {name}")
        if not rows:
            print("  評価できた戦略なし\n"); continue
        board = pd.DataFrame(rows).sort_values(
            ["port_pos_year%", "port_median_PF"], ascending=False).reset_index(drop=True)
        print(f"\n=== 🎯 目標スクリーン({tf})  ポート=7ペア等加重 / セル=ペア×年 ===")
        print(board.to_string(index=False))
        print(f"\n({len(rows)}戦略 / {time.time()-t0:.1f}s)\n")
        if args.save:
            p = config.RESULTS_DIR / f"screen_{tf}.csv"
            board.to_csv(p, index=False)
            print(f"保存: {p}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
