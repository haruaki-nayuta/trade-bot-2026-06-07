"""strategies/ の全戦略を横並び比較するリーダーボード。

  uv run python leaderboard.py                         # EURUSD H1 で全戦略を評価・比較
  uv run python leaderboard.py --pair USDJPY --tf H4
  uv run python leaderboard.py --size risk --size-value 0.01
  uv run python leaderboard.py --save

各戦略を 10年総合評価(evaluate)し、要点を1表に集約。OOS Sharpe 降順で並べる。
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import time

import pandas as pd

import strategies
from fxlab import config, evaluate as ev

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def _discover() -> list[str]:
    return sorted(m.name for m in pkgutil.iter_modules(strategies.__path__)
                  if not m.name.startswith("_"))


def main() -> int:
    ap = argparse.ArgumentParser(description="全戦略のリーダーボード")
    ap.add_argument("--pair", default="EURUSD")
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--size", default="full", choices=["full", "value", "amount", "risk"])
    ap.add_argument("--size-value", type=float)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    names = _discover()
    print(f"対象戦略: {', '.join(names)}  on {args.pair} {args.tf}\n")
    t0 = time.time()
    rows = []
    for name in names:
        try:
            mod = importlib.import_module(f"strategies.{name}")
            out = ev.evaluate(name, mod, primary_pair=args.pair, primary_tf=args.tf,
                              size_mode=args.size, size_value=args.size_value)
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: 評価失敗 ({e})")
            continue
        bm = out["best_metrics"]
        oos = out["is_oos"]["oos"]
        ap_df = out["all_pairs"]
        pos = int((ap_df["sharpe"] > 0).sum()) if not ap_df.empty else 0
        tot = len(ap_df) if not ap_df.empty else 0
        rows.append({
            "strategy": name,
            "params": str(out["best_params"]),
            "return_10y": round(bm["total_return"], 3),
            "sharpe": round(bm["sharpe"], 2),
            "max_dd": round(bm["max_drawdown"], 3),
            "oos_sharpe": round(oos["sharpe"], 2),
            "oos_return": round(oos["total_return"], 3),
            "pairs+": f"{pos}/{tot}",
            "trades": int(bm["num_trades"]),
        })
        print(f"  ✓ {name}  (OOS Sharpe {oos['sharpe']:+.2f})")

    if not rows:
        print("評価できた戦略がありません。")
        return 1

    board = pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False).reset_index(drop=True)
    print(f"\n=== 🏁 リーダーボード({args.pair} {args.tf}, OOS Sharpe 降順)===")
    print(board.to_string(index=False))
    print(f"\n(評価 {len(rows)}戦略 / {time.time()-t0:.1f}s)")
    if args.save:
        p = config.RESULTS_DIR / f"leaderboard_{args.pair}_{args.tf}.csv"
        board.to_csv(p, index=False)
        print(f"保存: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
