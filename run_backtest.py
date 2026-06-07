"""戦略を名前で指定して検証する CLI。

  # 単発(strategies/ma_cross.py の PARAMS を使用)
  uv run python run_backtest.py ma_cross --pair EURUSD --tf H1

  # パラメータ上書き
  uv run python run_backtest.py ma_cross --pair EURUSD --tf H1 --params fast=10,slow=100

  # パラメータ総当り探索(PARAM_GRID を使用、並列・高速)
  uv run python run_backtest.py ma_cross --pair EURUSD --tf H1 --sweep

  # 7ペア横断で同一手法を比較
  uv run python run_backtest.py ma_cross --all-pairs --tf H1

  # 結果CSV保存 / チャートHTML出力
  uv run python run_backtest.py ma_cross --pair EURUSD --tf H1 --save --plot
"""

from __future__ import annotations

import argparse
import importlib
import time

import pandas as pd

from fxlab import backtest, config

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def parse_params(s: str | None) -> dict:
    if not s:
        return {}
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        k = k.strip()
        v = v.strip()
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="FX 戦略バックテスト runner")
    ap.add_argument("strategy", help="strategies/ のモジュール名(拡張子なし)")
    ap.add_argument("--pair", default="EURUSD", help="通貨ペア")
    ap.add_argument("--tf", default="H1", help="時間足 (M1/M5/M15/M30/H1/H4/D1/W1)")
    ap.add_argument("--params", help="例: fast=10,slow=100")
    ap.add_argument("--sweep", action="store_true", help="PARAM_GRID で総当り探索")
    ap.add_argument("--all-pairs", action="store_true", help="7ペア横断で比較")
    ap.add_argument("--size", default="full", choices=["full", "value", "amount", "risk"],
                    help="サイジング: full(複利)/value(固定額)/amount(固定数量)/risk(リスク%)")
    ap.add_argument("--size-value", type=float, help="size の値(risk なら 0.01=1% 等)")
    ap.add_argument("--save", action="store_true", help="結果を results/ に CSV 保存")
    ap.add_argument("--plot", action="store_true", help="チャートを results/ に HTML 出力")
    args = ap.parse_args()

    mod = importlib.import_module(f"strategies.{args.strategy}")
    gen = mod.generate_signals
    szkw = {"size_mode": args.size, "size_value": args.size_value}
    t0 = time.time()

    if args.sweep:
        grid = getattr(mod, "PARAM_GRID", None)
        if not grid:
            print("この戦略に PARAM_GRID がありません")
            return 1
        n = 1
        for v in grid.values():
            n *= len(v)
        print(f"探索: {args.strategy} on {args.pair} {args.tf} — {n} 通り\n")
        res = backtest.sweep(args.pair, args.tf, gen, grid, **szkw)
        print(res.head(15).to_string())
        print(f"\n{time.time()-t0:.1f}s")
        if args.save:
            p = config.RESULTS_DIR / f"sweep_{args.strategy}_{args.pair}_{args.tf}.csv"
            res.to_csv(p)
            print(f"保存: {p}")
        return 0

    params = parse_params(args.params) or getattr(mod, "PARAMS", {})

    if args.all_pairs:
        print(f"横断比較: {args.strategy} {params} on {args.tf}\n")
        rows = {}
        for pair in config.PAIRS:
            try:
                pf = backtest.run(pair, args.tf, gen, params, **szkw)
                rows[pair] = backtest.metrics(pf).iloc[0]
            except FileNotFoundError:
                print(f"  {pair}: データ未取得 — スキップ")
        table = pd.DataFrame(rows).T
        print(table.to_string())
        print(f"\n{time.time()-t0:.1f}s")
        if args.save:
            p = config.RESULTS_DIR / f"allpairs_{args.strategy}_{args.tf}.csv"
            table.to_csv(p)
            print(f"保存: {p}")
        return 0

    # 単発
    print(f"単発: {args.strategy} {params} on {args.pair} {args.tf}\n")
    pf = backtest.run(args.pair, args.tf, gen, params, **szkw)
    print(backtest.metrics(pf).iloc[0].to_string())
    print(f"\n{time.time()-t0:.1f}s")
    if args.plot:
        p = config.RESULTS_DIR / f"chart_{args.strategy}_{args.pair}_{args.tf}.html"
        pf.plot().write_html(str(p))
        print(f"チャート: {p}")
    if args.save:
        p = config.RESULTS_DIR / f"run_{args.strategy}_{args.pair}_{args.tf}.csv"
        backtest.metrics(pf).to_csv(p)
        print(f"保存: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
