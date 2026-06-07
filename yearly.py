"""年次×ペア の合否判定 CLI — 目標「どの年・どのペアでも年間プラス / PF2.0 / 年100取引」を直接検査する。

  uv run python yearly.py rsi_meanrev                         # EURUSD H1、全ペア×年のPF/損益/取引
  uv run python yearly.py rsi_meanrev --tf H4 --params period=14,low=25,high=75
  uv run python yearly.py rsi_meanrev --size risk --size-value 0.01
  uv run python yearly.py rsi_meanrev --pf-target 2.0 --min-trades 100 --save

各ペアを全期間で 1 回バックテストし、決済年でグループ化して年次成績に分解。
PF マトリクス / リターン% マトリクス / 取引数マトリクス と、目標に対する合否を出力する。
既定サイジングは value(固定建玉=非複利)で年ごとを公平比較する(--size で変更可)。
"""

from __future__ import annotations

import argparse
import importlib
import time

import pandas as pd

from fxlab import config
from fxlab import yearly as ylib

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def parse_params(s: str | None) -> dict:
    if not s:
        return {}
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        k, v = k.strip(), v.strip()
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def _fmt(mat: pd.DataFrame, nd: int = 2) -> str:
    return mat.round(nd).to_string()


def main() -> int:
    ap = argparse.ArgumentParser(description="年次×ペア 合否判定")
    ap.add_argument("strategy", help="strategies/ のモジュール名")
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--params", help="例: period=14,low=25,high=75(省略時は PARAMS)")
    ap.add_argument("--size", default="value", choices=["full", "value", "amount", "risk"])
    ap.add_argument("--size-value", type=float)
    ap.add_argument("--pf-target", type=float, default=2.0)
    ap.add_argument("--min-trades", type=int, default=100, help="年間取引数の下限")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    mod = importlib.import_module(f"strategies.{args.strategy}")
    gen = mod.generate_signals
    params = parse_params(args.params) or getattr(mod, "PARAMS", {})
    kw = {"size_mode": args.size, "size_value": args.size_value}

    print(f"年次評価: {args.strategy} {params} on {args.tf}  (size={args.size})\n")
    t0 = time.time()

    pf_mat = ylib.yearly_matrix(args.tf, gen, params, metric="profit_factor", **kw)
    ret_mat = ylib.yearly_matrix(args.tf, gen, params, metric="return_pct", **kw)
    trd_mat = ylib.yearly_matrix(args.tf, gen, params, metric="trades", **kw)
    acc = ylib.acceptance(args.tf, gen, params, pf_target=args.pf_target,
                          min_trades_per_year=args.min_trades, **kw)

    print("=== PF(ペア×年) ===")
    print(_fmt(pf_mat, 2), "\n")
    print("=== リターン%(固定建玉, ペア×年) ===")
    print(_fmt(ret_mat, 1), "\n")
    print("=== 取引数(ペア×年) ===")
    print(_fmt(trd_mat, 0), "\n")

    print("=== ペア別サマリ ===")
    print(acc["per_pair"].round(2).to_string(), "\n")

    v = acc["verdict"]
    def mark(b):  # noqa: E306
        return "✅" if b else "❌"
    print("=== 🎯 目標に対する判定 ===")
    print(f"  全年プラス     : {mark(v['pass_positive'])}  (マイナス年セル {v.get('negative_cells','?')} 件)")
    print(f"  PF≥{args.pf_target}        : {mark(v['pass_pf'])}  (最小PF {v.get('min_pf_overall', float('nan')):.2f} / 達成率 {v.get('frac_cells_pf_ok', float('nan')):.0%})")
    print(f"  年{args.min_trades}取引以上 : {mark(v['pass_trades'])}")
    print(f"  ── 総合       : {mark(v['overall'])}")
    print(f"\n({time.time()-t0:.1f}s)")

    if args.save:
        stem = f"yearly_{args.strategy}_{args.tf}"
        pf_mat.to_csv(config.RESULTS_DIR / f"{stem}_pf.csv")
        ret_mat.to_csv(config.RESULTS_DIR / f"{stem}_return.csv")
        acc["cells"].to_csv(config.RESULTS_DIR / f"{stem}_cells.csv", index=False)
        print(f"保存: results/{stem}_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
