"""ユニバース・ポートフォリオ評価 CLI(メジャー7 + 合成クロス13 = 20対象)。

本リポジトリの到達点である「分散ポートフォリオ運用」を再現・評価する正式ツール。
チャンピオン `confluence_meanrev` をはじめ、close ベースの戦略をユニバース横断で等加重運用し、
年次成績と目標(PF/毎年プラス/年取引)に対する合否を出す。

  uv run python portfolio.py confluence_meanrev               # 既定 H4・20対象
  uv run python portfolio.py confluence_meanrev --no-crosses  # メジャー7のみ
  uv run python portfolio.py confluence_meanrev --cross-spread 4 --save
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np
import pandas as pd

from fxlab import config, universe as uni

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def parse_params(s):
    if not s:
        return None
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        try:
            out[k.strip()] = int(v)
        except ValueError:
            out[k.strip()] = float(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="ユニバース・ポートフォリオ評価")
    ap.add_argument("strategy")
    ap.add_argument("--tf", default="H4")
    ap.add_argument("--params", help="PARAMS 上書き(例 slow_z=1.75)")
    ap.add_argument("--no-crosses", action="store_true", help="メジャー7のみで評価")
    ap.add_argument("--exclude", nargs="+", default=[], help="除外する対象(例: AUDJPY=トレンド性が強く平均回帰に不適)")
    ap.add_argument("--cross-spread", type=float, default=uni.CROSS_SPREAD_PIPS, help="クロスの往復スプレッド(pips)")
    ap.add_argument("--pf-target", type=float, default=2.0)
    ap.add_argument("--min-trades", type=int, default=100)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    uni.register_cross_spreads(args.cross_spread)
    mod = importlib.import_module(f"strategies.{args.strategy}")
    params = parse_params(args.params) or getattr(mod, "PARAMS", {})
    instruments = [x for x in uni.universe(crosses=not args.no_crosses) if x not in set(args.exclude)]

    print(f"ポートフォリオ評価: {args.strategy} on {args.tf}  対象{len(instruments)} "
          f"(crosses={'off' if args.no_crosses else 'on'}, cross_spread={args.cross_spread}pips)")
    print(f"params: {params}\n")

    res = uni.acceptance(args.tf, mod.generate_signals, params, pf_target=args.pf_target,
                         min_trades=args.min_trades, instruments=instruments, size_mode="value")
    port = res["port"]
    v = res["verdict"]
    show = port.copy()
    show["profit_factor"] = show["profit_factor"].replace(np.inf, np.nan).round(2)
    show["pnl"] = show["pnl"].round(0)
    show["return_pct"] = show["return_pct"].round(2)
    print("=== ポートフォリオ年次 ===")
    print(show.to_string(), "\n")

    def mk(b):
        return "✅" if b else "❌"
    print("=== 🎯 目標判定(ポートフォリオ)===")
    print(f"  毎年プラス     : {mk(v['pass_positive'])}  (プラス年率 {v['positive_year_rate']:.0%})")
    print(f"  PF≥{args.pf_target}        : {mk(v['pass_pf'])}  (中央 {v['pf_median']:.2f} / 最小 {v['pf_min']:.2f})")
    print(f"  年{args.min_trades}取引以上 : {mk(v['pass_trades'])}  (年平均 {v['avg_trades']})")
    print(f"  ── 総合       : {mk(v['overall'])}")

    if args.save:
        p = config.RESULTS_DIR / f"portfolio_{args.strategy}_{args.tf}.csv"
        port.to_csv(p)
        print(f"\n保存: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
