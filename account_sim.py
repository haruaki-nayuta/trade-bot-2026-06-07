"""口座レベル・マネーマネジメント・シミュレータ(実運用化の核心)。

ユニバース全対象のトレードを1つの口座で時系列に処理し、複利・同時建玉上限・資金配分を
反映した「実際に運用したらどうなるか」を出す。チャンピオンは損切りを置かず平均回帰で手仕舞い
するため、リスク管理は **①1トレードの配分 ②同時建玉数上限 ③ボラフィルタ** が担う設計。

  uv run python account_sim.py                       # 推奨構成で口座シミュレーション
  uv run python account_sim.py --max-pos 6 --deploy 1.0
  uv run python account_sim.py --entry-delay 1       # 約定を1バー遅らせて頑健性確認

出力: 年次リターン / 通算 / CAGR / 最大DD / Sharpe / 同時建玉の分布。検証専用(発注しない)。
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np
import pandas as pd

from fxlab import universe as uni
from fxlab.backtest import run
from fxlab.trades import trade_table

pd.set_option("display.width", 200)
TF_DEFAULT = "H4"


def collect_trades(strategy, params, tf, instruments, entry_delay=0):
    """全対象のトレードを (entry, exit, ret) で集める。entry_delay>0 でエントリーをNバー遅延。"""
    gen = strategy.generate_signals
    all_tr = []
    for nm in instruments:
        data = uni.instrument_data(nm, tf)
        if entry_delay > 0:
            le, lx, se, sx = gen(data, **params)
            le = le.shift(entry_delay, fill_value=False)
            se = se.shift(entry_delay, fill_value=False)
            g2 = lambda d, _le=le, _lx=lx, _se=se, _sx=sx, **k: (_le, _lx, _se, _sx)  # noqa: E731
            pf = run(nm, tf, g2, {}, data=data, size_mode="value")
        else:
            pf = run(nm, tf, gen, params, data=data, size_mode="value")
        tt = trade_table(pf, data)
        for _, r in tt.iterrows():
            all_tr.append({"instr": nm, "entry": r["entry"], "exit": r["exit"],
                           "ret": r["return_pct"] / 100.0})
    df = pd.DataFrame(all_tr).sort_values("entry").reset_index(drop=True)
    return df


def simulate(trades, init=10_000.0, max_pos=6, deploy=1.0):
    """単一口座・複利・同時建玉上限のシミュレーション。

    各トレードに equity*(deploy/max_pos) を配分(満玉で deploy 比率を運用)。
    決済時に pnl=alloc*ret を equity に反映。max_pos 超過分は見送り(過大リスク回避)。
    """
    equity = init
    open_pos = []  # (exit_ts, alloc, ret)
    eq_curve = []  # (time, equity_realized)
    conc = []
    skipped = 0
    weight = deploy / max_pos
    for _, t in trades.iterrows():
        now = t["entry"]
        # これ以前に決済されるポジションを実現
        still = []
        for ex, alloc, ret in open_pos:
            if ex <= now:
                equity += alloc * ret
                eq_curve.append((ex, equity))
            else:
                still.append((ex, alloc, ret))
        open_pos = still
        if len(open_pos) >= max_pos:
            skipped += 1
            continue
        alloc = equity * weight
        open_pos.append((t["exit"], alloc, t["ret"]))
        conc.append(len(open_pos))
    for ex, alloc, ret in sorted(open_pos):
        equity += alloc * ret
        eq_curve.append((ex, equity))

    eq = pd.Series({t: v for t, v in eq_curve}).sort_index()
    eq = eq[~eq.index.duplicated(keep="last")]
    return eq, {"final": equity, "skipped": skipped, "max_conc": max(conc) if conc else 0,
                "avg_conc": float(np.mean(conc)) if conc else 0}


def report(eq, info, init):
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / init) ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    yearly = eq.groupby(eq.index.year).last()
    yr_ret = yearly.pct_change()
    yr_ret.iloc[0] = yearly.iloc[0] / init - 1
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(len(rets) / max(years, 1e-9)) if rets.std() > 0 else float("nan")
    print(f"通算リターン : {eq.iloc[-1]/init-1:+.1%}   最終資産: {eq.iloc[-1]:,.0f}(初期{init:,.0f})")
    print(f"CAGR        : {cagr:+.1%}")
    print(f"最大DD      : {dd:.1%}")
    print(f"Sharpe(概算): {sharpe:.2f}")
    print(f"同時建玉    : 最大{info['max_conc']} / 平均{info['avg_conc']:.1f} / 見送り{info['skipped']}件")
    print(f"プラス年率  : {(yr_ret>0).mean():.0%}  ({int((yr_ret>0).sum())}/{len(yr_ret)}年)")
    print("\n年次リターン:")
    print((yr_ret*100).round(1).to_string())


def main() -> int:
    ap = argparse.ArgumentParser(description="口座レベル資金管理シミュレーション")
    ap.add_argument("--strategy", default="confluence_meanrev")
    ap.add_argument("--tf", default=TF_DEFAULT)
    ap.add_argument("--exclude", nargs="+", default=["AUDJPY"])
    ap.add_argument("--max-pos", type=int, default=6)
    ap.add_argument("--deploy", type=float, default=1.0, help="満玉時の運用比率(1.0=同時建玉上限で全額)")
    ap.add_argument("--entry-delay", type=int, default=0, help="約定をNバー遅らせる(頑健性確認)")
    ap.add_argument("--cross-spread", type=float, default=uni.CROSS_SPREAD_PIPS)
    ap.add_argument("--init", type=float, default=10_000.0)
    args = ap.parse_args()

    uni.register_cross_spreads(args.cross_spread)
    mod = importlib.import_module(f"strategies.{args.strategy}")
    params = dict(getattr(mod, "PARAMS", {}))
    if args.strategy == "confluence_meanrev":
        params["slow_z"] = 1.75
    instruments = [x for x in uni.universe() if x not in set(args.exclude)]

    print(f"=== 口座シミュレーション: {args.strategy} on {args.tf}  対象{len(instruments)} ===")
    print(f"params={params}  max_pos={args.max_pos} deploy={args.deploy} entry_delay={args.entry_delay}\n")
    trades = collect_trades(mod, params, args.tf, instruments, entry_delay=args.entry_delay)
    print(f"総トレード数: {len(trades)}(年平均 {len(trades)/11:.0f})\n")
    eq, info = simulate(trades, init=args.init, max_pos=args.max_pos, deploy=args.deploy)
    report(eq, info, args.init)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
