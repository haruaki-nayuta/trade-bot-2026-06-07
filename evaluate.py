"""アイデアを放り込む入口 — 戦略を10年スパンで総合評価し、改善提案まで出す。

  uv run python evaluate.py ma_cross                       # EURUSD H1 で総合評価
  uv run python evaluate.py rsi_meanrev --pair USDJPY --tf H4
  uv run python evaluate.py ma_cross --save                # results/ に Markdown 保存

やること: パラメータ探索 / IS・OOS(過剰最適化) / 7ペア横断 / マルチ時間足 /
ロング・ショート分離 / 損切り・利確の自動テスト → データ駆動の改善提案。
"""

from __future__ import annotations

import argparse
import importlib
import time

import pandas as pd

from fxlab import config, evaluate as ev

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

_ROUND = {
    "total_return": 4, "sharpe": 2, "sortino": 2, "max_drawdown": 4,
    "win_rate": 3, "profit_factor": 2, "num_trades": 0, "expectancy": 2,
}


def _fmt(df: pd.DataFrame) -> str:
    df = df.copy()
    for c, n in _ROUND.items():
        if c in df.columns:
            df[c] = df[c].round(n)
    return df.to_string()


def _verdict(out: dict) -> str:
    oos = out["is_oos"]["oos"]["sharpe"]
    pf = out["best_metrics"].get("profit_factor")
    if pd.isna(oos):
        return "判定不能(取引が成立せず)"
    if oos > 0.5 and pd.notna(pf) and pf > 1.2:
        return f"有望 — OOS Sharpe {oos:+.2f} / PF {pf:.2f}。下記の改善で実運用検証へ。"
    if oos > 0:
        return f"改善の余地あり — OOS Sharpe {oos:+.2f}。下記提案で底上げを。"
    return f"現状では非推奨 — OOS Sharpe {oos:+.2f}(アウトサンプルで通用せず)。提案を参照。"


def build_report(out: dict) -> str:
    p0, p1 = out["period"]
    L = []
    L.append(f"# 検証レポート: {out['strategy']}  ({out['primary_pair']} {out['primary_tf']})")
    L.append("")
    L.append(f"- 期間: **{p0:%Y-%m-%d} 〜 {p1:%Y-%m-%d}(約 {out['years']} 年 / {out['bars']:,} 本)**")
    L.append(f"- 採用パラメータ: `{out['best_params']}`")
    sz = out.get("size_mode", "full")
    if sz != "full":
        L.append(f"- サイジング: **{sz}**" + (f" ({out.get('size_value')})" if out.get("size_value") is not None else ""))
    L.append(f"- **総合判定: {_verdict(out)}**")
    L.append("")

    # 改善提案を上部に(雑に投げて改善案が欲しいニーズ)
    L.append("## 🔧 改善提案")
    for i, d in enumerate(out["diagnostics"], 1):
        L.append(f"{i}. **{d['issue']}**")
        L.append(f"   - 根拠: {d['evidence']}")
        L.append(f"   - 提案: {d['suggestion']}")
    L.append("")

    L.append(f"## 📊 採用パラメータの成績(全期間 約{out['years']}年)")
    L.append("```")
    L.append(out["best_metrics"].round(4).to_string())
    L.append("```")
    L.append("")

    io = out["is_oos"]
    L.append("## イン/アウトサンプル(過剰最適化チェック)")
    L.append(f"- IS(前半70%): {io['is_period'][0]:%Y-%m-%d}〜{io['is_period'][1]:%Y-%m-%d} で最適化 → `{io['is_params']}`")
    L.append(f"- OOS(後半30%): {io['oos_period'][0]:%Y-%m-%d}〜{io['oos_period'][1]:%Y-%m-%d} で素の成績")
    L.append("```")
    L.append(_fmt(pd.DataFrame({"IS": io["is"], "OOS": io["oos"]}).T))
    L.append("```")
    L.append("")

    if not out["all_pairs"].empty:
        L.append("## 7ペア横断(同一パラメータの頑健性)")
        L.append("```")
        L.append(_fmt(out["all_pairs"]))
        L.append("```")
        L.append("")

    if not out["timeframes"].empty:
        L.append("## マルチ時間足")
        L.append("```")
        L.append(_fmt(out["timeframes"]))
        L.append("```")
        L.append("")

    L.append("## ロング/ショート分離")
    L.append("```")
    L.append(_fmt(out["sides"]))
    L.append("```")
    L.append("")

    L.append("## 損切り/利確/トレーリング 自動テスト")
    L.append("```")
    L.append(_fmt(out["stops"]))
    L.append("```")
    L.append("")

    if out["sweep"] is not None:
        L.append("## パラメータ探索 上位")
        L.append("```")
        L.append(_fmt(out["sweep"].head(10)))
        L.append("```")
    return "\n".join(L)


def _wf_section(wf: dict) -> str:
    L = ["## 🔁 ウォークフォワード最適化(毎回その時点までで最適化→次窓で素検証)"]
    folds = wf["folds"]
    if folds.empty:
        L.append("(データ不足でfoldを構成できず)")
        return "\n".join(L)
    s = wf["summary"]
    L.append(f"- OOSでプラスだったfold: **{wf['consistency']*100:.0f}%**  "
             f"(平均 Sharpe {s['sharpe']:+.2f} / 平均リターン {s['total_return']:+.2%} / 平均DD {s['max_drawdown']:.2%})")
    disp = folds.copy()
    disp["params"] = disp["params"].astype(str)
    for c in ("test_start", "test_end"):
        disp[c] = pd.to_datetime(disp[c]).dt.strftime("%Y-%m-%d")
    for c in ("total_return", "sharpe", "max_drawdown"):
        disp[c] = disp[c].round(3)
    L.append("```")
    L.append(disp.to_string(index=False))
    L.append("```")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="戦略を10年スパンで総合評価＋改善提案")
    ap.add_argument("strategy", help="strategies/ のモジュール名(拡張子なし)")
    ap.add_argument("--pair", default="EURUSD")
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--tfs", nargs="*", help="マルチ時間足の対象(既定: M15 H1 H4 D1)")
    ap.add_argument("--size", default="full", choices=["full", "value", "amount", "risk"],
                    help="サイジング: full(複利) / value(固定額) / amount(固定数量) / risk(リスク%)")
    ap.add_argument("--size-value", type=float, help="size の値(risk なら 0.01=1%, value なら金額 等)")
    ap.add_argument("--wf", action="store_true", help="ウォークフォワード最適化も実施")
    ap.add_argument("--folds", type=int, default=5, help="ウォークフォワードの分割数")
    ap.add_argument("--save", action="store_true", help="results/ に Markdown 保存")
    args = ap.parse_args()

    mod = importlib.import_module(f"strategies.{args.strategy}")
    t0 = time.time()
    out = ev.evaluate(
        args.strategy, mod,
        primary_pair=args.pair, primary_tf=args.tf, timeframes=args.tfs,
        size_mode=args.size, size_value=args.size_value,
    )
    report = build_report(out)
    if args.wf:
        wf = ev.walk_forward(args.strategy, mod, pair=args.pair, tf=args.tf, n_folds=args.folds)
        report += "\n\n" + _wf_section(wf)
    print(report)
    print(f"\n(評価時間 {time.time()-t0:.1f}s)")
    if args.save:
        p = config.RESULTS_DIR / f"eval_{args.strategy}_{args.pair}_{args.tf}.md"
        p.write_text(report, encoding="utf-8")
        print(f"保存: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
