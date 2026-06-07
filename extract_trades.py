"""特に良かった/悪かったトレードと、その「トレード前の値動き」を抽出する CLI。

  uv run python extract_trades.py ma_cross --pair EURUSD --tf H1
  uv run python extract_trades.py ma_cross --pair EURUSD --tf H1 --n 10 --lookback 80
  uv run python extract_trades.py ma_cross --pair EURUSD --tf H1 --params fast=30,slow=100
  uv run python extract_trades.py ma_cross --pair EURUSD --tf H1 --save --plot

出力:
  * ベスト/ワースト n 件のトレード(損益・保有・直前の特徴量)を表示
  * --save : 全トレード+特徴量CSV と、各トレードの値動きOHLCV(直前+建玉中)CSV
  * --plot : 各トレードのローソク足チャート(エントリー/エグジット印・直前の値動き付き)
"""

from __future__ import annotations

import argparse
import importlib
import time

import pandas as pd

from fxlab import backtest, config, trades

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

# コンソール表示する列(全列＋αは CSV に保存)
_SHOW = [
    "entry", "dir", "return_pct", "pnl", "bars_held",
    "pre_ret_%", "pre_trend_%/bar", "pre_vol_%", "up_bar_ratio",
    "rsi_at_entry", "atr_at_entry_%", "dist_from_high_%", "dist_from_low_%",
]


def _parse_params(s, mod):
    if not s:
        return getattr(mod, "PARAMS", {})
    out = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        try:
            out[k.strip()] = int(v)
        except ValueError:
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v.strip()
    return out


def _fmt(df: pd.DataFrame) -> str:
    d = df.copy()
    d["entry"] = pd.to_datetime(d["entry"]).dt.strftime("%Y-%m-%d %H:%M")
    for c in ("return_pct", "pnl"):
        d[c] = d[c].round(3)
    cols = [c for c in _SHOW if c in d.columns]
    return d[cols].to_string(index=False)


def _plot(ctx, row, title, path):
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Candlestick(
            x=ctx.index, open=ctx["open"], high=ctx["high"],
            low=ctx["low"], close=ctx["close"], name="OHLC",
        )
    )
    fig.add_vline(x=row["entry"], line_color="#2563eb", line_width=2)   # エントリー
    fig.add_vline(x=row["exit"], line_color="#f59e0b", line_width=2)    # エグジット
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, template="plotly_dark")
    fig.write_html(str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description="ベスト/ワーストのトレードと直前の値動きを抽出")
    ap.add_argument("strategy")
    ap.add_argument("--pair", default="EURUSD")
    ap.add_argument("--tf", default="H1")
    ap.add_argument("--params", help="例: fast=30,slow=100(省略時は戦略の PARAMS)")
    ap.add_argument("--n", type=int, default=5, help="ベスト/ワースト各何件")
    ap.add_argument("--lookback", type=int, default=50, help="トレード前に遡る本数")
    ap.add_argument("--by", default="return_pct", choices=["return_pct", "pnl"], help="ランク基準")
    ap.add_argument("--save", action="store_true", help="CSV 出力")
    ap.add_argument("--plot", action="store_true", help="チャート HTML 出力")
    args = ap.parse_args()

    mod = importlib.import_module(f"strategies.{args.strategy}")
    params = _parse_params(args.params, mod)
    t0 = time.time()

    from fxlab.data import load
    data = load(args.pair, args.tf)
    pf = backtest.run(args.pair, args.tf, mod.generate_signals, params)

    r = trades.analyze(pf, data, n=args.n, lookback=args.lookback, by=args.by)

    print(f"# {args.strategy} {params} on {args.pair} {args.tf}  "
          f"(全 {len(r['all'])} トレード / 直前 {args.lookback} 本)\n")
    print(f"===== 🏆 ベスト {args.n} トレード(by {args.by}) =====")
    print(_fmt(r["best"]))
    print(f"\n===== 💀 ワースト {args.n} トレード(by {args.by}) =====")
    print(_fmt(r["worst"]))

    # ベスト/ワーストの直前の値動きの傾向を要約
    print("\n===== 📈 トレード前の値動き: ベスト vs ワースト 平均 =====")
    cmp_cols = ["pre_ret_%", "pre_trend_%/bar", "pre_vol_%", "up_bar_ratio",
                "rsi_at_entry", "atr_at_entry_%", "dist_from_high_%", "dist_from_low_%"]
    cmp = pd.DataFrame({
        "best平均": r["best"][cmp_cols].mean().round(3),
        "worst平均": r["worst"][cmp_cols].mean().round(3),
    })
    print(cmp.to_string())

    print(f"\n(抽出時間 {time.time()-t0:.1f}s)")

    if args.save or args.plot:
        outdir = config.RESULTS_DIR / f"trades_{args.strategy}_{args.pair}_{args.tf}"
        outdir.mkdir(parents=True, exist_ok=True)
        if args.save:
            r["all"].to_csv(outdir / "summary_all.csv", index=False)
            for tag, ctx in r["contexts"].items():
                ctx.to_csv(outdir / f"{tag}_pricewindow.csv")
            print(f"保存: {outdir}/ (summary_all.csv + 各トレードの値動きCSV)")
        if args.plot:
            for tag, frame in (("best", r["best"]), ("worst", r["worst"])):
                for i, row in frame.iterrows():
                    key = f"{tag}{i+1}"
                    title = (f"{key}: {row['dir']} {row['return_pct']:+.2f}% "
                             f"@ {pd.to_datetime(row['entry']):%Y-%m-%d %H:%M}")
                    _plot(r["contexts"][key], row, title, outdir / f"{key}.html")
            print(f"チャート: {outdir}/ ({2*args.n} 枚の HTML)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
