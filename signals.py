"""実運用シグナル生成器 — 最新の確定足から「現在の建玉 / 新規シグナル / 各対象の状態」を出力。

チャンピオン(confluence_meanrev)を 7メジャー + 合成クロス に適用し、いま取るべきアクションを示す。
検証専用環境のため**自動発注はしない**(CLAUDE.md 準拠)。実弾接続は別途ブローカーAPIが必要。

  uv run python signals.py                       # 推奨構成で現在のシグナル一覧
  uv run python signals.py --strategy confluence_meanrev --tf H4
  uv run python signals.py --asof 2025-12-31     # 指定時点での状態(過去再現)

「新規」= 直近の確定足でエントリー条件が成立したもの(=今すぐ建てる候補)。
"""

from __future__ import annotations

import argparse
import importlib

import numpy as np
import pandas as pd

from fxlab import universe as uni

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def position_series(le, lx, se, sx) -> np.ndarray:
    """エントリー/エグジット bool から現在ポジション(1/-1/0)系列を再構成(ドテン対応)。"""
    le, lx, se, sx = le.values, lx.values, se.values, sx.values
    n = len(le)
    pos = np.zeros(n, dtype=int)
    cur = 0
    for i in range(n):
        if cur == 1 and (lx[i] or se[i]):
            cur = 0
        elif cur == -1 and (sx[i] or le[i]):
            cur = 0
        if cur == 0:
            if le[i]:
                cur = 1
            elif se[i]:
                cur = -1
        pos[i] = cur
    return pos


def main() -> int:
    ap = argparse.ArgumentParser(description="実運用シグナル生成器")
    ap.add_argument("--strategy", default="confluence_meanrev")
    ap.add_argument("--tf", default="H4")
    ap.add_argument("--exclude", nargs="+", default=["AUDJPY"])
    ap.add_argument("--params", help="PARAMS 上書き(例 slow_z=1.75)")
    ap.add_argument("--asof", help="この時点までで判定(YYYY-MM-DD、過去再現用)")
    ap.add_argument("--cross-spread", type=float, default=uni.CROSS_SPREAD_PIPS)
    args = ap.parse_args()

    uni.register_cross_spreads(args.cross_spread)
    mod = importlib.import_module(f"strategies.{args.strategy}")
    params = dict(getattr(mod, "PARAMS", {}))
    if args.strategy == "confluence_meanrev":
        params["slow_z"] = 1.75  # 推奨運用構成
    if args.params:
        for kv in args.params.split(","):
            k, v = kv.split("="); params[k.strip()] = float(v)
    gen = mod.generate_signals
    instruments = [x for x in uni.universe() if x not in set(args.exclude)]

    asof = pd.Timestamp(args.asof, tz="UTC") if args.asof else None
    rows = []
    for nm in instruments:
        data = uni.instrument_data(nm, args.tf)
        if asof is not None:
            data = data[data.index <= asof]
        if len(data) < 300:
            continue
        le, lx, se, sx = gen(data, **params)
        pos = position_series(le, lx, se, sx)
        close = data["close"]
        z = (close - close.rolling(params.get("window", 50)).mean()) / close.rolling(params.get("window", 50)).std()
        last = data.index[-1]
        rows.append({
            "instrument": nm,
            "asof": last,
            "position": {1: "LONG", -1: "SHORT", 0: "flat"}[int(pos[-1])],
            "new_entry": "★LONG" if le.iloc[-1] else ("★SHORT" if se.iloc[-1] else ""),
            "z": round(float(z.iloc[-1]), 2) if pd.notna(z.iloc[-1]) else np.nan,
            "close": round(float(close.iloc[-1]), 5),
        })

    df = pd.DataFrame(rows)
    print(f"=== シグナル: {args.strategy} {params} on {args.tf} ===")
    print(f"基準時刻(最新確定足): {df['asof'].max()}\n")

    opens = df[df["position"] != "flat"]
    news = df[df["new_entry"] != ""]
    print(f"■ 現在の建玉: {len(opens)} 件")
    if len(opens):
        print(opens[["instrument", "position", "z", "close"]].to_string(index=False))
    print(f"\n■ 直近足の新規シグナル(今すぐ建てる候補): {len(news)} 件")
    if len(news):
        print(news[["instrument", "new_entry", "z", "close"]].to_string(index=False))
    print("\n(全対象の状態)")
    print(df.to_string(index=False))
    print("\n※ 検証専用環境のため自動発注はしない。実弾はブローカーAPI接続が別途必要。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
