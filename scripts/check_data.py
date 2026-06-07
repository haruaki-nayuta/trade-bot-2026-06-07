"""価格データの品質チェック — 重複・欠損・時系列の乱れ・OHLC整合・異常ギャップを検査。

  uv run python scripts/check_data.py            # 全ペア
  uv run python scripts/check_data.py --pairs EURUSD USDJPY

各ペアについて:
  * 行数 / 期間
  * index の重複・単調増加
  * OHLCV の NaN
  * OHLC 整合(high>=low, high>=open/close, low<=open/close)
  * バー間ギャップ: 週末/祝日の正常な空白と、平日の異常欠損を区別して報告
判定: 致命的(重複/NaN/OHLC破れ/逆行)があれば NG、平日大欠損は WARN。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fxlab import config  # noqa: E402
from fxlab.data import load_m1, parquet_path  # noqa: E402


def check_pair(pair: str) -> dict:
    df = load_m1(pair)
    idx = df.index

    dups = int(idx.duplicated().sum())
    monotonic = bool(idx.is_monotonic_increasing)
    nan = int(df[["open", "high", "low", "close", "volume"]].isna().sum().sum())
    bad_hl = int((df["high"] < df["low"]).sum())
    bad_h = int(((df["high"] < df["open"]) | (df["high"] < df["close"])).sum())
    bad_l = int(((df["low"] > df["open"]) | (df["low"] > df["close"])).sum())

    # ギャップ分析
    s = idx.to_series()
    gmin = s.diff().dt.total_seconds() / 60
    big = gmin > 60                                  # 1時間超の空白
    prev_wd = s.shift().dt.weekday                   # 直前バーの曜日
    weekend = big & prev_wd.isin([4, 5])             # 金/土発 = 週末(正常)
    other = big & ~prev_wd.isin([4, 5])              # それ以外 = 祝日/欠損(要確認)

    top = (
        pd.DataFrame({"gap_h": (gmin[other] / 60).round(1), "after": s[other]})
        .assign(before=s.shift()[other])
        .sort_values("gap_h", ascending=False)
        .head(5)
    )

    fatal = dups > 0 or nan > 0 or bad_hl > 0 or bad_h > 0 or bad_l > 0 or not monotonic
    return {
        "pair": pair,
        "rows": len(df),
        "start": idx.min(),
        "end": idx.max(),
        "MB": round(parquet_path(pair).stat().st_size / 1e6, 1),
        "dups": dups,
        "monotonic": monotonic,
        "nan": nan,
        "bad_OHLC": bad_hl + bad_h + bad_l,
        "weekend_gaps": int(weekend.sum()),
        "other_gaps": int(other.sum()),
        "status": "NG" if fatal else ("WARN" if other.sum() > 50 else "OK"),
        "_top_other": top,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="価格データ品質チェック")
    ap.add_argument("--pairs", nargs="*", default=list(config.PAIRS))
    args = ap.parse_args()

    results = []
    details = []
    for pair in args.pairs:
        if not parquet_path(pair).exists():
            print(f"■ {pair}: 未取得 — スキップ")
            continue
        r = check_pair(pair)
        details.append((pair, r.pop("_top_other")))
        results.append(r)

    if not results:
        print("チェック対象のデータがありません。")
        return 1

    tbl = pd.DataFrame(results)
    print("=== データ品質サマリ ===")
    print(tbl.to_string(index=False))

    # 異常(平日)ギャップの詳細
    print("\n=== 平日の大ギャップ Top(祝日 or 欠損の可能性) ===")
    for pair, top in details:
        if len(top):
            print(f"\n■ {pair}")
            for _, row in top.iterrows():
                print(f"   {row['before']:%Y-%m-%d %H:%M} → {row['after']:%Y-%m-%d %H:%M}  ({row['gap_h']}h)")

    ng = (tbl["status"] == "NG").sum()
    warn = (tbl["status"] == "WARN").sum()
    print(f"\n=== 判定: OK {(tbl['status']=='OK').sum()} / WARN {warn} / NG {ng} ===")
    return 1 if ng else 0


if __name__ == "__main__":
    raise SystemExit(main())
