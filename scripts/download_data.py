"""7大メジャーの M1(1分足)を Dukascopy から過去10年分ダウンロードして parquet 保存。

  uv run python scripts/download_data.py                # 全ペア(既存はスキップ)
  uv run python scripts/download_data.py --pairs EURUSD USDJPY
  uv run python scripts/download_data.py --force        # 既存も再取得
  uv run python scripts/download_data.py --years 5      # 期間変更

年ごとに分割取得し、ペア単位で parquet 出力(途中失敗してもペア単位で再開可能)。
データは UTC・OHLCV・BID。スプレッド等の取引コストはバックテスト側で計上する。
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import dukascopy_python as dk
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fxlab import config  # noqa: E402
from fxlab.data import parquet_path  # noqa: E402


def fetch_pair(pair: str, instrument: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """1 ペアを年単位で分割取得して結合。"""
    frames = []
    seg_start = start
    while seg_start < end:
        seg_end = min(seg_start.replace(year=seg_start.year + 1), end)
        t0 = time.time()
        try:
            df = dk.fetch(instrument, dk.INTERVAL_MIN_1, dk.OFFER_SIDE_BID, seg_start, seg_end)
        except Exception as e:  # noqa: BLE001
            print(f"    [{seg_start:%Y}] 取得失敗 ({e}) — スキップ")
            seg_start = seg_end
            continue
        if df is not None and len(df):
            frames.append(df)
            print(f"    [{seg_start:%Y}] {len(df):>8,} 本  ({time.time()-t0:.1f}s)")
        else:
            print(f"    [{seg_start:%Y}] データなし")
        seg_start = seg_end

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Dukascopy から M1 を取得")
    ap.add_argument("--pairs", nargs="*", default=list(config.PAIRS), help="対象ペア")
    ap.add_argument("--years", type=int, default=config.HISTORY_YEARS, help="遡る年数")
    ap.add_argument("--force", action="store_true", help="既存ファイルも再取得")
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)  # ログ/リダイレクト時も進捗を即時出力

    end = config.default_end()
    start = end.replace(year=end.year - args.years)
    print(f"期間: {start:%Y-%m-%d} 〜 {end:%Y-%m-%d}  /  足: M1  /  ソース: Dukascopy(UTC, BID)\n")

    overall = time.time()
    for pair in args.pairs:
        if pair not in config.PAIRS:
            print(f"未知のペア: {pair} — スキップ")
            continue
        path = parquet_path(pair)
        if path.exists() and not args.force:
            existing = pd.read_parquet(path, columns=["close"])
            print(f"■ {pair}: 既存 {len(existing):,} 本 — スキップ(--force で再取得)")
            continue

        print(f"■ {pair}: 取得開始")
        t0 = time.time()
        df = fetch_pair(pair, config.PAIRS[pair], start, end)
        if df.empty:
            print(f"  {pair}: データ取得できず\n")
            continue
        df.to_parquet(path, compression="zstd")
        print(
            f"  {pair}: 完了 {len(df):,} 本  "
            f"({df.index.min():%Y-%m-%d}〜{df.index.max():%Y-%m-%d})  "
            f"{path.stat().st_size/1e6:.1f}MB  {time.time()-t0:.0f}s\n"
        )

    print(f"=== 全体 {time.time()-overall:.0f}s 完了 → {config.DATA_DIR} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
