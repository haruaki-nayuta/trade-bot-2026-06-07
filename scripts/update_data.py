"""既存の M1 parquet を最新まで追記更新する(差分のみ取得)。

  uv run python scripts/update_data.py            # 全ペアを最新化
  uv run python scripts/update_data.py --pairs EURUSD

各ペアの最終足の直後から現在までを取得して結合・重複排除して保存。
定期実行(cron 等)で常に最新データを保てる。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fxlab import config  # noqa: E402
from fxlab.data import parquet_path  # noqa: E402
from scripts.download_data import fetch_pair  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="M1 データを最新まで更新")
    ap.add_argument("--pairs", nargs="*", default=list(config.PAIRS))
    args = ap.parse_args()

    end = config.default_end()
    for pair in args.pairs:
        path = parquet_path(pair)
        if not path.exists():
            print(f"■ {pair}: 未取得 — まず download_data.py を実行してください")
            continue
        old = pd.read_parquet(path)
        last = old.index.max().tz_convert(None).to_pydatetime()
        if last >= end:
            print(f"■ {pair}: 既に最新 ({last:%Y-%m-%d %H:%M})")
            continue
        print(f"■ {pair}: {last:%Y-%m-%d %H:%M} 〜 最新 を取得")
        t0 = time.time()
        new = fetch_pair(pair, config.PAIRS[pair], last, end)
        if new.empty:
            print(f"  {pair}: 追加分なし\n")
            continue
        merged = pd.concat([old, new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.to_parquet(path, compression="zstd")
        print(f"  {pair}: +{len(merged)-len(old):,} 本 → 計 {len(merged):,}  ({time.time()-t0:.0f}s)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
