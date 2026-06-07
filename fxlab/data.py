"""データ読込・リサンプル・状態確認。

価格は data/raw/{PAIR}_M1.parquet に M1(UTC, OHLCV)で保存される。
上位足はここで M1 からリサンプルして得る(再ダウンロード不要)。
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from . import config


def parquet_path(pair: str) -> "config.Path":
    return config.DATA_DIR / f"{pair}_M1.parquet"


def available_pairs() -> list[str]:
    """ダウンロード済みのペア一覧。"""
    return [p for p in config.PAIRS if parquet_path(p).exists()]


@lru_cache(maxsize=8)
def _read_m1(pair: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path(pair))


def load_m1(pair: str) -> pd.DataFrame:
    """M1 の OHLCV を読み込む(UTC, DatetimeIndex)。プロセス内でキャッシュ。

    注意: 返り値はキャッシュ共有。破壊的に書き換えない(必要なら .copy())。
    """
    if not parquet_path(pair).exists():
        raise FileNotFoundError(
            f"{pair} の M1 データが見つかりません: {parquet_path(pair)}\n"
            f"先に `uv run python scripts/download_data.py` を実行してください。"
        )
    return _read_m1(pair)


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """M1 の OHLCV を上位足へ集約する。

    timeframe は config.TIMEFRAMES のキー("M5","H1","H4","D1" 等)。
    """
    if timeframe == "M1":
        return df
    if timeframe not in config.TIMEFRAMES:
        raise ValueError(
            f"未知の時間足: {timeframe}. 対応: {list(config.TIMEFRAMES)}"
        )
    rule = config.TIMEFRAMES[timeframe]
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


@lru_cache(maxsize=64)
def _load_cached(pair: str, timeframe: str) -> pd.DataFrame:
    return resample(load_m1(pair), timeframe)


def load(pair: str, timeframe: str = "M1") -> pd.DataFrame:
    """1 行で「ペア + 時間足」を取得する一番よく使う入口(キャッシュ付き)。

    例: df = load("EURUSD", "H1")
    注意: 返り値はキャッシュ共有。破壊的に書き換えない(必要なら .copy())。
    """
    return _load_cached(pair, timeframe)


def clear_cache() -> None:
    """データ更新後にプロセス内キャッシュを破棄する(長時間プロセス/ノートブック用)。"""
    _read_m1.cache_clear()
    _load_cached.cache_clear()


def summary() -> pd.DataFrame:
    """ダウンロード済み各ペアの行数・期間の一覧。"""
    rows = []
    for pair in config.PAIRS:
        path = parquet_path(pair)
        if not path.exists():
            rows.append({"pair": pair, "status": "未取得"})
            continue
        df = pd.read_parquet(path, columns=["close"])
        rows.append(
            {
                "pair": pair,
                "status": "OK",
                "rows": len(df),
                "start": df.index.min(),
                "end": df.index.max(),
                "MB": round(path.stat().st_size / 1e6, 1),
            }
        )
    return pd.DataFrame(rows)
