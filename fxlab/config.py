"""中央設定 — 通貨ペア・パス・取引コスト・期間。

ここを変えれば全パイプライン(取得・読込・検証)に反映される。
価格データは Dukascopy(高精度・信頼できる無料ソース)から M1(1分足)を基盤に取得し、
上位足は M1 からリサンプルする。
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from dukascopy_python.instruments import (
    INSTRUMENT_FX_MAJORS_AUD_USD,
    INSTRUMENT_FX_MAJORS_EUR_USD,
    INSTRUMENT_FX_MAJORS_GBP_USD,
    INSTRUMENT_FX_MAJORS_NZD_USD,
    INSTRUMENT_FX_MAJORS_USD_CAD,
    INSTRUMENT_FX_MAJORS_USD_CHF,
    INSTRUMENT_FX_MAJORS_USD_JPY,
)

# --- パス ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"        # M1 parquet 置き場
RESULTS_DIR = ROOT / "results"          # バックテスト結果の出力先
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- 取得対象: 7大メジャー --------------------------------------------
# キー = 内部名 / 値 = Dukascopy instrument 定数
PAIRS: dict[str, str] = {
    "EURUSD": INSTRUMENT_FX_MAJORS_EUR_USD,
    "USDJPY": INSTRUMENT_FX_MAJORS_USD_JPY,
    "GBPUSD": INSTRUMENT_FX_MAJORS_GBP_USD,
    "AUDUSD": INSTRUMENT_FX_MAJORS_AUD_USD,
    "USDCHF": INSTRUMENT_FX_MAJORS_USD_CHF,
    "USDCAD": INSTRUMENT_FX_MAJORS_USD_CAD,
    "NZDUSD": INSTRUMENT_FX_MAJORS_NZD_USD,
}

# --- 取得期間 -----------------------------------------------------------
HISTORY_YEARS = 10


def default_start() -> _dt.datetime:
    """今日から HISTORY_YEARS 年前(UTC)。"""
    today = _dt.datetime.now(_dt.timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None
    )
    return today.replace(year=today.year - HISTORY_YEARS)


def default_end() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)


# --- リサンプル対応の時間足 --------------------------------------------
# 名前 -> pandas offset alias。M1 を基盤に下記へ集約できる。
TIMEFRAMES: dict[str, str] = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
    "W1": "1W",
}

# --- 取引コスト(現実的な検証のため) ----------------------------------
# pip 単位の平均スプレッド。実際の口座に合わせて調整可。
# JPY ペアは pip = 0.01、それ以外は pip = 0.0001。
SPREADS_PIPS: dict[str, float] = {
    "EURUSD": 0.6,
    "USDJPY": 0.7,
    "GBPUSD": 0.9,
    "AUDUSD": 0.8,
    "USDCHF": 1.0,
    "USDCAD": 1.2,
    "NZDUSD": 1.4,
    # 金(XAUUSD): pip=0.01 USD として 35pips = 0.35 USD/oz(Dukascopy 実勢 0.25-0.45 の保守側)
    "XAUUSD": 35.0,
}

# 片道の手数料(約定代金に対する割合)。ECN想定の例: $30/100万 ≈ 0.00003。
# デフォルトはスプレッドのみで検証する想定で 0。
COMMISSION_FRACTION = 0.0


def pip_size(pair: str) -> float:
    """その通貨ペアの 1 pip の価格幅。"""
    if pair.startswith("XAU"):
        return 0.01  # 金は 1pip = 0.01 USD 慣行
    return 0.01 if pair.endswith("JPY") else 0.0001


def spread_pips(pair: str) -> float:
    return SPREADS_PIPS.get(pair, 1.0)
