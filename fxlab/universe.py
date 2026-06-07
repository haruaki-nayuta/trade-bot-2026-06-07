"""取引ユニバース = 7メジャー + 合成クロス。ポートフォリオ評価の中核。

検証で「メジャー単独だと USD 主導で悪い年が共通」「クロスを足すと分散で年次が安定」と判明。
クロスはメジャー(USD建て)から close を合成して扱う(EURGBP=EURUSD/GBPUSD など)。
close ベースの戦略(confluence_meanrev 等)はそのまま適用できる。

  * CROSS_DEFS         : クロス名 -> (脚1, 演算, 脚2)
  * instrument_close   : メジャーは実データ、クロスは合成 close
  * instrument_data    : 戦略に渡す OHLCV(クロスは close で OHLC 代用=close系戦略向け)
  * universe           : 評価対象一覧(メジャー+クロス)
  * portfolio_yearly   : ユニバース横断で決済年集計したポートフォリオ年次成績
  * acceptance         : 目標(PF/毎年プラス/年取引)に対する合否

クロスのスプレッドは config.SPREADS_PIPS 未登録なら既定 1.0。現実的には広めなので
CROSS_SPREAD_PIPS を config に反映して使う(下記 register_cross_spreads)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .backtest import run
from .data import available_pairs, load
from .trades import trade_table
from .yearly import yearly

# クロス定義(USD建てメジャーから合成)。op: '/' か '*'。
CROSS_DEFS: dict[str, tuple[str, str, str]] = {
    "EURGBP": ("EURUSD", "/", "GBPUSD"),
    "EURCHF": ("EURUSD", "*", "USDCHF"),
    "EURAUD": ("EURUSD", "/", "AUDUSD"),
    "EURCAD": ("EURUSD", "*", "USDCAD"),
    "GBPAUD": ("GBPUSD", "/", "AUDUSD"),
    "GBPCHF": ("GBPUSD", "*", "USDCHF"),
    "AUDNZD": ("AUDUSD", "/", "NZDUSD"),
    "AUDCAD": ("AUDUSD", "*", "USDCAD"),
    "NZDCAD": ("NZDUSD", "*", "USDCAD"),
    "AUDCHF": ("AUDUSD", "*", "USDCHF"),
    "EURJPY": ("EURUSD", "*", "USDJPY"),
    "GBPJPY": ("GBPUSD", "*", "USDJPY"),
    "AUDJPY": ("AUDUSD", "*", "USDJPY"),
}

CROSS_SPREAD_PIPS = 3.0  # 合成クロスの往復コスト見積り(厳しめ既定)


def register_cross_spreads(pips: float = CROSS_SPREAD_PIPS) -> None:
    """クロスのスプレッドを config に登録(コスト計上のため)。"""
    for nm in CROSS_DEFS:
        config.SPREADS_PIPS[nm] = pips


def instrument_close(name: str, tf: str) -> pd.Series:
    """メジャーは実 close、クロスは合成 close を返す。"""
    if name in CROSS_DEFS:
        a, op, b = CROSS_DEFS[name]
        ca, cb = load(a, tf)["close"], load(b, tf)["close"]
        df = pd.concat([ca, cb], axis=1).dropna()
        return df.iloc[:, 0] / df.iloc[:, 1] if op == "/" else df.iloc[:, 0] * df.iloc[:, 1]
    return load(name, tf)["close"]


def instrument_data(name: str, tf: str) -> pd.DataFrame:
    """戦略に渡す OHLCV。クロスは close で OHLC を代用(close ベース戦略向け)。"""
    if name not in CROSS_DEFS:
        return load(name, tf)
    c = instrument_close(name, tf)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=c.index)


def universe(crosses: bool = True) -> list[str]:
    pairs = list(available_pairs())
    return pairs + list(CROSS_DEFS) if crosses else pairs


def _pair_yearly(name: str, tf: str, gen, params, **kw) -> pd.DataFrame:
    if name in CROSS_DEFS:
        return yearly(name, tf, gen, params, data=instrument_data(name, tf), **kw)
    return yearly(name, tf, gen, params, **kw)


def portfolio_yearly(tf: str, gen, params: dict | None = None, *,
                     instruments: list[str] | None = None, init_cash: float = 10_000,
                     **kw) -> pd.DataFrame:
    """ユニバース等加重・決済年合算のポートフォリオ年次成績。"""
    instruments = instruments or universe()
    accum: dict[int, list[float]] = {}
    for name in instruments:
        try:
            y = _pair_yearly(name, tf, gen, params, init_cash=init_cash, **kw)
        except Exception:  # noqa: BLE001
            continue
        for year, r in y.iterrows():
            a = accum.setdefault(int(year), [0.0, 0.0, 0.0, 0.0, 0, 0])
            a[0] += r["gross_profit"]; a[1] += r["gross_loss"]; a[2] += r["pnl"]
            a[3] += r["trades"]; a[4] += int(r["pnl"] > 0); a[5] += 1
    rows = {}
    for year, (gp, gl, pnl, tr, pos, tot) in sorted(accum.items()):
        rows[year] = {
            "trades": int(tr),
            "profit_factor": (gp / gl) if gl > 0 else float("inf"),
            "pnl": pnl,
            "return_pct": pnl / (init_cash * max(tot, 1)) * 100,
            "instr_positive": f"{pos}/{tot}",
        }
    df = pd.DataFrame(rows).T
    for c in ["trades", "profit_factor", "pnl", "return_pct"]:  # .T で object 化するため数値へ戻す
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def acceptance(tf: str, gen, params: dict | None = None, *,
               pf_target: float = 2.0, min_trades: int = 100,
               instruments: list[str] | None = None, **kw) -> dict:
    """ポートフォリオ年次に対する目標合否。"""
    port = portfolio_yearly(tf, gen, params, instruments=instruments, **kw)
    if port.empty:
        return {"port": port, "verdict": {"overall": False}}
    pf = port["profit_factor"].replace(np.inf, np.nan)
    pos_rate = float((port["pnl"] > 0).mean())
    verdict = {
        "positive_year_rate": pos_rate,
        "pass_positive": bool((port["pnl"] > 0).all()),
        "pf_median": float(pf.median()),
        "pf_min": float(pf.min()),
        "pass_pf": bool((pf.fillna(0) >= pf_target).all()),
        "avg_trades": int(port["trades"].mean()),
        "pass_trades": bool(port["trades"].mean() >= min_trades),
    }
    verdict["overall"] = verdict["pass_positive"] and verdict["pass_pf"] and verdict["pass_trades"]
    return {"port": port, "verdict": verdict}
