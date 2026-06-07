"""キャリー(金利差)エッジ源 — 価格以外のファンダメンタル情報。

本環境は外部データ取得が遮断されているため、各通貨の短期政策金利の**年次近似値**
(マクロ知識ベース、%)を内蔵する。実データが手に入れば RATES を CSV から差し替えるだけで
精度が上がる(年×通貨の表)。

キャリー = base通貨金利 − quote通貨金利。ロングは年率でこれを受け取り(or支払い)、ショートは逆。
短期平均回帰(保有数日)では受取額は小さいが、「金利差で構造的にトレンドする局面」を避ける
フィルタとして機能しうる。

  carry_annual(instr, year) -> 年率%(例 EURUSD 2023 = r_EUR - r_USD)
  apply_carry(trades) -> トレード表に carry収益を加味した return を付与
"""

from __future__ import annotations

import pandas as pd

# 各通貨の短期政策金利・年次近似(%)。2016-2026。実CSVがあれば置換可。
RATES: dict[str, dict[int, float]] = {
    "USD": {2016: 0.5, 2017: 1.1, 2018: 2.1, 2019: 2.1, 2020: 0.4, 2021: 0.25, 2022: 2.0, 2023: 5.1, 2024: 5.1, 2025: 4.2, 2026: 3.5},
    "EUR": {2016: 0.0, 2017: 0.0, 2018: 0.0, 2019: -0.3, 2020: -0.5, 2021: -0.5, 2022: 0.5, 2023: 3.4, 2024: 3.6, 2025: 2.5, 2026: 2.0},
    "JPY": {2016: -0.05, 2017: -0.05, 2018: -0.05, 2019: -0.05, 2020: -0.05, 2021: -0.05, 2022: -0.05, 2023: -0.05, 2024: 0.1, 2025: 0.4, 2026: 0.5},
    "GBP": {2016: 0.4, 2017: 0.4, 2018: 0.7, 2019: 0.75, 2020: 0.2, 2021: 0.2, 2022: 1.5, 2023: 4.7, 2024: 5.1, 2025: 4.3, 2026: 3.75},
    "AUD": {2016: 1.75, 2017: 1.5, 2018: 1.5, 2019: 1.1, 2020: 0.3, 2021: 0.1, 2022: 1.4, 2023: 3.9, 2024: 4.35, 2025: 4.1, 2026: 3.6},
    "CHF": {2016: -0.75, 2017: -0.75, 2018: -0.75, 2019: -0.75, 2020: -0.75, 2021: -0.75, 2022: -0.2, 2023: 1.6, 2024: 1.4, 2025: 0.5, 2026: 0.25},
    "CAD": {2016: 0.5, 2017: 0.85, 2018: 1.5, 2019: 1.75, 2020: 0.4, 2021: 0.25, 2022: 2.3, 2023: 4.8, 2024: 4.8, 2025: 3.2, 2026: 2.75},
    "NZD": {2016: 2.25, 2017: 1.75, 2018: 1.75, 2019: 1.2, 2020: 0.3, 2021: 0.4, 2022: 2.5, 2023: 5.4, 2024: 5.4, 2025: 4.2, 2026: 3.5},
}


def _rate(cur: str, year: int) -> float:
    d = RATES.get(cur, {})
    if not d:
        return 0.0
    if year in d:
        return d[year]
    yrs = sorted(d)
    return d[yrs[0]] if year < yrs[0] else d[yrs[-1]]


def carry_annual(instr: str, year: int) -> float:
    """instr(base+quote, 6文字)の年率キャリー(%)= r_base − r_quote。ロング基準。"""
    base, quote = instr[:3], instr[3:]
    return _rate(base, year) - _rate(quote, year)


def apply_carry(trades: pd.DataFrame, instr: str) -> pd.DataFrame:
    """トレード表に保有期間ぶんのキャリー収益を加味した return_pct_carry / pnl_carry を付与。

    trades: trade_table の出力(dir, return_pct, pnl, hours, exit を含む)。
    """
    out = trades.copy()
    years = pd.DatetimeIndex(out["exit"]).year
    dir_sign = out["dir"].map(lambda d: 1.0 if str(d).lower().startswith("l") else -1.0)
    car_ann = pd.Series([carry_annual(instr, int(y)) for y in years], index=out.index)  # %/年, ロング基準
    days = out["hours"] / 24.0
    carry_ret = dir_sign * (car_ann / 100.0) * (days / 365.0)  # 保有ぶんの受取/支払(割合)
    out["carry_ret"] = carry_ret
    out["return_carry"] = out["return_pct"] / 100.0 + carry_ret
    return out
