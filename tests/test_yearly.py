"""年次分解(fxlab.yearly)の正しさを担保するテスト。

目標の合否判定は yearly() の集計に依存するため、ここで「年ごとに正しく分割・集計」
していることを確認する(年次PnLの合算=全期間PnL、PFの定義一致、など)。
"""

import numpy as np
import pandas as pd

from fxlab import backtest, yearly
from fxlab.trades import trade_table
from strategies.ma_cross import generate_signals as ma_gen


def _multiyear_h1(years=3, seed=0):
    """複数年にまたがる H1 合成OHLCV。"""
    n = years * 3000
    idx = pd.date_range("2018-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.RandomState(seed)
    close = 1.10 + np.cumsum(rng.randn(n) * 0.0008)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.randn(n)) * 0.0003
    low = np.minimum(open_, close) - np.abs(rng.randn(n)) * 0.0003
    vol = rng.randint(100, 1000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_profit_factor_definition():
    pnl = pd.Series([3.0, -1.0, 2.0, -1.0])
    assert yearly.profit_factor(pnl) == 5.0 / 2.0
    assert np.isinf(yearly.profit_factor(pd.Series([1.0, 2.0])))      # 損失なし
    assert np.isnan(yearly.profit_factor(pd.Series([], dtype=float)))  # 取引なし


def test_yearly_splits_by_exit_year():
    df = _multiyear_h1(years=3)
    y = yearly.yearly("EURUSD", "H1", ma_gen, {"fast": 5, "slow": 20}, data=df)
    # 3年(または端数年)に分割されている
    assert len(y) >= 2
    assert set(y.index).issubset({2018, 2019, 2020, 2021})
    for col in ("trades", "win_rate", "profit_factor", "pnl", "return_pct"):
        assert col in y.columns


def test_yearly_pnl_sums_to_total():
    """年次PnLの合算が、全トレードのPnL合計と一致する(漏れ・二重計上なし)。"""
    df = _multiyear_h1(years=3)
    pf = backtest.run("EURUSD", "H1", ma_gen, {"fast": 5, "slow": 20}, data=df,
                      size_mode="value")
    total_pnl = trade_table(pf, df)["pnl"].sum()
    y = yearly.yearly("EURUSD", "H1", ma_gen, {"fast": 5, "slow": 20}, data=df)
    assert np.isclose(y["pnl"].sum(), total_pnl, rtol=1e-6)


def test_yearly_trades_count_matches():
    df = _multiyear_h1(years=3)
    pf = backtest.run("EURUSD", "H1", ma_gen, {"fast": 5, "slow": 20}, data=df,
                      size_mode="value")
    n_total = len(trade_table(pf, df))
    y = yearly.yearly("EURUSD", "H1", ma_gen, {"fast": 5, "slow": 20}, data=df)
    assert int(y["trades"].sum()) == n_total


def test_empty_when_no_trades():
    """シグナルが出ない設定では空テーブルを返す(落ちない)。"""
    df = _multiyear_h1(years=1)
    # fast>slow は決して交差しない極端設定で取引ゼロ近くを狙う
    y = yearly.yearly("EURUSD", "H1", ma_gen, {"fast": 999, "slow": 1000}, data=df)
    assert isinstance(y, pd.DataFrame)
