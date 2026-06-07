"""バックテスト中核エンジンの正しさを担保するテスト。

ここが壊れると全結論が無効になるため、特に「先読み防止」「コスト計上」を重視。
"""

import numpy as np
import pandas as pd

from fxlab import backtest, config
from strategies.ma_cross import generate_signals as ma_gen


# --- 先読み(look-ahead)していないこと ---------------------------------
def test_no_lookahead(synth_h1):
    """過去のシグナルは未来データに依存しない:
    full データと truncated データで、共通区間のシグナルが完全一致するはず。"""
    k = len(synth_h1) - 100
    full = ma_gen(synth_h1, fast=5, slow=20)
    trunc = ma_gen(synth_h1.iloc[:k], fast=5, slow=20)
    for f_full, f_trunc in zip(full, trunc):
        # 共通区間 [0:k] で完全一致(未来を見ていない証拠)
        assert f_full.iloc[:k].equals(f_trunc.iloc[:k])


# --- 取引コストがリターンを下げること ----------------------------------
def test_cost_reduces_return(synth_h1, monkeypatch):
    p = {"fast": 5, "slow": 20}
    monkeypatch.setitem(config.SPREADS_PIPS, "EURUSD", 0.0)
    r0 = float(backtest.run("EURUSD", "H1", ma_gen, p, data=synth_h1).total_return())
    monkeypatch.setitem(config.SPREADS_PIPS, "EURUSD", 5.0)
    r5 = float(backtest.run("EURUSD", "H1", ma_gen, p, data=synth_h1).total_return())
    assert r5 < r0  # スプレッドを広げたら必ず成績が落ちる


def test_spread_is_applied(synth_h1):
    """スプレッド > 0 のときスリッページ系列が正(コストとして効く)。"""
    slip = backtest._slippage_series("EURUSD", synth_h1["close"])
    assert (slip > 0).all()
    # JPY ペアは pip=0.01 なので非JPYより1桁大きいスケール
    assert config.pip_size("USDJPY") == 0.01
    assert config.pip_size("EURUSD") == 0.0001


# --- メトリクス / sweep の形 --------------------------------------------
def test_metrics_columns(synth_h1):
    pf = backtest.run("EURUSD", "H1", ma_gen, {"fast": 5, "slow": 20}, data=synth_h1)
    m = backtest.metrics(pf)
    for col in ("total_return", "sharpe", "max_drawdown", "num_trades", "profit_factor"):
        assert col in m.columns


def test_sweep_ranked(synth_h1):
    grid = {"fast": [5, 10], "slow": [20, 40]}
    res = backtest.sweep("EURUSD", "H1", ma_gen, grid, data=synth_h1, objective="sharpe")
    assert len(res) == 4
    # objective 降順に並んでいる
    assert res["sharpe"].is_monotonic_decreasing


# --- サイジング ----------------------------------------------------------
def test_sizing_modes_run(synth_h1):
    p = {"fast": 5, "slow": 20}
    for mode, val in [("full", None), ("value", 10000), ("amount", 8000), ("risk", 0.01)]:
        pf = backtest.run("EURUSD", "H1", ma_gen, p, data=synth_h1,
                          size_mode=mode, size_value=val)
        assert np.isfinite(float(pf.total_return()))


def test_risk_sizing_bounds_loss(synth_h1):
    """リスク1%サイジングなら、ギャップの無い合成データでは
    1トレードの損失が極端に大きくならない(初期資金の数%以内)。"""
    from fxlab.trades import trade_table
    pf = backtest.run("EURUSD", "H1", ma_gen, {"fast": 5, "slow": 20}, data=synth_h1,
                      size_mode="risk", size_value=0.01)
    tt = trade_table(pf, synth_h1)
    if len(tt):
        assert tt["pnl"].min() > -0.05 * 10000  # 最悪でも初期資金の5%以内


def test_side_isolation(synth_h1):
    """side='long' はショート寄与を持たない(両建てと別物)。"""
    p = {"fast": 5, "slow": 20}
    both = backtest.run("EURUSD", "H1", ma_gen, p, data=synth_h1, side="both")
    longo = backtest.run("EURUSD", "H1", ma_gen, p, data=synth_h1, side="long")
    # ロングのみは取引数が両建て以下
    assert int(longo.trades.count()) <= int(both.trades.count())
