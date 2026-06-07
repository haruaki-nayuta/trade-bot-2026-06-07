"""トレード抽出(ベスト/ワースト + 直前の値動き)のテスト。"""

from fxlab import backtest, trades
from strategies.ma_cross import generate_signals as ma_gen


def _pf(synth_h1):
    return backtest.run("EURUSD", "H1", ma_gen, {"fast": 5, "slow": 20}, data=synth_h1)


def test_trade_table_shape(synth_h1):
    tt = trades.trade_table(_pf(synth_h1), synth_h1)
    for c in ("entry", "exit", "dir", "pnl", "return_pct", "bars_held", "hours"):
        assert c in tt.columns
    assert (tt["bars_held"] >= 0).all()


def test_analyze_best_worst_ordering(synth_h1):
    r = trades.analyze(_pf(synth_h1), synth_h1, n=3, lookback=30, by="return_pct")
    # ベストの最小リターン >= ワーストの最大リターン
    assert r["best"]["return_pct"].min() >= r["worst"]["return_pct"].max()


def test_context_phases(synth_h1):
    r = trades.analyze(_pf(synth_h1), synth_h1, n=2, lookback=20)
    for ctx in r["contexts"].values():
        phases = set(ctx["phase"].unique())
        assert phases <= {"pre", "trade"}
        assert "trade" in phases
        # 直前(pre)は lookback 本以内
        assert (ctx["phase"] == "pre").sum() <= 20


def test_pre_features_keys(synth_h1):
    pf = _pf(synth_h1)
    tt = trades.trade_table(pf, synth_h1)
    rsi = trades._rsi(synth_h1["close"])
    atr = trades._atr_pct(synth_h1)
    feat = trades.pre_features(synth_h1, tt["entry"].iloc[1], 30, rsi, atr)
    for k in ("pre_ret_%", "pre_vol_%", "rsi_at_entry", "atr_at_entry_%",
              "dist_from_high_%", "dist_from_low_%"):
        assert k in feat
