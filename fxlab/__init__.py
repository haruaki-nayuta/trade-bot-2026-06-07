"""fxlab — FX 手法のリサーチ〜検証パイプライン。

よく使う入口:
    from fxlab import load, run, sweep, metrics, summary, config

    df  = load("EURUSD", "H1")                 # データ読込(M1→任意足)
    pf  = run("EURUSD", "H1", gen, {...})       # 単発バックテスト
    res = sweep("EURUSD", "H1", gen, {...})     # 並列パラメータ探索
"""

from . import config, trades
from .backtest import metrics, run, sweep
from .data import available_pairs, load, load_m1, resample, summary

__all__ = [
    "config",
    "trades",
    "load",
    "load_m1",
    "resample",
    "summary",
    "available_pairs",
    "run",
    "sweep",
    "metrics",
]
