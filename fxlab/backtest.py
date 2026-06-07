"""バックテスト中核エンジン(vectorbt ベース)。

設計のキモ:
  * 戦略は「シグナル生成関数」だけ書けばよい(strategies/ 参照)。
  * 単発検証は run()、パラメータ探索は sweep()。
  * sweep() は全パラメータ組合せを 1 回のベクトル化シミュレーションで回す
    = numba 並列・超高速。シグナル生成は joblib で並列化。
  * スプレッド/手数料を現実的にコスト計上(config.SPREADS_PIPS)。

シグナル生成関数の契約:
    def generate_signals(data: pd.DataFrame, **params)
        -> (long_entries, long_exits)                      # ロングのみ
        または
        -> (long_entries, long_exits, short_entries, short_exits)  # 両建て可
    すべて data.index に整列した bool の pd.Series。
"""

from __future__ import annotations

import itertools
from typing import Callable

import pandas as pd
import vectorbt as vbt
from joblib import Parallel, delayed

from . import config


# --- コスト --------------------------------------------------------------
def _slippage_series(pair: str, close: pd.Series) -> pd.Series:
    """スプレッドの片側(半分)を価格比率のスリッページとして表現。

    エントリーで +半スプレッド、エグジットで +半スプレッド = 往復で全スプレッド。
    バーごとの終値で割るので、10年スパンの価格変動にも追随する厳密な割合。
    """
    half_spread_price = config.spread_pips(pair) * config.pip_size(pair) / 2.0
    return half_spread_price / close


def _normalize_signals(sig):
    """2要素(ロングのみ)を 4 要素に正規化。"""
    if len(sig) == 2:
        le, lx = sig
        empty = le & False
        return le, lx, empty, empty
    if len(sig) == 4:
        return sig
    raise ValueError(
        "generate_signals は (entries, exits) か "
        "(long_entries, long_exits, short_entries, short_exits) を返すこと"
    )


def _apply_side(le, lx, se, sx, side: str):
    """side='long'/'short'/'both' に応じて片側シグナルを無効化(寄与分析用)。"""
    if side == "long":
        return le, lx, se & False, sx & False
    if side == "short":
        return le & False, lx & False, se, sx
    return le, lx, se, sx


def _stop_kwargs(sl_stop, tp_stop, tsl_stop) -> dict:
    """損切り/利確/トレーリングを from_signals 用に整形(None は無効)。割合指定。

    vectorbt はトレーリングを sl_stop + sl_trail=True で表現する。
    """
    k = {}
    if tsl_stop is not None:
        k["sl_stop"] = tsl_stop
        k["sl_trail"] = True
    elif sl_stop is not None:
        k["sl_stop"] = sl_stop
    if tp_stop is not None:
        k["tp_stop"] = tp_stop
    return k


def _atr_frac(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR を価格比(割合)で返す。リスク%サイジングのストップ幅算出に使う。"""
    h, l, c = data["high"], data["low"], data["close"]
    pc = c.shift()
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean() / c


# --- ポジションサイジング -----------------------------------------------
_MAX_LEVERAGE = 30.0  # FX リテール想定の上限(リスク%モードの暴走防止)


def _size_kwargs(size_mode, size_value, atr_mult, data, init_cash, sl_stop):
    """サイジングを from_signals 用に整形。戻り値 (kwargs, 実効sl_stop)。

      full    : 既定。余力フル(複利・両建て反転に対応)= vectorbt 既定挙動
      value   : 固定キャッシュ建玉(既定 init_cash。非複利で時系列比較しやすい)
      amount  : 固定数量(単位)
      risk    : 1トレードのリスク = init_cash * size_value(既定1%)。
                ATR ストップ(atr_mult×ATR)到達でその損失になるよう建玉を調整。
                sl_stop 未指定なら ATR ストップを自動設定。

    注: 'percent'(余力割合)は from_signals のポジション反転に非対応のため非提供。
        フル複利は 'full'、固定割合相当は 'value' を使う。
    """
    if size_mode == "full":
        return {}, sl_stop  # size 未指定 = vectorbt 既定(余力フル・反転対応)
    if size_mode == "value":
        return {"size": init_cash if size_value is None else size_value, "size_type": "value"}, sl_stop
    if size_mode == "amount":
        if size_value is None:
            raise ValueError("size_mode='amount' は size_value(数量)が必要")
        return {"size": size_value, "size_type": "amount"}, sl_stop
    if size_mode == "risk":
        rp = 0.01 if size_value is None else size_value
        if sl_stop is not None:
            stop_frac = sl_stop  # 明示ストップがあればそれでリスク計算
        else:
            stop_frac = (atr_mult * _atr_frac(data)).clip(lower=1e-4).bfill()
        value = (init_cash * rp) / stop_frac
        if hasattr(value, "clip"):
            value = value.clip(upper=init_cash * _MAX_LEVERAGE)
        else:
            value = min(value, init_cash * _MAX_LEVERAGE)
        return {"size": value, "size_type": "value"}, stop_frac
    raise ValueError(f"未知の size_mode: {size_mode}")


# --- 標準メトリクス ------------------------------------------------------
def metrics(pf: "vbt.Portfolio") -> pd.DataFrame:
    """Portfolio から標準指標を抽出(単一/複数カラム両対応の DataFrame)。"""
    def s(x):
        return x if isinstance(x, pd.Series) else pd.Series({"value": x})

    df = pd.DataFrame(
        {
            "total_return": s(pf.total_return()),
            "sharpe": s(pf.sharpe_ratio()),
            "sortino": s(pf.sortino_ratio()),
            "max_drawdown": s(pf.max_drawdown()),
            "win_rate": s(pf.trades.win_rate()),
            "profit_factor": s(pf.trades.profit_factor()),
            "num_trades": s(pf.trades.count()),
            "expectancy": s(pf.trades.expectancy()),
        }
    )
    return df


# --- 単発検証 ------------------------------------------------------------
def run(
    pair: str,
    timeframe: str,
    generate_signals: Callable,
    params: dict | None = None,
    *,
    data: pd.DataFrame | None = None,
    init_cash: float = 10_000,
    side: str = "both",
    sl_stop: float | None = None,
    tp_stop: float | None = None,
    tsl_stop: float | None = None,
    size_mode: str = "full",
    size_value: float | None = None,
    atr_mult: float = 2.0,
) -> "vbt.Portfolio":
    """1 つの (ペア, 時間足, パラメータ) で検証し Portfolio を返す。

    data を渡すとその DataFrame を使う(期間スライス/IS・OOS 検証用)。
    side='long'/'short' で片側のみ、sl_stop/tp_stop/tsl_stop(割合)で
    損切り・利確・トレーリングを付与できる(改善案の自動検証用)。
    size_mode で建玉量を制御: full(既定=余力フル/複利) / value(固定キャッシュ) /
    amount(固定数量) / risk(1トレード size_value=リスク%、ATR ストップ基準)。

    例:
        from strategies.ma_cross import generate_signals
        pf = run("EURUSD", "H1", generate_signals, {"fast": 20, "slow": 50})
        # 1トレード1%リスク(ATR2倍ストップ)で現実的に:
        pf = run("EURUSD", "H1", generate_signals, {"fast":20,"slow":50},
                 size_mode="risk", size_value=0.01)
    """
    from .data import load

    params = params or {}
    if data is None:
        data = load(pair, timeframe)
    le, lx, se, sx = _normalize_signals(generate_signals(data, **params))
    le, lx, se, sx = _apply_side(le, lx, se, sx, side)
    close = data["close"]
    freq = config.TIMEFRAMES[timeframe]
    size_kw, sl_eff = _size_kwargs(size_mode, size_value, atr_mult, data, init_cash, sl_stop)

    pf = vbt.Portfolio.from_signals(
        close,
        entries=le,
        exits=lx,
        short_entries=se,
        short_exits=sx,
        slippage=_slippage_series(pair, close),
        fees=config.COMMISSION_FRACTION,
        init_cash=init_cash,
        freq=freq,
        **size_kw,
        **_stop_kwargs(sl_eff, tp_stop, tsl_stop),
    )
    return pf


# --- パラメータ探索(並列・ベクトル化) --------------------------------
def sweep(
    pair: str,
    timeframe: str,
    generate_signals: Callable,
    param_grid: dict[str, list],
    *,
    data: pd.DataFrame | None = None,
    init_cash: float = 10_000,
    objective: str = "sharpe",
    side: str = "both",
    sl_stop: float | None = None,
    tp_stop: float | None = None,
    tsl_stop: float | None = None,
    size_mode: str = "full",
    size_value: float | None = None,
    atr_mult: float = 2.0,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """パラメータ総当りを並列・高速に検証し、結果を objective 降順で返す。

    全組合せのシグナルを joblib で並列生成 → 1 回のベクトル化シミュレーションで
    全カラムを同時に回す(numba 並列)。data を渡すと期間スライス検証(IS/OOS)に使える。

    例:
        from strategies.ma_cross import generate_signals
        res = sweep("EURUSD", "H1", generate_signals,
                    {"fast": [10,20,30], "slow": [50,100,200]})
        print(res.head())
    """
    from .data import load

    if data is None:
        data = load(pair, timeframe)
    close = data["close"]
    freq = config.TIMEFRAMES[timeframe]

    names = list(param_grid)
    combos = [dict(zip(names, vals)) for vals in itertools.product(*param_grid.values())]

    def _gen(combo):
        return _apply_side(*_normalize_signals(generate_signals(data, **combo)), side)

    sigs = Parallel(n_jobs=n_jobs)(delayed(_gen)(c) for c in combos)

    cols = pd.MultiIndex.from_tuples(
        [tuple(c[n] for n in names) for c in combos], names=names
    )
    le = pd.concat([s[0] for s in sigs], axis=1); le.columns = cols
    lx = pd.concat([s[1] for s in sigs], axis=1); lx.columns = cols
    se = pd.concat([s[2] for s in sigs], axis=1); se.columns = cols
    sx = pd.concat([s[3] for s in sigs], axis=1); sx.columns = cols

    size_kw, sl_eff = _size_kwargs(size_mode, size_value, atr_mult, data, init_cash, sl_stop)
    pf = vbt.Portfolio.from_signals(
        close,
        entries=le,
        exits=lx,
        short_entries=se,
        short_exits=sx,
        slippage=_slippage_series(pair, close),
        fees=config.COMMISSION_FRACTION,
        init_cash=init_cash,
        freq=freq,
        **size_kw,
        **_stop_kwargs(sl_eff, tp_stop, tsl_stop),
    )
    res = metrics(pf)
    if objective in res.columns:
        res = res.sort_values(objective, ascending=False)
    return res
