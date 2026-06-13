"""seed主張(M30 lb48=+0.523 / H1 lb24=+0.512 GROSS Sharpe)の再現確認。

per-pair Sharpeの単純平均 vs 7ペア等加重ポートフォリオ(日次リターン合算)のSharpe
のどちらでseed数値が出るかを確認し、breadthの解釈を確定する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import fxlab.config as C

_ORIG_SPREADS = dict(C.SPREADS_PIPS)
_ORIG_COMM = C.COMMISSION_FRACTION
C.SPREADS_PIPS = {k: 0.0 for k in C.SPREADS_PIPS}
C.COMMISSION_FRACTION = 0.0

from fxlab import load, run, metrics  # noqa: E402
from strategies.tsmom import generate_signals  # noqa: E402

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]


def portfolio_sharpe(tf, lookback):
    """7ペア等加重: 各ペアのbar-returns系列を合算しSharpe(年率)を計算。"""
    rets = []
    per_pair = {}
    for pair in PAIRS:
        data = load(pair, tf)
        pf = run(pair, tf, generate_signals, {"lookback": lookback, "band": 0.0},
                 data=data, size_mode="value", side="both")
        r = pf.returns()
        if isinstance(r, pd.DataFrame):
            r = r.iloc[:, 0]
        rets.append(r.rename(pair))
        m = metrics(pf)
        per_pair[pair] = float(m["sharpe"])
    mat = pd.concat(rets, axis=1).fillna(0.0)
    port = mat.mean(axis=1)  # 等加重
    # 年率Sharpe: barあたり -> 年率。bars/yearを推定
    bars_per_year = {"H1": 24 * 252, "M30": 48 * 252, "M15": 96 * 252}[tf]
    mu = port.mean()
    sd = port.std()
    ann = (mu / sd) * np.sqrt(bars_per_year) if sd > 0 else 0.0
    return ann, per_pair, np.mean(list(per_pair.values()))


def main():
    for tf, lb in [("H1", 24), ("M30", 48), ("M30", 96), ("H1", 48)]:
        ann, per_pair, mean_sh = portfolio_sharpe(tf, lb)
        print(f"--- {tf} lb{lb} ---")
        print(f"  equal-weight PORTFOLIO ann Sharpe = {ann:.4f}")
        print(f"  mean of per-pair Sharpe          = {mean_sh:.4f}")
        print(f"  per-pair: " + ", ".join(f"{p}={v:.3f}" for p, v in per_pair.items()))


if __name__ == "__main__":
    main()
    C.SPREADS_PIPS = _ORIG_SPREADS
    C.COMMISSION_FRACTION = _ORIG_COMM
