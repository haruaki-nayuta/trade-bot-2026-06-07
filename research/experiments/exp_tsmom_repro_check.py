"""セッション報告値(EURUSD H1 lb24 GROSS Sharpe +0.512 等)の再現確認。

報告値: M15 lb96=+0.517 / M30 lb48=+0.523 / H1 lb24=+0.512。
これがEURUSD単独なのか7ペア平均なのか、size_modeで変わるかを切り分ける。
strategies/tsmom.py をそのまま使い、size_mode と side を変えて測る。
"""

from __future__ import annotations

import fxlab.config as C
from fxlab import load, metrics, run
from strategies.tsmom import generate_signals


def set_gross():
    C.SPREADS_PIPS = {k: 0.0 for k in C.SPREADS_PIPS}
    C.COMMISSION_FRACTION = 0.0


def sh(pf):
    return float(metrics(pf)["sharpe"].iloc[0])


def main():
    set_gross()
    configs = [("H1", 24), ("M30", 48), ("M15", 96)]
    for tf, lb in configs:
        data = load("EURUSD", tf)
        for sm in ["full", "value"]:
            pf = run("EURUSD", tf, generate_signals, {"lookback": lb, "band": 0.0},
                     data=data, size_mode=sm, side="both")
            print(f"EURUSD {tf} lb{lb} size={sm}: sharpe={sh(pf):+.4f}")


if __name__ == "__main__":
    main()
