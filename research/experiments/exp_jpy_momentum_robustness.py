"""短期モメンタムのエッジは「JPY全般」(頑健)か「USDJPY×2022の運」(脆弱)か。

決定的な robustness ゲート: H1 ~1日tsmom の GROSS Sharpe を JPYファミリー4本 + 対照ペアで、
FULL / IS(2016-2020) / OOS(2021-2026) に分けて測る。
  JPY全般に広く・両期間で安定 → 構造的な「JPYトレンド/モメンタム」ファクター=採用候補。
  USDJPY単独・OOS偏在 → 2022 BoJ起因の単一標本=脆弱=不採用。
さらに lb{12,24,36} で plateau も確認。
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from fxlab import run, universe as uni
import fxlab.config as C
from strategies.tsmom import generate_signals as tsmom_sig

ORIG = dict(C.SPREADS_PIPS)
uni.register_cross_spreads(3.0)


def gross_sharpe(pair, lookback, period=None):
    C.SPREADS_PIPS = {k: 0.0 for k in ORIG}; C.COMMISSION_FRACTION = 0.0
    data = uni.instrument_data(pair, "H1")
    if period:
        data = data.loc[period[0]:period[1]]
    try:
        pf = run(pair, "H1", tsmom_sig, {"lookback": lookback, "band": 0.0},
                 data=data, size_mode="value", side="both")
        from fxlab import metrics
        s = metrics(pf)["sharpe"]
        s = float(s) if not hasattr(s, "__len__") else float(np.asarray(s).ravel()[0])
    except Exception:  # noqa: BLE001
        s = np.nan
    C.SPREADS_PIPS = dict(ORIG)
    return s


def main():
    JPY = ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY"]
    OTHER3 = ["EURUSD", "GBPUSD"]      # 3ペアバスケットの非JPY構成
    CONTRA = ["AUDUSD", "NZDUSD", "USDCHF"]  # 短期反転で負ける対照
    print("=== H1 lb24 ~1日モメンタム GROSS Sharpe: FULL / IS(16-20) / OOS(21-26) ===\n")
    print(f"{'pair':>8} {'group':>7} {'FULL':>8} {'IS':>8} {'OOS':>8} {'両期正?':>7}")
    for grp, pairs in [("JPY", JPY), ("EUR/GBP", OTHER3), ("対照", CONTRA)]:
        for p in pairs:
            f = gross_sharpe(p, 24)
            i = gross_sharpe(p, 24, ("2016-01-01", "2020-12-31"))
            o = gross_sharpe(p, 24, ("2021-01-01", "2026-12-31"))
            both = "✓" if (i > 0 and o > 0) else ""
            print(f"{p:>8} {grp:>7} {f:>8.3f} {i:>8.3f} {o:>8.3f} {both:>7}")
        print()
    print("=== lookback plateau (FULL GROSS Sharpe) ===")
    print(f"{'pair':>8} {'lb12':>8} {'lb24':>8} {'lb36':>8} {'lb48':>8}")
    for p in JPY + OTHER3:
        vals = [gross_sharpe(p, lb) for lb in (12, 24, 36, 48)]
        print(f"{p:>8} " + " ".join(f"{v:>8.3f}" for v in vals))
    print("\n判定: JPY4本がFULL+両期間で広く正 & lookback plateau → 構造的JPYモメンタム=採用候補。")
    print("      USDJPYのみ/OOS偏在/単一lb → 2022単一標本=脆弱=不採用。")


if __name__ == "__main__":
    main()
