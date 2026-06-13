"""短い時間足(M5/M15/M30/H1)でトレンドが拾えるか + スプレッドが食うか。

ユーザーの問い: 検証はD1中心だった。もっと短い足ならトレンドを拾えるか? 今度はスプレッド負けするか?
事前予想(reports/12): 短期FXは反転支配でトレンドはグロスでも負ける。→ 主張でなく実測で確認。

決定的軸 = GROSS(コスト0):
  GROSS正 → トレンドエッジ実在 → NETで残るか(スプレッド負けか)を見る。
  GROSS負 → 短期足でもトレンド不在(反転が支配)= スプレッド以前の問題。
低スプレッドの主要3ペア(EURUSD0.6/USDJPY0.7/GBPUSD0.9 pips)で測る(=トレンドに最も有利な条件)。
注意: 短期足はUTC20-22ロールオーバーBIDアーティファクトの偽ロングに注意(reports/12)。本スキャンは
  両建てtsmomなので片側バイアスは出にくいが、グロス正が出たら時間帯除染が必須。
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from fxlab import load, run, metrics
import fxlab.config as C
from strategies.tsmom import generate_signals as tsmom_sig

ORIG_SPREADS = dict(C.SPREADS_PIPS)
ORIG_COMM = C.COMMISSION_FRACTION
PAIRS = ["EURUSD", "USDJPY", "GBPUSD"]

# TFごとの lookback(バー単位)。日内〜数日のトレンドを狙う。
TF_LOOKBACKS = {
    "M5":  [48, 144, 288, 576],     # 4h / 12h / 1d / 2d
    "M15": [32, 96, 288],           # 8h / 1d / 3d
    "M30": [24, 48, 144],           # 12h / 1d / 3d
    "H1":  [24, 72, 168],           # 1d / 3d / 1w
}


def _f(x):
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return float(np.asarray(x).ravel()[0])


def measure(pair, tf, lookback, gross):
    if gross:
        C.SPREADS_PIPS = {k: 0.0 for k in ORIG_SPREADS}
        C.COMMISSION_FRACTION = 0.0
    else:
        C.SPREADS_PIPS = dict(ORIG_SPREADS)
        C.COMMISSION_FRACTION = ORIG_COMM
    data = load(pair, tf)
    pf = run(pair, tf, tsmom_sig, {"lookback": lookback, "band": 0.0},
             data=data, size_mode="value", side="both")
    m = metrics(pf)
    return _f(m["total_return"]), _f(m["sharpe"]), _f(m["num_trades"])


def main():
    print("=== 短い足でのトレンド(tsmom両建て)gross/net 実測 ===")
    print("低スプレッド主要3ペア平均。GROSS_Sh>0 ならトレンドエッジ有→NET差がスプレッド負け分。\n")
    best_gross = -9
    for tf, lbs in TF_LOOKBACKS.items():
        print(f"--- {tf} ---")
        print(f"{'lookback':>9} {'GROSS_tr':>9} {'GROSS_Sh':>9} {'NET_tr':>8} {'NET_Sh':>8} {'cost_drag':>10} {'ntr':>6}")
        for lb in lbs:
            grs, nts = [], []
            gtr_l, gsh_l, ntr_l, ntrd_l = [], [], [], []
            for p in PAIRS:
                gtr, gsh, gn = measure(p, tf, lb, gross=True)
                ntr_, nsh, nn = measure(p, tf, lb, gross=False)
                gtr_l.append(gtr); gsh_l.append(gsh); ntr_l.append(ntr_)
                grs.append(gsh); nts.append(nsh); ntrd_l.append(nn)
            gtr_avg = np.mean(gtr_l); gsh_avg = np.mean(gsh_l)
            ntr_avg = np.mean(ntr_l); nsh_avg = np.mean(nts)
            cost_drag = gtr_avg - ntr_avg
            ntrd = int(np.mean(ntrd_l))
            best_gross = max(best_gross, gsh_avg)
            print(f"{lb:>9} {gtr_avg:>9.3f} {gsh_avg:>9.3f} {ntr_avg:>8.3f} {nsh_avg:>8.3f} {cost_drag:>10.3f} {ntrd:>6}")
        print()
    print(f"全TF・全lookbackの最大 GROSS Sharpe = {best_gross:.3f}")
    print("判定: 最大GROSS Sharpe<=0 なら短期足でもトレンド不在(反転支配)=スプレッド以前の問題。")
    print("      GROSS>0 だが NET<0 なら『スプレッドが食う』=低スプレッド口座/指値で救える可能性→要時間帯除染。")
    C.SPREADS_PIPS = dict(ORIG_SPREADS)
    C.COMMISSION_FRACTION = ORIG_COMM


if __name__ == "__main__":
    main()
