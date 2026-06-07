"""イテレーション13: 時間ストップで「塩漬けの負け」を外科的に除去 → PF2.0を狙う。

exp12_worst_trades の発見: 保有期間とリターンは強い負相関(-0.85)。0-30本は勝ち、
31本以上で急速に負け化(=平均回帰が外れトレンド化した塩漬け)。ワースト10%(平均55本保有)が
総損失の72.5%。タイトな価格損切りは勝ちを刈って逆効果だったが、**時間ストップ**は速攻反転の
勝ちを残し塩漬けの負けだけ切るので、PFを大きく改善しうる(経済合理・非カーブフィット)。

エントリーから max_hold 本で強制手仕舞い(Z回帰の手仕舞いと併用、早い方)。
取引『数』は減らさない(出口を早めるだけ)ので年100取引は維持されるはず。
19対象ポートで max_hold を振り、PF/毎年プラス/取引 の高原を確認する。
実行: uv run python exp13.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import universe as uni
from strategies.confluence_meanrev import generate_signals as cmr

pd.set_option("display.width", 200)
TF = "H4"
P = {"window": 50, "entry_z": 2.0, "exit_z": 0.5, "rsi_p": 14, "rsi_low": 35, "rsi_high": 65,
     "vol_win": 100, "vol_pct": 0.70, "slow_win": 250, "slow_z": 1.75}


def make_gen(max_hold):
    """confluence + 時間ストップ(entryから max_hold 本で強制exit)。"""
    def gen(data, **params):
        le, lx, se, sx = cmr(data, **params)
        if max_hold and max_hold > 0:
            lx = lx | le.shift(max_hold, fill_value=False)
            sx = sx | se.shift(max_hold, fill_value=False)
        return le, lx, se, sx
    return gen


def evalp(instruments, max_hold):
    port = uni.portfolio_yearly(TF, make_gen(max_hold), P, instruments=instruments, size_mode="value")
    if port.empty:
        return None
    pf = port["profit_factor"].replace(np.inf, np.nan)
    return (port["pnl"] > 0).mean(), pf.median(), pf.min(), int(port["trades"].mean()), port


def main():
    uni.register_cross_spreads(3.0)
    instruments = [x for x in uni.universe() if x != "AUDJPY"]
    print(f"対象 {len(instruments)} / {TF}  (★ = PF中央≥2.0 & 年取引≥100 & 毎年プラス100%)\n")
    print(f"{'max_hold(本)':>12} {'毎年+':>6} {'PF中央':>7} {'PF最小':>7} {'年取引':>7}")
    best = None
    for mh in [0, 60, 45, 40, 35, 30, 25, 20, 15]:
        r = evalp(instruments, mh)
        if r is None:
            continue
        pos, med, mn, tr, port = r
        star = " ★" if (pos == 1.0 and tr >= 100 and med >= 2.0) else ""
        label = "なし" if mh == 0 else f"{mh}"
        print(f"{label:>12} {pos:>6.0%} {med:>7.2f} {mn:>7.2f} {tr:>7d}{star}")
        if pos == 1.0 and tr >= 100 and med >= 2.0 and (best is None or med > best[1]):
            best = (mh, med, port)

    if best:
        mh, med, port = best
        print(f"\n=== ★ 3条件達成: max_hold={mh}本 (PF中央 {med:.2f}) — 年次 ===")
        b = port.copy(); b["profit_factor"] = b["profit_factor"].replace(np.inf, np.nan).round(2)
        b["pnl"] = b["pnl"].round(0); b["return_pct"] = b["return_pct"].round(2)
        print(b.to_string())
    else:
        print("\n3条件同時達成は時間ストップでも未達。")


if __name__ == "__main__":
    main()
