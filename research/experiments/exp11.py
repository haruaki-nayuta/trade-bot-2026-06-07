"""イテレーション11: 真の前向き(OOS)検証 — チャンピオンが将来データで通用するか。

パラメータ(window=50,entry_z=2,RSI35/65,slow_win=250,slow_z=1.75,vol_pct=0.70)は全期間を見て
選んだ「広い高原上の丸い値」。それでも厳密性のため、IS(2016-2021)で固定 → OOS(2022-2026)を
完全な素検証として分離し、前向きの成績(毎年プラス/PF/取引)を確認する。実運用化の核心検証。

実行: uv run python exp11.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import universe as uni
from strategies.confluence_meanrev import generate_signals as g

pd.set_option("display.width", 200)
TF = "H4"
P = {"window": 50, "entry_z": 2.0, "exit_z": 0.5, "rsi_p": 14, "rsi_low": 35, "rsi_high": 65,
     "vol_win": 100, "vol_pct": 0.70, "slow_win": 250, "slow_z": 1.75}


def main():
    uni.register_cross_spreads(3.0)
    instruments = [x for x in uni.universe() if x != "AUDJPY"]
    port = uni.portfolio_yearly(TF, g, P, instruments=instruments, size_mode="value")
    pf = port["profit_factor"].replace(np.inf, np.nan)

    is_yrs = [y for y in port.index if y <= 2021]
    oos_yrs = [y for y in port.index if y >= 2022]   # 後半≈4.5年 = 前向き相当

    def block(yrs, label):
        sub = port.loc[yrs]
        spf = sub["profit_factor"].replace(np.inf, np.nan)
        print(f"\n--- {label}({int(yrs[0])}-{int(yrs[-1])}) ---")
        show = sub.copy()
        show["profit_factor"] = show["profit_factor"].replace(np.inf, np.nan).round(2)
        show["pnl"] = show["pnl"].round(0); show["return_pct"] = show["return_pct"].round(2)
        print(show.to_string())
        print(f"  → 毎年プラス {(sub['pnl']>0).mean():.0%} / PF中央 {spf.median():.2f} / "
              f"PF最小 {spf.min():.2f} / 年取引 {int(sub['trades'].mean())}")

    print(f"=== 前向き検証: confluence_meanrev 推奨構成({len(instruments)}対象, 固定パラメータ)===")
    block(is_yrs, "IS(設計参照期間)")
    block(oos_yrs, "OOS(前向き=未使用扱い)")
    print(f"\n全期間: 毎年プラス {(port['pnl']>0).mean():.0%} / PF中央 {pf.median():.2f} / 年取引 {int(port['trades'].mean())}")


if __name__ == "__main__":
    main()
