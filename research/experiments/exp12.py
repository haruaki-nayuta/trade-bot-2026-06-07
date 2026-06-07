"""イテレーション12: 低時間足 × 高選別 で PF2.0 を狙う(鉄の三角形の突破試行)。

H4 では「選別を上げる=取引数<100」でPF2.0に届かない。M30/H1 は足数が数倍あるので、
時間軸(窓)を比例拡大して経済的horizonを揃えつつ、より厳しい閾値で選別すれば、
ポート合算の取引数100+ を保ったまま 1対象あたりPFを上げられる可能性がある。

19対象(メジャー+主要クロス, AUDJPY除外)で、TF×選別 を走査し、
「PF中央≥2.0 かつ 年取引≥100 かつ 毎年プラス100%」を満たす構成を探す。
実行: uv run python exp12.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import universe as uni
from strategies.confluence_meanrev import generate_signals as g

pd.set_option("display.width", 200)


def evalcfg(tf, instruments, **p):
    base = dict(window=50, entry_z=2.0, exit_z=0.5, rsi_p=14, rsi_low=35, rsi_high=65,
                vol_win=100, vol_pct=0.70, slow_win=250, slow_z=1.75)
    base.update(p)
    port = uni.portfolio_yearly(tf, g, base, instruments=instruments, size_mode="value")
    if port.empty:
        return None
    pf = port["profit_factor"].replace(np.inf, np.nan)
    return {
        "pos_rate": (port["pnl"] > 0).mean(),
        "pf_med": pf.median(),
        "pf_min": pf.min(),
        "trades": int(port["trades"].mean()),
        "port": port,
    }


def main():
    uni.register_cross_spreads(3.0)
    instruments = [x for x in uni.universe() if x != "AUDJPY"]

    # TF ごとに「同一horizon」になるよう窓を比例拡大(H4基準: window=50≈8日, slow=250≈41日)
    # H1 は H4 の4倍の足、M30 は8倍。
    configs = {
        "H4": dict(window=50, slow_win=250),
        "H1": dict(window=200, slow_win=1000),
        "M30": dict(window=400, slow_win=2000),
    }
    # 選別レベル(entry_z, slow_z, vol_pct)。低TFほど取引が増えるので厳選を強められる。
    selects = [
        ("標準", dict(entry_z=2.0, slow_z=1.75, vol_pct=0.70)),
        ("やや厳", dict(entry_z=2.25, slow_z=2.0, vol_pct=0.70)),
        ("厳選", dict(entry_z=2.5, slow_z=2.25, vol_pct=0.65)),
        ("超厳選", dict(entry_z=2.75, slow_z=2.5, vol_pct=0.60)),
    ]

    print(f"対象 {len(instruments)}  (★ = PF中央≥2.0 & 年取引≥100 & 毎年プラス100%)\n")
    print(f"{'TF':>4} {'選別':>8} {'毎年+':>6} {'PF中央':>7} {'PF最小':>7} {'年取引':>7}")
    hits = []
    for tf, win in configs.items():
        for label, sel in selects:
            r = evalcfg(tf, instruments, **win, **sel)
            if r is None:
                print(f"{tf:>4} {label:>8}  (取引なし)"); continue
            star = ""
            if r["pos_rate"] == 1.0 and r["trades"] >= 100 and r["pf_med"] >= 2.0:
                star = " ★"; hits.append((tf, label, r))
            print(f"{tf:>4} {label:>8} {r['pos_rate']:>6.0%} {r['pf_med']:>7.2f} "
                  f"{r['pf_min']:>7.2f} {r['trades']:>7d}{star}")

    if hits:
        print("\n=== ★ 3条件同時達成の構成 ===")
        for tf, label, r in hits:
            print(f"\n--- {tf} / {label} ---")
            b = r["port"].copy()
            b["profit_factor"] = b["profit_factor"].replace(np.inf, np.nan).round(2)
            b["pnl"] = b["pnl"].round(0)
            print(b.to_string())
    else:
        print("\n3条件同時達成の構成は見つからず(鉄の三角形を再確認)。")


if __name__ == "__main__":
    main()
