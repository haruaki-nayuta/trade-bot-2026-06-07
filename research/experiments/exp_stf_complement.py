"""短期足モメンタム(EUR/JPY/GBP ~1日tsmom, 半スプレッド)はチャンピオンの補完になるか。

Workflow確定: 短期tsmomのグロス+0.5は EUR/JPY/GBP の3ペア(実質USDJPY)に集中、非アーティファクト、
NET半スプレッドで3ペア+0.29。だが単一通貨依存・plateauなし。
ここで最後の問い: これがチャンピオン(H4平均回帰)と低/負相関で、失血窓を守り、2スリーブでCAGRを上げるか。
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from fxlab import load, run
from fxlab.trades import trade_table
import fxlab.config as C
from strategies.tsmom import generate_signals as tsmom_sig
import bleed_lab as bl

ORIG = dict(C.SPREADS_PIPS); ORIG_COMM = C.COMMISSION_FRACTION
NOTIONAL = 10_000.0
OOS = pd.Period("2022-01", "M")


def mom_monthly(pairs, lookback=24, half=True):
    if half:
        C.SPREADS_PIPS = {k: v * 0.5 for k, v in ORIG.items()}
    else:
        C.SPREADS_PIPS = dict(ORIG)
    C.COMMISSION_FRACTION = ORIG_COMM
    rows = []
    for p in pairs:
        data = load(p, "H1")
        pf = run(p, "H1", tsmom_sig, {"lookback": lookback, "band": 0.0},
                 data=data, size_mode="value", side="both")
        tt = trade_table(pf, data)
        for _, r in tt.iterrows():
            rows.append((r["exit"], r["pnl"]))
    C.SPREADS_PIPS = dict(ORIG)
    df = pd.DataFrame(rows, columns=["exit", "pnl"])
    m = pd.PeriodIndex(pd.to_datetime(df["exit"]).dt.to_period("M"))
    return df.assign(m=m).groupby("m")["pnl"].sum()


def main():
    eqm, eqr, pool, _ = bl.champion_mtm(max_pos=8)
    mask, dd = bl.bleed_mask_monthly(eqm)
    cm = pool.copy(); cm["m"] = pd.PeriodIndex(pd.to_datetime(cm["exit"]).dt.to_period("M"))
    champ = cm.groupby("m")["ret"].sum().reindex(mask.index).fillna(0.0) * NOTIONAL

    for label, pairs in [("EUR/JPY/GBP", ["EURUSD", "USDJPY", "GBPUSD"]),
                         ("USDJPY単独", ["USDJPY"])]:
        mo = mom_monthly(pairs, half=True).reindex(mask.index).fillna(0.0)
        cr = float(np.corrcoef(mo.values, champ.values)[0, 1])
        inb = mo[mask.values]; out = mo[~mask.values]
        yr = mo.groupby([p.year for p in mo.index]).sum()
        # 月次vol正規化2スリーブ: 合成Sharpe vs champion単独
        rc = champ / champ.std(); rx = mo / mo.std()
        sc = champ.mean() / champ.std() * np.sqrt(12)
        print(f"\n=== {label} ~1日モメンタム(H1 lb24, 半スプレッド)===")
        print(f"  単体: total={mo.sum():.0f}  プラス年率={(yr>0).mean():.0%}  worst_yr={yr.min():.0f}")
        print(f"  チャンピオン月次相関 = {cr:+.3f}")
        print(f"  失血窓 平均月次P&L = {inb.mean():+.1f}  (平時 {out.mean():+.1f})  → 失血窓で{'稼ぐ' if inb.mean()>0 else '沈む'}")
        print(f"  2022(最大失血年) P&L = {yr.get(2022, float('nan')):+.0f}")
        # 合成Sharpe(月次)
        champ_sh = champ.mean()/champ.std()*np.sqrt(12)
        for w in (0.1, 0.2, 0.3):
            comb = (1-w)*rc + w*rx
            csh = comb.mean()/comb.std()*np.sqrt(12)
            print(f"    w={w}: 合成月次Sharpe={csh:.3f} (champ単独={champ_sh:.3f}, Δ={csh-champ_sh:+.3f})")
    print("\n判定: 相関が負&失血窓で稼ぐ&合成SharpeがΔ正なら補完価値。単一通貨依存・半スプ依存は実装リスクとして併記。")


if __name__ == "__main__":
    main()
