"""診断1: グロス vs ネット切り分け。

tsmom(lb100,band0) / breakout_trend / adx_trend を 7メジャー x D1 で
GROSS(コスト0)と NET(通常コスト)両方で回し、7ペア平均 total_return / sharpe を比較。

問い: 順張りは GROSS でも負け/ゼロ(=エッジ不在)か、GROSS は正だが NET で負け(=コスト殺し)か。
"""
from __future__ import annotations

import copy
import importlib

import numpy as np

import fxlab.config as C
from fxlab import load, run, metrics

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
TF = "D1"

# 戦略名 -> (module path, params)
STRATS = {
    "tsmom": ("strategies.tsmom", {"lookback": 100, "band": 0.0}),
    "breakout_trend": ("strategies.breakout_trend", {"entry": 40, "exit": 20, "trend": 200}),
    "adx_trend": ("strategies.adx_trend", {"fast": 20, "slow": 50, "adx_period": 14, "adx_th": 25}),
}

# データは一度だけ読む(キャッシュ済み)。コスト変更はrun()内のconfig参照で反映される。
DATA = {p: load(p, TF) for p in PAIRS}


def get_gen(modpath):
    mod = importlib.import_module(modpath)
    return mod.generate_signals


def measure(label):
    """現在のconfig設定下で全戦略x全ペアを回し結果dictを返す。"""
    out = {}
    for sname, (modpath, params) in STRATS.items():
        gen = get_gen(modpath)
        rows = {}
        for p in PAIRS:
            pf = run(p, TF, gen, params, data=DATA[p], size_mode="value", side="both")
            m = metrics(pf)
            # value/single col -> first row
            r = m.iloc[0]
            rows[p] = {
                "total_return": float(r["total_return"]),
                "sharpe": float(r["sharpe"]),
                "num_trades": int(r["num_trades"]),
            }
        out[sname] = rows
        sr = np.array([rows[p]["sharpe"] for p in PAIRS])
        tr = np.array([rows[p]["total_return"] for p in PAIRS])
        nt = np.array([rows[p]["num_trades"] for p in PAIRS])
        print(f"[{label}] {sname}: avg_sharpe={np.nanmean(sr):+.3f} "
              f"avg_total_return={np.nanmean(tr):+.4f} avg_trades={nt.mean():.1f}")
        for p in PAIRS:
            rr = rows[p]
            print(f"    {p}: sharpe={rr['sharpe']:+.3f} tr={rr['total_return']:+.4f} n={rr['num_trades']}")
    return out


# --- 元のコスト設定を退避 ---
ORIG_SPREADS = copy.deepcopy(C.SPREADS_PIPS)
ORIG_COMM = C.COMMISSION_FRACTION

# === NET (通常コスト) ===
print("=" * 70)
print("NET (normal cost: SPREADS_PIPS + COMMISSION_FRACTION)")
print("=" * 70)
C.SPREADS_PIPS = copy.deepcopy(ORIG_SPREADS)
C.COMMISSION_FRACTION = ORIG_COMM
net = measure("NET")

# === GROSS (コスト0) ===
print("=" * 70)
print("GROSS (zero cost)")
print("=" * 70)
C.SPREADS_PIPS = {k: 0.0 for k in C.SPREADS_PIPS}
C.COMMISSION_FRACTION = 0.0
gross = measure("GROSS")

# 復元
C.SPREADS_PIPS = ORIG_SPREADS
C.COMMISSION_FRACTION = ORIG_COMM

# === サマリ ===
print("=" * 70)
print("SUMMARY: 7-pair averages (sharpe / total_return)")
print("=" * 70)
gross_sharpes_all = []
net_sharpes_all = []
for sname in STRATS:
    gsr = np.nanmean([gross[sname][p]["sharpe"] for p in PAIRS])
    nsr = np.nanmean([net[sname][p]["sharpe"] for p in PAIRS])
    gtr = np.nanmean([gross[sname][p]["total_return"] for p in PAIRS])
    ntr = np.nanmean([net[sname][p]["total_return"] for p in PAIRS])
    gross_sharpes_all.append(gsr)
    net_sharpes_all.append(nsr)
    print(f"{sname:16s} GROSS sharpe={gsr:+.3f} tr={gtr:+.4f}  |  NET sharpe={nsr:+.3f} tr={ntr:+.4f}")

print("-" * 70)
print(f"3-STRATEGY MEAN  GROSS sharpe={np.mean(gross_sharpes_all):+.4f}  "
      f"NET sharpe={np.mean(net_sharpes_all):+.4f}")
