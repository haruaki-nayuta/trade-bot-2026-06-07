"""USDバスケット・トレンドのゲート版: 平時フラット化で「単体黒字 × 失血窓保護」の両立を狙う。

scout(exp_usdbasket_scout)の発見:
  - usdbasket_D1_ens は失血窓で稼ぐ(IS+123/OOS+381)が常時建玉で平時whipsaw=単体赤字-4364。
  - 単一3moは単体+1684。→ 平時のチョップを「ゲートでフラット化」すれば両立しうる。

ゲート2種(因果):
  consensus: 各lookbackのモメンタム符号の合算 |Σ| が閾値以上のときだけ建玉(弱/対立トレンド=フラット)。
  er_gate  : USDバスケットの効率比ER(er_win)が閾値以上(=一直線の強ドルトレンド)のときだけ建玉。
両ゲートとも「ドルが明確にトレンドしている時だけ順張り、レンジは休む」=平時ドラッグの除去。

scanで高原(広い設定でプラス)を確認 → 単一点の過剰最適化を回避。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from fxlab import config, universe as uni
import bleed_lab as bl

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 60)

NOTIONAL = 10_000.0
MAJORS = list(config.PAIRS)
USD_BASE = {"USDJPY", "USDCHF", "USDCAD"}
OOS = pd.Period("2022-01", "M")


def usd_sign(p): return +1.0 if p in USD_BASE else -1.0


def load_majors(tf):
    return pd.DataFrame({p: uni.instrument_close(p, tf) for p in MAJORS}).dropna()


def _basket(tf):
    cl = load_majors(tf)
    signs = np.array([usd_sign(p) for p in cl.columns])
    logret = np.log(cl).diff()
    usd_lr = (logret.to_numpy() * signs).mean(axis=1)
    usd_idx = pd.Series(np.nancumsum(np.nan_to_num(usd_lr)), index=cl.index)
    half = {p: config.spread_pips(p) * config.pip_size(p) / 2.0 / float(cl[p].mean()) for p in cl.columns}
    return cl, signs, usd_idx, half


def _basket_er(usd_idx: pd.Series, w: int) -> pd.Series:
    direction = (usd_idx - usd_idx.shift(w)).abs()
    vol = usd_idx.diff().abs().rolling(w).sum()
    return (direction / vol).replace([np.inf, -np.inf], np.nan)


def basket_tsmom_gated(tf, lookbacks, consensus_min=1, er_gate=None, er_win=63,
                       min_hold=1) -> pd.DataFrame:
    cl, signs, usd_idx, half = _basket(tf)
    n = len(cl)
    consensus = np.zeros(n)
    for L in lookbacks:
        consensus = consensus + np.sign((usd_idx - usd_idx.shift(L)).fillna(0.0).to_numpy())
    raw = np.where(np.abs(consensus) >= consensus_min, np.sign(consensus), 0.0)
    if er_gate is not None:
        er = _basket_er(usd_idx, er_win).fillna(0.0).to_numpy()
        raw = np.where(er >= er_gate, raw, 0.0)
    pos = pd.Series(raw, index=cl.index).shift(1).fillna(0.0).to_numpy()

    arr = cl.to_numpy(); idx = cl.index; cols = list(cl.columns)
    maxL = max(lookbacks)
    recs = []
    t = maxL + 1
    while t < n - 1:
        d = pos[t]
        if d == 0.0:
            t += 1; continue
        e = t
        while e < n - 1 and pos[e] == d:
            e += 1
        if e - t >= min_hold:
            ts = idx[e]
            for j, p in enumerate(cols):
                fwd = arr[e, j] / arr[t, j] - 1.0
                recs.append((ts, (d * signs[j] * fwd - 2 * half[p]) * NOTIONAL))
        t = e
    return pd.DataFrame(recs, columns=["exit", "pnl"])


def to_monthly(tr):
    if len(tr) == 0:
        return pd.Series(dtype=float)
    m = pd.PeriodIndex(pd.to_datetime(tr["exit"]).dt.to_period("M"))
    return tr.assign(m=m).groupby("m")["pnl"].sum()


def metrics(monthly, mask, champ_monthly):
    monthly = monthly.reindex(mask.index).fillna(0.0)
    yr = monthly.groupby([p.year for p in monthly.index]).sum()
    pos = monthly[monthly > 0].sum(); neg = -monthly[monthly < 0].sum()
    inb = monthly[mask.values]; out = monthly[~mask.values]
    is_m = mask & (mask.index < OOS); oos_m = mask & (mask.index >= OOS)
    cr = float(np.corrcoef(monthly.values, champ_monthly.values)[0, 1])
    return dict(
        total=float(monthly.sum()), pos_yr=float((yr > 0).mean()),
        PF=float(pos / neg) if neg > 0 else np.nan,
        IS=float(monthly[monthly.index < OOS].sum()),
        OOS=float(monthly[monthly.index >= OOS].sum()),
        worst_yr=float(yr.min()), n_trades_yr=len(monthly),
        corr=cr, bleed_mean=float(inb.mean()), normal_mean=float(out.mean()),
        bleed_IS=float(monthly[is_m.values].mean()) if is_m.sum() else np.nan,
        bleed_OOS=float(monthly[oos_m.values].mean()) if oos_m.sum() else np.nan,
        n_active_months=int((monthly != 0).sum()),
    )


def main():
    eqm, eqr, pool, _ = bl.champion_mtm(max_pos=8)
    mask, dd = bl.bleed_mask_monthly(eqm)
    cm = pool.copy()
    cm["m"] = pd.PeriodIndex(pd.to_datetime(cm["exit"]).dt.to_period("M"))
    champ_monthly = cm.groupby("m")["ret"].sum().reindex(mask.index).fillna(0.0) * NOTIONAL

    LB = [21, 63, 126, 252]   # D1 ensemble 1/3/6/12mo
    print("=== ゲート・スキャン(D1 ensemble lookbacks=1/3/6/12mo, 各脚$10k)===")
    print("consensus=|Σsign|の最小値(4本中) / er_gate=バスケットER下限\n")
    rows = []
    for cons in (1, 2, 3, 4):
        for erg in (None, 0.20, 0.30, 0.40):
            tr = basket_tsmom_gated("D1", LB, consensus_min=cons, er_gate=erg, er_win=63)
            mo = to_monthly(tr)
            if len(mo) == 0:
                continue
            mm_ = metrics(mo, mask, champ_monthly)
            rows.append({
                "cons": cons, "er_gate": erg if erg else "-",
                "total": round(mm_["total"], 0), "pos_yr": f"{mm_['pos_yr']:.0%}",
                "PF": round(mm_["PF"], 2), "IS": round(mm_["IS"], 0), "OOS": round(mm_["OOS"], 0),
                "worst_yr": round(mm_["worst_yr"], 0), "corr": round(mm_["corr"], 3),
                "bleedIS": round(mm_["bleed_IS"], 0), "bleedOOS": round(mm_["bleed_OOS"], 0),
                "active_mo": mm_["n_active_months"],
            })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\n探索目標: total>0 & pos_yr高 & IS/OOS両プラス & bleedIS/bleedOOS両プラス(失血窓保護)が")
    print("広い(cons,er_gate)で成立する高原。単一セルだけ良いのは過剰最適化。")


if __name__ == "__main__":
    main()
