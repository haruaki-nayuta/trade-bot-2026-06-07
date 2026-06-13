"""補完エッジ・スカウト: 「単体ロバスト性 × チャンピオン相関 × 失血窓貢献」を一括測定。

目的(ユーザー goal): チャンピオン(confluence_meanrev_v2, 平均回帰・無ストップ)を補う
**通貨限定**の相補手法を探す。利益の正体(reports/15)から、理想の補完は
「中庸ボラ × 持続USDトレンド(高ER)の失血窓で稼ぐトレンド追随」かつ「単体でロバスト」。

未検証の核心アイデア = **USDバスケット・トレンド追随**:
  単ペアFXトレンドは「67構成全滅」(idiosyncraticノイズのwhipsaw)。だが7メジャーの
  **共通ドルファクターを集約**すればS/N比が上がり、チャンピオンの失血窓(=持続USDトレンド)を
  ピンポイントで捉えうる。これをTSMOM(複数lookbackアンサンブル)で構築し、
  既存候補(xsec / per-pair tsmom / trend_insurance)と公平比較する。

実行: PYTHONPATH=lab:money_management:repo  uv run python -m research.experiments.exp_usdbasket_scout
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from fxlab import config, universe as uni
import mm_lab as mm
import bleed_lab as bl

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 50)

NOTIONAL = 10_000.0
MAJORS = list(config.PAIRS)                 # 7 USD majors
USD_BASE = {"USDJPY", "USDCHF", "USDCAD"}   # USDが基軸 → 上昇=USD高
OOS = pd.Period("2022-01", "M")


def usd_sign(p: str) -> float:
    """そのペアのlogリターンがUSD指数に寄与する符号(+1=上昇でUSD高)。"""
    return +1.0 if p in USD_BASE else -1.0


def load_majors(tf: str) -> pd.DataFrame:
    return pd.DataFrame({p: uni.instrument_close(p, tf) for p in MAJORS}).dropna()


def half_spreads(cl: pd.DataFrame) -> dict:
    return {p: config.spread_pips(p) * config.pip_size(p) / 2.0 / float(cl[p].mean())
            for p in cl.columns}


# ── USDバスケット TSMOM(セグメント=トレンド継続区間で1往復コスト)──────────
def usd_basket_tsmom(tf: str, lookbacks, min_hold=1) -> pd.DataFrame:
    """共通ドルファクターのアンサンブルTSMOM。トレード単位PnL(各脚$10k×7脚)を返す。

    先読みなし: t時点で確定の lookback リターン符号→t+1バーから建玉。方向が変わるまで保持し、
    保持区間(セグメント)ごとに各脚 1往復スプレッドを計上。
    """
    cl = load_majors(tf)
    signs = np.array([usd_sign(p) for p in cl.columns])
    logret = np.log(cl).diff()
    usd_lr = (logret.to_numpy() * signs).mean(axis=1)          # USD指数のバーlogリターン
    usd_idx = pd.Series(np.nancumsum(np.nan_to_num(usd_lr)), index=cl.index)

    # アンサンブル方向: 各lookbackのモメンタム符号の合算→純符号
    sig = np.zeros(len(cl))
    for L in lookbacks:
        sig = sig + np.sign((usd_idx - usd_idx.shift(L)).fillna(0.0).to_numpy())
    sig = np.sign(sig)
    pos = pd.Series(sig, index=cl.index).shift(1).fillna(0.0).to_numpy()   # 因果(t-1で決定)

    half = half_spreads(cl)
    arr = cl.to_numpy(); idx = cl.index; cols = list(cl.columns)
    maxL = max(lookbacks)

    recs = []
    t = maxL + 1
    n = len(cl)
    while t < n - 1:
        d = pos[t]
        if d == 0.0:
            t += 1
            continue
        # 方向dが続く区間 [t, e) を特定
        e = t
        while e < n - 1 and pos[e] == d:
            e += 1
        if e - t >= min_hold:
            ts = idx[e]
            for j, p in enumerate(cols):
                fwd = arr[e, j] / arr[t, j] - 1.0
                pnl = (d * signs[j] * fwd - 2 * half[p]) * NOTIONAL
                recs.append((ts, pnl))
        t = e
    return pd.DataFrame(recs, columns=["exit", "pnl"])


# ── per-pair 戦略の月次PnL(value $10k/銘柄)──────────────────────────────
def perpair_monthly(strategy: str, params=None, side="both", instruments=None) -> pd.Series:
    return bl.strategy_monthly_pnl(strategy, params=params, side=side,
                                   instruments=instruments)


# ── xsec(クロスセクション平均回帰)──────────────────────────────────────
def xsec_monthly(tf="H4", lookback=9, hold=24, max_legs=4) -> pd.Series:
    import xsec_meanrev as xs
    uni.register_cross_spreads(3.0)
    close = xs.universe_close(tf)
    tr = xs.backtest(close, lookback=lookback, hold=hold, max_legs=max_legs)
    m = pd.PeriodIndex(pd.to_datetime(tr["exit"]).dt.to_period("M"))
    return tr.assign(m=m).groupby("m")["pnl"].sum()


def to_monthly(trades: pd.DataFrame) -> pd.Series:
    m = pd.PeriodIndex(pd.to_datetime(trades["exit"]).dt.to_period("M"))
    return trades.assign(m=m).groupby("m")["pnl"].sum()


def yearly_metrics(monthly: pd.Series) -> dict:
    """月次PnL→年次PF/プラス年率/総PnL/IS/OOS。"""
    yr = monthly.groupby([p.year for p in monthly.index]).sum()
    # PFは月次の正負で近似(トレード単位でないが頑健性の代理)
    pos_y = float((yr > 0).mean())
    total = float(monthly.sum())
    is_tot = float(monthly[monthly.index < OOS].sum())
    oos_tot = float(monthly[monthly.index >= OOS].sum())
    # 月次PFの中央値(年内)
    pf_by_year = {}
    for y, grp in monthly.groupby([p.year for p in monthly.index]):
        pos = grp[grp > 0].sum(); neg = -grp[grp < 0].sum()
        pf_by_year[y] = pos / neg if neg > 0 else np.nan
    pf_med = float(np.nanmedian(list(pf_by_year.values())))
    return dict(total=total, pos_year_rate=pos_y, pf_month_med=pf_med,
                is_total=is_tot, oos_total=oos_tot, n_pos_years=int((yr > 0).sum()),
                n_years=int(len(yr)), worst_year=float(yr.min()))


def conditional(monthly: pd.Series, mask: pd.Series) -> dict:
    s = monthly.reindex(mask.index).fillna(0.0)
    inb = s[mask.values]; out = s[~mask.values]
    is_m = mask & (mask.index < OOS); oos_m = mask & (mask.index >= OOS)
    return dict(mean_in_bleed=float(inb.mean()), mean_normal=float(out.mean()),
                total_in_bleed=float(inb.sum()),
                winrate_in_bleed=float((inb > 0).mean()),
                mean_bleed_IS=float(s[is_m.values].mean()) if is_m.sum() else np.nan,
                mean_bleed_OOS=float(s[oos_m.values].mean()) if oos_m.sum() else np.nan)


def corr_with(monthly: pd.Series, champ_monthly: pd.Series) -> float:
    a = monthly.reindex(champ_monthly.index).fillna(0.0)
    return float(np.corrcoef(a.values, champ_monthly.values)[0, 1])


def main():
    print("=== チャンピオン基準(失血窓 + 月次PnL)===")
    eqm, eqr, pool, _ = bl.champion_mtm(max_pos=8)
    mask, dd = bl.bleed_mask_monthly(eqm)
    cm = pool.copy()
    cm["m"] = pd.PeriodIndex(pd.to_datetime(cm["exit"]).dt.to_period("M"))
    champ_monthly = cm.groupby("m")["ret"].sum() * NOTIONAL
    champ_monthly = champ_monthly.reindex(mask.index).fillna(0.0)
    cy = yearly_metrics(champ_monthly)
    print(f"  champion total={cy['total']:.0f}  pos_yr={cy['pos_year_rate']:.0%}  "
          f"PFmed(月次)={cy['pf_month_med']:.2f}  失血窓={int(mask.sum())}ヶ月")

    # 候補定義
    cands = {}
    print("\n... 候補生成中 ...")
    # 1) USDバスケットTSMOM(D1, アンサンブル 1/3/6/12ヶ月)
    cands["usdbasket_D1_ens"] = to_monthly(usd_basket_tsmom("D1", [21, 63, 126, 252]))
    # 2) USDバスケットTSMOM(H4, アンサンブル ~1/2/4/8週)
    cands["usdbasket_H4_ens"] = to_monthly(usd_basket_tsmom("H4", [60, 120, 240, 480]))
    # 3) USDバスケットTSMOM(D1, 単一 3ヶ月)— アンサンブル効果の対照
    cands["usdbasket_D1_63"] = to_monthly(usd_basket_tsmom("D1", [63]))
    # 4) per-pair tsmom(D1, lookback100)
    cands["perpair_tsmom_D1"] = perpair_monthly("tsmom", params={"lookback": 100, "band": 0.0})
    # 5) per-pair trend_insurance(H4, 既定)
    cands["trend_insurance_H4"] = perpair_monthly("trend_insurance")
    # 6) xsec クロスセクションMR
    cands["xsec_H4"] = xsec_monthly()

    rows = []
    for name, mser in cands.items():
        if mser is None or len(mser) == 0:
            print(f"  [skip] {name}: empty")
            continue
        mser = mser.reindex(mask.index).fillna(0.0)
        ym = yearly_metrics(mser)
        cd = conditional(mser, mask)
        cr = corr_with(mser, champ_monthly)
        rows.append({
            "candidate": name,
            "total": round(ym["total"], 0),
            "pos_yr": f"{ym['pos_year_rate']:.0%}",
            "PFmed": round(ym["pf_month_med"], 2),
            "IS": round(ym["is_total"], 0),
            "OOS": round(ym["oos_total"], 0),
            "worst_yr": round(ym["worst_year"], 0),
            "corr_champ": round(cr, 3),
            "bleed_mean": round(cd["mean_in_bleed"], 1),
            "normal_mean": round(cd["mean_normal"], 1),
            "bleed_tot": round(cd["total_in_bleed"], 0),
            "bleed_win%": f"{cd['winrate_in_bleed']:.0%}",
            "bleed_IS": round(cd["mean_bleed_IS"], 1),
            "bleed_OOS": round(cd["mean_bleed_OOS"], 1),
        })
    df = pd.DataFrame(rows).set_index("candidate")
    print("\n=== 補完候補スカウト(各脚$10k同尺; bleed=失血窓での月次平均PnL)===")
    print(df.to_string())
    print("\n判定軸: ①単体ロバスト(pos_yr高/PFmed>1/worst_yr浅/IS・OOS両プラス)")
    print("        ②直交(corr_champ低/負)  ③失血窓で稼ぐ(bleed_mean>0 かつ IS/OOS両方)")
    print("理想の補完 = ①②③を同時に満たす。チャンピオンの失血窓は2021-22のUSDラリー(高ER)。")


if __name__ == "__main__":
    main()
