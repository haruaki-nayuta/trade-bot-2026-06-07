"""二スリーブ・ポートフォリオ評価: champion + xsec を「独立資本スリーブ」で合成し、足を引っ張るか判定。

reports/29(記憶)は xsec を **共有DD口座**(champion とDD予算を奪い合う)で評価し純減と結論。
本実験は **独立スリーブ**レンズ(ユーザーの「ポートフォリオ化」に対応):
  各スリーブを単独で 20% MtM DD に較正 → 重み w でリスクパリティ合成 →
  ① 合成Sharpe/最悪年/プラス年/DD が champion 単独より良いか
  ② 合成を 20%DD にレバ調整した CAGR が champion 単独を超えるか(DD厳格レンズの再確認)
  ③ 2022除外/IS-OOS/年別で「足を引っ張る」兆候(チャンピオンの負け年に xsec も負ける等)

判定: ①が改善 & ③でドラッグ無し → 「相補的に底上げ・足を引っ張らない」を満たす。
       ②が同等以下でも、ユーザー目的は CAGR最大化でなく分散なら ① が本筋。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from fxlab import config, universe as uni
import mm_lab as mm

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 50)

TF = "H4"
BPY = 6 * 252


# ── xsec をトレードプール化(mm_lab.simulate にかけられる形)─────────────
def build_xsec_pool(tf=TF, lookback=9, hold=24, max_legs=4, vol_win=50) -> pd.DataFrame:
    import xsec_meanrev as xs
    uni.register_cross_spreads(3.0)
    close = xs.universe_close(tf)
    names = list(close.columns)
    mom = close.pct_change(lookback)
    vol = close.pct_change().rolling(vol_win).std()
    mp = close.mean()
    hs = {p: config.spread_pips(p) * config.pip_size(p) / 2.0 / mp[p] for p in names}
    rows = []
    for t in range(max(lookback, vol_win) + 1, len(close) - hold, hold):
        score = mom.iloc[t] / vol.iloc[t]
        if score.isna().any():
            continue
        score = score - score.mean()
        s = score.sort_values()
        longs = s[s < 0].index[:max_legs]
        shorts = s[s > 0].index[-max_legs:]
        ets, xts = close.index[t], close.index[t + hold]
        for p in longs:
            fwd = close[p].iloc[t + hold] / close[p].iloc[t] - 1.0
            rows.append((p, ets, xts, 1, float(close[p].iloc[t]), float(fwd - 2 * hs[p]), hold))
        for p in shorts:
            fwd = close[p].iloc[t + hold] / close[p].iloc[t] - 1.0
            rows.append((p, ets, xts, -1, float(close[p].iloc[t]), float(-fwd - 2 * hs[p]), hold))
    pool = pd.DataFrame(rows, columns=["instr", "entry", "exit", "dir", "entry_price", "ret", "bars_held"])
    pool["z_entry"] = 1.0
    pool["vol_entry"] = 0.01
    return pool.sort_values("entry").reset_index(drop=True)


def _eq_cagr(rets, L):
    eq = (1.0 + L * rets).cumprod()
    years = (rets.index[-1] - rets.index[0]).days / 365.25
    return eq, eq.iloc[-1] ** (1 / years) - 1.0


def lever_to_dd(rets: pd.Series, target_dd=0.20, lo=0.05, hi=20.0, iters=40, robust=False):
    """バーリターン列を定数倍 L して DD==target に揃える L と CAGR。

    robust=False: 経験的(単一パス)最大DD を target に。
    robust=True : ブロックブートストラップ p95(理論DD)を target に=レバ偽装に強い厳格較正。
    """
    def dd_of(L):
        eq = (1.0 + L * rets).cumprod()
        if robust:
            return abs(mm.bootstrap_maxdd(eq, n_boot=400, block=63)["p95"])
        return abs(float((eq / eq.cummax() - 1.0).min()))
    if dd_of(hi) <= target_dd:
        L = hi
    else:
        for _ in range(iters):
            mid = (lo + hi) / 2
            if dd_of(mid) > target_dd:
                hi = mid
            else:
                lo = mid
        L = lo
    _, cagr = _eq_cagr(rets, L)
    return L, cagr


def p95_at_lever(rets: pd.Series, L: float) -> float:
    eq = (1.0 + L * rets).cumprod()
    return abs(mm.bootstrap_maxdd(eq, n_boot=400, block=63)["p95"])


def series_stats(rets: pd.Series, label=""):
    eq = (1.0 + rets).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    sharpe = rets.mean() / rets.std() * np.sqrt(BPY) if rets.std() > 0 else np.nan
    yr = eq.groupby(eq.index.year).last().pct_change()
    yr.iloc[0] = eq.groupby(eq.index.year).last().iloc[0] - 1.0
    return dict(label=label, maxdd=dd, sharpe=float(sharpe),
                pos_yr=float((yr > 0).mean()), worst_yr=float(yr.min()), yr=yr)


def main():
    print("=== スリーブ構築(各々 20% 経験的MtM DD に較正)===")
    closes = mm.load_closes(TF)

    # champion sleeve
    from mm_production import champion_sizing
    pool_c = mm.build_pool()
    mk_c = champion_sizing(pool_c, max_pos=8)
    kc, eqm_c, eqr_c, info_c = mm.calibrate(pool_c, closes, mk_c, target_dd=0.20, max_pos=8)
    sc = mm.stats(eqm_c, eqr_c, info_c)
    print(f"  champion: k={kc:.2f} CAGR={sc['cagr']:.1%} DD={sc['maxdd_mtm']:.1%} "
          f"Sharpe={sc['sharpe']:.2f} pos_yr={sc['pos_year_rate']:.0%} worst_yr={sc['worst_year']:.1%}")

    # xsec sleeve
    pool_x = build_xsec_pool()
    def mk_x(k):
        base = k / 8
        return lambda ctx: ctx["equity_real"] * base
    kx, eqm_x, eqr_x, info_x = mm.calibrate(pool_x, closes, mk_x, target_dd=0.20, max_pos=8)
    sx = mm.stats(eqm_x, eqr_x, info_x)
    print(f"  xsec:     k={kx:.2f} CAGR={sx['cagr']:.1%} DD={sx['maxdd_mtm']:.1%} "
          f"Sharpe={sx['sharpe']:.2f} pos_yr={sx['pos_year_rate']:.0%} worst_yr={sx['worst_year']:.1%}")

    # 共通グリッドのバーリターン
    idx = eqm_c.index.intersection(eqm_x.index)
    r_c = eqm_c.reindex(idx).pct_change().fillna(0.0)
    r_x = eqm_x.reindex(idx).pct_change().fillna(0.0)
    corr = float(np.corrcoef(r_c.values, r_x.values)[0, 1])
    # 月次相関も(バー相関は0に張り付きやすい)
    mc = eqm_c.reindex(idx).resample("ME").last().pct_change().dropna()
    mx = eqm_x.reindex(idx).resample("ME").last().pct_change().dropna()
    j = mc.index.intersection(mx.index)
    mcorr = float(np.corrcoef(mc.reindex(j), mx.reindex(j))[0, 1])
    print(f"\n  バー相関={corr:.3f}  月次相関={mcorr:.3f}")

    print("\n=== 重み w スイープ: 経験的 vs robust(p95)較正 + レバ偽装チェック ===")
    print("empCAGR=経験的20%DDにレバ調整したCAGR / robCAGR=bootp95=20%に較正したCAGR")
    print("p95@empL=経験的レバ時のbootstrap p95(champ比で悪化=レバ偽装署名)\n")
    rows = []
    Lc_emp, cagr_c_emp = lever_to_dd(r_c, 0.20, robust=False)
    Lc_rob, cagr_c_rob = lever_to_dd(r_c, 0.20, robust=True)
    p95_c = p95_at_lever(r_c, Lc_emp)
    for w in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3):
        rp = r_c if w == 0 else (1 - w) * r_c + w * r_x
        Le, cge = lever_to_dd(rp, 0.20, robust=False)
        Lr, cgr = lever_to_dd(rp, 0.20, robust=True)
        st = series_stats(rp)
        rows.append(dict(w=w, empCAGR=cge, robCAGR=cgr, sharpe=st["sharpe"],
                         emp_maxdd=st["maxdd"], p95_at_empL=p95_at_lever(rp, Le),
                         worst_yr=st["worst_yr"], pos_yr=st["pos_yr"]))
    df = pd.DataFrame(rows)
    df["Δemp"] = df["empCAGR"] - cagr_c_emp
    df["Δrob"] = df["robCAGR"] - cagr_c_rob
    print(df.round({"w": 2, "empCAGR": 4, "robCAGR": 4, "sharpe": 3, "emp_maxdd": 3,
                    "p95_at_empL": 3, "worst_yr": 3, "pos_yr": 2, "Δemp": 4, "Δrob": 4}).to_string(index=False))
    print(f"\n  champion p95@empL={p95_c:.3f}。合成のp95@empLがこれ以下&robCAGRも純増ならレバ偽装でない。")

    # ── 年別の足引っ張りチェック(w=0.2 固定スリーブの実額配分でなく、リスクパリティ年次寄与)──
    print("\n=== 年次リターン(champion / xsec / 合成 w=0.2)— 同符号で沈む年があるか ===")
    yc = (1 + r_c).cumprod().groupby(idx.year).last().pct_change()
    yc.iloc[0] = (1 + r_c).cumprod().groupby(idx.year).last().iloc[0] - 1
    yx = (1 + r_x).cumprod().groupby(idx.year).last().pct_change()
    yx.iloc[0] = (1 + r_x).cumprod().groupby(idx.year).last().iloc[0] - 1
    rp2 = 0.8 * r_c + 0.2 * r_x
    yp = (1 + rp2).cumprod().groupby(idx.year).last().pct_change()
    yp.iloc[0] = (1 + rp2).cumprod().groupby(idx.year).last().iloc[0] - 1
    ytab = pd.DataFrame({"champ": yc, "xsec": yx, "combo_w20": yp}).round(3)
    print(ytab.to_string())
    bad_c = yc[yc < 0].index.tolist()
    print(f"\n  championの負け年: {bad_c}")
    for y in bad_c:
        print(f"    {y}: champ={yc[y]:+.1%}  xsec={yx.get(y, float('nan')):+.1%}  "
              f"→ xsecは{'助けた' if yx.get(y, 0) > 0 else '一緒に沈んだ'}")

    # ── 2022除外でSharpe改善が残るか ──
    print("\n=== 2022除外でも分散効果(Sharpe)が残るか ===")
    m2 = idx.year != 2022
    r_c2, r_x2 = r_c[m2], r_x[m2]
    s_champ_ex = series_stats(r_c2)["sharpe"]
    s_combo_ex = series_stats(0.8 * r_c2 + 0.2 * r_x2)["sharpe"]
    print(f"  champion Sharpe(2022除外)={s_champ_ex:.3f}  combo_w20 Sharpe(2022除外)={s_combo_ex:.3f}  "
          f"Δ={s_combo_ex - s_champ_ex:+.3f}")


if __name__ == "__main__":
    main()
