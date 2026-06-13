"""exp64: ドル中立 残差平均回帰(Avellaneda-Lee 型 統計裁定)— 失血窓を守る分散候補の検証。

動機(web リサーチ + 自己分析):
  チャンピオンの失血窓は「持続USDトレンドで複数ペアの USDショートが一斉に含み損」(reports/15)。
  D1/xsec が失血窓を守れなかった([[29]])のは、結局ドル方向の素抜けを持つから。
  Avellaneda-Lee 型の**残差平均回帰**は設計上ドル中立(共通ドルファクターを除去した
  残差だけを売買)なので、原理的に**ドルトレンドでは沈まない**=失血窓で稼ぐ可能性がある。
  xsec はその粗い近似(固定hold・betaなし・OU s-score なし)。本実験は proper 版を作る。

機構:
  ① 7 USD メジャーを「外貨の対USD価値」の対数価格に統一(USDJPY/CHF/CAD は反転)。
  ② バーごとに等加重ドルファクター m_t = mean_i(r_i)。残差 e_i = r_i - beta_i * m_t
     (beta_i は窓 W のローリング回帰。等加重版 beta=1 も比較)。
  ③ 残差の累積 = OU 過程。s-score_i = zscore(cumsum e_i, 窓 W)。
  ④ |s|>entry で残差をフェード(s高→short, s低→long)、|s|<exit で手仕舞い。
     ポジションは残差に対して建てるので Σ ≈ 0 = 自動的にドル中立。
  ⑤ バー損益 = p_i * e_i - コスト(ポジション変化時に半スプレッド)。
     残差 PnL を直接集計 → equity。Σe=0 ゆえドル素抜けなし(設計どおり)。

判定:
  ・単独 Sharpe がチャンピオン(1.37)に対抗できるか / プラス年率。
  ・**失血窓テスト**: チャンピオン最悪十分位月で本戦略はプラスか(D1=0/xsec=-0.24% だった)。
  ・相関。高原性(W/entry/exit スイープ)。
  → ここで「強い単独 or 失血窓プラス」が出たら exp65 で DD口座統合へ。さもなくば閉鎖。
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))

import mm_lab as mm  # noqa: E402
from mm_production import Z0, P, CLIP_LO, CLIP_HI, build_pool_d1  # noqa: E402
from fxlab import config, universe as uni  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

# 7 USD メジャー。invert=True は「USD建て(USDXXX)」=外貨対USD価値に反転が必要。
MAJORS = {"EURUSD": False, "GBPUSD": False, "AUDUSD": False, "NZDUSD": False,
          "USDJPY": True, "USDCHF": True, "USDCAD": True}
BARS_PER_YEAR = 6 * 252


def foreign_in_usd_logprice(tf="H4"):
    """各通貨の『対USD価値』対数価格(共通ドル動意が同符号で効く形)。"""
    cols = {}
    halfspread = {}
    for pair, invert in MAJORS.items():
        px = uni.instrument_close(pair, tf)
        lp = -np.log(px) if invert else np.log(px)
        cols[pair] = lp
        # 半スプレッド(価格比)。対数残差PnLに対する往復コストの近似。
        halfspread[pair] = config.spread_pips(pair) * config.pip_size(pair) / 2.0 / px.mean()
    df = pd.DataFrame(cols).dropna()
    return df, halfspread


def statarb_equity(logp, halfspread, *, W=60, entry=1.25, exit_=0.5, use_beta=True,
                   init=10_000.0, gross=8.0):
    """残差平均回帰の equity(バー駆動・複利・ドル中立)。

    gross = 常時の総建玉/資産(脚数で割って1脚配分)。脚数は最大7。
    """
    r = logp.diff()
    m = r.mean(axis=1)  # 等加重ドルファクター(リターン)
    names = list(logp.columns)
    # 残差 e_i
    e = pd.DataFrame(index=logp.index, columns=names, dtype=float)
    if use_beta:
        for nm in names:
            cov = r[nm].rolling(W).cov(m)
            var = m.rolling(W).var()
            beta = (cov / var).clip(-3, 3)
            e[nm] = r[nm] - beta * m
    else:
        e = r.sub(m, axis=0)
    # s-score: cumsum 残差の z(窓W)
    cum = e.cumsum()
    s = (cum - cum.rolling(W).mean()) / cum.rolling(W).std()

    pos = pd.DataFrame(0.0, index=logp.index, columns=names)
    prev = {nm: 0.0 for nm in names}
    idx = logp.index
    for t in range(W + 1, len(idx)):
        for nm in names:
            sv = s[nm].iat[t]
            p_ = prev[nm]
            if not np.isfinite(sv):
                newp = 0.0
            elif p_ == 0.0:
                newp = -1.0 if sv > entry else (1.0 if sv < -entry else 0.0)
            else:  # 建玉中: |s|<exit で手仕舞い、さもなくば維持
                newp = 0.0 if abs(sv) < exit_ else p_
            pos[nm].iat[t] = newp
            prev[nm] = newp

    # バー損益: p_i(前バーのポジ)× e_i(当バー残差リターン) − コスト(ポジ変化)
    pos_lag = pos.shift(1).fillna(0.0)
    pnl_gross = (pos_lag * e).sum(axis=1)
    dpos = pos.diff().abs().fillna(0.0)
    cost = sum(dpos[nm] * halfspread[nm] for nm in names)
    n_legs = pos.abs().sum(axis=1).replace(0, np.nan)
    # 1脚あたり配分 = gross/最大脚数(7)。総建玉は脚数に比例(常時フルでない)
    leg_alloc = gross / 7.0
    bar_ret = leg_alloc * (pnl_gross - cost)  # equity に対する比率
    eq = (1.0 + bar_ret.fillna(0.0)).cumprod() * init
    return eq, pos


def stats_of(eq, tf="H4"):
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    dd = float((eq / eq.cummax() - 1).min())
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(BARS_PER_YEAR) if rets.std() > 0 else float("nan")
    yr = eq.groupby(eq.index.year).last().pct_change()
    yr.iloc[0] = eq.groupby(eq.index.year).last().iloc[0] / eq.iloc[0] - 1
    return {"cagr": cagr, "maxdd": dd, "sharpe": sharpe,
            "pos_year": float((yr > 0).mean()), "worst_year": float(yr.min())}


def champion_monthly(tf="H4"):
    pool = build_pool_d1(tf="H4")
    closes = mm.load_closes(tf="H4")
    fbar = float(np.mean([np.clip((z / Z0) ** P, CLIP_LO, CLIP_HI) for z in pool["z_entry"] if np.isfinite(z)]))
    sz = lambda c: c["equity_real"] * (1 / 8) * (np.clip((c["z"] / Z0) ** P, CLIP_LO, CLIP_HI) / fbar if np.isfinite(c["z"]) else 1.0 / fbar)
    eqm, _, _ = mm.simulate(pool, closes, sz, max_pos=8)
    return eqm.resample("ME").last().pct_change().dropna()


def main():
    print("=" * 72)
    print("  exp64: ドル中立 残差平均回帰(Avellaneda-Lee型 統計裁定)— 失血窓分散候補")
    print("=" * 72)
    uni.register_cross_spreads(3.0)
    logp, hs = foreign_in_usd_logprice("H4")
    print(f"  7 USD majors / bars {len(logp)} / {logp.index[0].date()}..{logp.index[-1].date()}")

    mh4 = champion_monthly()

    print("\n[①] パラメータ高原スキャン(robustでなく素のSharpe/CAGR・gross=8で同尺)")
    print(f"  {'W':>4} {'entry':>6} {'exit':>5} {'beta':>5} {'CAGR':>8} {'Sharpe':>7} {'maxDD':>8} {'posY':>6} {'corr':>7} {'bleed':>8}")
    best = None
    for W in [40, 60, 90]:
        for entry in [1.0, 1.25, 1.5]:
            for use_beta in [True, False]:
                eq, pos = statarb_equity(logp, hs, W=W, entry=entry, exit_=0.5, use_beta=use_beta)
                st = stats_of(eq)
                mxs = eq.resample("ME").last().pct_change().dropna()
                j = pd.concat([mh4.rename("h4"), mxs.rename("xs")], axis=1).dropna()
                corr = j["h4"].corr(j["xs"])
                thr = j["h4"].quantile(0.10)
                bleed = j[j["h4"] <= thr]["xs"].mean()
                tagb = "Y" if use_beta else "N"
                print(f"  {W:>4} {entry:>6.2f} {0.5:>5.2f} {tagb:>5} {st['cagr']:>+7.2%} "
                      f"{st['sharpe']:>7.2f} {st['maxdd']:>+7.1%} {st['pos_year']:>5.0%} "
                      f"{corr:>+7.3f} {bleed:>+7.3%}")
                cand = (st["sharpe"], W, entry, use_beta, st, corr, bleed)
                if best is None or cand[0] > best[0]:
                    best = cand

    print("\n" + "-" * 72)
    sh, W, entry, ub, st, corr, bleed = best
    print(f"  最良(Sharpe基準): W={W} entry={entry} beta={ub} → Sharpe {sh:.2f} / CAGR {st['cagr']:+.2%} / "
          f"prosY {st['pos_year']:.0%}")
    print(f"  チャンピオン相関 {corr:+.3f} / 失血窓十分位の平均 {bleed:+.3%}")
    print("\n  判定: 単独Sharpeがチャンピオン1.37に迫る or 失血窓で明確プラス → exp65でDD統合。")
    print("        弱い&失血窓非プラスなら xsec/D1 と同じ閉鎖。")
    print("=" * 72)


if __name__ == "__main__":
    main()
