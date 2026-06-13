"""exp59: マルチ時間足アンサンブル・スカウト — 唯一未検証のレバーの生死判定。

仮説: チャンピオン(confluence_meanrev_v2_d1, 平均回帰)を H4 だけでなく D1(同一バーパラメータ)
でも回し、両方のトレードを1つの DD較正口座に統合すると、時間軸の分散で「同一 p95=20% でより高い k」
が引けて CAGR が上がるのではないか。これは reports/10 のオーバーレイ(net負の弱いトレンド戦略を
混ぜる=保険)とは別物で、「実証済みの正のエッジを別時間軸で足す=加法的分散」。

判定の物差し(reports/19 のプロトコル):
  ・ベースライン = H4-only d1 pool, P=4.0, mp8, robust(ブートp95 DD=20%)較正の CAGR。
  ・統合が勝つ条件 = 同一 p95=20% 契約で CAGR がベースラインを +10% 相対(≒ +2pp)以上上回る。
  ・レバ偽装チェック(empirical CAGR↑ かつ p95 悪化 でないか)・単年依存は後段で。

本スカウトは「生死」を見るだけ:
  (1) H4/D1/H1 単独の robust CAGR・トレード数・プラス年率・標準パスの p95。
  (2) D1 を H4 グリッドで回したときと D1 グリッドで回したときの整合(グリッド近似の健全性チェック)。
  (3) 月次リターンの H4↔D1 相関(分散の素地があるか)。
  (4) 統合 H4+D1(mp8 / mp12 / mp16)の robust CAGR。
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
from mm_production import build_pool_d1, champion_sizing  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

SEEDS = [0, 1, 2]
N_BOOT_CAL = 600


def robust_cagr(pool, closes, max_pos, tf_for_stats="H4", target=0.20, seeds=SEEDS):
    """robust 較正(p95 DD=target)を複数シードで回し CAGR / p95 / empirical DD を返す。"""
    out = []
    for sd in seeds:
        mk = champion_sizing(pool, max_pos=max_pos)
        # calibrate_robust は内部で seed 固定の bootstrap を使う(mm_lab は seed=0 固定)。
        # シード差を出すため、ここでは calibrate_robust を呼んだ後に標準パスを別シードで再ブートして p95 を確認。
        k, eqm, eqr, info, p95 = mm.calibrate_robust(
            pool, closes, mk, target_dd=target, max_pos=max_pos, n_boot=N_BOOT_CAL)
        s = mm.stats(eqm, eqr, info, tf=tf_for_stats)
        bs = mm.bootstrap_maxdd(eqm, n_boot=1500, seed=sd)
        out.append({"seed": sd, "k": k, "cagr": s["cagr"], "p95": bs["p95"],
                    "emp_dd": s["maxdd_mtm"], "pos_year": s["pos_year_rate"],
                    "worst_year": s["worst_year"], "sharpe": s["sharpe"],
                    "n_taken": s["n_taken"], "max_conc": s["max_conc"]})
    df = pd.DataFrame(out)
    return df


def summarize(name, df):
    print(f"\n  [{name}]  (seeds {list(df['seed'])})")
    print(f"    k            : {df['k'].mean():.2f}  (range {df['k'].min():.2f}-{df['k'].max():.2f})")
    print(f"    robust CAGR  : {df['cagr'].mean():+.2%}  (range {df['cagr'].min():+.2%}..{df['cagr'].max():+.2%})")
    print(f"    boot p95 DD  : {df['p95'].mean():+.1%}   empirical DD: {df['emp_dd'].mean():+.1%}")
    print(f"    pos-year     : {df['pos_year'].mean():.0%}   worst-year: {df['worst_year'].mean():+.1%}   Sharpe: {df['sharpe'].mean():.2f}")
    print(f"    n_taken      : {df['n_taken'].mean():.0f}   max_conc: {df['max_conc'].mean():.0f}")
    return df["cagr"].mean()


def main():
    print("=" * 70)
    print("  exp59: マルチ時間足アンサンブル・スカウト")
    print("=" * 70)

    # --- プール構築(同一チャンピオン d1, 各時間足) ---
    print("\n[1] プール構築 ...")
    pool_h4 = build_pool_d1(tf="H4")
    pool_d1 = build_pool_d1(tf="D1")
    pool_h1 = build_pool_d1(tf="H1")
    print(f"    H4: {len(pool_h4)} trades (年{len(pool_h4)/11:.0f})")
    print(f"    D1: {len(pool_d1)} trades (年{len(pool_d1)/11:.0f})")
    print(f"    H1: {len(pool_h1)} trades (年{len(pool_h1)/11:.0f})")

    closes_h4 = mm.load_closes(tf="H4")
    closes_d1 = mm.load_closes(tf="D1")
    closes_h1 = mm.load_closes(tf="H1")

    # --- 単独 robust CAGR ---
    print("\n[2] 単独 robust(p95 DD=20%)較正 ...")
    base_h4 = robust_cagr(pool_h4, closes_h4, max_pos=8, tf_for_stats="H4")
    base = summarize("H4-only mp8 (=ベースライン)", base_h4)

    d1_solo = robust_cagr(pool_d1, closes_d1, max_pos=8, tf_for_stats="D1")
    summarize("D1-only mp8", d1_solo)

    h1_solo = robust_cagr(pool_h1, closes_h1, max_pos=8, tf_for_stats="H1")
    summarize("H1-only mp8", h1_solo)

    # --- グリッド近似チェック: D1 pool を H4 グリッドで回す ---
    print("\n[3] グリッド近似チェック(D1 pool を H4 グリッドで MtM)...")
    d1_on_h4grid = robust_cagr(pool_d1, closes_h4, max_pos=8, tf_for_stats="H4")
    summarize("D1-pool on H4-grid mp8", d1_on_h4grid)
    print("    ↑ D1-only(D1グリッド)と CAGR が大きくずれなければグリッド統合は健全")

    # --- 月次相関(分散の素地) ---
    print("\n[4] H4↔D1 月次リターン相関 ...")
    mk_h4 = champion_sizing(pool_h4, max_pos=8)
    _, eqr_h4, _ = mm.simulate(pool_h4, closes_h4, mk_h4(1.0), max_pos=8)
    mk_d1 = champion_sizing(pool_d1, max_pos=8)
    _, eqr_d1, _ = mm.simulate(pool_d1, closes_d1, mk_d1(1.0), max_pos=8)
    m_h4 = eqr_h4.resample("ME").last().pct_change().dropna()
    m_d1 = eqr_d1.resample("ME").last().pct_change().dropna()
    j = pd.concat([m_h4.rename("h4"), m_d1.rename("d1")], axis=1).dropna()
    corr = j["h4"].corr(j["d1"])
    print(f"    月次相関 corr(H4, D1) = {corr:+.3f}   (n={len(j)}ヶ月)")
    print(f"    → 低いほど統合の分散メリット大。diversification gain目安 sqrt(2/(1+corr))={np.sqrt(2/(1+corr)):.3f}x Sharpe")

    # --- 統合 H4+D1(H4 グリッド) ---
    print("\n[5] 統合 H4+D1(H4 グリッドで MtM, robust p95=20%)...")
    pool_mix = pd.concat([pool_h4, pool_d1], ignore_index=True).sort_values("entry").reset_index(drop=True)
    print(f"    統合 trades: {len(pool_mix)}")
    for mp in [8, 12, 16]:
        mix = robust_cagr(pool_mix, closes_h4, max_pos=mp, tf_for_stats="H4")
        c = summarize(f"H4+D1 mp{mp}", mix)
        print(f"      → ベースライン(+{base:+.2%})比: {c - base:+.2f}pp  ({(c/base-1)*100:+.1f}% 相対)")

    print("\n" + "=" * 70)
    print(f"  ベースライン H4-only mp8 robust = {base:+.2%}")
    print("  判定: 統合がベースを +2pp(+10%相対)以上、かつ p95 非悪化なら次段(敵対検証)へ")
    print("=" * 70)


if __name__ == "__main__":
    main()
