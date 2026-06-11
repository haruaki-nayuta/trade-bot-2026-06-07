"""敵対検証8: BIDアーティファクト署名「ロング20時エントリーのみ・ショート20時イグジット無し」の独立再計算。

主張:
- 1本目グロスリターン(dir*(次H4 close/エントリーバーclose - 1)): 20時エントリー +3.72bps vs 他 -0.76bps (MWU p=0.0057)
- ロング限定: +6.97 vs -1.83bps (Welch p=0.0245)
- ショート限定: 差なし (p=0.79)
- ショート20時イグジットの平均ret +12.4bps は全ショート +17.1bps より低い(異常優位なし)
"""
import numpy as np
import pandas as pd
from scipy import stats

from fxlab import universe as uni

POOL = "results/mm_pool_v2_H4_19.parquet"


def main():
    df = pd.read_parquet(POOL)

    # --- ベースライン検算 ---
    print(f"baseline: n={len(df)} sum_ret={df['ret'].sum():+.4f} "
          f"win={(df['ret'] > 0).mean():.3f}")
    assert len(df) == 1214
    assert abs(df["ret"].sum() - 1.9086) < 1e-3

    # --- 1本目グロスリターンの計算 ---
    uni.register_cross_spreads(3.0)
    closes = {}
    for instr in df["instr"].unique():
        closes[instr] = uni.instrument_close(instr, "H4")

    gross1 = np.full(len(df), np.nan)
    miss = 0
    for k, (instr, entry, d) in enumerate(zip(df["instr"], df["entry"], df["dir"])):
        c = closes[instr]
        try:
            i = c.index.get_loc(entry)
        except KeyError:
            miss += 1
            continue
        if i + 1 >= len(c):
            miss += 1
            continue
        gross1[k] = d * (c.iloc[i + 1] / c.iloc[i] - 1.0)
    df = df.assign(gross1_bps=gross1 * 1e4)
    print(f"gross1: computed={np.isfinite(gross1).sum()} missing={miss}")

    hr = df["entry"].dt.hour
    is20 = hr == 20
    g = df["gross1_bps"]
    ok = g.notna()

    def cmp(mask, label):
        a = g[ok & mask & is20]
        b = g[ok & mask & ~is20]
        mwu = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        wel = stats.ttest_ind(a, b, equal_var=False).pvalue
        print(f"{label}: n20={len(a)} mean20={a.mean():+.2f}bps | "
              f"n_oth={len(b)} mean_oth={b.mean():+.2f}bps | "
              f"MWU p={mwu:.4f} Welch p={wel:.4f}")
        return a.mean(), b.mean(), mwu, wel

    print("\n--- 1本目グロス: 20時エントリー vs 他 ---")
    cmp(pd.Series(True, index=df.index), "ALL  ")
    cmp(df["dir"] == 1, "LONG ")
    cmp(df["dir"] == -1, "SHORT")

    # --- ショート 20時イグジット ---
    print("\n--- ショートの20時イグジット(実現ret, bps) ---")
    sh = df[df["dir"] == -1]
    ex20 = sh[sh["exit"].dt.hour == 20]
    print(f"short exit@20: n={len(ex20)} mean_ret={ex20['ret'].mean() * 1e4:+.1f}bps")
    print(f"all shorts   : n={len(sh)} mean_ret={sh['ret'].mean() * 1e4:+.1f}bps")
    mwu = stats.mannwhitneyu(
        ex20["ret"], sh[sh["exit"].dt.hour != 20]["ret"], alternative="two-sided"
    ).pvalue
    print(f"short exit@20 vs other-exit shorts: MWU p={mwu:.3f}")

    # 参考: ロング20時エントリーの件数規模(総純益への寄与感)
    l20 = df[(df["dir"] == 1) & is20]
    print(f"\n(ref) long entry@20: n={len(l20)} sum_ret={l20['ret'].sum():+.4f} "
          f"(= {l20['ret'].sum() / 1.9086 * 100:.1f}% of total)")


if __name__ == "__main__":
    main()
