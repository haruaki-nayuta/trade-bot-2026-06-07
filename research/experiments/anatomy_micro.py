"""anatomy_micro.py — チャンピオン confluence_meanrev_v2 トレード内部のミクロ構造解剖。

各トレードの entry〜exit close経路から path = dir*(close/entry_close - 1) を構築し、
MAE/MFE・time-to・z勾配・保有経過期待値・取りこぼし・サイジング妥当性を全件実測する。

実行: uv run python -m research.experiments.anatomy_micro
読み取り専用。ファイルは書かない(標準出力のみ)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from fxlab import universe as uni

POOL = "results/mm_pool_v2_H4_19.parquet"
BASE_SUM = 1.9086  # 検算ベースライン


# ---------------------------------------------------------------- helpers
def _zscore(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def _efficiency_ratio(close: pd.Series, w: int) -> pd.Series:
    direction = (close - close.shift(w)).abs()
    volatility = close.diff().abs().rolling(w).sum()
    return (direction / volatility).replace([np.inf, -np.inf], np.nan)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score_pos > score_neg)。label=1が陽性。"""
    score = np.asarray(score, float)
    label = np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = sps.rankdata(np.concatenate([pos, neg]))
    rp = r[: len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def dist_stats(x: np.ndarray, w: np.ndarray | None = None) -> dict:
    x = np.asarray(x, float)
    if w is None:
        mean, std = x.mean(), x.std(ddof=1)
        sk, ku = sps.skew(x), sps.kurtosis(x)  # 超過尖度
        q = np.percentile(x, [1, 5, 50, 95, 99])
        mn, mx = x.min(), x.max()
    else:
        w = np.asarray(w, float) / np.sum(w)
        mean = np.sum(w * x)
        var = np.sum(w * (x - mean) ** 2)
        std = np.sqrt(var)
        sk = np.sum(w * (x - mean) ** 3) / std**3
        ku = np.sum(w * (x - mean) ** 4) / std**4 - 3
        order = np.argsort(x)
        cw = np.cumsum(w[order])
        q = np.interp([0.01, 0.05, 0.50, 0.95, 0.99], cw, x[order])
        mn, mx = x.min(), x.max()
    return dict(mean=mean, std=std, skew=sk, kurt=ku,
                p1=q[0], p5=q[1], p50=q[2], p95=q[3], p99=q[4], min=mn, max=mx)


def f_z(z: np.ndarray) -> np.ndarray:
    """本番 z-power サイジング: clip((|z|/2.2)^4.0, 0.3, 3.0)"""
    return np.clip((np.abs(z) / 2.2) ** 4.0, 0.3, 3.0)


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

    uni.register_cross_spreads(3.0)
    pool = pd.read_parquet(POOL)
    instrs = sorted(pool["instr"].unique())

    # ---- 価格系列とトレード時点参照用の指標(戦略仕様と同定義) ----
    closes: dict[str, pd.Series] = {}
    ind: dict[str, pd.DataFrame] = {}
    for it in instrs:
        c = uni.instrument_close(it, "H4")
        closes[it] = c
        ind[it] = pd.DataFrame({
            "z50": _zscore(c, 50),
            "z250": _zscore(c, 250),
            "er40": _efficiency_ratio(c, 40),
        })

    # ---- 全トレードの経路構築 ----
    rec = []          # per-trade micro features
    paths = []        # np.ndarray path (k=0..bars_held)
    mismatch = 0
    for t in pool.itertuples():
        c = closes[t.instr]
        i0 = c.index.get_loc(t.entry)
        i1 = c.index.get_loc(t.exit)
        seg = c.iloc[i0: i1 + 1].to_numpy(float)
        nb = i1 - i0
        if nb != t.bars_held:
            mismatch += 1
        path = t.dir * (seg / seg[0] - 1.0)
        paths.append(path)
        body = path[1:]  # エントリーバー(=0)を除く走行
        mae_raw = body.min() if len(body) else 0.0
        mfe_raw = body.max() if len(body) else 0.0
        mae = min(mae_raw, 0.0)
        mfe = max(mfe_raw, 0.0)
        t_mae = int(np.argmin(body)) + 1 if mae < 0 else 0
        t_mfe = int(np.argmax(body)) + 1 if mfe > 0 else 0
        gross = path[-1]
        rec.append(dict(
            instr=t.instr, entry=t.entry, exit=t.exit, dir=t.dir,
            ret=t.ret, bars=t.bars_held, z=t.z_entry, vol=t.vol_entry,
            gross=gross, mae=mae, mfe=mfe, t_mae=t_mae, t_mfe=t_mfe,
            first_adverse=bool(path[1] < 0) if len(path) > 1 else False,
            win=t.ret > 0,
        ))
    df = pd.DataFrame(rec)
    n = len(df)

    print("=" * 100)
    print("S0. 整合チェック")
    print("=" * 100)
    print(f"n={n}  sum(ret)={df['ret'].sum():+.4f} (基準 {BASE_SUM:+.4f})  "
          f"勝率={df['win'].mean():.3f}  経路長不一致={mismatch}件")
    cost = df["gross"] - df["ret"]
    print(f"sum(gross)={df['gross'].sum():+.4f}  暗黙コスト(gross-ret): "
          f"mean={cost.mean()*1e4:.2f}bps median={cost.median()*1e4:.2f}bps "
          f"sum={cost.sum():+.4f}  負のコスト件数={int((cost < -1e-12).sum())}")

    # ============================================================ Q1
    print()
    print("=" * 100)
    print("Q1. MAE/MFE と time-to の分布、早期判別可能性")
    print("=" * 100)
    for lbl, sub in [("全体", df), ("勝ち", df[df.win]), ("負け", df[~df.win])]:
        d = sub
        print(f"[{lbl}] n={len(d)}")
        print(f"  MAE%  : mean={d.mae.mean()*100:.3f} p25={d.mae.quantile(.25)*100:.3f} "
              f"median={d.mae.median()*100:.3f} p75={d.mae.quantile(.75)*100:.3f} "
              f"p5={d.mae.quantile(.05)*100:.3f} min={d.mae.min()*100:.3f}")
        print(f"  MFE%  : mean={d.mfe.mean()*100:.3f} median={d.mfe.median()*100:.3f} "
              f"p95={d.mfe.quantile(.95)*100:.3f} max={d.mfe.max()*100:.3f}")
        print(f"  t_MAE : median={d.t_mae.median():.0f}本 p75={d.t_mae.quantile(.75):.0f}  "
              f"t_MFE: median={d.t_mfe.median():.0f}本 p75={d.t_mfe.quantile(.75):.0f}  "
              f"保有: median={d.bars.median():.0f}本")
        # time-to を保有比で
        frac_mae = (d.t_mae / d.bars.clip(lower=1)).median()
        frac_mfe = (d.t_mfe / d.bars.clip(lower=1)).median()
        print(f"  t_MAE/保有 median={frac_mae:.2f}  t_MFE/保有 median={frac_mfe:.2f}")
    adv_first = df["first_adverse"].mean()
    mae_before_mfe = ((df.mae < 0) & (df.t_mae < df.t_mfe)).mean()
    never_neg = (df.mae == 0).mean()
    never_pos = (df.mfe == 0).mean()
    print(f"\n逆行が先行: 1本目が含み損 {adv_first*100:.1f}% / "
          f"MAEがMFEより先 {mae_before_mfe*100:.1f}% / 一度も含み損なし {never_neg*100:.1f}% / "
          f"一度も含み益なし {never_pos*100:.1f}%")
    print(f"\n最終MAE(全期間)で勝敗を判別: AUC={auc(-df.mae.values, ~df.win.values):.3f} "
          f"(深いMAE→負け)")
    print("早期判別(k本目時点で建玉中のトレードのみ、その時点までの情報):")
    print(f"{'k本':>5} {'残存n':>6} {'AUC(走行PnL)':>13} {'AUC(走行MAE)':>13} "
          f"{'その時点勝率':>11}")
    for k in [2, 4, 6, 12, 18, 24, 36]:
        m = df.bars.values > k
        if m.sum() < 30:
            continue
        pnl_k = np.array([p[k] for p, mm in zip(paths, m) if mm])
        mae_k = np.array([min(p[1:k + 1].min(), 0.0) for p, mm in zip(paths, m) if mm])
        win_k = df.win.values[m]
        print(f"{k:>5} {m.sum():>6} {auc(pnl_k, win_k):>13.3f} "
              f"{auc(-mae_k, ~win_k):>13.3f} {win_k.mean():>11.3f}")

    # ============================================================ Q2
    print()
    print("=" * 100)
    print("Q2. E[最終ret | MAE≤-x] と E[残り期待値 | k本目で建玉中](損切り/時間ストップ全滅の物理)")
    print("=" * 100)
    print(f"{'x%':>5} {'n(到達)':>8} {'E[最終net ret]bps':>18} {'勝率':>7} "
          f"{'E[回復: 最終-初到達点]bps':>26} {'回復>0率':>9} {'sum(ret)寄与':>13}")
    for x in [0.005, 0.010, 0.015, 0.020, 0.030]:
        hit_idx, recov = [], []
        for i, p in enumerate(paths):
            body = p[1:]
            kk = np.nonzero(body <= -x)[0]
            if len(kk):
                hit_idx.append(i)
                recov.append(p[-1] - body[kk[0]])
        if not hit_idx:
            print(f"{x*100:>5.1f}  到達なし")
            continue
        sub = df.iloc[hit_idx]
        recov = np.array(recov)
        print(f"{x*100:>5.1f} {len(sub):>8} {sub.ret.mean()*1e4:>18.1f} "
              f"{sub.win.mean():>7.3f} {recov.mean()*1e4:>26.1f} "
              f"{(recov > 0).mean():>9.3f} {sub.ret.sum():>+13.4f}")
    # MAE 排他バケツで sum(ret) 分解(検算)
    print("\nMAEバケツ別の sum(ret) 分解(排他・検算用):")
    edges = [0.0, -0.005, -0.010, -0.015, -0.020, -0.030, -np.inf]
    names = ["MAE=0〜-0.5%", "-0.5〜-1%", "-1〜-1.5%", "-1.5〜-2%", "-2〜-3%", "≤-3%"]
    tot = 0.0
    for nm, hi, lo in zip(names, edges[:-1], edges[1:]):
        m = (df.mae <= hi + 1e-15) & (df.mae > lo)
        s = df.ret[m].sum()
        tot += s
        print(f"  {nm:<14} n={int(m.sum()):>4}  sum={s:+.4f} ({s/BASE_SUM*100:+.1f}% of total)  "
              f"勝率={df.win[m].mean() if m.sum() else float('nan'):.3f}  "
              f"平均保有={df.bars[m].mean() if m.sum() else float('nan'):.1f}本")
    print(f"  検算: バケツ合計={tot:+.4f} vs 全体={df.ret.sum():+.4f}")

    print("\nE[残り期待値 | k本目で建玉中](gross, bps) — 時間ストップの物理:")
    print(f"{'k本':>5} {'残存n':>6} {'平均走行PnL(bps)':>17} {'E[残り](bps)':>13} "
          f"{'残り>0率':>9} {'E[残り|走行<0]':>15} {'E[残り|走行>0]':>15}")
    for k in [1, 2, 4, 6, 12, 18, 24, 36, 48, 72, 96, 120]:
        m = df.bars.values > k
        if m.sum() < 15:
            continue
        cur = np.array([p[k] for p, mm in zip(paths, m) if mm])
        rem = np.array([p[-1] - p[k] for p, mm in zip(paths, m) if mm])
        neg, pos = cur < 0, cur >= 0
        print(f"{k:>5} {m.sum():>6} {cur.mean()*1e4:>17.1f} {rem.mean()*1e4:>13.1f} "
              f"{(rem > 0).mean():>9.3f} "
              f"{rem[neg].mean()*1e4 if neg.sum() else float('nan'):>15.1f} "
              f"{rem[pos].mean()*1e4 if pos.sum() else float('nan'):>15.1f}")

    # ---- Q2b: 反実仮想(close基準の楽観近似ですら悪化するかを実測) ----
    print("\nQ2b. 反実仮想の総和(close基準=現実のSLはintrabar約定でさらに悪い):")
    cost_arr = cost.values
    ret_arr = df.ret.values
    for x in [0.005, 0.010, 0.015, 0.020, 0.030]:
        tot_sl = 0.0
        n_hit = 0
        for i, p in enumerate(paths):
            body = p[1:]
            kk = np.nonzero(body <= -x)[0]
            if len(kk):
                tot_sl += body[kk[0]] - cost_arr[i]  # 到達バーcloseで決済(楽観)
                n_hit += 1
            else:
                tot_sl += ret_arr[i]
        print(f"  SL -{x*100:.1f}%: sum(ret) {df.ret.sum():+.4f} → {tot_sl:+.4f} "
              f"({(tot_sl-df.ret.sum())/BASE_SUM*100:+.1f}% of total, 発動{n_hit}件)")
    for kmax in [12, 24, 36, 48, 72]:
        tot_ts = 0.0
        n_cut = 0
        for i, p in enumerate(paths):
            if len(p) - 1 > kmax:
                tot_ts += p[kmax] - cost_arr[i]
                n_cut += 1
            else:
                tot_ts += ret_arr[i]
        print(f"  時間ストップ {kmax}本: sum(ret) → {tot_ts:+.4f} "
              f"({(tot_ts-df.ret.sum())/BASE_SUM*100:+.1f}% of total, 発動{n_cut}件)")
    # 深い含み損×長期保有の複合セル
    print("\n深い含み損×長期保有の複合セル E[残り](gross, bps):")
    for k in [12, 24, 36]:
        for x in [0.01, 0.02]:
            cur, rem = [], []
            for i, p in enumerate(paths):
                if len(p) - 1 > k and p[k] <= -x:
                    cur.append(p[k]); rem.append(p[-1] - p[k])
            if len(rem) >= 10:
                rem = np.array(rem)
                print(f"  k={k}本 & 走行≤-{x*100:.0f}%: n={len(rem)} "
                      f"E[残り]={rem.mean()*1e4:+.1f}bps 残り>0率={(rem>0).mean():.3f}")

    # ============================================================ Q3
    print()
    print("=" * 100)
    print("Q3. z_entry 細ビン × 成績(P=4サイジングの独立確認)")
    print("=" * 100)
    bins = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, np.inf]
    lab = [f"{a:.1f}-{b:.1f}" if np.isfinite(b) else f"{a:.1f}+" for a, b in zip(bins[:-1], bins[1:])]
    df["zbin"] = pd.cut(df.z, bins, labels=lab, right=False)
    g = df.groupby("zbin", observed=True)
    tab = pd.DataFrame({
        "n": g.size(),
        "平均ret_bps": g.ret.mean() * 1e4,
        "median_bps": g.ret.median() * 1e4,
        "勝率": g.win.mean(),
        "平均保有": g.bars.mean(),
        "ret_std_bps": g.ret.std() * 1e4,
        "sum_ret": g.ret.sum(),
        "寄与%": g.ret.sum() / BASE_SUM * 100,
        "f(z)中心": [float(f_z(np.array([(a + min(b, 4.0)) / 2]))[0])
                     for a, b in zip(bins[:-1], bins[1:])],
    })
    tab["mean/var比"] = tab["平均ret_bps"] / (tab["ret_std_bps"] ** 2) * 1e4  # ケリー比例項
    print(tab.to_string())
    print(f"検算: zビン sum_ret 合計={tab.sum_ret.sum():+.4f} vs 全体={df.ret.sum():+.4f}")
    base = tab.loc[lab[0], "平均ret_bps"]
    print("\n平均retの対ベース倍率(=理論的に最適なサイズ比のラフ近似) vs f(z)実装倍率:")
    f0 = tab.loc[lab[0], "f(z)中心"]
    for l_ in lab:
        r = tab.loc[l_, "平均ret_bps"] / base if base else np.nan
        fr = tab.loc[l_, "f(z)中心"] / f0
        print(f"  z {l_:<8} 実測mean比 x{r:>5.2f}   f(z)比 x{fr:>5.2f}   n={int(tab.loc[l_,'n'])}")

    # z×vol 交互作用
    print("\nz(粗ビン) × vol_entry(銘柄内ランク三分位) の平均ret_bps / [勝率] / (n):")
    df["volr"] = pool.groupby("instr")["vol_entry"].rank(pct=True).values
    df["volb"] = pd.cut(df.volr, [0, 1 / 3, 2 / 3, 1.0001], labels=["低vol", "中vol", "高vol"])
    df["zb3"] = pd.cut(df.z, [2.0, 2.4, 2.8, np.inf], labels=["z2.0-2.4", "z2.4-2.8", "z2.8+"], right=False)
    piv_m = df.pivot_table(index="zb3", columns="volb", values="ret", aggfunc="mean", observed=True) * 1e4
    piv_w = df.pivot_table(index="zb3", columns="volb", values="win", aggfunc="mean", observed=True)
    piv_n = df.pivot_table(index="zb3", columns="volb", values="ret", aggfunc="size", observed=True)
    for zb in piv_m.index:
        row = "  ".join(f"{c}: {piv_m.loc[zb,c]:6.1f} [{piv_w.loc[zb,c]:.2f}] (n={int(piv_n.loc[zb,c])})"
                        for c in piv_m.columns)
        print(f"  {zb:<9} {row}")

    # ============================================================ Q4
    print()
    print("=" * 100)
    print("Q4. ret 分布統計(素 vs 本番f(z)加重)")
    print("=" * 100)
    w = f_z(df.z.values)
    w_norm = w / w.mean()
    ret_w = df.ret.values * w_norm
    for lbl, x, ww in [("素のret", df.ret.values, None),
                       ("f(z)加重 ret_w=ret*f/mean(f)", ret_w, None)]:
        st = dist_stats(x, ww)
        print(f"[{lbl}]")
        print(f"  mean={st['mean']*1e4:+.1f}bps std={st['std']*1e4:.1f}bps "
              f"skew={st['skew']:+.2f} 超過kurt={st['kurt']:.1f}")
        print(f"  p1={st['p1']*100:+.2f}% p5={st['p5']*100:+.2f}% median={st['p50']*1e4:+.1f}bps "
              f"p95={st['p95']*100:+.2f}% p99={st['p99']*100:+.2f}% "
              f"min={st['min']*100:+.2f}% max={st['max']*100:+.2f}%")
        print(f"  左テールのσ距離: (p1-mean)/std={(st['p1']-st['mean'])/st['std']:.2f}σ  "
              f"(min-mean)/std={(st['min']-st['mean'])/st['std']:.2f}σ")
    print(f"\n加重後合計={ret_w.sum():+.4f}(素 {df.ret.sum():+.4f})  "
          f"加重ワースト1件={ret_w.min()*100:+.2f}%(素ワースト {df.ret.min()*100:+.2f}%)")
    wi = np.argmin(ret_w)
    print(f"  加重ワーストの正体: {df.instr.iloc[wi]} dir={df.dir.iloc[wi]} "
          f"entry={df.entry.iloc[wi].date()} z={df.z.iloc[wi]:.2f} w={w_norm[wi]:.2f} "
          f"素ret={df.ret.iloc[wi]*100:+.2f}%")

    # ============================================================ Q5
    print()
    print("=" * 100)
    print("Q5. 取りこぼし: Σ(MFE - 実現gross) と MFE回収率")
    print("=" * 100)
    miss = df.mfe - df.gross
    print(f"Σ MFE = {df.mfe.sum():+.4f}  Σ gross = {df.gross.sum():+.4f}  "
          f"Σ(MFE-gross) = {miss.sum():+.4f} = 総純益(+{BASE_SUM:.4f})の {miss.sum()/BASE_SUM:.2f}倍")
    print(f"MFE回収率 Σgross/ΣMFE = {df.gross.sum()/df.mfe.sum()*100:.1f}%  "
          f"(close基準MFE=intrabar高値ではない控えめな値)")
    pos = df[df.mfe > 0]
    cap = (pos.gross / pos.mfe)
    print(f"per-trade回収率(MFE>0のn={len(pos)}): median={cap.median()*100:.1f}% "
          f"mean={cap.mean()*100:.1f}%  勝ちのみmedian={(df[df.win].gross/df[df.win].mfe).median()*100:.1f}%")
    print(f"勝ちトレードの取りこぼし Σ(MFE-gross|win)={miss[df.win].sum():+.4f} / "
          f"負けトレードの取りこぼし Σ(MFE-gross|loss)={miss[~df.win].sum():+.4f}")
    # MFE時点でz出口(|z|<=0.5)は既に発火していたか
    fired, not_fired, give_f, give_nf = 0, 0, [], []
    for i, t in enumerate(df.itertuples()):
        if t.mfe <= 0 or t.t_mfe == 0:
            continue
        c = closes[t.instr]
        i0 = c.index.get_loc(t.entry)
        zt = ind[t.instr]["z50"].iloc[i0 + t.t_mfe]
        give = t.mfe - t.gross
        if np.isfinite(zt) and abs(zt) <= 0.5:
            fired += 1; give_f.append(give)
        else:
            not_fired += 1; give_nf.append(give)
    print(f"\nMFE時点のz状態: |z|≤0.5(出口圏)だった={fired}件(givebackΣ={np.sum(give_f):+.4f}) / "
          f"|z|>0.5(出口未発火)={not_fired}件(givebackΣ={np.sum(give_nf):+.4f})")
    print("→ giveback の大半が『出口ルール未発火のままMFEから垂れた』のか『出口圏到達後の同バー差』かを判別")

    # ============================================================ Q6
    print()
    print("=" * 100)
    print("Q6. 含み損ピーク(MAE)時点の情報は回復/失敗を予測するか(MAE≤-1%の塩漬け帯)")
    print("=" * 100)
    deep = df[df.mae <= -0.01].copy()
    feats = {"er40_at_mae": [], "dir_z250_at_mae": [], "dir_z50_at_mae": [],
             "elapsed_at_mae": [], "mae_depth": []}
    for t in deep.itertuples():
        c = closes[t.instr]
        i0 = c.index.get_loc(t.entry)
        row = ind[t.instr].iloc[i0 + t.t_mae]
        feats["er40_at_mae"].append(row["er40"])
        feats["dir_z250_at_mae"].append(t.dir * row["z250"])
        feats["dir_z50_at_mae"].append(t.dir * row["z50"])
        feats["elapsed_at_mae"].append(t.t_mae)
        feats["mae_depth"].append(-t.mae)
    for k_, v in feats.items():
        deep[k_] = v
    rec_lbl = deep.win.values
    print(f"対象 n={len(deep)} (回復=最終ret>0 は {rec_lbl.mean()*100:.1f}%)  "
          f"sum(ret)={deep.ret.sum():+.4f} ({deep.ret.sum()/BASE_SUM*100:+.1f}% of total)")
    print(f"{'特徴量(MAE時点)':<18} {'回復群mean':>11} {'失敗群mean':>11} {'AUC(回復予測)':>14}")
    for k_ in feats:
        v = deep[k_].values.astype(float)
        ok = np.isfinite(v)
        a = auc(v[ok], rec_lbl[ok])
        print(f"{k_:<18} {v[ok][rec_lbl[ok]].mean():>11.3f} {v[ok][~rec_lbl[ok]].mean():>11.3f} "
              f"{a:>14.3f}")
    # 最有望特徴の上下分割で E[最終ret]
    for k_ in ["er40_at_mae", "dir_z250_at_mae"]:
        v = deep[k_].astype(float)
        med = v.median()
        lo_m, hi_m = v <= med, v > med
        print(f"\n  {k_} ≤median({med:.3f}): n={int(lo_m.sum())} E[最終ret]={deep.ret[lo_m].mean()*1e4:+.1f}bps "
              f"勝率={deep.win[lo_m].mean():.3f} | >median: n={int(hi_m.sum())} "
              f"E[最終ret]={deep.ret[hi_m].mean()*1e4:+.1f}bps 勝率={deep.win[hi_m].mean():.3f}")

    print("\n完了。")


if __name__ == "__main__":
    main()
