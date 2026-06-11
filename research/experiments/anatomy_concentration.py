"""anatomy_concentration.py — チャンピオン手法 confluence_meanrev_v2 トレードプールの
利益の集中度と依存構造の解剖。

実行: uv run python -m research.experiments.anatomy_concentration

クエスチョン:
 Q1 勝ち上位1/5/10%の利益集中、純益ゼロ化に必要な上位勝ちk件
 Q2 銘柄別19行表 + leave-one-out + 上位3シェア
 Q3 銘柄×決済年セルのプラス率（広く浅いか狭く深いか）
 Q4 同時クラスタ（5営業日以内開始+保有重複）→ 実効独立ベット数/年、通貨レッグ重複
 Q5 本番加重 f(z)=clip((|z|/2.2)^4,0.3,3) で 1-4 の主要数値の変化
 Q6 ブートストラップ95%CI / t統計量 / クラスタ考慮の実効n
"""
import numpy as np
import pandas as pd

POOL = "results/mm_pool_v2_H4_19.parquet"
TOTAL_EXPECTED = 1.9086
SEED = 42
N_BOOT = 10000


# ---------------------------------------------------------------- helpers
def fz(z):
    return np.clip((np.abs(z) / 2.2) ** 4.0, 0.3, 3.0)


def pf(r):
    g = r[r > 0].sum()
    l = r[r < 0].sum()
    return g / abs(l) if l < 0 else np.inf


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_clusters(df, busdays=5):
    """エントリーが互いに5営業日以内 AND 保有期間が重なるトレードを連結成分でクラスタ化。"""
    d = df.sort_values("entry").reset_index(drop=True)
    n = len(d)
    ent = d["entry"].values
    exi = d["exit"].values
    ent_dates = d["entry"].dt.date.values.astype("datetime64[D]")
    dsu = DSU(n)
    # 5営業日 <= 7暦日なので 8日窓でスキャン
    win = np.timedelta64(8, "D")
    for i in range(n):
        j = i + 1
        while j < n and ent[j] - ent[i] <= win:
            if ent[j] < exi[i]:  # 保有が重なる
                if np.busday_count(ent_dates[i], ent_dates[j]) <= busdays:
                    dsu.union(i, j)
            j += 1
    d["cluster"] = [dsu.find(i) for i in range(n)]
    # relabel 0..K-1 in chronological order
    order = d.groupby("cluster")["entry"].min().sort_values().index
    remap = {c: k for k, c in enumerate(order)}
    d["cluster"] = d["cluster"].map(remap)
    return d


def concentration(rets, total_label, weights_note=""):
    """勝ちトレード上位p%のグロス利益・純益シェアと、純益ゼロ化のk。"""
    r = np.sort(rets[rets > 0])[::-1]
    gross = r.sum()
    net = rets.sum()
    out = {}
    for p in (0.01, 0.05, 0.10):
        k = max(1, int(np.ceil(len(r) * p)))
        s = r[:k].sum()
        out[p] = (k, s, s / gross, s / net)
    cum = np.cumsum(r)
    kzero = int(np.searchsorted(cum, net) + 1)  # cum[k-1] >= net -> 除外で純益<=0
    print(f"  [{total_label}{weights_note}] 勝ちn={len(r)} グロス利益={gross:+.4f} 純益={net:+.4f}")
    for p, (k, s, gs, ns) in out.items():
        print(f"    勝ち上位{p*100:>4.0f}% (k={k:>3}): 合計{s:+.4f} = グロス利益の{gs*100:5.1f}% / 純益の{ns*100:6.1f}%")
    print(f"    純益ゼロ化に必要な上位勝ち除外 k = {kzero} 件 (= 勝ちの{kzero/len(r)*100:.1f}%, 全体の{kzero/len(rets)*100:.1f}%)")
    return out, kzero


def currency_exposure(d, wcol=None):
    """イベントグリッド上の通貨別ネットエクスポージャ(+base/-quote×dir×w)の最大絶対値。"""
    ccys = sorted({i[:3] for i in d["instr"]} | {i[3:] for i in d["instr"]})
    ent_ns = d["entry"].values.astype("datetime64[ns]").astype(np.int64)
    exi_ns = d["exit"].values.astype("datetime64[ns]").astype(np.int64)
    events = np.sort(np.unique(np.concatenate([ent_ns, exi_ns])))
    ei_arr = np.searchsorted(events, ent_ns)
    xi_arr = np.searchsorted(events, exi_ns)
    expo = {c: np.zeros(len(events)) for c in ccys}
    openw = np.zeros(len(events))
    instrs = d["instr"].values
    dirs = d["dir"].values
    ws = np.ones(len(d)) if wcol is None else d[wcol].values
    for k in range(len(d)):
        b, q = instrs[k][:3], instrs[k][3:]
        ei, xi, dr, w = ei_arr[k], xi_arr[k], dirs[k], ws[k]
        expo[b][ei] += dr * w
        expo[b][xi] -= dr * w
        expo[q][ei] -= dr * w
        expo[q][xi] += dr * w
        openw[ei] += w
        openw[xi] -= w
    paths = {c: np.cumsum(v) for c, v in expo.items()}
    best_c, best_v, best_t = None, 0.0, None
    for c, p in paths.items():
        i = int(np.argmax(np.abs(p)))
        if abs(p[i]) > best_v:
            best_c, best_v, best_t = c, abs(p[i]), events[i]
    open_path = np.cumsum(openw)
    return best_c, best_v, pd.Timestamp(best_t, tz="UTC"), open_path.max(), paths


def boot_ci(values, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(values)
    means = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5]), means.std()


def cluster_boot_ci(d, retcol, n_boot=N_BOOT, seed=SEED):
    """クラスタ単位リサンプル: mean = sum(ret)/sum(n) のブートストラップ。"""
    g = d.groupby("cluster").agg(s=(retcol, "sum"), n=(retcol, "size"))
    s, n = g["s"].values, g["n"].values.astype(float)
    rng = np.random.default_rng(seed)
    K = len(s)
    idx = rng.integers(0, K, size=(n_boot, K))
    means = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    return np.percentile(means, [2.5, 97.5]), means.std()


def cluster_robust_t(d, retcol):
    """クラスタロバストSEによるt値と実効n。"""
    r = d[retcol].values
    n = len(r)
    mu = r.mean()
    se_iid = r.std(ddof=1) / np.sqrt(n)
    g = d.groupby("cluster")[retcol]
    resid_sums = g.sum().values - mu * g.size().values
    se_cl = np.sqrt((resid_sums ** 2).sum()) / n
    n_eff = n * (se_iid / se_cl) ** 2
    return mu, se_iid, se_cl, mu / se_iid, mu / se_cl, n_eff


# ---------------------------------------------------------------- main
def main():
    df = pd.read_parquet(POOL)
    total = df["ret"].sum()
    print("=" * 88)
    print("ベースライン検算")
    print("=" * 88)
    gross_p = df.loc[df.ret > 0, "ret"].sum()
    gross_l = df.loc[df.ret < 0, "ret"].sum()
    print(f"n={len(df)}  sum(ret)={total:+.4f} (期待 {TOTAL_EXPECTED:+.4f})  "
          f"平均={total/len(df)*1e4:.1f}bps  勝率={(df.ret>0).mean()*100:.1f}%  PF={gross_p/abs(gross_l):.3f}")
    assert abs(total - TOTAL_EXPECTED) < 1e-3

    years_span = (df["exit"].max() - df["entry"].min()).days / 365.25

    # ---------------- Q1 勝ち集中度 ----------------
    print("\n" + "=" * 88)
    print("Q1. 勝ちトレードの利益集中度（等加重）")
    print("=" * 88)
    concentration(df["ret"].values, "等加重")
    # 対称性: ベスト10% vs ワースト10%（全トレード基準）
    k10 = int(np.ceil(len(df) * 0.10))
    r_sorted = np.sort(df["ret"].values)
    best10 = r_sorted[-k10:].sum()
    worst10 = r_sorted[:k10].sum()
    print(f"  対称性チェック(全トレード基準): ベスト10%({k10}件)=グロス利益の{best10/gross_p*100:.1f}% "
          f"/ ワースト10%({k10}件)=総損失の{worst10/gross_l*100:.1f}%（既知72.5%との照合）")

    # ---------------- Q2 銘柄別 ----------------
    print("\n" + "=" * 88)
    print("Q2. 銘柄別の純益・PF・取引数（19行）と leave-one-out")
    print("=" * 88)
    rows = []
    for instr, g in df.groupby("instr"):
        rows.append({
            "instr": instr, "n": len(g), "net": g["ret"].sum(),
            "PF": pf(g["ret"]), "win%": (g["ret"] > 0).mean() * 100,
            "net_share%": g["ret"].sum() / total * 100,
            "LOO_net": total - g["ret"].sum(),
        })
    tab = pd.DataFrame(rows).sort_values("net", ascending=False).reset_index(drop=True)
    tab["LOO_drop%"] = (1 - tab["LOO_net"] / total) * 100
    with pd.option_context("display.float_format", lambda v: f"{v:8.4f}"):
        print(tab.to_string(index=False))
    assert abs(tab["net"].sum() - total) < 1e-9, "銘柄分解の合計が不一致"
    print(f"\n  検算: 銘柄別net合計 = {tab['net'].sum():+.4f} ✓")
    top3 = tab.head(3)
    print(f"  上位3銘柄 {list(top3['instr'])} の純益シェア = {top3['net'].sum()/total*100:.1f}%")
    worst_loo = tab.iloc[0]
    print(f"  leave-one-out最大: {worst_loo['instr']} を除くと純益 {total:+.4f} → {worst_loo['LOO_net']:+.4f} "
          f"({worst_loo['LOO_drop%']:.1f}%減)")
    n_pos = (tab["net"] > 0).sum()
    print(f"  純益プラス銘柄: {n_pos}/19")

    # ---------------- Q3 銘柄×決済年 ----------------
    print("\n" + "=" * 88)
    print("Q3. 銘柄×決済年セルのプラス率")
    print("=" * 88)
    d3 = df.copy()
    d3["year"] = d3["exit"].dt.year
    pv = d3.pivot_table(index="instr", columns="year", values="ret", aggfunc="sum")
    cnt = d3.pivot_table(index="instr", columns="year", values="ret", aggfunc="size")
    filled = pv.notna()
    pos = (pv > 0) & filled
    print(f"  セル(銘柄×年, 取引あり): {filled.values.sum()}個 / プラス {pos.values.sum()}個 "
          f"= プラス率 {pos.values.sum()/filled.values.sum()*100:.1f}%")
    instr_posrate = (pos.sum(axis=1) / filled.sum(axis=1)).sort_values(ascending=False)
    print("  銘柄別プラス年率（プラス年数/取引あり年数）:")
    for instr, v in instr_posrate.items():
        yrs = int(filled.loc[instr].sum())
        net = pv.loc[instr].sum()
        print(f"    {instr}: {v*100:5.1f}% ({int(pos.loc[instr].sum())}/{yrs}年)  net={net:+.4f}")
    assert abs(pv.fillna(0).values.sum() - total) < 1e-9
    # 集中度指標: 正セルだけで上位セルシェア
    cellvals = pv.stack().sort_values(ascending=False)
    topcells = cellvals.head(10)
    print(f"  上位10セル(全{len(cellvals)}セル)の合計 = {topcells.sum():+.4f} = 純益の{topcells.sum()/total*100:.1f}%")
    print("  上位5セル:", [(f"{i[0]}-{i[1]}", round(v, 3)) for i, v in cellvals.head(5).items()])

    # ---------------- Q4 同時クラスタ ----------------
    print("\n" + "=" * 88)
    print("Q4. 同時クラスタ分析（5営業日以内開始 AND 保有重複 → 連結成分）")
    print("=" * 88)
    dc = build_clusters(df)
    cl = dc.groupby("cluster").agg(
        n=("ret", "size"), net=("ret", "sum"),
        start=("entry", "min"), end=("exit", "max"),
        instrs=("instr", lambda s: len(set(s))),
    )
    K = len(cl)
    print(f"  クラスタ数 K = {K}（n=1214 → 縮約率 {K/len(df)*100:.1f}%）")
    print(f"  実効独立ベット数 = {K/years_span:.1f} 個/年（期間 {years_span:.2f}年）")
    sizes = cl["n"].value_counts().sort_index()
    print(f"  クラスタサイズ分布: " + ", ".join(f"{s}件×{c}" for s, c in sizes.items()))
    print(f"  サイズ中央値={cl['n'].median():.0f} / 平均={cl['n'].mean():.2f} / 最大={cl['n'].max()}")
    assert abs(cl["net"].sum() - total) < 1e-9
    big = cl.sort_values("n", ascending=False).head(3)
    print("  最大クラスタ(サイズ順):")
    for cid, row in big.iterrows():
        print(f"    #{cid}: {int(row['n'])}トレード/{int(row['instrs'])}銘柄 "
              f"{row['start'].date()}→{row['end'].date()} net={row['net']:+.4f} (純益の{row['net']/total*100:+.1f}%)")
    bestc = cl.sort_values("net", ascending=False).head(3)
    worstc = cl.sort_values("net").head(3)
    print("  純益寄与トップ3クラスタ:")
    for cid, row in bestc.iterrows():
        mem = dc[dc.cluster == cid]
        print(f"    #{cid}: net={row['net']:+.4f} ({row['net']/total*100:+.1f}%) "
              f"{int(row['n'])}件 {sorted(set(mem['instr']))} {row['start'].date()}")
    print("  損失寄与ワースト3クラスタ:")
    for cid, row in worstc.iterrows():
        mem = dc[dc.cluster == cid]
        print(f"    #{cid}: net={row['net']:+.4f} ({row['net']/total*100:+.1f}%) "
              f"{int(row['n'])}件 {sorted(set(mem['instr']))} {row['start'].date()}")
    yearly_clusters = cl.groupby(cl["start"].dt.year).size()
    print("  年別クラスタ数:", dict(yearly_clusters))

    # 通貨レッグ重複
    print("\n  -- 通貨レッグ重複（同時保有時の実効エクスポージャ）--")
    c, v, t, maxopen, paths = currency_exposure(dc)
    print(f"  等加重: 最大同時オープン建玉 = {maxopen:.0f}本")
    print(f"  等加重: 通貨別ネットエクスポージャ最大 = {c} {v:.0f}単位 @ {t}")
    peaks = {cc: np.abs(p).max() for cc, p in paths.items()}
    print("  通貨別ピーク|エクスポージャ|:", {k: round(v2, 1) for k, v2 in sorted(peaks.items(), key=lambda x: -x[1])})

    # ---------------- Q5 本番加重 ----------------
    print("\n" + "=" * 88)
    print("Q5. 本番加重 f(z)=clip((|z|/2.2)^4, 0.3, 3) での再計算")
    print("=" * 88)
    w = fz(df["z_entry"].values)
    print(f"  重み統計: mean={w.mean():.3f} median={np.median(w):.3f} "
          f"min={w.min():.3f} max={w.max():.3f} / クリップ下限{(w<=0.3).mean()*100:.0f}% 上限{(w>=3.0).mean()*100:.0f}%")
    dw = df.copy()
    dw["w"] = w
    dw["ret_w"] = dw["ret"] * dw["w"] / w.mean()
    total_w = dw["ret_w"].sum()
    gross_pw = dw.loc[dw.ret_w > 0, "ret_w"].sum()
    gross_lw = dw.loc[dw.ret_w < 0, "ret_w"].sum()
    print(f"  加重後: sum(ret_w)={total_w:+.4f}  平均={total_w/len(dw)*1e4:.1f}bps  PF={gross_pw/abs(gross_lw):.3f}")
    print("\n  Q1' 加重後の勝ち集中度:")
    concentration(dw["ret_w"].values, "本番加重")
    rs = np.sort(dw["ret_w"].values)
    print(f"  ベスト10%={rs[-k10:].sum()/gross_pw*100:.1f}%ofグロス / ワースト10%={rs[:k10].sum()/gross_lw*100:.1f}%of総損失")
    tabw = dw.groupby("instr")["ret_w"].sum().sort_values(ascending=False)
    print(f"\n  Q2' 加重後 上位3銘柄 {list(tabw.index[:3])} シェア = {tabw.iloc[:3].sum()/total_w*100:.1f}% "
          f"(等加重 {top3['net'].sum()/total*100:.1f}%)")
    loo_w = tabw.iloc[0] / total_w * 100
    print(f"  加重後 leave-one-out最大: {tabw.index[0]} 除外で {loo_w:.1f}%減 (等加重 {worst_loo['LOO_drop%']:.1f}%)")
    dcw = dc.merge(dw[["instr", "entry", "ret_w", "w"]], on=["instr", "entry"], how="left")
    assert abs(dcw["ret_w"].sum() - total_w) < 1e-6
    clw = dcw.groupby("cluster")["ret_w"].sum()
    print(f"\n  Q4' 加重後 純益寄与最大クラスタ = {clw.max():+.4f} ({clw.max()/total_w*100:+.1f}%) "
          f"/ ワースト = {clw.min():+.4f} ({clw.min()/total_w*100:+.1f}%)")
    print(f"      (等加重: 最大 {cl['net'].max()/total*100:+.1f}% / ワースト {cl['net'].min()/total*100:+.1f}%)")
    cw, vw, tw, maxopen_w, _ = currency_exposure(dcw, wcol="w")
    print(f"  加重後 通貨レッグ最大 = {cw} {vw:.1f}重みユニット @ {tw} "
          f"(等価トレード本数換算 {vw/w.mean():.1f}本) / 最大同時オープン重み = {maxopen_w:.1f}")

    # ---------------- Q6 統計的頑健性 ----------------
    print("\n" + "=" * 88)
    print("Q6. 統計的頑健性（ブートストラップ・クラスタロバスト）")
    print("=" * 88)
    r = df["ret"].values
    (lo, hi), se_b = boot_ci(r)
    mu, se_iid, se_cl, t_iid, t_cl, n_eff = cluster_robust_t(dc, "ret")
    print(f"  等加重: 平均={mu*1e4:.2f}bps  iid bootstrap 95%CI=[{lo*1e4:.2f}, {hi*1e4:.2f}]bps  t(iid)={t_iid:.2f}")
    (lo_c, hi_c), _ = cluster_boot_ci(dc, "ret")
    print(f"  クラスタbootstrap 95%CI=[{lo_c*1e4:.2f}, {hi_c*1e4:.2f}]bps  t(cluster-robust)={t_cl:.2f}")
    print(f"  実効n = {n_eff:.0f} (名目 {len(r)} → {n_eff/len(r)*100:.0f}%, SE膨張率 x{se_cl/se_iid:.2f})")
    rw = dw["ret_w"].values
    (lo_w, hi_w), _ = boot_ci(rw)
    mu_w, se_iid_w, se_cl_w, t_iid_w, t_cl_w, n_eff_w = cluster_robust_t(dcw, "ret_w")
    print(f"  本番加重: 平均={mu_w*1e4:.2f}bps  iid 95%CI=[{lo_w*1e4:.2f}, {hi_w*1e4:.2f}]bps  t(iid)={t_iid_w:.2f}")
    (lo_cw, hi_cw), _ = cluster_boot_ci(dcw, "ret_w")
    print(f"  クラスタbootstrap 95%CI=[{lo_cw*1e4:.2f}, {hi_cw*1e4:.2f}]bps  t(cluster-robust)={t_cl_w:.2f}")
    print(f"  実効n = {n_eff_w:.0f} (名目比 {n_eff_w/len(rw)*100:.0f}%, SE膨張率 x{se_cl_w/se_iid_w:.2f})")


if __name__ == "__main__":
    main()
