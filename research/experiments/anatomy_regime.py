"""anatomy_regime.py — チャンピオンv2トレードプールの「利益のレジーム依存」解剖。

対象: results/mm_pool_v2_H4_19.parquet(n=1214, sum ret=+1.9086)。読み取り専用。
価格: results/mm_closes_H4_19.parquet(mm_lab.load_closes のキャッシュ=H4 close 行列, ffill 済)。
      vol_entry の再計算が pool と median diff=0 / max 2.5e-5(典型ボラの1.6%)で一致することを確認済み。

必答クエスチョン:
  Q1 vol_entry 四分位 / ER(40) 再計算の十分位 × 純益・平均ret・勝率
  Q2 時代分割(2016-2021 vs 2022-2026)+ ret の時間線形トレンド(減衰検定)
  Q3 危機/高ボラ窓(COVID, 2022-09/10, 実現ボラ上位5%日)の窓内 vs 窓外
  Q4 等加重 MtM 日次損益のトップ5ドローダウン窓と地合い(ER/ボラ/USD指数トレンド)
  Q5 通貨レッグ寄与(±0.5 配賦)等加重 / f(z) 本番加重
  Q6 月次戦略リターン × |USD指数月次変化| / 平均実現ボラ の相関(ショートボラ性)

計上規約(明記):
  - 等加重 = 1トレード1単位(sum(ret) と同じ土俵)。分解は常に合計=+1.9086 に整合検算。
  - Q4 の日次損益は MtM: トレード i のバー t 損益 = dir*(C_t - C_{t-1})/entry_price
    (起点は entry_price、決済バーは「トレード合計=ret」になる残差=スプレッド片道分を計上)。
    バー損益を UTC 日付で集計 → 日次。よって日次合計 = sum(ret) と厳密一致。
  - Q3/Q6 のトレード割当は entry 日基準(別途 MtM 窓内損益も併記)。
  - 月次戦略リターン(Q2/Q6)は決済月計上(exit 月)。トレード無し月は 0。

実行: uv run python -m research.experiments.anatomy_regime
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

POOL_PATH = "results/mm_pool_v2_H4_19.parquet"
CLOSES_PATH = "results/mm_closes_H4_19.parquet"
TOTAL_EXPECTED = 1.9086

# USD 等加重指数: USD が分子のペアは +、分母のペアは -
USD_SIGN = {"EURUSD": -1, "GBPUSD": -1, "AUDUSD": -1, "NZDUSD": -1,
            "USDJPY": +1, "USDCHF": +1, "USDCAD": +1}


def f_z(z: np.ndarray) -> np.ndarray:
    """本番 z-power サイジング: clip((|z|/2.2)^4, 0.3, 3.0)。"""
    return np.clip((np.abs(z) / 2.2) ** 4.0, 0.3, 3.0)


def efficiency_ratio(close: pd.Series, w: int = 40) -> pd.Series:
    """Kaufman ER = |Δw本| / Σ|Δ1本|(strategies/confluence_meanrev_filtered と同一式)。"""
    direction = (close - close.shift(w)).abs()
    volatility = close.diff().abs().rolling(w).sum()
    return (direction / volatility).replace([np.inf, -np.inf], np.nan)


def grp_stats(df: pd.DataFrame, key) -> pd.DataFrame:
    """グループ別: n / 純益 / 総益比% / 平均ret(bps) / 勝率 / PF。"""
    rows = []
    for g, sub in df.groupby(key, observed=True):
        r = sub["ret"]
        pos, neg = r[r > 0].sum(), -r[r <= 0].sum()
        rows.append({
            "group": g, "n": len(sub), "sum_ret": r.sum(),
            "share_of_total_%": r.sum() / TOTAL_EXPECTED * 100,
            "mean_bps": r.mean() * 1e4, "win_rate_%": (r > 0).mean() * 100,
            "PF": pos / neg if neg > 0 else np.inf,
            "med_bars": sub["bars_held"].median(),
        })
    return pd.DataFrame(rows).set_index("group")


def main() -> None:
    pool = pd.read_parquet(POOL_PATH).copy()
    closes = pd.read_parquet(CLOSES_PATH)
    grid = closes.index
    instruments = list(closes.columns)

    total = pool["ret"].sum()
    print("=" * 100)
    print(f"ベースライン検算: n={len(pool)}  sum(ret)={total:+.4f}  "
          f"平均={pool['ret'].mean()*1e4:+.1f}bps  勝率={(pool['ret']>0).mean()*100:.1f}%  "
          f"期間 {pool['entry'].min().date()} 〜 {pool['exit'].max().date()}")
    assert abs(total - TOTAL_EXPECTED) < 1e-3, "ベースライン不一致"

    # --- 共通指標(全銘柄): ER(40), vol20, 日次平均, USD 指数 -----------------
    er_mat = pd.DataFrame({nm: efficiency_ratio(closes[nm], 40) for nm in instruments})
    vol_mat = pd.DataFrame({nm: closes[nm].pct_change().rolling(20).std() for nm in instruments})

    # エントリー時点 lookup(同一バー約定=シグナルバーの値。先読みなし)
    pool["er_entry"] = np.nan
    pool["vol_chk"] = np.nan
    for nm, g in pool.groupby("instr"):
        pool.loc[g.index, "er_entry"] = er_mat[nm].reindex(g["entry"]).to_numpy()
        pool.loc[g.index, "vol_chk"] = vol_mat[nm].reindex(g["entry"]).to_numpy()
    vdiff = (pool["vol_chk"] - pool["vol_entry"]).abs()
    print(f"vol_entry 再計算整合: median diff={vdiff.median():.2e}  max={vdiff.max():.2e} "
          f"(典型ボラ {pool['vol_entry'].median():.2e} の {vdiff.max()/pool['vol_entry'].median()*100:.1f}%)")
    n_over = (pool["er_entry"] > 0.55).sum()
    print(f"ER 再計算: NaN={pool['er_entry'].isna().sum()}  >0.55(フィルタ矛盾)={n_over} 件 "
          f"(ffill 起因の微差。0.56 超={(pool['er_entry'] > 0.56).sum()} 件)")

    # 日次(UTC)クロスセクション平均: ボラ・ER。USD 等加重ログ指数。
    dates = grid.floor("D")
    vol_daily = vol_mat.mean(axis=1).groupby(dates).mean()        # 全銘柄平均実現ボラ(日次)
    er_daily = er_mat.mean(axis=1).groupby(dates).mean()          # 全銘柄平均 ER(日次)
    usd_bar = sum(USD_SIGN[p] * np.log(closes[p]).diff() for p in USD_SIGN) / len(USD_SIGN)
    usd_idx_daily = usd_bar.groupby(dates).sum().cumsum()         # USD 等加重ログ指数(日次)

    # =========================================================================
    print("\n" + "=" * 100)
    print("Q1. vol_entry 四分位 × 成績")
    pool["vol_q"] = pd.qcut(pool["vol_entry"], 4, labels=["Q1(低ボラ)", "Q2", "Q3", "Q4(高ボラ)"])
    t1 = grp_stats(pool, "vol_q")
    print(t1.to_string())
    print(f"  検算: 四分位合計 = {t1['sum_ret'].sum():+.4f}")

    print("\nQ1b. ER(40) エントリー時点・十分位 × 成績(フィルタ通過域 ≤0.55 内部の勾配)")
    er_ok = pool.dropna(subset=["er_entry"])
    er_ok = er_ok.assign(er_d=pd.qcut(er_ok["er_entry"], 10, labels=False, duplicates="drop"))
    edges = er_ok.groupby("er_d")["er_entry"].agg(["min", "max"])
    t1b = grp_stats(er_ok, "er_d")
    t1b.insert(0, "er_range", [f"{edges.loc[i,'min']:.3f}-{edges.loc[i,'max']:.3f}" for i in t1b.index])
    print(t1b.to_string())
    print(f"  検算: 十分位合計 = {t1b['sum_ret'].sum():+.4f} (ER NaN {pool['er_entry'].isna().sum()} 件除く)")
    rho, p_rho = stats.spearmanr(er_ok["er_entry"], er_ok["ret"])
    print(f"  Spearman(er_entry, ret) = {rho:+.3f} (p={p_rho:.3f})")

    # =========================================================================
    print("\n" + "=" * 100)
    print("Q2. 時代分割(決済日基準): 2016-2021(低金利) vs 2022-2026(利上げ以降)")
    pool["era"] = np.where(pool["exit"] < pd.Timestamp("2022-01-01", tz="UTC"),
                           "2016-2021", "2022-2026")
    t2 = grp_stats(pool, "era")
    # 年あたり純益(サンプル実期間で割る)
    yrs1 = (pd.Timestamp("2022-01-01", tz="UTC") - pool["entry"].min()).days / 365.25
    yrs2 = (pool["exit"].max() - pd.Timestamp("2022-01-01", tz="UTC")).days / 365.25
    t2["years"] = [yrs1, yrs2]
    t2["net_per_year"] = t2["sum_ret"] / t2["years"]
    t2["trades_per_year"] = t2["n"] / t2["years"]
    print(t2.to_string())
    print(f"  検算: 時代合計 = {t2['sum_ret'].sum():+.4f}")

    # トレード単位 ret の時間線形トレンド(exit 時刻を年に換算)
    t_yr = (pool["exit"] - pool["exit"].min()).dt.total_seconds() / (365.25 * 24 * 3600)
    lr = stats.linregress(t_yr, pool["ret"])
    print(f"  ret ~ time 線形回帰: slope={lr.slope*1e4:+.3f} bps/年 (p={lr.pvalue:.3f}, "
          f"R²={lr.rvalue**2:.4f})  切片={lr.intercept*1e4:+.1f}bps")
    rho2, p2 = stats.spearmanr(t_yr, pool["ret"])
    print(f"  Spearman(time, ret) = {rho2:+.3f} (p={p2:.3f})")
    ym = pool.groupby(pool["exit"].dt.year)["ret"].agg(["sum", "mean", "count"])
    ym["mean_bps"] = ym["mean"] * 1e4
    print("  年次(決済年):")
    print(ym[["count", "sum", "mean_bps"]].to_string())

    # =========================================================================
    print("\n" + "=" * 100)
    print("Q3. 危機/高ボラ窓: 窓内 vs 窓外")
    vol_p95 = vol_daily.quantile(0.95)
    hi_vol_days = set(vol_daily[vol_daily >= vol_p95].index)
    print(f"  全銘柄平均実現ボラ(日次) p95 閾値 = {vol_p95:.5f}(上位5% = {len(hi_vol_days)} 日)")
    hv = sorted(hi_vol_days)
    spans, s0, prev = [], hv[0], hv[0]
    for d in hv[1:]:
        if (d - prev).days > 7:
            spans.append((s0, prev)); s0 = d
        prev = d
    spans.append((s0, prev))
    big = sorted(spans, key=lambda t: -(t[1] - t[0]).days)[:6]
    print("  上位5%日の主なクラスタ:", ", ".join(f"{a.date()}〜{b.date()}" for a, b in sorted(big)))

    windows = {
        "COVID 2020-02〜04": ("2020-02-01", "2020-05-01"),
        "2022-09〜10": ("2022-09-01", "2022-11-01"),
    }
    pnl_daily = build_daily_mtm(pool, closes)   # MtM 日次損益(等加重)
    print(f"  MtM 日次損益 検算: 合計 = {pnl_daily.sum():+.4f}(= sum(ret) と一致必須)")
    assert abs(pnl_daily.sum() - total) < 1e-9

    rows = []
    for name, (a, b) in windows.items():
        a_, b_ = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        in_e = pool[(pool["entry"] >= a_) & (pool["entry"] < b_)]          # entry 基準
        ov = pool[(pool["entry"] < b_) & (pool["exit"] >= a_)]             # 保有重複基準
        mtm = pnl_daily[(pnl_daily.index >= a_) & (pnl_daily.index < b_)].sum()
        rows.append({"window": name, "def": "entry内", "n": len(in_e),
                     "sum_ret": in_e["ret"].sum(), "mean_bps": in_e["ret"].mean() * 1e4 if len(in_e) else np.nan,
                     "win_%": (in_e["ret"] > 0).mean() * 100 if len(in_e) else np.nan, "mtm_in_window": mtm})
        rows.append({"window": name, "def": "保有重複", "n": len(ov),
                     "sum_ret": ov["ret"].sum(), "mean_bps": ov["ret"].mean() * 1e4 if len(ov) else np.nan,
                     "win_%": (ov["ret"] > 0).mean() * 100 if len(ov) else np.nan, "mtm_in_window": mtm})
    # 高ボラ上位5%日(entry 基準)
    in_hv = pool[pool["entry"].dt.floor("D").isin(hi_vol_days)]
    out_hv = pool[~pool["entry"].dt.floor("D").isin(hi_vol_days)]
    mtm_hv = pnl_daily[pnl_daily.index.isin(hi_vol_days)].sum()
    rows.append({"window": "実現ボラ上位5%日", "def": "entry内", "n": len(in_hv),
                 "sum_ret": in_hv["ret"].sum(), "mean_bps": in_hv["ret"].mean() * 1e4,
                 "win_%": (in_hv["ret"] > 0).mean() * 100, "mtm_in_window": mtm_hv})
    rows.append({"window": "同・窓外", "def": "entry外", "n": len(out_hv),
                 "sum_ret": out_hv["ret"].sum(), "mean_bps": out_hv["ret"].mean() * 1e4,
                 "win_%": (out_hv["ret"] > 0).mean() * 100, "mtm_in_window": total - mtm_hv})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"  検算(上位5%日 entry 基準): 窓内+窓外 = {in_hv['ret'].sum()+out_hv['ret'].sum():+.4f}")

    # =========================================================================
    print("\n" + "=" * 100)
    print("Q4. ブリード窓の解剖(等加重 MtM 日次損益、累積カーブのトップ5ドローダウン)")
    cum = pnl_daily.cumsum()
    runmax = cum.cummax()
    dd = cum - runmax
    # ドローダウンのエピソード分割(新高値間)
    episodes = []
    in_dd, start = False, None
    for d in dd.index:
        if dd.loc[d] < -1e-12 and not in_dd:
            in_dd, start = True, d
        elif dd.loc[d] >= -1e-12 and in_dd:
            seg = dd.loc[start:d]
            episodes.append((start, seg.idxmin(), d, seg.min()))
            in_dd = False
    if in_dd:
        seg = dd.loc[start:]
        episodes.append((start, seg.idxmin(), None, seg.min()))
    top5 = sorted(episodes, key=lambda e: e[3])[:5]

    q4 = []
    vol_med_all = vol_daily.median()
    er_med_all = er_daily.median()
    for (s, trough, rec, depth) in sorted(top5, key=lambda e: e[0]):
        win_vol = vol_daily.loc[s:trough].mean()
        win_er = er_daily.loc[s:trough].mean()
        usd_chg = (usd_idx_daily.loc[:trough].iloc[-1] - usd_idx_daily.loc[:s].iloc[-1]) * 100
        # 窓内の最悪寄与銘柄
        contrib = trade_window_contrib(pool, closes, s, trough)
        worst3 = ", ".join(f"{k} {v:+.3f}" for k, v in contrib.head(3).items())
        q4.append({
            "peak": s.date(), "trough": trough.date(),
            "recover": rec.date() if rec is not None else "未回復",
            "depth": depth, "depth_%of_total": depth / total * 100,
            "len_days": (trough - s).days,
            "avg_vol": win_vol, "vol_pctile": (vol_daily < win_vol).mean() * 100,
            "avg_ER": win_er, "ER_pctile": (er_daily < win_er).mean() * 100,
            "USDidx_chg_%": usd_chg, "worst_instr(MtM)": worst3,
        })
    q4df = pd.DataFrame(q4)
    print(q4df.to_string(index=False))
    print(f"  (全期間中央値: 平均ボラ={vol_med_all:.5f}, 平均ER={er_med_all:.3f})")

    # =========================================================================
    print("\n" + "=" * 100)
    print("Q5. 通貨レッグ寄与(各トレードの ret を base/quote に 0.5 ずつ配賦)")
    pool["fz"] = f_z(pool["z_entry"].to_numpy())
    pool["ret_w"] = pool["ret"] * pool["fz"] / pool["fz"].mean()
    legs = []
    for _, r in pool.iterrows():
        base, quote = r["instr"][:3], r["instr"][3:]
        # dir=+1(ペアをロング)= base をロング, quote をショート
        legs.append((base, "long" if r["dir"] == 1 else "short", 0.5 * r["ret"], 0.5 * r["ret_w"]))
        legs.append((quote, "short" if r["dir"] == 1 else "long", 0.5 * r["ret"], 0.5 * r["ret_w"]))
    ldf = pd.DataFrame(legs, columns=["ccy", "side", "ret05", "ret05_w"])
    t5 = ldf.pivot_table(index="ccy", columns="side", values="ret05", aggfunc="sum").fillna(0)
    t5["total"] = t5.sum(axis=1)
    t5["share_%"] = t5["total"] / total * 100
    t5w = ldf.pivot_table(index="ccy", columns="side", values="ret05_w", aggfunc="sum").fillna(0)
    t5["total_fz"] = t5w.sum(axis=1)
    t5["share_fz_%"] = t5["total_fz"] / pool["ret_w"].sum() * 100
    t5["n_legs"] = ldf.groupby("ccy").size()
    print(t5.sort_values("total", ascending=False).to_string())
    print(f"  検算: 通貨合計 = {t5['total'].sum():+.4f}(等加重) / {t5['total_fz'].sum():+.4f}"
          f"(f(z)加重, 全体={pool['ret_w'].sum():+.4f})")

    # =========================================================================
    print("\n" + "=" * 100)
    print("Q6. マクロ感応度: 月次戦略リターン(決済月) × (a)|USD指数月次変化| (b)平均実現ボラ")
    mret = pool.groupby(pool["exit"].dt.to_period("M"))["ret"].sum()
    all_months = pd.period_range(pool["entry"].min(), pool["exit"].max(), freq="M")
    mret = mret.reindex(all_months, fill_value=0.0)
    usd_m = usd_idx_daily.groupby(usd_idx_daily.index.to_period("M")).last()
    usd_chg_abs = usd_m.diff().abs().reindex(all_months) * 100
    vol_m = vol_daily.groupby(vol_daily.index.to_period("M")).mean().reindex(all_months)
    dfm = pd.DataFrame({"strat": mret, "abs_usd_chg": usd_chg_abs, "vol": vol_m}).dropna()
    print(f"  月数 = {len(dfm)}(取引なし月は 0 計上)  月次平均 = {dfm['strat'].mean()*1e4:+.1f}bps")
    for cn in ["abs_usd_chg", "vol"]:
        pr, pp = stats.pearsonr(dfm["strat"], dfm[cn])
        sr, sp = stats.spearmanr(dfm["strat"], dfm[cn])
        print(f"  corr(strat, {cn:12s}): Pearson {pr:+.3f} (p={pp:.4f}) / Spearman {sr:+.3f} (p={sp:.4f})")
    # ボラ月の三分位で戦略月次リターンを比較(ショートボラ性の素朴な見方)
    dfm["vol_t"] = pd.qcut(dfm["vol"], 3, labels=["低ボラ月", "中", "高ボラ月"])
    print(dfm.groupby("vol_t", observed=True)["strat"]
          .agg(n="count", mean_bps=lambda s: s.mean() * 1e4, sum="sum").to_string())
    dfm["usd_t"] = pd.qcut(dfm["abs_usd_chg"], 3, labels=["USD静穏月", "中", "USD大変動月"])
    print(dfm.groupby("usd_t", observed=True)["strat"]
          .agg(n="count", mean_bps=lambda s: s.mean() * 1e4, sum="sum").to_string())


# --- MtM 日次損益(等加重)。トレード合計=ret を厳密保証 ---------------------
_DAILY_CACHE: dict[int, pd.Series] = {}


def build_daily_mtm(pool: pd.DataFrame, closes: pd.DataFrame) -> pd.Series:
    if 0 in _DAILY_CACHE:
        return _DAILY_CACHE[0]
    grid = closes.index
    gi = grid.to_numpy()
    n = len(grid)
    pnl_bar = np.zeros(n)
    carr = closes.to_numpy()
    col_of = {c: i for i, c in enumerate(closes.columns)}
    e_pos = np.clip(np.searchsorted(gi, pool["entry"].to_numpy(), side="left"), 0, n - 1)
    x_pos = np.clip(np.searchsorted(gi, pool["exit"].to_numpy(), side="left"), 0, n - 1)
    for i in range(len(pool)):
        ep, xp = int(e_pos[i]), int(x_pos[i])
        col = col_of[pool["instr"].iat[i]]
        d, p0, ret = pool["dir"].iat[i], pool["entry_price"].iat[i], pool["ret"].iat[i]
        if xp <= ep:
            pnl_bar[xp] += ret
            continue
        path = carr[ep:xp + 1, col]
        marks = d * (path - p0) / p0           # entry_price 起点の評価損益
        pnl_bar[ep] += marks[0]                 # エントリーバー: 約定価格→当該バーclose(入口コスト計上)
        inc = np.diff(marks)                    # バー毎の増分(ep+1 .. xp)
        inc[-1] = ret - marks[-2]               # 決済バー: 合計=ret になる残差(出口コスト込み)
        pnl_bar[ep + 1: xp + 1] += inc
    s = pd.Series(pnl_bar, index=grid).groupby(grid.floor("D")).sum()
    _DAILY_CACHE[0] = s
    return s


def trade_window_contrib(pool: pd.DataFrame, closes: pd.DataFrame, a, b) -> pd.Series:
    """窓 [a,b] 内の銘柄別 MtM 寄与(build_daily_mtm と同じ規約、銘柄別に集計)。"""
    grid = closes.index
    gi = grid.to_numpy()
    n = len(grid)
    carr = closes.to_numpy()
    col_of = {c: i for i, c in enumerate(closes.columns)}
    e_pos = np.clip(np.searchsorted(gi, pool["entry"].to_numpy(), side="left"), 0, n - 1)
    x_pos = np.clip(np.searchsorted(gi, pool["exit"].to_numpy(), side="left"), 0, n - 1)
    in_win = (grid >= a) & (grid <= b + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    out: dict[str, float] = {}
    for i in range(len(pool)):
        ep, xp = int(e_pos[i]), int(x_pos[i])
        nm = pool["instr"].iat[i]
        d, p0, ret = pool["dir"].iat[i], pool["entry_price"].iat[i], pool["ret"].iat[i]
        if xp <= ep:
            if in_win[xp]:
                out[nm] = out.get(nm, 0.0) + ret
            continue
        path = carr[ep:xp + 1, col_of[nm]]
        marks = d * (path - p0) / p0
        acc = float(marks[0]) if in_win[ep] else 0.0
        inc = np.diff(marks)
        inc[-1] = ret - marks[-2]
        mask = in_win[ep + 1: xp + 1]
        if mask.any():
            acc += float(inc[mask].sum())
        if acc != 0.0:
            out[nm] = out.get(nm, 0.0) + acc
    return pd.Series(out).sort_values()


if __name__ == "__main__":
    main()
