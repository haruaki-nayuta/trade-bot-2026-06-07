"""anatomy_time.py — チャンピオン(confluence_meanrev_v2)利益の時間構造の解剖
+ ロールオーバー・アーティファクト(UTC20-22)の H4 汚染監査。

リサンプル規約(fxlab/data.py L62 で確認済み):
  resample(rule, label="left", closed="left") → H4 バーのラベルは「開始時刻」
  (0/4/8/12/16/20)。close はバー内最後の M1 close(20:00 バーなら 23:5x)。
  約定=シグナルバー終値=この 23:5x close ± 半スプレッド。

実行: uv run python -m research.experiments.anatomy_time
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import config, universe as uni

try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

POOL_PATH = config.RESULTS_DIR / "mm_pool_v2_H4_19.parquet"
TOTAL_BASE = 1.9086  # ベースライン総純益(検算用)


def sec(title: str) -> None:
    print("\n" + "=" * 100)
    print(f"## {title}")
    print("=" * 100)


def fmt_share(x: float) -> str:
    return f"{x:+.4f} ({x / TOTAL_BASE * 100:+.1f}% of total)"


def zpower(z: np.ndarray) -> np.ndarray:
    """本番 z-power サイジング f(z) = clip((|z|/2.2)^4.0, 0.3, 3.0)."""
    return np.clip((np.abs(z) / 2.2) ** 4.0, 0.3, 3.0)


def group_table(df: pd.DataFrame, key, ret_col="ret") -> pd.DataFrame:
    g = df.groupby(key)[ret_col]
    out = pd.DataFrame({
        "n": g.size(),
        "sum_ret": g.sum(),
        "mean_bps": g.mean() * 1e4,
        "win_rate": g.apply(lambda s: (s > 0).mean()),
        "share_%": g.sum() / TOTAL_BASE * 100,
    })
    return out


def pf_of(s: pd.Series) -> float:
    gains = s[s > 0].sum()
    losses = -s[s < 0].sum()
    return gains / losses if losses > 0 else np.inf


# ---------------------------------------------------------------- M1 access
_M1_CACHE: dict[str, pd.Series] = {}


def m1_close(name: str) -> pd.Series:
    """メジャーは実 M1 close、クロスは脚の M1 close を inner-join で合成。"""
    if name in _M1_CACHE:
        return _M1_CACHE[name]
    if name in uni.CROSS_DEFS:
        a, op, b = uni.CROSS_DEFS[name]
        ca, cb = m1_close(a), m1_close(b)
        df = pd.concat([ca.rename("a"), cb.rename("b")], axis=1, join="inner").dropna()
        s = df["a"] / df["b"] if op == "/" else df["a"] * df["b"]
    else:
        s = pd.read_parquet(config.DATA_DIR / f"{name}_M1.parquet", columns=["close"])["close"]
    _M1_CACHE[name] = s
    return s


def exec_and_window(idx_vals: np.ndarray, closes: np.ndarray, bar_start: pd.Timestamp,
                    win_min: int = 60):
    """バー [start, start+4h) 内最後の M1 close(=約定)と、その後 win_min 分の M1 close 群。

    返り値: (exec_time, exec_close, window_closes(時間窓), window_closes_60bars(位置窓))
    """
    t0 = np.datetime64(bar_start.tz_convert(None))
    t1 = np.datetime64((bar_start + pd.Timedelta(hours=4)).tz_convert(None))
    i0 = np.searchsorted(idx_vals, t0, side="left")
    i1 = np.searchsorted(idx_vals, t1, side="left")
    if i1 <= i0:
        return None, np.nan, np.array([]), np.array([])
    iexec = i1 - 1
    texec = idx_vals[iexec]
    pexec = closes[iexec]
    tw = texec + np.timedelta64(win_min, "m")
    j1 = np.searchsorted(idx_vals, tw, side="right")
    w_time = closes[iexec + 1: j1]
    w_bars = closes[iexec + 1: iexec + 1 + 60]
    return texec, pexec, w_time, w_bars


# =================================================================== main
def main() -> None:
    df = pd.read_parquet(POOL_PATH).copy()
    df["entry"] = pd.to_datetime(df["entry"], utc=True)
    df["exit"] = pd.to_datetime(df["exit"], utc=True)
    df["e_hour"] = df["entry"].dt.hour
    df["x_hour"] = df["exit"].dt.hour
    df["e_dow"] = df["entry"].dt.dayofweek  # 0=Mon
    df["e_month"] = df["entry"].dt.month
    df["x_year"] = df["exit"].dt.year
    df["side"] = np.where(df["dir"] > 0, "long", "short")

    sec("0. ベースライン検算(タスク前提と一致するか)")
    tot = df["ret"].sum()
    print(f"n={len(df)}  sum(ret)={tot:+.4f}  mean={df['ret'].mean()*1e4:+.2f}bps  "
          f"win={(df['ret']>0).mean()*100:.1f}%  PF={pf_of(df['ret']):.3f}  "
          f"med_hold={df['bars_held'].median():.0f}bars  long%={(df['dir']>0).mean()*100:.1f}")
    ys = df.groupby("x_year")["ret"].sum()
    print("年次sum(決済年):")
    print(ys.round(3).to_string())
    assert abs(tot - TOTAL_BASE) < 1e-3, "ベースライン不一致"

    # ------------------------------------------------------------- Q1
    sec("Q1. エントリーバー開始時刻別(0/4/8/12/16/20)の成績")
    print(group_table(df, "e_hour").round(3).to_string())
    print("\n-- ロング/ショート別 --")
    print(group_table(df, ["e_hour", "side"]).round(3).to_string())
    h20 = df[df["e_hour"] == 20]
    print(f"\n20:00バー起点: n={len(h20)}  sum={fmt_share(h20['ret'].sum())}  "
          f"long n={len(h20[h20.dir>0])} sum={h20[h20.dir>0]['ret'].sum():+.4f} / "
          f"short n={len(h20[h20.dir<0])} sum={h20[h20.dir<0]['ret'].sum():+.4f}")
    # z-power 加重での寄与
    w = zpower(df["z_entry"].to_numpy())
    df["ret_w"] = df["ret"] * w / w.mean()
    tw_ = df["ret_w"].sum()
    print(f"\nz-power加重(本番ウェイト近似): 加重総益={tw_:+.4f}  "
          f"20:00バー寄与={df.loc[df.e_hour==20,'ret_w'].sum():+.4f} "
          f"({df.loc[df.e_hour==20,'ret_w'].sum()/tw_*100:+.1f}% of weighted)")
    print("\n検算: 時刻別sumの合計 =", round(df.groupby("e_hour")["ret"].sum().sum(), 4))

    # 曜日×時刻クロス表(日曜20時=薄商いバーの混入確認)
    print("\n-- エントリー曜日×時刻 件数(0=月…6=日) --")
    print(pd.crosstab(df["e_dow"], df["e_hour"]).to_string())

    # ------------------------------------------------------------- Q2
    sec("Q2. イグジットバー開始時刻別の成績(BID低下アーティファクトの署名探索)")
    print(group_table(df, "x_hour").round(3).to_string())
    print("\n-- ロング/ショート別 --")
    print(group_table(df, ["x_hour", "side"]).round(3).to_string())
    sig_le = df[(df.e_hour == 20) & (df.dir > 0)]
    sig_sx = df[(df.x_hour == 20) & (df.dir < 0)]
    print(f"\n署名チェック: ロングの20時エントリー n={len(sig_le)} sum={fmt_share(sig_le['ret'].sum())} "
          f"mean={sig_le['ret'].mean()*1e4:+.1f}bps")
    print(f"            ショートの20時イグジット n={len(sig_sx)} sum={fmt_share(sig_sx['ret'].sum())} "
          f"mean={sig_sx['ret'].mean()*1e4:+.1f}bps")
    print(f"参考: 全ロング mean={df[df.dir>0]['ret'].mean()*1e4:+.1f}bps / "
          f"全ショート mean={df[df.dir<0]['ret'].mean()*1e4:+.1f}bps")

    # ------------------------------------------------------------- Q3
    sec("Q3. 1本目グロスリターン(即時リバウンド)の時刻別比較 + 除染テスト")
    # H4 close で entry バー → 次バーのグロスリターン(方向符号付き)
    uni.register_cross_spreads(3.0)
    g1 = np.full(len(df), np.nan)
    for nm, sub in df.groupby("instr"):
        c = uni.instrument_close(nm, "H4")
        idx = c.index
        pos = idx.get_indexer(sub["entry"])
        if (pos < 0).any():
            bad = (pos < 0).sum()
            print(f"[warn] {nm}: {bad} entries not on H4 grid")
        nxt = np.where((pos >= 0) & (pos + 1 < len(c)), pos + 1, -1)
        vals = np.where(nxt >= 0,
                        sub["dir"].to_numpy() * (c.to_numpy()[nxt] / c.to_numpy()[pos] - 1.0),
                        np.nan)
        g1[sub.index] = vals
    df["gross1"] = g1
    t = df.groupby("e_hour")["gross1"].agg(["count", "mean", "median", "std"])
    t[["mean", "median", "std"]] *= 1e4
    print("1本目グロスリターン(bps, 方向符号付き) by entry hour:")
    print(t.round(2).to_string())
    a = df.loc[df.e_hour == 20, "gross1"].dropna()
    b = df.loc[df.e_hour != 20, "gross1"].dropna()
    print(f"\n20時 vs 他: mean {a.mean()*1e4:+.2f} vs {b.mean()*1e4:+.2f} bps")
    if HAVE_SCIPY:
        tt = sps.ttest_ind(a, b, equal_var=False)
        mw = sps.mannwhitneyu(a, b, alternative="two-sided")
        print(f"Welch t: t={tt.statistic:+.2f} p={tt.pvalue:.4f} | MWU p={mw.pvalue:.4f}")
        al = df.loc[(df.e_hour == 20) & (df.dir > 0), "gross1"].dropna()
        bl = df.loc[(df.e_hour != 20) & (df.dir > 0), "gross1"].dropna()
        ttl = sps.ttest_ind(al, bl, equal_var=False)
        print(f"ロング限定: 20時 n={len(al)} mean={al.mean()*1e4:+.2f} vs 他 n={len(bl)} "
              f"mean={bl.mean()*1e4:+.2f} bps | Welch t p={ttl.pvalue:.4f}")
        ash = df.loc[(df.e_hour == 20) & (df.dir < 0), "gross1"].dropna()
        bsh = df.loc[(df.e_hour != 20) & (df.dir < 0), "gross1"].dropna()
        tts = sps.ttest_ind(ash, bsh, equal_var=False)
        print(f"ショート限定: 20時 n={len(ash)} mean={ash.mean()*1e4:+.2f} vs 他 n={len(bsh)} "
              f"mean={bsh.mean()*1e4:+.2f} bps | Welch t p={tts.pvalue:.4f}")

    print("\n-- 除染テスト: 20:00バーエントリーを全除外 --")
    dec = df[df.e_hour != 20]
    print(f"除染後: n={len(dec)}  sum={dec['ret'].sum():+.4f} "
          f"(除外分 {fmt_share(h20['ret'].sum())})  mean={dec['ret'].mean()*1e4:+.2f}bps  "
          f"win={(dec['ret']>0).mean()*100:.1f}%  PF={pf_of(dec['ret']):.3f}")
    ysd = dec.groupby("x_year")["ret"].sum()
    cmp_y = pd.DataFrame({"base": ys, "decon": ysd, "diff": ysd - ys})
    print(cmp_y.round(3).to_string())
    print("全暦年プラス維持:", bool((ysd > 0).all()))

    # ------------------------------------------------------------- Q4
    sec("Q4. M1 精査: 約定価格 vs その後60分の M1 close 中央値(全件集計+20時サンプル10件)")
    # 全トレードについて entry 側 dev を計測(20時だけでなく全時間帯=対照群)
    ent_dev_t = np.full(len(df), np.nan)   # 時間窓60分
    ent_dev_b = np.full(len(df), np.nan)   # 位置窓60本
    ent_nwin = np.zeros(len(df))
    exit_dev_t = np.full(len(df), np.nan)  # 出口側: dir*(p_exit/median - 1)
    exec_info: dict[int, tuple] = {}
    for nm, sub in df.groupby("instr"):
        s = m1_close(nm)
        iv = s.index.values
        cv = s.to_numpy()
        for i, row in sub.iterrows():
            te, pe, wt, wb = exec_and_window(iv, cv, row["entry"])
            if te is not None and len(wt) >= 10:
                med = np.median(wt)
                ent_dev_t[i] = row["dir"] * (med / pe - 1.0)
            if te is not None and len(wb) >= 10:
                ent_dev_b[i] = row["dir"] * (np.median(wb) / pe - 1.0)
            ent_nwin[i] = len(wt)
            exec_info[i] = (te, pe, np.median(wt) if len(wt) else np.nan)
            tx, px, wxt, _ = exec_and_window(iv, cv, row["exit"])
            if tx is not None and len(wxt) >= 10:
                exit_dev_t[i] = row["dir"] * (px / np.median(wxt) - 1.0)
        # メジャーのキャッシュは残し、クロス合成は捨てる(メモリ)
        if nm in uni.CROSS_DEFS:
            _M1_CACHE.pop(nm, None)
    df["ent_dev_t"] = ent_dev_t
    df["ent_dev_b"] = ent_dev_b
    df["exit_dev_t"] = exit_dev_t

    print("entry側 dev = dir*(60分後M1中央値/約定close - 1) [bps] 正=約定が有利方向に乖離")
    te_ = df.groupby("e_hour")["ent_dev_t"].agg(["count", "mean", "median"])
    te_[["mean", "median"]] *= 1e4
    print(te_.round(2).to_string())
    print("\nexit側 dev = dir*(約定close/60分後M1中央値 - 1) [bps] 正=決済が有利方向に乖離")
    tx_ = df.groupby("x_hour")["exit_dev_t"].agg(["count", "mean", "median"])
    tx_[["mean", "median"]] *= 1e4
    print(tx_.round(2).to_string())
    if HAVE_SCIPY:
        a4 = df.loc[df.e_hour == 20, "ent_dev_t"].dropna()
        b4 = df.loc[df.e_hour != 20, "ent_dev_t"].dropna()
        tt4 = sps.ttest_ind(a4, b4, equal_var=False)
        print(f"\nentry dev 20時 vs 他: {a4.mean()*1e4:+.2f} vs {b4.mean()*1e4:+.2f} bps, "
              f"Welch p={tt4.pvalue:.4f}")
        a5 = df.loc[(df.x_hour == 20) & (df.dir < 0), "exit_dev_t"].dropna()
        b5 = df.loc[(df.x_hour != 20) & (df.dir < 0), "exit_dev_t"].dropna()
        if len(a5) >= 5:
            tt5 = sps.ttest_ind(a5, b5, equal_var=False)
            print(f"ショート exit dev 20時 vs 他: {a5.mean()*1e4:+.2f} vs {b5.mean()*1e4:+.2f} bps, "
                  f"Welch p={tt5.pvalue:.4f}")

    print("\n-- 20:00バーエントリーのサンプル10件(時系列で等間隔抽出) --")
    s20 = df[df.e_hour == 20].sort_values("entry")
    pick = s20.iloc[np.unique(np.linspace(0, len(s20) - 1, 10).astype(int))]
    rows = []
    for i, row in pick.iterrows():
        te, pe, med = exec_info[i]
        rows.append({
            "instr": row["instr"], "entry_bar": str(row["entry"])[:16], "dir": row["dir"],
            "exec_t(M1)": str(pd.Timestamp(te))[:16] if te is not None else "-",
            "exec_close": pe, "entry_price(pool)": row["entry_price"],
            "med60m": med,
            "dev_bps": row["ent_dev_t"] * 1e4 if np.isfinite(row["ent_dev_t"]) else np.nan,
            "n_win": int(ent_nwin[i]), "ret_bps": row["ret"] * 1e4,
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ------------------------------------------------------------- Q4b
    sec("Q4b. 20:00バー深掘り: 約定時刻の内訳とアーティファクト寄与のbps定量化")
    # 約定実時刻(exec_info から)。金曜・祝日の20:00バーは21:5x クローズ=アーティファクト窓内
    exec_t = pd.Series({i: v[0] for i, v in exec_info.items()})
    df["exec_hour"] = pd.to_datetime(exec_t.reindex(df.index).values).hour
    d20 = df[df.e_hour == 20].copy()
    d20["exec_in_2022"] = d20["exec_hour"].isin([20, 21, 22])
    d20["dow_name"] = d20["e_dow"].map({0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 6: "Sun"})
    agg = d20.groupby(["dow_name", "exec_in_2022"]).agg(
        n=("ret", "size"), sum_ret=("ret", "sum"), mean_ret_bps=("ret", lambda s: s.mean() * 1e4),
        mean_g1_bps=("gross1", lambda s: s.mean() * 1e4),
        mean_dev_t=("ent_dev_t", lambda s: s.mean() * 1e4),
        mean_dev_b=("ent_dev_b", lambda s: s.mean() * 1e4),
    )
    print("20時エントリーの曜日×(約定がUTC20-22窓内か)別:")
    print(agg.round(2).to_string())
    print("\n20時エントリーの約定時刻分布:", d20["exec_hour"].value_counts().sort_index().to_dict())

    # 位置窓60本ベースの dev by hour(金曜21:5xクローズ分も含む=週末ギャップ込み)
    tb_ = df.groupby("e_hour")["ent_dev_b"].agg(["count", "mean", "median"])
    tb_[["mean", "median"]] *= 1e4
    print("\nentry側 dev(位置窓60本, 週末跨ぎ含む) by entry hour [bps]:")
    print(tb_.round(2).to_string())

    # アーティファクト寄与の点推定: (20時devの超過) × 件数
    dev20 = df.loc[df.e_hour == 20, "ent_dev_t"].dropna()
    devR = df.loc[df.e_hour != 20, "ent_dev_t"].dropna()
    ex_all = (dev20.mean() - devR.mean()) * len(df[df.e_hour == 20])
    dev20L = df.loc[(df.e_hour == 20) & (df.dir > 0), "ent_dev_t"].dropna()
    devRL = df.loc[(df.e_hour != 20) & (df.dir > 0), "ent_dev_t"].dropna()
    dev20S = df.loc[(df.e_hour == 20) & (df.dir < 0), "ent_dev_t"].dropna()
    devRS = df.loc[(df.e_hour != 20) & (df.dir < 0), "ent_dev_t"].dropna()
    ex_long = (dev20L.mean() - devRL.mean()) * len(df[(df.e_hour == 20) & (df.dir > 0)])
    ex_short = (dev20S.mean() - devRS.mean()) * len(df[(df.e_hour == 20) & (df.dir < 0)])
    g20 = df.loc[df.e_hour == 20, "gross1"].dropna()
    gR = df.loc[df.e_hour != 20, "gross1"].dropna()
    ex_g1 = (g20.mean() - gR.mean()) * len(df[df.e_hour == 20])
    xd20 = df.loc[df.x_hour == 20, "exit_dev_t"].dropna()
    xdR = df.loc[df.x_hour != 20, "exit_dev_t"].dropna()
    ex_exit = (xd20.mean() - xdR.mean()) * len(df[df.x_hour == 20])
    print("\nアーティファクト寄与の点推定(超過dev×件数):")
    print(f"  entry側(全体, M1 60分窓):  {fmt_share(ex_all)}")
    print(f"    内訳 ロング: {ex_long:+.4f} / ショート: {ex_short:+.4f}")
    print(f"  entry側(1本目グロス基準):  {fmt_share(ex_g1)}")
    print(f"  exit側(M1 60分窓):        {fmt_share(ex_exit)}")
    print(f"  合計(entry M1 + exit M1): {fmt_share(ex_all + ex_exit)}")
    print(f"\n参考: 20時エントリーの平均ret {d20['ret'].mean()*1e4:+.1f}bps のうち "
          f"dev超過は {dev20.mean()*1e4 - devR.mean()*1e4:+.1f}bps")

    # 日曜20時(週明けオープンバー=薄商い)単独
    sun = df[(df.e_hour == 20) & (df.e_dow == 6)]
    print(f"\n日曜20時バー(週明けオープン, 実質21-24時の薄商いバー): n={len(sun)} "
          f"sum={fmt_share(sun['ret'].sum())} mean={sun['ret'].mean()*1e4:+.1f}bps "
          f"win={(sun['ret']>0).mean()*100:.0f}%")

    # ------------------------------------------------------------- Q5
    sec("Q5. 曜日・週末持ち越し・月別の季節性")
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    tdow = group_table(df, "e_dow").round(3)
    tdow.index = [dow_names[i] for i in tdow.index]
    print("-- エントリー曜日別 --")
    print(tdow.to_string())
    fri = df[df.e_dow == 4]
    print(f"\n金曜エントリー: n={len(fri)} sum={fmt_share(fri['ret'].sum())} "
          f"mean={fri['ret'].mean()*1e4:+.1f}bps win={(fri['ret']>0).mean()*100:.1f}%")

    # 週末持ち越し = entry と exit の間に土曜が1日以上挟まる
    def n_weekends(e: pd.Timestamp, x: pd.Timestamp) -> int:
        days = pd.date_range(e.normalize() + pd.Timedelta(days=1), x.normalize(), freq="D")
        return int((days.dayofweek == 5).sum())

    df["n_wknd"] = [n_weekends(e, x) for e, x in zip(df["entry"], df["exit"])]
    df["held_wknd"] = df["n_wknd"] > 0
    print("\n-- 週末持ち越し有無別 --")
    print(group_table(df, "held_wknd").round(3).to_string())
    print("\n-- 持ち越し週末回数別 --")
    print(group_table(df, "n_wknd").round(3).to_string())

    print("\n-- エントリー月別 --")
    print(group_table(df, "e_month").round(3).to_string())
    aug, dec_m = df[df.e_month == 8], df[df.e_month == 12]
    print(f"\n8月: n={len(aug)} sum={fmt_share(aug['ret'].sum())} mean={aug['ret'].mean()*1e4:+.1f}bps")
    print(f"12月: n={len(dec_m)} sum={fmt_share(dec_m['ret'].sum())} mean={dec_m['ret'].mean()*1e4:+.1f}bps")
    print("\n検算: 月別sumの合計 =", round(df.groupby("e_month")["ret"].sum().sum(), 4))

    # ------------------------------------------------------------- Q6
    sec("Q6. 統合: アーティファクト疑い寄与の定量化")
    sus_e = df[(df.e_hour == 20)]
    sus_x = df[(df.x_hour == 20) & (df.e_hour != 20)]
    both = df[(df.e_hour == 20) | (df.x_hour == 20)]
    print(f"20時エントリー: {fmt_share(sus_e['ret'].sum())}")
    print(f"20時イグジット(エントリー20時以外): {fmt_share(sus_x['ret'].sum())}")
    print(f"20時が入口または出口: {fmt_share(both['ret'].sum())}")
    strict = df[(df.e_hour != 20) & (df.x_hour != 20)]
    print(f"\n最厳格除染(入口も出口も20時を含むトレードを全除外): n={len(strict)} "
          f"sum={strict['ret'].sum():+.4f} PF={pf_of(strict['ret']):.3f} "
          f"win={(strict['ret']>0).mean()*100:.1f}%")
    yss = strict.groupby("x_year")["ret"].sum()
    print("年次(最厳格):")
    print(yss.round(3).to_string())
    print("全暦年プラス維持:", bool((yss > 0).all()))


if __name__ == "__main__":
    main()
