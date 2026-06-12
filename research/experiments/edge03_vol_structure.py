"""edge03: ボラ構造のエッジケース監査(コスト算術レンズ)— d1 チャンピオンプール。

問い: 「ボラがおかしい」局面のエントリーで、エッジがスプレッドコストに構造的に
食われている帯はないか? これは負け予測(閉鎖済み, reports/20)ではなく**コスト算術**:
期待戻り幅はボラ(std50)に比例する一方スプレッドは固定 → 極端な低ボラでは
net エッジが構造的に薄い。機構が先にあり、データで定量するだけ=カーブフィットでない。

計測(すべてシグナルバー = entry-1本 の因果値のみ):
  1. 期待グロス戻り幅%: (|z_entry| - 0.5) × std50 / close  (z が exit 閾値 0.5 まで
     戻った場合の値幅。rolling mean/std 固定の近似)
  2. 往復コスト%: SPREADS_PIPS[instr](クロスは register_cross_spreads(3.0)) × pip_size / close。
     fxlab/backtest._slippage_series は半スプレッド/close を entry/exit 両 fill に計上
     = 往復 1 スプレッド。ret_gross ≈ ret + cost_pct は半分が exit 価格基準である分の
     近似誤差(<数%相対)を含む。
  3. cost_ratio = コスト% / 期待戻り幅%。分位+固定閾値でコホート分解し、
     グロス/net 両方の成績を併記 → 「グロスでは勝つが net で負ける」帯の特定。
  4. ボラ異常系: vol20/vol100 比、vol-of-vol(vol20 の 20 本 CV)、vol20 の銘柄内
     ヒストリカル分位(expanding rank, 因果)の極端低(<5%)。
  5. 静穏ゲートぎわ: vol20 / Q70ゲート(rolling100 quantile 0.70)比。
     ぎりぎり通過(0.9-1.0)vs 深い静穏(<0.5)。

規約: コホートごとに n / 平均bps / 合計PnLシェア(対 +1.9622) / 勝率 / IS(-2021)・
OOS(2022-) / 年次分解(単年>50%で単年依存フラグ) / L・S 別 / ブートストラップCI(1000)。
n<30 は統計的に当てにならないと明記。プール段の解剖と veto 候補指名まで(採用判断なし)。

実行: uv run python research/experiments/edge03_vol_structure.py
出力: research/outputs/edge03_trades.csv / edge03_summary.json

結論(2026-06-13, n=1207 / +1.9622 検算一致):
  ★ 構造的コスト支配帯は**存在しない**。cost_ratio は max 0.180 / p99 0.115 で、
    指示の >0.3 / >0.5 帯は空(n=0)。機構的理由: エントリーは |z|>=2.0 クロス
    = 期待戻り幅が常に 1.5σ50 以上あり、calm ゲートは上からボラを切るだけで
    幅の下限(z 距離 × std50)はスプレッドの 5 倍超に保たれる。設計が自衛している。
  ★ プール全体でコストはグロス +2.219 の 11.6%(クロス 14.2% / メジャー 6.2%)。
    どの帯でも符号は反転しない。
  ・cost_ratio 五分位に勾配はある(Q1 +31.6bps vs Q5 +5.9bps、上位20% vs 残り
    差CI [-21,-5]bps で有意)が、**グロス側も同じだけ薄い**(32.5→9.2bps。コスト差は
    勾配 25.7bps 中わずか 2.5bps)= 「コストに食われる帯」でなく「もともと薄い帯」。
    Q5 自身は net 正・OOS(+10.5bps)>IS(+1.7bps)で veto 不適。エッジ薄帯の入口フィルタ化は
    閉鎖済み軸(reports/08/20)の言い換えなので提案しない。
  ・唯一の『グロス+/net−』帯 = cost_ratio 10-20%(n=26<30, net -0.0066 = シェア -0.3%,
    CI [-23,+13]bps, 単年フラグ)= ノイズ。複合コーナー(vol_hist<5% × cr上位20%, n=42)も
    net -0.011 / CI [-21.5,+14.3] / 2018 単年依存で不適。
  ・ボラ異常系(vol20/vol100 極端・vol-of-vol 極端・銘柄内ヒストリカル分位<5%・
    ゲートぎわ 0.9-1.0 vs 深い静穏<0.5)はすべて net 正または CI が 0 跨ぎ。
    vol_hist<5%(n=105, +3.1bps)は残りより有意に薄い(差CI [-27,-2.4]bps)が
    グロス差も同一 → コストでなくエッジの薄さ。IS/OOS とも正で veto 不適。
  ⇒ **veto 候補の指名なし**。コスト算術レンズではチャンピオンの数字は防衛された。
    注意: クロスのスプレッド 3pips 想定が実弾で2倍でも最悪帯 cost_ratio≈0.23 と
    支配閾値 0.3 に届かない(頭金あり)。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "money_management"))

from fxlab import config  # noqa: E402
from fxlab import universe as uni  # noqa: E402

OUT_DIR = ROOT / "research" / "outputs"
POOL_PATH = ROOT / "results" / "mm_pool_v2d1_H4_19.parquet"
SPLIT = pd.Timestamp("2022-01-01", tz="UTC")
TOTAL_EXPECTED = 1.9622
N_BOOT = 1000
EXIT_Z = 0.5  # チャンピオンの exit 閾値(出口は |z|→0.5 回帰)

pd.set_option("display.width", 230)


def sec(t):
    print("\n" + "=" * 96 + f"\n{t}\n" + "=" * 96)


# --------------------------------------------------------------------------
# シグナルバー特徴量(因果: feature.shift(1).reindex(entry) = entry-4h の確定値)
# --------------------------------------------------------------------------
def signal_features(pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for instr, g in pool.groupby("instr"):
        close = uni.instrument_data(instr, "H4")["close"]
        ret1 = close.pct_change()
        vol20 = ret1.rolling(20).std()
        vol100 = ret1.rolling(100).std()
        std50 = close.rolling(50).std()
        gate70 = vol20.rolling(100).quantile(0.70)          # 静穏ゲートの水準(戦略と同一定義)
        vov_cv = vol20.rolling(20).std() / vol20.rolling(20).mean()  # vol-of-vol(無次元)
        # 銘柄内ヒストリカル分位(expanding=因果)。min 250 本(約2ヶ月)
        vol_hist_pct = vol20.expanding(min_periods=250).rank(pct=True)

        def sig(s):  # シグナルバー(=entryの1本前)の値
            return s.shift(1).reindex(g["entry"]).to_numpy()

        spread_pct_at = (config.spread_pips(instr) * config.pip_size(instr)) / close
        rows.append(pd.DataFrame({
            "_idx": g.index,
            "sig_close": sig(close),
            "std50": sig(std50),
            "vol20": sig(vol20),
            "vol100": sig(vol100),
            "gate70": sig(gate70),
            "vov_cv": sig(vov_cv),
            "vol_hist_pct": sig(vol_hist_pct),
            "cost_pct": sig(spread_pct_at),   # 往復スプレッド% (固定spread/シグナル時close)
        }))
    f = pd.concat(rows).set_index("_idx").sort_index()
    f.index.name = None
    return f


# --------------------------------------------------------------------------
# 規約コホート統計
# --------------------------------------------------------------------------
def boot_ci_mean(x: np.ndarray, seed=0, n_boot=N_BOOT):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def boot_ci_diff(a: np.ndarray, b: np.ndarray, seed=0, n_boot=N_BOOT):
    """mean(a) - mean(b) の CI(独立リサンプル)。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    return tuple(np.percentile(da - db, [2.5, 97.5]))


def cohort_row(df: pd.DataFrame, mask: pd.Series, name: str) -> dict:
    sub = df[mask.fillna(False)]
    n = len(sub)
    if n == 0:
        return {"cohort": name, "n": 0}
    r = sub["ret"]
    g = sub["ret_gross"]
    is_m = sub["entry"] < SPLIT
    ysum = r.groupby(sub["entry"].dt.year).sum()
    tot = r.sum()
    if abs(tot) > 1e-12 and len(ysum):
        dom = ysum.loc[ysum.abs().idxmax()]
        dom_year = int(ysum.abs().idxmax())
        dom_share = float(dom / tot)
    else:
        dom_year, dom_share = -1, np.nan
    lo, hi = boot_ci_mean(r.to_numpy())
    L, S = sub[sub["dir"] > 0], sub[sub["dir"] < 0]
    return {
        "cohort": name, "n": n,
        "net_mean_bps": r.mean() * 1e4, "net_sum": tot,
        "share_pct": tot / TOTAL_EXPECTED * 100, "win_pct": (r > 0).mean() * 100,
        "gross_mean_bps": g.mean() * 1e4, "gross_sum": g.sum(),
        "cost_sum": sub["cost_pct"].sum(),
        "ci_lo_bps": lo * 1e4, "ci_hi_bps": hi * 1e4,
        "IS_n": int(is_m.sum()), "IS_sum": r[is_m].sum(),
        "IS_mean_bps": (r[is_m].mean() * 1e4) if is_m.any() else np.nan,
        "OOS_n": int((~is_m).sum()), "OOS_sum": r[~is_m].sum(),
        "OOS_mean_bps": (r[~is_m].mean() * 1e4) if (~is_m).any() else np.nan,
        "L_n": len(L), "L_sum": L["ret"].sum(),
        "S_n": len(S), "S_sum": S["ret"].sum(),
        "dom_year": dom_year, "dom_share": dom_share,
        "single_year_flag": bool(dom_share > 0.5) if np.isfinite(dom_share) else False,
        "small_n_flag": bool(n < 30),
        "yearly": {int(k): float(v) for k, v in ysum.items()},
    }


def protocol_table(df, cats: pd.Series, title: str) -> list[dict]:
    rows = []
    for lab in cats.cat.categories if hasattr(cats, "cat") else sorted(cats.dropna().unique()):
        rows.append(cohort_row(df, cats == lab, str(lab)))
    na = cats.isna()
    if na.any():
        rows.append(cohort_row(df, na, "n/a(欠損)"))
    rows = [x for x in rows if x and x.get("n", 0) > 0]
    t = pd.DataFrame(rows).drop(columns=["yearly"])
    cols = ["cohort", "n", "net_mean_bps", "ci_lo_bps", "ci_hi_bps", "net_sum", "share_pct",
            "win_pct", "gross_mean_bps", "gross_sum", "IS_mean_bps", "OOS_mean_bps",
            "IS_sum", "OOS_sum", "L_sum", "S_sum", "dom_year", "dom_share",
            "single_year_flag", "small_n_flag"]
    print(f"\n--- {title} ---")
    print(t[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return rows


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)

    # --- プール検算 ---------------------------------------------------------
    pool = pd.read_parquet(POOL_PATH)
    sec(f"0. プール検算: n={len(pool)}  sum(ret)={pool['ret'].sum():+.4f} "
        f"(期待 n=1207 / +1.9622)")
    assert len(pool) == 1207 and abs(pool["ret"].sum() - 1.9622) < 1e-3

    # --- 特徴量 -------------------------------------------------------------
    feats = signal_features(pool)
    df = pd.concat([pool, feats], axis=1)
    # 期待グロス戻り幅% と cost_ratio
    df["width_pct"] = (df["z_entry"] - EXIT_Z) * df["std50"] / df["sig_close"]
    df["cost_ratio"] = df["cost_pct"] / df["width_pct"]
    df["ret_gross"] = df["ret"] + df["cost_pct"]
    df["ratio2010"] = df["vol20"] / df["vol100"]
    df["gate_ratio"] = df["vol20"] / df["gate70"]
    df["year"] = df["entry"].dt.year

    sec("1. コスト算術の分布(シグナルバー因果値)")
    desc = df[["width_pct", "cost_pct", "cost_ratio", "ratio2010", "gate_ratio",
               "vov_cv", "vol_hist_pct"]].describe(
        percentiles=[0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]).T
    print(desc.to_string(float_format=lambda x: f"{x:.5f}"))
    nan_counts = df[["std50", "cost_ratio", "vov_cv", "vol_hist_pct", "gate_ratio"]].isna().sum()
    print("\n欠損数:", nan_counts.to_dict())
    print(f"gate_ratio > 1.0 の件数(戦略ゲート整合チェック, 0 が期待): "
          f"{(df['gate_ratio'] > 1.0 + 1e-9).sum()}")
    print(f"プール全体: net {df['ret'].sum():+.4f} / gross {df['ret_gross'].sum():+.4f} / "
          f"総コスト {df['cost_pct'].sum():.4f} (= gross の {df['cost_pct'].sum()/df['ret_gross'].sum():.1%})")

    # --- 銘柄別コスト構造 -----------------------------------------------------
    sec("2. 銘柄別: cost_ratio とグロス/net(クロス3pips vs メジャー実スプレッド)")
    by_i = df.groupby("instr").agg(
        n=("ret", "size"), cost_pct_med=("cost_pct", "median"),
        width_med=("width_pct", "median"), cr_med=("cost_ratio", "median"),
        cr_p90=("cost_ratio", lambda s: s.quantile(0.9)),
        gross_sum=("ret_gross", "sum"), net_sum=("ret", "sum"),
        cost_sum=("cost_pct", "sum"), win=("ret", lambda s: (s > 0).mean()),
    ).sort_values("cr_med", ascending=False)
    print(by_i.to_string(float_format=lambda x: f"{x:.4f}"))
    cross_m = df["instr"].isin(uni.CROSS_DEFS)
    for nm, m in [("クロス(spread=3p)", cross_m), ("メジャー(実スプレッド)", ~cross_m)]:
        s = df[m]
        print(f"{nm}: n={len(s)} cr_med={s['cost_ratio'].median():.4f} "
              f"gross={s['ret_gross'].sum():+.3f} net={s['ret'].sum():+.3f} "
              f"cost={s['cost_pct'].sum():.3f}")

    # --- 3. cost_ratio コホート(グロス/net 両方) ----------------------------
    sec("3. cost_ratio コホート(機構量: 往復コスト ÷ 期待戻り幅)")
    q = df["cost_ratio"].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
    cats_q = pd.cut(df["cost_ratio"], [-np.inf, *q, np.inf],
                    labels=[f"Q1(<{q[0]:.3f})", "Q2", "Q3", "Q4",
                            f"Q5(>{q[3]:.3f})"])
    rows_crq = protocol_table(df, cats_q, "cost_ratio 五分位")
    fixed_edges = [0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, np.inf]
    cats_f = pd.cut(df["cost_ratio"], fixed_edges,
                    labels=["<2%", "2-5%", "5-10%", "10-20%", "20-30%", "30-50%", ">50%"])
    rows_crf = protocol_table(df, cats_f, "cost_ratio 固定閾値帯(指示の >0.3 / >0.5 を含む)")
    # 指示の特定帯
    for thr in (0.3, 0.5):
        sub = df[df["cost_ratio"] > thr]
        print(f"\ncost_ratio > {thr}: n={len(sub)}"
              + ("" if len(sub) == 0 else
                 f" net={sub['ret'].sum():+.4f} gross={sub['ret_gross'].sum():+.4f}"))

    # 「グロスで勝って net で負ける」帯の明示チェック
    gross_pos_net_neg = [r for r in rows_crq + rows_crf
                         if r["n"] > 0 and r["gross_sum"] > 0 and r["net_sum"] < 0]
    print("\n『グロス+ / net −』のコホート: "
          + (", ".join(f"{r['cohort']}(n={r['n']})" for r in gross_pos_net_neg)
             if gross_pos_net_neg else "なし"))

    # --- 4. ボラ異常系コホート ------------------------------------------------
    sec("4. ボラ異常系(シグナルバー時点)")
    qr = df["ratio2010"].quantile([0.1, 0.3, 0.7, 0.9]).to_numpy()
    rows_r = protocol_table(
        df, pd.cut(df["ratio2010"], [-np.inf, *qr, np.inf],
                   labels=[f"低10%(<{qr[0]:.2f})", "10-30%", "30-70%", "70-90%",
                           f"高10%(>{qr[3]:.2f})"]),
        "vol20/vol100 比(短期ボラの相対水準。calm ゲートで上は切られている)")
    qv = df["vov_cv"].quantile([0.1, 0.3, 0.7, 0.9]).to_numpy()
    rows_v = protocol_table(
        df, pd.cut(df["vov_cv"], [-np.inf, *qv, np.inf],
                   labels=[f"低10%(<{qv[0]:.2f})", "10-30%", "30-70%", "70-90%",
                           f"高10%(>{qv[3]:.2f})"]),
        "vol-of-vol(vol20 の 20本CV)")
    rows_h = protocol_table(
        df, pd.cut(df["vol_hist_pct"], [0, 0.05, 0.20, 0.50, 1.0],
                   labels=["<5%(極端低)", "5-20%", "20-50%", ">50%"]),
        "vol20 の銘柄内ヒストリカル分位(expanding=因果)")
    # 極端低ボラ帯の cost_ratio との関係
    low5 = df["vol_hist_pct"] < 0.05
    print(f"\nvol_hist_pct<5% 帯の cost_ratio 中央値: {df.loc[low5, 'cost_ratio'].median():.4f} "
          f"vs 残り {df.loc[~low5.fillna(False), 'cost_ratio'].median():.4f}")

    # --- 5. 静穏ゲートぎわ -----------------------------------------------------
    sec("5. 静穏ゲートぎわ(vol20 / Q70ゲート水準)")
    rows_g = protocol_table(
        df, pd.cut(df["gate_ratio"], [0, 0.5, 0.7, 0.9, 1.0 + 1e-9],
                   labels=["深い静穏(<0.5)", "0.5-0.7", "0.7-0.9", "ぎわ(0.9-1.0)"]),
        "ゲート比コホート")
    edge_m = (df["gate_ratio"] > 0.9) & (df["gate_ratio"] <= 1.0 + 1e-9)
    deep_m = df["gate_ratio"] < 0.5
    lo, hi = boot_ci_diff(df.loc[edge_m, "ret"].to_numpy(), df.loc[deep_m, "ret"].to_numpy())
    print(f"\nぎわ(0.9-1.0, n={edge_m.sum()}) − 深い静穏(<0.5, n={deep_m.sum()}) "
          f"平均差CI95: [{lo*1e4:+.1f}, {hi*1e4:+.1f}] bps")

    # --- 6. 主要コホートの対残り CI ------------------------------------------
    sec("6. 主要コホート vs 残り: 平均差ブートストラップCI(1000)")
    key_masks = {
        "cost_ratio 上位10%": df["cost_ratio"] >= df["cost_ratio"].quantile(0.9),
        "cost_ratio 上位20%": df["cost_ratio"] >= df["cost_ratio"].quantile(0.8),
        "vol_hist_pct < 5%": low5,
        "ratio2010 下位10%": df["ratio2010"] <= df["ratio2010"].quantile(0.1),
        "vov_cv 上位10%": df["vov_cv"] >= df["vov_cv"].quantile(0.9),
        "ゲートぎわ 0.9-1.0": edge_m,
    }
    ci_rows = []
    for nm, m in key_masks.items():
        m = m.fillna(False)
        a, b = df.loc[m, "ret"].to_numpy(), df.loc[~m, "ret"].to_numpy()
        lo, hi = boot_ci_diff(a, b)
        glo, ghi = boot_ci_diff(df.loc[m, "ret_gross"].to_numpy(),
                                df.loc[~m, "ret_gross"].to_numpy())
        ci_rows.append({"cohort": nm, "n": int(m.sum()),
                        "net_mean_bps": df.loc[m, "ret"].mean() * 1e4,
                        "diff_ci_lo_bps": lo * 1e4, "diff_ci_hi_bps": hi * 1e4,
                        "gross_diff_ci_lo_bps": glo * 1e4, "gross_diff_ci_hi_bps": ghi * 1e4,
                        "sig_neg": bool(hi < 0)})
    cit = pd.DataFrame(ci_rows)
    print(cit.to_string(index=False, float_format=lambda x: f"{x:+.1f}"))

    # --- 7. veto 候補スイープ: cost_ratio > X で見送り -------------------------
    sec("7. veto スイープ『cost_ratio > X で見送り』(指名基準: 除外コホート net CI<0 "
        "かつ IS/OOS 両方負 かつ 単年依存なし かつ n>=30)")
    veto_rows = []
    thr_list = sorted(set(
        [round(float(df["cost_ratio"].quantile(p)), 4) for p in (0.7, 0.8, 0.9, 0.95)]
        + [0.02, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50]))
    for thr in thr_list:
        m = df["cost_ratio"] > thr
        if m.sum() == 0:
            veto_rows.append({"thr": thr, "n_excl": 0, "verdict": "空(該当なし)"})
            continue
        row = cohort_row(df, m, f"cr>{thr}")
        ok = (row["ci_hi_bps"] < 0 and row["IS_sum"] < 0 and row["OOS_sum"] < 0
              and not row["single_year_flag"] and row["n"] >= 30)
        veto_rows.append({"thr": thr, "n_excl": row["n"],
                          "excl_net_sum": row["net_sum"], "excl_gross_sum": row["gross_sum"],
                          "excl_mean_bps": row["net_mean_bps"],
                          "ci": f"[{row['ci_lo_bps']:+.0f},{row['ci_hi_bps']:+.0f}]bps",
                          "IS_sum": row["IS_sum"], "OOS_sum": row["OOS_sum"],
                          "dom_year": row["dom_year"], "dom_share": row["dom_share"],
                          "remain_net": df.loc[~m, "ret"].sum(),
                          "verdict": "指名可" if ok else "不適"})
    vt = pd.DataFrame(veto_rows)
    print(vt.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- 保存 -----------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "edge03_trades.csv", index=False)
    summary = {
        "pool_check": {"n": int(len(pool)), "sum_ret": float(pool["ret"].sum())},
        "totals": {"net": float(df["ret"].sum()), "gross": float(df["ret_gross"].sum()),
                   "cost": float(df["cost_pct"].sum())},
        "cost_ratio_dist": {k: float(v) for k, v in
                            df["cost_ratio"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).items()},
        "cohorts": {"cost_ratio_quintile": rows_crq, "cost_ratio_fixed": rows_crf,
                    "ratio2010": rows_r, "vov_cv": rows_v, "vol_hist_pct": rows_h,
                    "gate_ratio": rows_g},
        "key_ci": ci_rows,
        "veto_sweep": veto_rows,
    }
    (OUT_DIR / "edge03_summary.json").write_text(
        json.dumps(summary, indent=2, default=float, ensure_ascii=False))
    print(f"\nsaved -> {OUT_DIR / 'edge03_trades.csv'} / edge03_summary.json")
    print(f"総経過 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
