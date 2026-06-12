"""edge08: veto候補「金曜遅バーエントリー禁止」の敵対検証(殺しに行く側)。

殺すべき主張(edge05 C7/C11): 「entryラベル=金曜20:00 の29件は、実約定がロールオーバー窓内に
落ちる唯一のコホートで EV が負(-5.3bps, ×3再価格 -7.3bps)。スキップすれば何も失わない」

攻撃ベクトル(全部実行):
  A1 セル選択の恣意性: 全曜日×全ラベル時刻グリッドのセル平均分布 + ランダム n=29 帰無分布の
     %タイル + 置換検定(最弱セル統計=多重比較補正後の family-wise p)。
  A2 2018依存: 2018除外でコホートEVの符号が変わるか。
  A3 プラセボ・機構整合: 隣接セル(金16:00ラベル=20:00close約定)/ 他曜日20:00ラベル /
     シグナル=金曜遅バー(=日曜オープン建玉)/ 同一執行窓の出口側(C8)との整合。
  A4 再価格×3の独立証拠: M1 連続バー間ジャンプ分布(金曜20-21時 vs 平日ロールオーバー窓
     vs 流動時間帯)+ バー密度(薄商いの直接証拠)。
  A5 時間反転 + 口座レベル再シミュレート: コホートはOOS(2022-)で正。29件を落とした変種プールを
     ペアシード(0-4) robust 較正でベースと比較(建玉枠の解放・kの変化込み)+ G3(IS較正→OOS)
     + レバ偽装署名 + 年次分割感度。

実行: uv run python research/experiments/edge08_friday_latebar_adversarial.py
出力: research/outputs/edge08_cells.csv / edge08_summary.json

結論(2026-06-13 実行): **veto候補は死亡(survives=false)。机上では棄却。**
  A1 後知恵セル選択の確定: -5.3bps はランダム n=29 サブセット帰無分布の 7.9%タイル
     (下位5%に入らない=単独でも有意でない)。曜日×時刻グリッド(n≥10 の30セル)の置換検定で
     「最弱セルが -5.3bps 以下になる」family-wise p = 0.900 = 30セルから最弱を選べば 9割の
     確率でこの程度は出る。しかも金20:00 は実際には最弱ですらない(Mon-08 が -21.7bps, n=31。
     機構ストーリーが無いから誰も禁止を提案しないだけ)。致命傷。
  A2 単年依存の確定: 2018(n=4, -0.0190)除外でコホート合計 +0.0036 / 平均 +1.42bps に符号反転。
  A3 機構不整合: 同一執行窓の出口11件は +28.7bps(窓全体では +0.0161=正)、シグナル=金曜遅バー
     (日曜オープン建玉, n=12)は +29.7bps、隣接セル金16:00(n=70, 約定=金20:00close)は
     +10.7bps、月-木20:00ラベル(n=135)は +16.9bps。「窓内執行が損」の機構はエントリー29件に
     しか現れない=ノイズの署名。
  A4 ×3再価格に金曜固有の独立証拠なし: 連続M1ジャンプ median は金曜20-21h 0.46bps vs 平日
     ロールオーバー窓 0.40bps(1.14x)、流動時間帯比ではむしろ 0.57x と小さい(p90 も同等)。
     薄さの直接証拠はバー密度のみ(金21h 0.34 = DST起因の早終い、水21h 0.94)。BID close-to-close
     はスプレッド拡大を直接測れない点は留保するが、「平日窓と同じ既知の薄さ」を超える金曜固有の
     ペナルティは検出されず(edge05 S4 はその窓の×3ストレス込みで全暦年プラスを確認済み)。
  A5 時間反転+口座レベルで採用バー未達: コホート 2022- は +13.97bps(n=13)=スキップはOOSでは損
     (プール段の除去利益も後半2021-で -0.0157 と符号反転、偶数/奇数年も符号不安定)。
     29件除去を再シミュレート(枠解放込み)したペアシード5本 robust: base +18.57% → skip +19.05%
     = +0.48pp(全シード正だが 0.15-0.90pp で較正ノイズ帯 ±0.4-0.8pp 内、reports/22 の不採用判例
     +0.34pp と同階級)。empirical は +2.45pp だが p95 が -27.3%→-28.8% に悪化=**レバ偽装署名**。
     G3(IS較正→OOS)は rob +0.59pp / emp +1.85pp と表面上は正だが、これは IS から負け玉を
     抜いて k_is が 6.49→6.68 / 10.85→11.31 に上がった分のレバ増で、OOS DD も -11.5→-11.8% /
     -18.8→-19.5% と深化=同じレバ偽装の経路。エッジの純増ではない。
  → 統計は全滅(A1 後知恵セル選択の確定 + A2 単年依存の確定 + A5 ノイズ帯未満&署名)。
     「実弾運用で金曜引け際の薄商い+週末ギャップ直前に成行を投げない」は衛生ルールとして
     言及可だが、バックテスト上の veto(プール除外で性能改善という主張)としては採用不可。
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
sys.path.insert(0, str(ROOT / "research" / "lab"))

from fxlab import config  # noqa: E402
from fxlab import universe as uni  # noqa: E402
from fxlab.data import load_m1  # noqa: E402
from fxlab.universe import CROSS_DEFS  # noqa: E402

import mm_lab as mm  # noqa: E402
from mm_production import champion_sizing  # noqa: E402
from tail_protocol import (  # noqa: E402
    boot_dd,
    cagr_of,
    calibrate_empirical,
    calibrate_robust_seeded,
    max_dd,
    protocol_eval,
    yearly_returns,
)

pd.set_option("display.width", 240)

OUT_DIR = ROOT / "research" / "outputs"
POOL_PATH = ROOT / "results" / "mm_pool_v2d1_H4_19.parquet"
N_EXPECT, TOTAL_EXPECT = 1207, 1.9622
MAX_POS = 8
SEEDS = (0, 1, 2, 3, 4)
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
B_PERM = 10_000     # 置換検定(最弱セル統計)
B_DRAW = 20_000     # ランダム n=29 帰無分布
NOISE_PP = (0.4, 0.8)  # robust較正ノイズ帯(pp)
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def boot_mean_ci(x, seed=0, n_boot=2000):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cohort_row(name, r):
    r = np.asarray(r, float)
    lo, hi = boot_mean_ci(r) if len(r) else (np.nan, np.nan)
    return {"name": name, "n": int(len(r)), "mean_bps": float(r.mean() * 1e4) if len(r) else np.nan,
            "sum": float(r.sum()), "win_pct": float((r > 0).mean() * 100) if len(r) else np.nan,
            "ci_lo_bps": lo * 1e4, "ci_hi_bps": hi * 1e4}


# --- 口座レベル評価(exp47 evaluate の軽量版) ------------------------------
def make_eq_fn(pool_v, closes, mk):
    cache = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            eqm, _, _ = mm.simulate(pool_v, closes, mk(kk), max_pos=MAX_POS)
            cache[kk] = eqm
        return cache[kk]

    return eq_of_k


def account_eval(label, pool_v, closes):
    mk = champion_sizing(pool_v, max_pos=MAX_POS)
    eq_of_k = make_eq_fn(pool_v, closes, mk)
    res = protocol_eval(eq_of_k, label=label, seeds=SEEDS)
    yr_e = yearly_returns(eq_of_k(res["emp_k"]))
    res["yr_emp"] = {int(y): float(v) for y, v in yr_e.items()}
    res["neg_years_emp"] = int((yr_e < 0).sum())
    yr_r0 = yearly_returns(eq_of_k(res["rob"][0]["k"]))
    res["yr_rob0"] = {int(y): float(v) for y, v in yr_r0.items()}
    res["neg_years_rob0"] = int((yr_r0 < 0).sum())
    # G3: IS(-2021)較正 → OOS(2022-)素検証
    isp = pool_v[pool_v["entry"] < OOS_START].reset_index(drop=True)
    oop = pool_v[pool_v["entry"] >= OOS_START].reset_index(drop=True)
    iscl = closes[closes.index < OOS_START]
    oocl = closes[closes.index >= OOS_START]
    eq_is = make_eq_fn(isp, iscl, mk)
    eq_oos = make_eq_fn(oop, oocl, mk)
    k_ie = calibrate_empirical(eq_is, 0.20)
    res["k_is_emp"] = k_ie
    res["is_emp_cagr"] = cagr_of(eq_is(k_ie))
    res["oos_emp_cagr"] = cagr_of(eq_oos(k_ie))
    res["oos_emp_dd"] = max_dd(eq_oos(k_ie))
    k_ir = calibrate_robust_seeded(eq_is, 0.20, seed=0)
    res["k_is_rob"] = k_ir
    res["is_rob_cagr"] = cagr_of(eq_is(k_ir))
    res["oos_rob_cagr"] = cagr_of(eq_oos(k_ir))
    res["oos_rob_dd"] = max_dd(eq_oos(k_ir))
    print(f"      IS emp k={k_ie:5.2f} ISC={res['is_emp_cagr']:+7.2%} -> OOS {res['oos_emp_cagr']:+7.2%} "
          f"(DD {res['oos_emp_dd']:+5.1%}) | IS rob k={k_ir:5.2f} ISC={res['is_rob_cagr']:+7.2%} -> "
          f"OOS {res['oos_rob_cagr']:+7.2%} (DD {res['oos_rob_dd']:+5.1%})")
    return res


# --- M1 close(edge05/exp52 方式: クロスは脚 inner-join 合成) --------------
_LEG: dict[str, pd.Series] = {}


def m1_close(name: str) -> pd.Series:
    if name in CROSS_DEFS:
        a, op, b = CROSS_DEFS[name]
        ca, cb = m1_close(a), m1_close(b)
        df = pd.concat([ca.rename("a"), cb.rename("b")], axis=1, join="inner").dropna()
        return df["a"] / df["b"] if op == "/" else df["a"] * df["b"]
    if name not in _LEG:
        c = load_m1(name)["close"]
        _LEG[name] = pd.Series(c.to_numpy(), index=c.index.tz_localize(None))
    return _LEG[name]


def m1_jump_audit(instr: str) -> dict:
    """連続M1バー(Δt=1分)の |close-to-close| ジャンプ(bps)をバケット別に集計 + バー密度。"""
    s = m1_close(instr)
    t = s.index
    px = s.to_numpy()
    dt_min = np.diff(t.values).astype("timedelta64[s]").astype(np.int64) / 60.0
    jump = np.abs(np.diff(px) / px[:-1]) * 1e4
    wd = t.dayofweek.to_numpy()[1:]
    hr = t.hour.to_numpy()[1:]
    consec = dt_min == 1.0
    buckets = {
        "fri_close(20-21h)": (wd == 4) & ((hr == 20) | (hr == 21)),
        "wd_roll(Mon-Thu 20-21h)": (wd <= 3) & ((hr == 20) | (hr == 21)),
        "fri_liquid(14-16h)": (wd == 4) & (hr >= 14) & (hr <= 16),
        "liquid(Mon-Thu 8-16h)": (wd <= 3) & (hr >= 8) & (hr <= 16),
    }
    rec = {"instr": instr}
    for bname, bmask in buckets.items():
        x = jump[consec & bmask]
        rec[f"{bname}|med"] = float(np.median(x)) if len(x) else np.nan
        rec[f"{bname}|p90"] = float(np.percentile(x, 90)) if len(x) else np.nan
        rec[f"{bname}|n"] = int(len(x))
    # バー密度: 当該スロットの実在M1バー数 / (該当曜日の日数 × 60分)
    days = pd.Series(t.normalize()).groupby(t.dayofweek).nunique()
    full_wd = t.dayofweek.to_numpy()
    full_hr = t.hour.to_numpy()
    for bname, w_set, h_set in [("dens_fri20h", (4,), (20,)), ("dens_fri21h", (4,), (21,)),
                                ("dens_wed21h", (2,), (21,)), ("dens_wed14h", (2,), (14,))]:
        m = np.isin(full_wd, w_set) & np.isin(full_hr, h_set)
        nd = sum(int(days.get(w, 0)) for w in w_set)
        rec[bname] = float(m.sum() / (nd * 60.0)) if nd else np.nan
    return rec


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = pd.read_parquet(POOL_PATH)
    total = float(pool["ret"].sum())
    print(f"=== edge08: 金曜遅バーエントリー禁止 veto の敵対検証 (n={len(pool)}, sum={total:+.4f}) ===")
    assert len(pool) == N_EXPECT and abs(total - TOTAL_EXPECT) < 1e-3, "プール検算失敗"

    e_wd = pool["entry"].dt.dayofweek
    e_hr = pool["entry"].dt.hour
    target = (e_wd == 4) & (e_hr == 20)
    sub = pool[target]
    r_t = sub["ret"].to_numpy()
    obs_mean = float(r_t.mean())
    print(f"標的コホート(entryラベル=金20:00): n={len(sub)}  mean={obs_mean*1e4:+.2f}bps  "
          f"sum={r_t.sum():+.4f}")
    assert len(sub) == 29 and abs(obs_mean * 1e4 - (-5.3246)) < 0.01, "コホート検算失敗"

    # 再価格 ×3 の確認(エントリー側半スプレッド hs_e を entry_close から再構成)
    ec = np.full(len(pool), np.nan)
    for instr, g in pool.groupby("instr"):
        s = uni.instrument_close(instr, "H4")
        ie = s.index.get_indexer(g["entry"])
        assert (ie >= 0).all()
        ec[g.index.to_numpy()] = s.to_numpy()[ie]
    hs_e = pool["instr"].map(lambda i: config.spread_pips(i) * config.pip_size(i)).to_numpy() / 2.0 / ec
    repriced = float((r_t - 2.0 * hs_e[target.to_numpy()]).mean() * 1e4)
    print(f"×3再価格(自コスト2半スプレッド追加)後 mean={repriced:+.2f}bps(主張は -7.3bps)")

    # ====================================================================
    sec("A1. セル選択の恣意性: 曜日×ラベル時刻グリッド + 帰無分布 + 置換検定")
    cells = pool.groupby([e_wd.rename("wd"), e_hr.rename("hr")])["ret"].agg(["size", "mean", "sum"])
    cells["mean_bps"] = cells["mean"] * 1e4
    cells = cells.reset_index()
    cells["cell"] = cells.apply(lambda x: f"{WD[int(x.wd)]}-{int(x.hr):02d}", axis=1)
    cells_s = cells.sort_values("mean_bps")
    print("全セル(平均bps昇順):")
    print(cells_s[["cell", "size", "mean_bps", "sum"]].to_string(index=False,
          float_format=lambda x: f"{x:+.2f}"))
    valid = cells[cells["size"] >= 10]
    rank = int((valid["mean"] <= obs_mean).sum())
    print(f"\nn≥10 の {len(valid)} セル中、金20:00 は最弱から {rank} 位 "
          f"(負のセル {int((valid['mean'] < 0).sum())} 個)")

    rng = np.random.default_rng(0)
    r_all = pool["ret"].to_numpy()
    # (a) ランダム n=29 サブセットの平均の帰無分布
    null29 = np.empty(B_DRAW)
    for b in range(B_DRAW):
        null29[b] = r_all[rng.choice(len(r_all), 29, replace=False)].mean()
    pct_single = float((null29 <= obs_mean).mean() * 100)
    print(f"(a) ランダム n=29 帰無分布(B={B_DRAW}): mean={null29.mean()*1e4:+.1f}bps "
          f"p5={np.percentile(null29,5)*1e4:+.1f}bps | 観測 -5.32bps の%タイル = {pct_single:.1f}%"
          f"  -> {'下位5%未満=単独でも異常' if pct_single < 5 else '下位5%に入らない=単独でも有意でない'}")
    # (b) 置換検定: 最弱セル統計(n≥10 セル family の多重比較補正)
    key = (e_wd * 100 + e_hr).to_numpy()
    codes, uniq = pd.factorize(key)
    counts = np.bincount(codes)
    vmask = counts >= 10
    mins = np.empty(B_PERM)
    for b in range(B_PERM):
        rp = rng.permutation(r_all)
        means = np.bincount(codes, weights=rp) / counts
        mins[b] = means[vmask].min()
    p_fw = float((mins <= obs_mean).mean())
    print(f"(b) 置換検定(B={B_PERM}, n≥10 の {int(vmask.sum())} セル): "
          f"min-cell 帰無分布 median={np.median(mins)*1e4:+.1f}bps | "
          f"P(最弱セル ≤ -5.32bps) = {p_fw:.3f}"
          f"  -> {'family-wise でも有意' if p_fw < 0.05 else '最弱セル選択の範囲内=後知恵'}")
    kill_a1 = (pct_single >= 5.0) or (p_fw >= 0.05)

    # ====================================================================
    sec("A2. 2018依存: 単年テール除去にすぎないか")
    yr_t = sub.groupby(sub["entry"].dt.year)["ret"].agg(["size", "sum", "mean"])
    yr_t["mean_bps"] = yr_t["mean"] * 1e4
    print(yr_t[["size", "sum", "mean_bps"]].to_string(float_format=lambda x: f"{x:+.4f}"))
    ex2018 = sub[sub["entry"].dt.year != 2018]["ret"]
    worst_y = int(yr_t["sum"].idxmin())
    ex_worst = sub[sub["entry"].dt.year != worst_y]["ret"]
    print(f"\n2018除外: n={len(ex2018)} mean={ex2018.mean()*1e4:+.2f}bps sum={ex2018.sum():+.4f}"
          f"  -> {'符号反転=2018単年依存' if ex2018.sum() > 0 else '依然負'}")
    print(f"最悪年({worst_y})除外: mean={ex_worst.mean()*1e4:+.2f}bps sum={ex_worst.sum():+.4f}")
    kill_a2 = float(ex2018.sum()) > 0

    # ====================================================================
    sec("A3. プラセボ・機構整合: 同じ執行窓の他コホートと整合するか")
    x_wd = pool["exit"].dt.dayofweek
    x_hr = pool["exit"].dt.hour
    # シグナルバー=金曜遅バー(エントリーの1グリッド前バーのラベルが金20:00)
    sig_fri = np.zeros(len(pool), bool)
    for instr, g in pool.groupby("instr"):
        s = uni.instrument_close(instr, "H4")
        ie = s.index.get_indexer(g["entry"])
        ok = ie > 0
        lab = pd.DatetimeIndex(s.index.values[np.maximum(ie - 1, 0)])
        m = ok & (lab.dayofweek == 4) & (lab.hour == 20)
        sig_fri[g.index.to_numpy()[m]] = True
    rows = [
        cohort_row("標的: entryラベル=金20:00 (窓内約定)", r_t),
        cohort_row("隣接: entryラベル=金16:00 (約定=金20:00close)", pool.loc[(e_wd == 4) & (e_hr == 16), "ret"]),
        cohort_row("他曜日: entryラベル=月-木20:00 (約定=翌0:00)", pool.loc[(e_wd <= 3) & (e_hr == 20), "ret"]),
        cohort_row("プラセボ: シグナル=金曜遅バー (建玉=日曜オープン)", pool.loc[sig_fri, "ret"]),
        cohort_row("日曜オープンバーentry", pool.loc[e_wd == 6, "ret"]),
        cohort_row("同一窓の出口側: exitラベル=金20:00 (C8)", pool.loc[(x_wd == 4) & (x_hr == 20), "ret"]),
    ]
    t3 = pd.DataFrame(rows)
    print(t3.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    win_total = float(r_t.sum() + pool.loc[(x_wd == 4) & (x_hr == 20), "ret"].sum())
    print(f"\n金曜遅バー執行窓に触れる全PnL(エントリー29件+出口11件)= {win_total:+.4f} "
          f"-> 窓全体では{'正' if win_total > 0 else '負'}")
    print("機構整合性: 「窓内執行が損」なら出口側にも同符号で現れるはず。エントリーのみ負・"
          "出口/シグナル側/隣接セルは正 = 機構ではなくサンプリングノイズの署名。")
    a3_incoherent = win_total > 0

    # ====================================================================
    sec("A4. 再価格×3の独立証拠: M1連続バー間ジャンプ + バー密度(コホート銘柄)")
    instrs = sorted(sub["instr"].unique())
    print(f"対象 {len(instrs)} 銘柄: {instrs}")
    jrows = []
    for nm in instrs:
        jrows.append(m1_jump_audit(nm))
        print(f"  {nm} done [{time.time()-t0:.0f}s]")
    jt = pd.DataFrame(jrows).set_index("instr")
    med_cols = [c for c in jt.columns if c.endswith("|med")]
    p90_cols = [c for c in jt.columns if c.endswith("|p90")]
    dens_cols = [c for c in jt.columns if c.startswith("dens_")]
    print("\n銘柄横断平均(連続M1 |Δclose| bps):")
    print("  median:", {c.split("|")[0]: round(float(jt[c].mean()), 2) for c in med_cols})
    print("  p90   :", {c.split("|")[0]: round(float(jt[c].mean()), 2) for c in p90_cols})
    print("  バー密度:", {c: round(float(jt[c].mean()), 3) for c in dens_cols})
    fri_med = float(jt["fri_close(20-21h)|med"].mean())
    roll_med = float(jt["wd_roll(Mon-Thu 20-21h)|med"].mean())
    liq_med = float(jt["liquid(Mon-Thu 8-16h)|med"].mean())
    fri_vs_roll = fri_med / roll_med if roll_med else np.nan
    print(f"\n金曜20-21h median / 平日ロールオーバー窓 median = {fri_vs_roll:.2f}x "
          f"(流動時間帯比 {fri_med/liq_med:.2f}x)")
    print("-> 比が ≈1 なら『金曜固有の執行困難』の独立証拠なし(平日窓と同じ既知の薄さ。"
          "edge05 S4 はその窓の×3ストレスを既に通している)。")

    # ====================================================================
    sec("A5. 時間反転 + 口座レベル再シミュレート(ペアシード0-4, mp8, P=4.0)")
    is_m = sub["entry"] < OOS_START
    print(f"コホート IS(-2021): n={int(is_m.sum())} mean={sub.loc[is_m,'ret'].mean()*1e4:+.2f}bps "
          f"sum={sub.loc[is_m,'ret'].sum():+.4f}")
    print(f"コホート OOS(2022-): n={int((~is_m).sum())} mean={sub.loc[~is_m,'ret'].mean()*1e4:+.2f}bps "
          f"sum={sub.loc[~is_m,'ret'].sum():+.4f}  -> OOSは{'正=スキップはOOSで損' if sub.loc[~is_m,'ret'].sum() > 0 else '負'}")
    oos_cohort_sum = float(sub.loc[~is_m, "ret"].sum())

    # 年次分割感度(プール段: 除去の利益 = -コホート年次合計)
    ben = -sub.groupby(sub["entry"].dt.year)["ret"].sum()
    even = float(ben[ben.index % 2 == 0].sum())
    odd = float(ben[ben.index % 2 == 1].sum())
    h1 = float(ben[ben.index <= 2020].sum())
    h2 = float(ben[ben.index >= 2021].sum())
    print(f"\n除去利益の年次分割感度: 偶数年 {even:+.4f} / 奇数年 {odd:+.4f} | "
          f"前半(-2020) {h1:+.4f} / 後半(2021-) {h2:+.4f} "
          f"-> {'符号不安定' if (even * odd < 0) or (h1 * h2 < 0) else '符号安定'}")
    split_unstable = bool((even * odd < 0) or (h1 * h2 < 0))

    closes = mm.load_closes()
    pool_skip = pool[~target].reset_index(drop=True)
    print(f"\n変種プール(29件除去): n={len(pool_skip)} sum={pool_skip['ret'].sum():+.4f}")
    print("--- base (d1 そのまま) ---")
    base = account_eval("base_d1", pool, closes)
    print(f"    [{time.time()-t0:.0f}s]")
    print("--- skip_fri_late (29件除去・枠解放込み再シミュ) ---")
    var = account_eval("skip_fri_late", pool_skip, closes)
    print(f"    [{time.time()-t0:.0f}s]")

    print("\n--- ペアシード robust(p95=20%) CAGR ---")
    diffs = []
    for sd in SEEDS:
        d = (var["rob"][sd]["cagr"] - base["rob"][sd]["cagr"]) * 100
        diffs.append(d)
        print(f"  s{sd}: base {base['rob'][sd]['cagr']:+.2%}  skip {var['rob'][sd]['cagr']:+.2%}  "
              f"diff {d:+.2f}pp")
    gain_pp = (var["rob_cagr_mean"] - base["rob_cagr_mean"]) * 100
    print(f"  mean: base {base['rob_cagr_mean']:+.2%}  skip {var['rob_cagr_mean']:+.2%}  "
          f"diff {gain_pp:+.2f}pp (較正ノイズ帯 ±{NOISE_PP[0]}-{NOISE_PP[1]}pp)")
    emp_diff = (var["emp_cagr"] - base["emp_cagr"]) * 100
    sig = (var["emp_cagr"] > base["emp_cagr"]) and (abs(var["emp_p95"]) > abs(base["emp_p95"]) + 0.005)
    print(f"  empirical: base {base['emp_cagr']:+.2%} (p95 {base['emp_p95']:+.1%})  "
          f"skip {var['emp_cagr']:+.2%} (p95 {var['emp_p95']:+.1%})  diff {emp_diff:+.2f}pp  "
          f"レバ偽装署名 {'あり' if sig else 'なし'}")
    g3_rob = (var["oos_rob_cagr"] - base["oos_rob_cagr"]) * 100
    g3_emp = (var["oos_emp_cagr"] - base["oos_emp_cagr"]) * 100
    print(f"  G3(IS較正→OOS素): rob {g3_rob:+.2f}pp / emp {g3_emp:+.2f}pp "
          f"-> {'OOSで負ける=採用根拠なし' if max(g3_rob, g3_emp) < 0 else 'OOSでも一部勝つ'}")
    yd = {y: (var["yr_emp"].get(y, np.nan) - base["yr_emp"].get(y, np.nan)) * 100
          for y in base["yr_emp"]}
    print("  年次差分(emp較正, pp): " + "  ".join(f"{y}:{v:+.2f}" for y, v in yd.items()))

    in_noise = abs(gain_pp) < NOISE_PP[1]
    g3_fail = (g3_rob < 0) and (g3_emp < 0)
    kill_a5 = (oos_cohort_sum > 0) and (in_noise or g3_fail)

    # ====================================================================
    sec("判定")
    attacks = {
        "A1_cell_selection": {"pct_single": pct_single, "p_fw": p_fw, "rank_weakest": rank,
                              "n_cells_family": int(vmask.sum()), "kill": bool(kill_a1)},
        "A2_2018_dependence": {"sum_ex2018": float(ex2018.sum()), "mean_ex2018_bps": float(ex2018.mean() * 1e4),
                               "n_2018": int(yr_t.loc[2018, "size"]) if 2018 in yr_t.index else 0,
                               "sum_2018": float(yr_t.loc[2018, "sum"]) if 2018 in yr_t.index else 0.0,
                               "kill": bool(kill_a2)},
        "A3_placebo_mechanism": {"window_total_pnl": win_total, "cohorts": rows,
                                 "incoherent": bool(a3_incoherent)},
        "A4_reprice_evidence": {"fri_med_bps": fri_med, "wdroll_med_bps": roll_med,
                                "liquid_med_bps": liq_med, "fri_vs_roll": fri_vs_roll,
                                "density": {c: float(jt[c].mean()) for c in dens_cols}},
        "A5_temporal_account": {"oos_cohort_sum": oos_cohort_sum,
                                "split_unstable": split_unstable,
                                "even": even, "odd": odd, "h1": h1, "h2": h2,
                                "rob_gain_pp": gain_pp, "rob_diffs_pp": diffs,
                                "emp_diff_pp": emp_diff, "signature": bool(sig),
                                "g3_rob_pp": g3_rob, "g3_emp_pp": g3_emp,
                                "in_noise_band": bool(in_noise), "g3_fail": bool(g3_fail),
                                "kill": bool(kill_a5)},
    }
    survives = not (kill_a1 or kill_a2 or kill_a5)
    print(f"A1 後知恵セル選択: {'致命傷' if kill_a1 else '生存'} "
          f"(単独%タイル {pct_single:.1f}%, family-wise p {p_fw:.3f})")
    print(f"A2 2018単年依存: {'致命傷(符号反転)' if kill_a2 else '生存'} "
          f"(2018除外 sum {float(ex2018.sum()):+.4f})")
    print(f"A3 機構整合: {'不整合(窓全体は正・出口/シグナル側は正)' if a3_incoherent else '整合'}")
    print(f"A4 独立証拠: 金曜窓/平日窓ジャンプ比 {fri_vs_roll:.2f}x(≈1なら金曜固有の困難なし)")
    print(f"A5 時間反転+口座: {'致命傷' if kill_a5 else '生存'} "
          f"(OOSコホート {oos_cohort_sum:+.4f}, rob利得 {gain_pp:+.2f}pp, G3 rob/emp "
          f"{g3_rob:+.2f}/{g3_emp:+.2f}pp)")
    print(f"\n>>> veto候補は{'生存' if survives else '死亡(机上では棄却。実弾の衛生ルールとしての言及のみ可)'}")

    # 保存
    cells.to_csv(OUT_DIR / "edge08_cells.csv", index=False)
    summary = {
        "claim": "entryラベル=金曜20:00 の29件はEV負(-5.3bps/再価格-7.3bps)、スキップは無料",
        "cohort": {"n": int(len(sub)), "mean_bps": obs_mean * 1e4, "sum": float(r_t.sum()),
                   "repriced_mean_bps": repriced},
        "attacks": attacks,
        "account": {"base": {k: ({str(s): v for s, v in vv.items()} if k == "rob" else vv)
                             for k, vv in base.items()},
                    "skip": {k: ({str(s): v for s, v in vv.items()} if k == "rob" else vv)
                             for k, vv in var.items()}},
        "survives": bool(survives),
    }
    (OUT_DIR / "edge08_summary.json").write_text(
        json.dumps(summary, indent=2, default=float, ensure_ascii=False))
    print(f"\nsaved -> {OUT_DIR / 'edge08_cells.csv'} / edge08_summary.json")
    print(f"総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
