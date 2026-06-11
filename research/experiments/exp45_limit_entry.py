"""exp45: エントリー指値化(スプレッド片道化)の実測 — fill率 / 逆選択 / mm層同一テール判定。

背景(reports/15 cost/mechanism):
  ・エントリー側スプレッド = 純益の 6.8%(理論上限)。
  ・エントリー1バー遅延は +2.7%(即時約定に非依存)。
  ・エントリー直後1〜3本は平均逆行 → 指値は現値より有利に置けて刺さりやすい。
  ・古典的な罠 = 逆選択(刺さらないのは即走った勝ちトレード)。未約定コホートの
    「成行なら得られた ret」を直接実測して、逆選択込みネットで判定する。

fill 判定の規約(BID のみデータ):
  ・メジャー7ペア: M1 OHLC(load_m1)の low/high。
  ・合成クロス12: 脚 M1 close の分単位合成 → 「M1 close が指値水準に達したか」で判定
    (分内の極値を見ない=保守的)。
  ・ロング買い指値 L: 約定には ASK≈BID+spread が L に達する必要
      → M1 BID low ≤ L − spread で約定(保守側)。
  ・ショート売り指値 L: BID ≥ L → M1 BID high ≥ L で約定。
  ・約定価格 = L(エントリー側半スプレッドが消える)。出口は従来通り
    (exit close ∓ 半スプレッド)= スプレッド片道化。
  ・有効期間 = シグナルバーの次の1本(H4=4時間)。シグナルバー close = C0。
  ・ロールオーバー感度: 指値窓が UTC20:00 バーのトレードだけ判定を厳格化
    (long: lo ≤ L − 2·sp / short: hi ≥ L + sp)した場合の差を1行報告。

執行ポリシー3案(本番ベースライン = シグナルバー close 成行 = C0 + dir·半スプレッド):
  (a) 指値@C0、未約定 → 次バー close で成行(フォールバック)
  (b) 指値@C0、未約定 → 見送り
  (c) 指値@C0 − dir·0.25·spread、未約定 → 次バー close で成行

mm層判定: 最良案(IS<2022 選択を正)で ret/entry_price を差し替えたプールを作り、
tail_protocol のペアシード較正(seeds 0-4, mp8)で robust/empirical をベースラインと比較。
adopt 基準: 逆選択込みネットで robust 平均差 +0.5pp 以上 + レバ偽装署名なし
+ 2022(最良年)除外で符号維持 + 負け年が増えない。

実行: PYTHONPATH=. uv run python research/experiments/exp45_limit_entry.py [--pool-only]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))

import mm_lab as mm  # noqa: E402
from mm_production import champion_sizing  # noqa: E402
from tail_protocol import (  # noqa: E402
    calibrate_empirical,
    cagr_of,
    max_dd,
    protocol_eval,
    yearly_returns,
)
from fxlab import config, data, universe as uni  # noqa: E402

pd.set_option("display.width", 240)

BASE_NET = 1.9086
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
MAX_POS = 8
SEEDS = (0, 1, 2, 3, 4)
OUT_CSV = ROOT / "research" / "outputs" / "exp45_limit_entry.csv"
OUT_JSON = ROOT / "research" / "outputs" / "exp45_limit_entry.json"


def sec(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --- M1 → H4ビンの BID 極値(fill 判定の材料) ---------------------------
def m1_bins(instr: str) -> pd.DataFrame:
    """H4ビン(floor 4h)ごとの M1 BID 極値。メジャー=low/high、クロス=合成closeのmin/max。"""
    if instr in uni.CROSS_DEFS:
        a, op, b = uni.CROSS_DEFS[instr]
        ca, cb = data.load_m1(a)["close"], data.load_m1(b)["close"]
        df = pd.concat([ca, cb], axis=1, join="inner").dropna()
        c = df.iloc[:, 0] / df.iloc[:, 1] if op == "/" else df.iloc[:, 0] * df.iloc[:, 1]
        g = c.groupby(c.index.floor("4h"))
        return pd.DataFrame({"lo": g.min(), "hi": g.max()})
    m1 = data.load_m1(instr)
    key = m1.index.floor("4h")
    return pd.DataFrame({"lo": m1["low"].groupby(key).min(),
                         "hi": m1["high"].groupby(key).max()})


def build_exec_table(pool: pd.DataFrame) -> pd.DataFrame:
    """トレードごとの執行シミュレーション材料を1表に集める。"""
    n = len(pool)
    C0 = np.full(n, np.nan)        # シグナルバー close(=ベースライン約定の素価格)
    Cx = np.full(n, np.nan)        # exit バー close
    C1 = np.full(n, np.nan)        # 次バー close(フォールバック成行)
    lo = np.full(n, np.nan)        # 指値窓(次バー)の M1 BID 安値
    hi = np.full(n, np.nan)        # 同 高値
    skip = np.zeros(n, dtype=bool)  # 次バー=exitバー以降(フォールバック不能=見送り)
    t1h = np.full(n, -1)           # 指値窓バーの UTC 時刻(時)
    sp = np.full(n, np.nan)        # スプレッド(価格単位)

    for instr, g in pool.groupby("instr"):
        h4 = uni.instrument_close(instr, "H4")
        hv = h4.to_numpy()
        idx_e = h4.index.get_indexer(g["entry"])
        idx_x = h4.index.get_indexer(g["exit"])
        assert (idx_e >= 0).all() and (idx_x >= 0).all(), f"{instr}: timestamp miss"
        rows = g.index.to_numpy()
        ie1 = np.minimum(idx_e + 1, len(h4) - 1)
        C0[rows] = hv[idx_e]
        Cx[rows] = hv[idx_x]
        C1[rows] = hv[ie1]
        skip[rows] = (idx_e + 1) >= idx_x
        t1 = pd.DatetimeIndex(h4.index.to_numpy()[ie1])
        t1h[rows] = t1.hour
        bins = m1_bins(instr)
        lo[rows] = bins["lo"].reindex(t1).to_numpy()
        hi[rows] = bins["hi"].reindex(t1).to_numpy()
        sp[rows] = config.spread_pips(instr) * config.pip_size(instr)

    t = pool.copy()
    t["C0"], t["Cx"], t["C1"] = C0, Cx, C1
    t["lo"], t["hi"], t["skip"], t["t1h"], t["sp"] = lo, hi, skip, t1h, sp
    t["year"] = t["exit"].dt.year  # 決済年集計(ベースライン規約)
    return t


# --- ポリシー評価(プール段) --------------------------------------------
def policy_pool(t: pd.DataFrame, name: str, limit_off: float, fallback: bool,
                rollover_strict: bool = False,
                dip_long: float = 1.0, dip_short: float = 0.0) -> dict:
    """1ポリシーのプール段評価。limit_off: L = C0 − dir·limit_off·sp。

    fill 条件(sp 単位): long lo ≤ L − dip_long·sp / short hi ≥ L + dip_short·sp。
    既定 (1.0, 0.0) = 規約(close=BID, ASK=BID+sp)。感度用に
    (0.5, 0.5) = close=mid 解釈 / (0.0, 0.0) = 楽観上限 も取れる。
    返り値: kept マスク・新 ret/entry_price とサマリ統計。
    ベースライン口座系: entry = C0 + dir·hsp / exit = Cx − dir·hsp(fxlab 半スプレッド×2)。
    指値約定 = L(エントリー側スプレッド消滅)。フォールバック = C1 + dir·hsp。
    """
    d = t["dir"].to_numpy().astype(float)
    sp = t["sp"].to_numpy()
    hsp = sp / 2.0
    C0, C1, Cx = t["C0"].to_numpy(), t["C1"].to_numpy(), t["Cx"].to_numpy()
    lo, hi = t["lo"].to_numpy(), t["hi"].to_numpy()
    ret0 = t["ret"].to_numpy()

    L = C0 - d * limit_off * sp
    extra = np.where(rollover_strict & (t["t1h"].to_numpy() == 20), 1.0, 0.0)
    filled = np.where(
        d > 0,
        lo <= L - sp * (dip_long + extra),    # long: ASK=BID+sp が L に到達(規約)
        hi >= L + sp * (dip_short + extra),   # short: BID が L に到達(規約)
    )
    filled &= np.isfinite(lo) & np.isfinite(hi)

    exit_eff = Cx - d * hsp
    entry_base = C0 + d * hsp
    recon = d * (exit_eff / entry_base - 1.0)
    ret_fill = ret0 + d * (exit_eff / L - 1.0) - recon
    entry_fb = C1 + d * hsp
    ret_fb = ret0 + d * (exit_eff / entry_fb - 1.0) - recon

    skip = t["skip"].to_numpy()
    if fallback:
        kept = filled | ~skip                  # 未約定でも次バーで成行(exit到達済みは見送り)
        ret_new = np.where(filled, ret_fill, ret_fb)
        eprice_new = np.where(filled, L, entry_fb)
    else:
        kept = filled
        ret_new = ret_fill
        eprice_new = L

    unf = ~filled
    base_sum = ret0.sum()
    new_sum = ret_new[kept].sum()
    impr_bps_fill = float(np.mean(d[filled] * (entry_base[filled] - L[filled])
                                  / entry_base[filled]) * 1e4) if filled.any() else np.nan
    is_mask = (t["entry"] < OOS_START).to_numpy()
    diff_tr = np.where(kept, ret_new - ret0, -ret0)   # トレードごとの差分(見送り=失う ret)
    yr = pd.Series(diff_tr).groupby(t["year"]).sum()
    ynew = pd.Series(np.where(kept, ret_new, 0.0)).groupby(t["year"]).sum()

    out = {
        "name": name,
        "fill_rate": float(filled.mean()),
        "fill_rate_long": float(filled[d > 0].mean()),
        "fill_rate_short": float(filled[d < 0].mean()),
        "n_dropped": int((~kept).sum()),
        "impr_bps_filled": impr_bps_fill,
        "unfilled_n": int(unf.sum()),
        "unfilled_base_ret_mean_bps": float(ret0[unf].mean() * 1e4) if unf.any() else np.nan,
        "unfilled_base_ret_med_bps": float(np.median(ret0[unf]) * 1e4) if unf.any() else np.nan,
        "unfilled_base_ret_sum": float(ret0[unf].sum()),
        "unfilled_winrate": float((ret0[unf] > 0).mean()) if unf.any() else np.nan,
        "filled_base_ret_mean_bps": float(ret0[filled].mean() * 1e4) if filled.any() else np.nan,
        "pool_net": float(new_sum),
        "pool_diff": float(new_sum - base_sum),
        "pool_diff_pct": float((new_sum - base_sum) / BASE_NET * 100),
        "diff_is": float(diff_tr[is_mask].sum()),
        "diff_oos": float(diff_tr[~is_mask].sum()),
        "neg_years_new": int((ynew < 0).sum()),
        "yearly_diff": yr,
        "kept": kept, "ret_new": ret_new, "eprice_new": eprice_new, "filled": filled,
    }
    return out


def print_policy(r: dict):
    print(f"\n--- {r['name']} ---")
    print(f"  fill率 {r['fill_rate']:.1%} (long {r['fill_rate_long']:.1%} / short {r['fill_rate_short']:.1%})"
          f"  見送り {r['n_dropped']}件  約定価格改善(filled平均) {r['impr_bps_filled']:+.2f}bps")
    print(f"  逆選択(未約定 n={r['unfilled_n']}): 成行なら ret 平均 {r['unfilled_base_ret_mean_bps']:+.1f}bps"
          f" / 中央値 {r['unfilled_base_ret_med_bps']:+.1f}bps / 勝率 {r['unfilled_winrate']:.0%}"
          f" / 合計 {r['unfilled_base_ret_sum']:+.4f}"
          f"   (約定コホート平均 {r['filled_base_ret_mean_bps']:+.1f}bps)")
    print(f"  プール純益 {r['pool_net']:+.4f} (基準 {BASE_NET:+.4f}, 差 {r['pool_diff']:+.4f}"
          f" = {r['pool_diff_pct']:+.2f}%)   IS差 {r['diff_is']:+.4f} / OOS差 {r['diff_oos']:+.4f}")
    print(f"  年次(決済年)差分: " + "  ".join(f"{y}:{v*1e2:+.2f}%" for y, v in r["yearly_diff"].items()))
    print(f"  新プールのマイナス年(決済年 sum(ret)<0): {r['neg_years_new']}")


# --- mm層評価(exp37 流) -------------------------------------------------
def eval_mm(label: str, pool: pd.DataFrame, closes: pd.DataFrame, seeds=SEEDS) -> dict:
    mk = champion_sizing(pool, max_pos=MAX_POS)
    cache: dict[float, pd.Series] = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            eqm, _, _ = mm.simulate(pool, closes, mk(kk), max_pos=MAX_POS)
            cache[kk] = eqm
        return cache[kk]

    res = protocol_eval(eq_of_k, label=label, seeds=seeds)
    eq_emp = eq_of_k(res["emp_k"])
    yr = yearly_returns(eq_emp)
    res["yearly_emp"] = yr
    res["worst_year"] = float(yr.min())
    res["neg_years"] = int((yr < 0).sum())

    # IS 較正 → OOS 素検証(empirical)
    is_pool = pool[pool["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]
    oos_cl = closes[closes.index >= OOS_START]

    def eq_is(k):
        eqm, _, _ = mm.simulate(is_pool, is_cl, mk(k), max_pos=MAX_POS)
        return eqm

    k_is = calibrate_empirical(eq_is, 0.20)
    eqo, _, _ = mm.simulate(oos_pool, oos_cl, mk(k_is), max_pos=MAX_POS)
    res["k_is"] = k_is
    res["oos_emp_cagr"] = cagr_of(eqo)
    res["oos_emp_dd"] = max_dd(eqo)
    print(f"      IS k={k_is:5.2f} -> OOS CAGR={res['oos_emp_cagr']:+7.2%} DD={res['oos_emp_dd']:+6.1%}"
          f"   worst_year={res['worst_year']:+.1%} neg_years={res['neg_years']}")
    return res


def make_mod_pool(pool: pd.DataFrame, r: dict) -> pd.DataFrame:
    """ポリシー結果でプールの ret / entry_price を差し替え(見送りは除外)。

    entry タイムスタンプは据え置き: mm.simulate は entry バーの翌バーから MtM 評価する
    ので、「指値が次バー内で約定し fill 価格 L から時価評価」が正しく表現される。
    """
    mod = pool.copy()
    mod["ret"] = r["ret_new"]
    mod["entry_price"] = r["eprice_new"]
    mod = mod[r["kept"]].reset_index(drop=True)
    return mod


def main() -> int:
    t_start = time.time()
    uni.register_cross_spreads(3.0)
    pool = mm.build_pool()
    closes = mm.load_closes()
    print(f"pool {len(pool)} trades  sum(ret)={pool['ret'].sum():+.4f}  (基準 {BASE_NET:+.4f})")

    sec("0. 執行テーブル構築(M1ビン極値 + ベースライン口座系の検算)")
    t = build_exec_table(pool)
    n_nan = int((~np.isfinite(t["lo"].to_numpy())).sum())
    print(f"指値窓の M1 ビン欠損: {n_nan}件(欠損=未約定扱い)  "
          f"窓=20:00バー: {(t['t1h'] == 20).sum()}件  次バー=exitバー以降(skip): {t['skip'].sum()}件")
    # 検算: pool.entry_price ≈ C0 + dir·半スプレッド / ret ≈ dir·(exit_eff/entry_eff − 1)
    d = t["dir"].to_numpy().astype(float)
    hsp = t["sp"].to_numpy() / 2.0
    eb = t["C0"].to_numpy() + d * hsp
    ep_err = np.abs(eb - t["entry_price"].to_numpy()) / t["entry_price"].to_numpy()
    recon = d * ((t["Cx"].to_numpy() - d * hsp) / eb - 1.0)
    ret_err = np.abs(recon - t["ret"].to_numpy())
    print(f"entry_price 再構成誤差: median {np.median(ep_err):.2e} / max {ep_err.max():.2e}")
    print(f"ret 再構成誤差:        median {np.median(ret_err):.2e} / max {ret_err.max():.2e}")
    # 理論上限(全件 fill率100%で L=C0 が刺さった場合)
    exit_eff = t["Cx"].to_numpy() - d * hsp
    ub = (t["ret"].to_numpy() + d * (exit_eff / t["C0"].to_numpy() - 1.0) - recon).sum() - BASE_NET
    print(f"理論上限(fill率100%@C0): プール差 {ub:+.4f} = 純益の {ub/BASE_NET*100:+.2f}%")

    sec("1. 執行ポリシー3案のプール段実測(fill率 / 逆選択 / 純益差分 / 年次)")
    pol_a = policy_pool(t, "(a) 指値@C0 → 未約定は次バー成行", 0.0, fallback=True)
    pol_b = policy_pool(t, "(b) 指値@C0 → 未約定は見送り", 0.0, fallback=False)
    pol_c = policy_pool(t, "(c) 指値@C0−0.25sp → 未約定は次バー成行", 0.25, fallback=True)
    pols = {"a": pol_a, "b": pol_b, "c": pol_c}
    for r in pols.values():
        print_policy(r)

    # IS-argmax(ポリシー選択を IS のみで行う)
    best_full = max(pols, key=lambda k: pols[k]["pool_diff"])
    best_is = max(pols, key=lambda k: pols[k]["diff_is"])
    print(f"\nフル期間選択: ({best_full})  /  IS(<2022)選択: ({best_is})"
          + ("  → 一致" if best_full == best_is else "  → 不一致: IS 選択を正とする"))
    chosen = best_is

    sec(f"2. ロールオーバー感度(採用案 ({chosen}) の 20:00 窓 fill 判定を spread×2 に厳格化)")
    off = {"a": 0.0, "b": 0.0, "c": 0.25}[chosen]
    fb = {"a": True, "b": False, "c": True}[chosen]
    r_strict = policy_pool(t, f"({chosen}) rollover-strict", off, fallback=fb, rollover_strict=True)
    print(f"  fill率 {pols[chosen]['fill_rate']:.1%} → {r_strict['fill_rate']:.1%}, "
          f"プール差 {pols[chosen]['pool_diff']:+.4f} → {r_strict['pool_diff']:+.4f} "
          f"(変化 {r_strict['pool_diff']-pols[chosen]['pool_diff']:+.4f})")

    sec(f"2b. fill判定規約の感度(採用案 ({chosen})): BID規約 / mid解釈 / 楽観上限")
    sens = {}
    for tag, dl, ds in [("BID規約(long −sp / short ±0)", 1.0, 0.0),
                        ("mid解釈(両側 −sp/2)", 0.5, 0.5),
                        ("楽観上限(タッチで約定)", 0.0, 0.0)]:
        rs = policy_pool(t, tag, off, fallback=fb, dip_long=dl, dip_short=ds)
        sens[tag] = rs
        print(f"  {tag:28s} fill率 {rs['fill_rate']:5.1%}  プール差 {rs['pool_diff']:+.4f} "
              f"({rs['pool_diff_pct']:+.2f}%)  未約定の成行ret平均 {rs['unfilled_base_ret_mean_bps']:+.1f}bps")

    if "--pool-only" in sys.argv:
        print("\n--pool-only: mm層はスキップ")
        return 0

    sec(f"3. mm層 同一テール判定(ペアシード {SEEDS}, mp{MAX_POS}): ベースライン vs ({chosen})")
    res_base = eval_mm("baseline_mp8(成行)", pool, closes)
    mod = make_mod_pool(pool, pols[chosen])
    res_cand = eval_mm(f"limit_({chosen})_mp8", mod, closes)
    # フル期間選択がISと違う場合は参考でフル選択も回す
    res_alt = None
    if best_full != chosen:
        mod_alt = make_mod_pool(pool, pols[best_full])
        res_alt = eval_mm(f"limit_({best_full})_mp8(参考:フル期間選択)", mod_alt, closes)

    sec("4. プロトコル判定表")
    rows = []
    for lab, r in [("baseline", res_base), (f"limit_{chosen}", res_cand)] + (
            [(f"limit_{best_full}_ref", res_alt)] if res_alt else []):
        rows.append({
            "config": lab, "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"],
            "emp_p95": r["emp_p95"], "rob_mean5": r["rob_cagr_mean"],
            **{f"rob_s{s}": r["rob"][s]["cagr"] for s in SEEDS},
            "worst_year": r["worst_year"], "neg_years": r["neg_years"],
            "k_is": r["k_is"], "oos_emp_cagr": r["oos_emp_cagr"],
        })
    tab = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(tab.to_string(index=False))

    dseed = {s: res_cand["rob"][s]["cagr"] - res_base["rob"][s]["cagr"] for s in SEEDS}
    dmean = float(np.mean(list(dseed.values())))
    print("\nper-seed robust 差 (candidate − baseline): "
          + "  ".join(f"s{s}:{v:+.2%}" for s, v in dseed.items()) + f"   平均 {dmean:+.2%}")
    demp = res_cand["emp_cagr"] - res_base["emp_cagr"]
    dp95 = res_cand["emp_p95"] - res_base["emp_p95"]  # p95は負値。さらに負=悪化
    print(f"empirical 差 {demp:+.2%}   p95: base {res_base['emp_p95']:+.1%} → cand {res_cand['emp_p95']:+.1%}"
          f" (差 {dp95:+.2%})")
    lev_disguise = (demp > 0) and (dp95 < -0.005)
    print(f"レバ偽装署名(emp↑ & p95悪化>0.5pp): {'あり=reject' if lev_disguise else 'なし'}")

    sec("5. 年次分解(empirical 較正パスの年次差)と 2022(最良年)除外チェック")
    ydiff = (res_cand["yearly_emp"] - res_base["yearly_emp"]).dropna()
    print("mm層 年次差(emp較正): " + "  ".join(f"{y}:{v:+.2%}" for y, v in ydiff.items()))
    best_y = ydiff.idxmax()
    print(f"mm層: 最良年 {best_y} を除いた残差合計 {ydiff.drop(best_y).sum():+.2%} "
          f"(符号{'維持' if ydiff.drop(best_y).sum() > 0 else '反転=単年依存'})")
    yp = pols[chosen]["yearly_diff"]
    best_yp = yp.idxmax()
    print(f"プール段: 最良年 {best_yp} 除外後の差分合計 {yp.drop(best_yp).sum():+.4f} "
          f"(符号{'維持' if yp.drop(best_yp).sum() > 0 else '反転'})")

    # --- 保存 ---------------------------------------------------------------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT_CSV, index=False)
    payload = {
        "chosen": chosen, "best_full": best_full,
        "policies": {k: {kk: vv for kk, vv in v.items()
                         if kk not in ("kept", "ret_new", "eprice_new", "filled", "yearly_diff")}
                     | {"yearly_diff": {int(y): float(x) for y, x in v["yearly_diff"].items()}}
                     for k, v in pols.items()},
        "rollover_strict_diff": r_strict["pool_diff"],
        "fill_convention_sensitivity": {k: {"fill_rate": v["fill_rate"],
                                            "pool_diff": v["pool_diff"],
                                            "pool_diff_pct": v["pool_diff_pct"]}
                                        for k, v in sens.items()},
        "robust_diff_per_seed": {int(s): float(v) for s, v in dseed.items()},
        "robust_diff_mean": dmean, "emp_diff": float(demp), "p95_diff": float(dp95),
        "lev_disguise": bool(lev_disguise),
        "mm_yearly_diff": {int(y): float(v) for y, v in ydiff.items()},
        "base": {k: v for k, v in res_base.items() if k not in ("rob", "yearly_emp")},
        "cand": {k: v for k, v in res_cand.items() if k not in ("rob", "yearly_emp")},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nsaved -> {OUT_CSV}\n        -> {OUT_JSON}\n経過 {time.time()-t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
