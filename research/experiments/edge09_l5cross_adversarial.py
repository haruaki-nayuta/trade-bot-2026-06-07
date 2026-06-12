"""edge09: veto候補「末尾5分限界クロス拒否(spike_made_l5, n=98)」の敵対検証(殺しに行く側)。

殺すべき主張(edge01 が指名した唯一の veto 候補):
  「シグナルの存在が末尾5分のスライバーに依存する限界クロス98件はエッジゼロ
   (+1.8bps, diff CI95=[-30.1,-2.9])の死荷重。拒否でプール質が上がる」

攻撃ベクトル(全部実行):
  A. 閉鎖済み軸の言い換え検定(本命): M1 を一切使わず「m = |z_sig|-2.0 < c」だけで作った
     プラセボ限界コホートの成績曲線に対し、spike_made_l5 の +1.8bps が区別できるか。
     (a) マッチド n プラセボ(m 最小98件)との重なり・平均比較
     (b) m 層化置換検定(マージナリティ条件付きで膜を張り、l5 ラベルが増分情報を持つか)
     (c) 最近傍 m マッチドペア(l5 − 対照)のブート CI
     区別できなければ「z 深さの言い換え」= reports/20 で ML(z深さ込み)拒否不能と確定済みの
     軸の再訴訟 = 既知で死亡。
  B. dose-response: 置換点を 3/5/10/15/30 分前にした除外コホートの EV 曲線。
     5分だけ特別なら後知恵、単調なら「マージナリティのダイヤル」=機構的に言い換え。
  C. L/S 非対称(-11.7 vs +13.3bps)の崩壊試験: IS/OOS・偶奇年・前後半の分割で符号安定か。
  D. 総PnL負効果と k 較正機構: 98件の建玉期間と DD 谷の重なり / 固定 k での DD・p95 変化 /
     除外プールの再シミュレート(ペアシード robust 較正 seeds 0-4 + G3 IS→OOS + レバ偽装署名)。
     規約: トレード除外は必ず再シミュレート(単純引き算禁止)。

判定: プラセボと区別不能 or メンバーシップ脆弱 or k較正利得の機構不在なら survives=False。

実行: uv run python research/experiments/edge09_l5cross_adversarial.py(リポジトリ直下)
出力: research/outputs/edge09_result.json / edge09_dose_response.csv
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
from fxlab import config  # noqa: E402
from fxlab.data import load_m1  # noqa: E402
from fxlab import universe as uni  # noqa: E402
from fxlab.universe import CROSS_DEFS  # noqa: E402

pd.set_option("display.width", 240)

OUT_DIR = ROOT / "research" / "outputs"
OUT_JSON = OUT_DIR / "edge09_result.json"
OUT_DOSE = OUT_DIR / "edge09_dose_response.csv"

POOL_PATH = ROOT / "results" / "mm_pool_v2d1_H4_19.parquet"
E01_CSV = OUT_DIR / "edge01_trades.csv"
EXPECT_N = 1207
EXPECT_SUM = 1.9622
EXPECT_L5_N = 98

WINDOW = 50
ENTRY_Z = 2.0
MAX_POS = 8
SEEDS = (0, 1, 2, 3, 4)
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
H4NS = np.timedelta64(4, "h").astype("timedelta64[ns]").astype(np.int64)
TAUS = [3, 5, 10, 15, 30]  # 置換点(バー終端からの分)
NOISE_BAND_PP = 0.4        # robust 較正ノイズの下端(±0.4-0.8pp)

RESULT: dict = {}


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# ---------------------------------------------------------------------------
# 共通: ブート CI
# ---------------------------------------------------------------------------
def boot_mean_ci(x: np.ndarray, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    if len(x) < 2:
        return (np.nan, np.nan)
    ms = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(ms, [2.5, 97.5])
    return float(lo), float(hi)


def boot_diff_ci(a: np.ndarray, b: np.ndarray, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    ma = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    mb = rng.choice(b, size=(n_boot, len(b)), replace=True).mean(axis=1)
    lo, hi = np.percentile(ma - mb, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# A. プラセボ(M1 不使用・z マージナリティのみ)
# ---------------------------------------------------------------------------
def attack_A(df: pd.DataFrame) -> dict:
    sec("A. 閉鎖済み軸の言い換え検定: |z_sig|-2.0 プラセボ vs spike_made_l5")
    ret = df["ret"].to_numpy()
    m = np.abs(df["z_sig"].to_numpy()) - ENTRY_Z
    l5 = df["l5"].to_numpy()
    n5 = int(l5.sum())
    obs_mean = float(ret[l5].mean())
    out: dict = {"l5_n": n5, "l5_mean_bps": obs_mean * 1e4}

    print(f"l5: n={n5}  mean={obs_mean*1e4:+.1f}bps  "
          f"m(=|z_sig|-2.0) median {np.median(m[l5]):.4f} vs rest {np.median(m[~l5]):.4f}")

    # (a) マッチド n プラセボ: m 最小 98 件
    order = np.argsort(m)
    placebo = np.zeros(len(df), dtype=bool)
    placebo[order[:n5]] = True
    inter = int((placebo & l5).sum())
    p_mean = float(ret[placebo].mean())
    ci = boot_diff_ci(ret[l5], ret[placebo], seed=1)
    out["placebo_matched_n"] = {
        "n": n5, "overlap_with_l5": inter, "jaccard": inter / int((placebo | l5).sum()),
        "placebo_mean_bps": p_mean * 1e4,
        "diff_l5_minus_placebo_bps_ci95": [ci[0] * 1e4, ci[1] * 1e4],
        "c_threshold": float(m[order[n5 - 1]]),
    }
    print(f"\n(a) プラセボ(m 最小{n5}件, c={out['placebo_matched_n']['c_threshold']:.4f}): "
          f"mean={p_mean*1e4:+.1f}bps  l5との重なり {inter}/{n5}  "
          f"(Jaccard {out['placebo_matched_n']['jaccard']:.2f})")
    print(f"    diff(l5 − placebo) CI95 = [{ci[0]*1e4:+.1f}, {ci[1]*1e4:+.1f}]bps  "
          f"-> {'区別不能' if ci[0] < 0 < ci[1] else '区別可能'}")

    # プラセボ EV 曲線(m 最小 n 件の平均 bps)
    curve = []
    for nn in [50, 75, 98, 125, 150, 200, 300, 500, 725]:
        sel = np.zeros(len(df), dtype=bool)
        sel[order[:nn]] = True
        clo, chi = boot_mean_ci(ret[sel], seed=2)
        curve.append({"n": nn, "mean_bps": float(ret[sel].mean() * 1e4),
                      "ci95_bps": [clo * 1e4, chi * 1e4]})
    out["placebo_curve"] = curve
    print("    プラセボ EV 曲線(m 最小 n 件):")
    for c in curve:
        mark = "  <- l5 と同 n" if c["n"] == n5 else ""
        print(f"      n={c['n']:>4}: {c['mean_bps']:+6.1f}bps  "
              f"CI[{c['ci95_bps'][0]:+6.1f},{c['ci95_bps'][1]:+6.1f}]{mark}")
    in_band = curve[2]["ci95_bps"][0] <= obs_mean * 1e4 <= curve[2]["ci95_bps"][1]
    print(f"    l5 の {obs_mean*1e4:+.1f}bps はプラセボ n=98 の CI 内: {in_band}")
    out["l5_within_placebo98_ci"] = bool(in_band)

    # (b) m 層化置換検定: マージナリティを固定して l5 ラベルだけ並べ替える
    n_bins = 40
    qs = np.unique(np.quantile(m, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.searchsorted(qs, m, side="right") - 1, 0, len(qs) - 2)
    rng = np.random.default_rng(7)
    n_perm = 4000
    null_means = np.empty(n_perm)
    idx_by_bin = {b: np.flatnonzero(bins == b) for b in np.unique(bins)}
    k_by_bin = {b: int(l5[v].sum()) for b, v in idx_by_bin.items()}
    for i in range(n_perm):
        tot, cnt = 0.0, 0
        for b, v in idx_by_bin.items():
            k = k_by_bin[b]
            if k == 0:
                continue
            pick = rng.choice(v, size=k, replace=False)
            tot += ret[pick].sum()
            cnt += k
        null_means[i] = tot / cnt
    p_low = float((null_means <= obs_mean).mean())   # 観測がヌルより低い側
    p_two = float((np.abs(null_means - null_means.mean())
                   >= abs(obs_mean - null_means.mean())).mean())
    out["stratified_permutation"] = {
        "n_bins": int(len(qs) - 1), "n_perm": n_perm,
        "null_mean_bps": float(null_means.mean() * 1e4),
        "null_ci95_bps": [float(np.percentile(null_means, 2.5) * 1e4),
                          float(np.percentile(null_means, 97.5) * 1e4)],
        "obs_bps": obs_mean * 1e4, "p_one_sided_low": p_low, "p_two_sided": p_two,
    }
    sp = out["stratified_permutation"]
    print(f"\n(b) m 層化置換検定(bins={sp['n_bins']}, perm={n_perm}): "
          f"ヌル平均 {sp['null_mean_bps']:+.1f}bps CI[{sp['null_ci95_bps'][0]:+.1f},"
          f"{sp['null_ci95_bps'][1]:+.1f}] / 観測 {sp['obs_bps']:+.1f}bps")
    print(f"    p(片側 観測≤ヌル)={p_low:.3f}  p(両側)={p_two:.3f}  -> "
          f"{'l5 ラベルに増分情報なし(マージナリティで説明済み)' if p_low > 0.05 else 'l5 はマージナリティ超の低EV'}")

    # (c) 最近傍 m マッチドペア(without replacement)
    l5_idx = np.flatnonzero(l5)
    pool_idx = np.flatnonzero(~l5)
    used = np.zeros(len(df), dtype=bool)
    pairs = []
    for ti in l5_idx[np.argsort(m[l5_idx])]:
        cand = pool_idx[~used[pool_idx]]
        j = cand[np.argmin(np.abs(m[cand] - m[ti]))]
        used[j] = True
        pairs.append((ti, j))
    d = np.array([ret[a] - ret[b] for a, b in pairs])
    dm = np.array([abs(m[a] - m[b]) for a, b in pairs])
    plo, phi = boot_mean_ci(d, seed=3)
    out["matched_pairs"] = {
        "n_pairs": len(pairs), "mean_diff_bps": float(d.mean() * 1e4),
        "ci95_bps": [plo * 1e4, phi * 1e4], "median_m_gap": float(np.median(dm)),
    }
    mp = out["matched_pairs"]
    print(f"\n(c) 最近傍 m マッチドペア(n={mp['n_pairs']}, m差中央値 {mp['median_m_gap']:.4f}): "
          f"l5−対照 = {mp['mean_diff_bps']:+.1f}bps CI95[{mp['ci95_bps'][0]:+.1f},"
          f"{mp['ci95_bps'][1]:+.1f}]")
    print(f"    -> {'0 を跨ぐ = l5 メンバーシップ自体に独自エッジ差なし' if plo < 0 < phi else '独自差あり'}")

    out["indistinguishable_from_placebo"] = bool(
        (ci[0] < 0 < ci[1]) and p_low > 0.05 and (plo < 0 < phi))
    return out


# ---------------------------------------------------------------------------
# M1 ロード(edge01 と同一方式)+ 反実仮想 z 計算基盤
# ---------------------------------------------------------------------------
def load_m1_closes(instruments: list[str]) -> dict[str, pd.Series]:
    majors = sorted({leg for nm in instruments if nm in CROSS_DEFS
                     for leg in (CROSS_DEFS[nm][0], CROSS_DEFS[nm][2])}
                    | {nm for nm in instruments if nm not in CROSS_DEFS})
    raw = {}
    for p in majors:
        c = load_m1(p)["close"]
        raw[p] = pd.Series(c.to_numpy(), index=c.index.tz_localize(None))
    out = {}
    for nm in instruments:
        if nm in CROSS_DEFS:
            a, op, b = CROSS_DEFS[nm]
            d = pd.concat([raw[a].rename("a"), raw[b].rename("b")],
                          axis=1, join="inner").dropna()
            out[nm] = d["a"] / d["b"] if op == "/" else d["a"] * d["b"]
        else:
            out[nm] = raw[nm]
    return out


def build_cf_base(pool: pd.DataFrame):
    """各トレードの sig バー: 置換点 close(τ別)+ H4 rolling 窓 seg を前計算。"""
    instruments = sorted(pool["instr"].unique())
    m1 = load_m1_closes(instruments)
    m1_arr = {nm: (s.index.to_numpy().astype(np.int64), s.to_numpy()) for nm, s in m1.items()}
    h4 = {}
    for nm in instruments:
        close = uni.instrument_data(nm, "H4")["close"]
        h4[nm] = {"idx": close.index, "close": close.to_numpy()}
    n = len(pool)
    surro = {tau: np.full(n, np.nan) for tau in TAUS}
    segs = np.full((n, WINDOW), np.nan)
    dirs = pool["dir"].to_numpy()
    for instr, g in pool.groupby("instr"):
        times, vals = m1_arr[instr]
        H = h4[instr]
        e_pos = H["idx"].get_indexer(g["entry"])
        assert (e_pos > 0).all(), f"{instr}: entry not found"
        for ti, ep in zip(g.index, e_pos):
            sig_pos = ep - 1
            if sig_pos - WINDOW + 1 < 0:
                continue
            seg = H["close"][sig_pos - WINDOW + 1: sig_pos + 1].astype(float)
            if not np.isfinite(seg).all():
                continue
            segs[ti] = seg
            t0 = H["idx"][sig_pos].tz_localize(None).value
            t1 = t0 + H4NS
            a = int(np.searchsorted(times, t0, "left"))
            b = int(np.searchsorted(times, t1, "left"))
            if b - a < 5:
                continue
            t = times[a:b]
            v = vals[a:b]
            for tau in TAUS:
                k = int(np.searchsorted(
                    t, t1 - np.timedelta64(tau, "m").astype("timedelta64[ns]").astype(np.int64),
                    "left"))
                if k > 0:
                    surro[tau][ti] = v[k - 1]
    return {"segs": segs, "surro": surro, "dirs": dirs}


def cf_membership(cfb: dict, tau: int, bump: np.ndarray | None = None) -> np.ndarray:
    """surro(τ)(+bump)で sig close を置換 → z クロス不成立なら True(=コホート員)。"""
    segs, dirs = cfb["segs"], cfb["dirs"]
    s = cfb["surro"][tau].copy()
    if bump is not None:
        s = s + bump
    n = len(dirs)
    member = np.zeros(n, dtype=bool)
    ok = np.isfinite(s) & np.isfinite(segs).all(axis=1)
    idx = np.flatnonzero(ok)
    s2 = segs[idx].copy()
    s2[:, -1] = s[idx]
    mu = s2.mean(axis=1)
    sd = s2.std(axis=1, ddof=1)
    z = np.where(sd > 0, (s[idx] - mu) / sd, np.nan)
    crossed = np.where(dirs[idx] == 1, z < -ENTRY_Z, z > ENTRY_Z)
    member[idx] = ~crossed & np.isfinite(z)
    return member


# ---------------------------------------------------------------------------
# B. dose-response(τ = 3/5/10/15/30 分前置換)
# ---------------------------------------------------------------------------
def attack_B(df: pd.DataFrame, cfb: dict) -> dict:
    sec("B. dose-response: 置換点 τ 分前の除外コホート EV 曲線")
    ret = df["ret"].to_numpy()
    m = np.abs(df["z_sig"].to_numpy()) - ENTRY_Z
    ent = df["entry"]
    if ent.dt.tz is None:
        ent = ent.dt.tz_localize("UTC")
    oos = (ent >= OOS_START).to_numpy()
    rows = []
    members = {}
    for tau in TAUS:
        mem = cf_membership(cfb, tau)
        members[tau] = mem
        nn = int(mem.sum())
        rest = ~mem
        ci = boot_diff_ci(ret[mem], ret[rest], seed=10 + tau) if nn >= 2 else (np.nan, np.nan)
        rows.append({
            "tau_min": tau, "n": nn, "mean_bps": float(ret[mem].mean() * 1e4),
            "rest_mean_bps": float(ret[rest].mean() * 1e4),
            "diff_ci95_lo_bps": ci[0] * 1e4, "diff_ci95_hi_bps": ci[1] * 1e4,
            "sum_pnl": float(ret[mem].sum()),
            "oos_mean_bps": float(ret[mem & oos].mean() * 1e4) if (mem & oos).any() else np.nan,
            "median_marginality": float(np.median(m[mem])) if nn else np.nan,
        })
    dose = pd.DataFrame(rows)
    print(dose.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    dose.to_csv(OUT_DOSE, index=False)

    # τ=5 が edge01 の 98 件と一致するか(検算)
    l5 = df["l5"].to_numpy()
    match5 = bool((members[5] == l5).all())
    print(f"\n検算: τ=5 メンバーシップ == edge01 spike_made_l5: {match5} "
          f"(n={int(members[5].sum())})")

    ns = dose["n"].to_numpy()
    mono_n = bool(np.all(np.diff(ns) >= 0))
    means = dose["mean_bps"].to_numpy()
    print(f"n 単調増加(τ↑): {mono_n} / 平均bps 曲線: " +
          " ".join(f"τ{t}:{v:+.1f}" for t, v in zip(TAUS, means)))
    # 「5分が特別か」: τ=5 の mean が両隣(3,10)の単調補間から大きく外れるか
    interp5 = means[0] + (means[2] - means[0]) * (5 - 3) / (10 - 3)
    dev5 = float(means[1] - interp5)
    print(f"τ=5 の隣接補間からの乖離: {dev5:+.1f}bps "
          f"(コホート CI 幅 ~{(dose.loc[1,'diff_ci95_hi_bps']-dose.loc[1,'diff_ci95_lo_bps'])/2:.0f}bps)")
    return {"table": rows, "tau5_matches_edge01": match5, "n_monotone": mono_n,
            "tau5_dev_from_interp_bps": dev5, "members5": members[5]}


# ---------------------------------------------------------------------------
# C. L/S 非対称の崩壊試験
# ---------------------------------------------------------------------------
def _ls_diff(sub: pd.DataFrame):
    lo = sub[sub["dir"] == 1]["ret"]
    sh = sub[sub["dir"] == -1]["ret"]
    return (len(lo), float(lo.mean() * 1e4) if len(lo) else np.nan,
            len(sh), float(sh.mean() * 1e4) if len(sh) else np.nan)


def attack_C(df: pd.DataFrame) -> dict:
    sec("C. L/S 非対称(-11.7 / +13.3bps)の分割安定性")
    sub = df[df["l5"]]
    nl, ml, ns_, ms_ = _ls_diff(sub)
    lo = sub[sub["dir"] == 1]["ret"].to_numpy()
    sh = sub[sub["dir"] == -1]["ret"].to_numpy()
    ci = boot_diff_ci(sh, lo, seed=21)
    print(f"全期間: Long n={nl} {ml:+.1f}bps / Short n={ns_} {ms_:+.1f}bps  "
          f"diff(S−L)={ms_-ml:+.1f}bps CI95[{ci[0]*1e4:+.1f},{ci[1]*1e4:+.1f}]")
    splits = {}
    ent = pd.DatetimeIndex(sub["entry"])
    med_date = sub["entry"].median()
    defs = {
        "IS": sub["entry"] < OOS_START, "OOS": sub["entry"] >= OOS_START,
        "even_year": pd.Series(ent.year % 2 == 0, index=sub.index),
        "odd_year": pd.Series(ent.year % 2 == 1, index=sub.index),
        "first_half": sub["entry"] <= med_date, "second_half": sub["entry"] > med_date,
    }
    signs = []
    for tag, mask in defs.items():
        s = sub[mask.to_numpy() if hasattr(mask, "to_numpy") else mask]
        a, b, c, d = _ls_diff(s)
        diff = (d - b) if np.isfinite(b) and np.isfinite(d) else np.nan
        splits[tag] = {"n_long": a, "long_bps": b, "n_short": c, "short_bps": d,
                       "sl_diff_bps": diff}
        if np.isfinite(diff):
            signs.append(diff > 0)
        print(f"  {tag:12s}: L n={a:>3} {b:+7.1f} | S n={c:>3} {d:+7.1f} | S−L {diff:+7.1f}")
    stable = bool(all(signs)) if signs else False
    sig_full = not (ci[0] < 0 < ci[1])
    print(f"S−L 符号が全分割で同符号: {stable} / 全期間 CI が 0 を跨がない: {sig_full}")
    # 参考: プール全体とプラセボ帯の L/S
    m = np.abs(df["z_sig"].to_numpy()) - ENTRY_Z
    order = np.argsort(m)
    pl = df.iloc[order[:98]]
    a, b, c, d = _ls_diff(pl)
    print(f"参考 プラセボ(m最小98): L n={a} {b:+.1f} / S n={c} {d:+.1f} (S−L {d-b:+.1f})")
    a, b, c, d = _ls_diff(df)
    print(f"参考 プール全体     : L n={a} {b:+.1f} / S n={c} {d:+.1f} (S−L {d-b:+.1f})")
    return {"full_ci_sl_diff_bps": [ci[0] * 1e4, ci[1] * 1e4], "splits": splits,
            "sign_stable_all_splits": stable, "full_significant": bool(sig_full)}


# ---------------------------------------------------------------------------
# D-pre. 実装ノイズ耐性(±0.5pip 摂動でのメンバーシップ脆弱性)
# ---------------------------------------------------------------------------
def attack_noise(df: pd.DataFrame, cfb: dict, base_mem: np.ndarray) -> dict:
    sec("4. 実装ノイズ耐性: surro(5分前 close)±0.5pip 摂動のメンバーシップ入替")
    pips = np.array([config.pip_size(i) for i in df["instr"]])
    res = {}
    n0 = int(base_mem.sum())
    for tag, bump in [("+0.5pip", 0.5 * pips), ("-0.5pip", -0.5 * pips)]:
        mem = cf_membership(cfb, 5, bump=bump)
        stay = int((mem & base_mem).sum())
        leave = n0 - stay
        join = int((mem & ~base_mem).sum())
        res[tag] = {"n": int(mem.sum()), "stay": stay, "leave": leave, "join": join,
                    "turnover_pct": (leave + join) / n0 * 100}
        print(f"  {tag}: n={int(mem.sum())}  残留 {stay}/{n0}  離脱 {leave}  新加入 {join}  "
              f"入替率 {(leave+join)/n0*100:.0f}%")
    rng = np.random.default_rng(42)
    tos = []
    for s in range(20):
        bump = rng.uniform(-0.5, 0.5, len(df)) * pips
        mem = cf_membership(cfb, 5, bump=bump)
        leave = int((base_mem & ~mem).sum())
        join = int((mem & ~base_mem).sum())
        tos.append((leave + join) / n0 * 100)
    res["uniform_noise_20seeds"] = {"mean_turnover_pct": float(np.mean(tos)),
                                    "max_turnover_pct": float(np.max(tos))}
    print(f"  U(-0.5,+0.5)pip ×20seed: 平均入替率 {np.mean(tos):.0f}% / 最大 {np.max(tos):.0f}%")
    # 反実 z の閾値距離分布(83% が ±0.1 の確認)
    segs, dirs = cfb["segs"], cfb["dirs"]
    s = cfb["surro"][5]
    ok = np.isfinite(s) & np.isfinite(segs).all(axis=1)
    s2 = segs[ok].copy()
    s2[:, -1] = s[ok]
    z = (s[ok] - s2.mean(axis=1)) / s2.std(axis=1, ddof=1)
    dist = np.abs(np.abs(z) - ENTRY_Z)
    within = float((dist[base_mem[ok]] < 0.1).mean() * 100)
    res["cf_z_within_0p1_pct"] = within
    print(f"  l5 員の反実 z 閾値距離 <0.1: {within:.0f}%(edge01 の 83% 主張の検算)")
    res["fragile"] = bool(max(res["+0.5pip"]["turnover_pct"], res["-0.5pip"]["turnover_pct"],
                              res["uniform_noise_20seeds"]["mean_turnover_pct"]) > 15.0)
    return res


# ---------------------------------------------------------------------------
# D. 口座レベル: DD 谷の重なり + 固定 k 機構 + 除外再シミュレート(ペアシード)
# ---------------------------------------------------------------------------
class Cfg:
    def __init__(self, label, pool, closes):
        self.label = label
        self.pool, self.closes = pool.reset_index(drop=True), closes
        self.mk = champion_sizing(self.pool, max_pos=MAX_POS)
        self._c = {}

    def _sim(self, k):
        kk = round(float(k), 10)
        if kk not in self._c:
            self._c[kk] = mm.simulate(self.pool, self.closes, self.mk(kk), max_pos=MAX_POS)
        return self._c[kk]

    def eq(self, k):
        return self._sim(k)[0]

    def info(self, k):
        return self._sim(k)[2]


def dd_episode_list(eq: pd.Series, n_top=3):
    dd = eq / eq.cummax() - 1.0
    arr = dd.to_numpy()
    under = arr < 0
    segs = []
    i, n = 0, len(arr)
    while i < n:
        if under[i]:
            j = i
            while j < n and under[j]:
                j += 1
            k = i + int(np.argmin(arr[i:j]))
            segs.append({"peak": dd.index[max(i - 1, 0)], "trough": dd.index[k],
                         "recover": dd.index[min(j, n - 1)], "depth": float(arr[i:j].min())})
            i = j
        else:
            i += 1
    segs.sort(key=lambda s: s["depth"])
    return segs[:n_top]


def is_oos_eval(cfg: Cfg):
    is_pool = cfg.pool[cfg.pool["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = cfg.pool[cfg.pool["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = cfg.closes[cfg.closes.index < OOS_START]
    oos_cl = cfg.closes[cfg.closes.index >= OOS_START]

    def eq_fn(pool, cl):
        cache = {}

        def f(k):
            kk = round(float(k), 10)
            if kk not in cache:
                cache[kk] = mm.simulate(pool, cl, cfg.mk(kk), max_pos=MAX_POS)[0]
            return cache[kk]
        return f

    eis, eoos = eq_fn(is_pool, is_cl), eq_fn(oos_pool, oos_cl)
    k_emp = calibrate_empirical(eis, 0.20)
    k_rob = calibrate_robust_seeded(eis, 0.20, seed=0)
    return {"k_is_emp": k_emp, "is_emp_cagr": cagr_of(eis(k_emp)),
            "oos_emp_cagr": cagr_of(eoos(k_emp)), "oos_emp_dd": max_dd(eoos(k_emp)),
            "k_is_rob": k_rob, "is_rob_cagr": cagr_of(eis(k_rob)),
            "oos_rob_cagr": cagr_of(eoos(k_rob)), "oos_rob_dd": max_dd(eoos(k_rob))}


def attack_D(pool: pd.DataFrame, df: pd.DataFrame, l5: np.ndarray,
             placebo: np.ndarray) -> dict:
    sec("D. 口座レベル: DD谷重なり / 固定k機構 / 除外プール再シミュレート(seeds 0-4)")
    closes = mm.load_closes()
    base = Cfg("base", pool, closes)
    out: dict = {}

    print("--- ベース較正(検算: rob5シード平均 ≈ +18.63% / emp ≈ +27.50%) ---")
    rb = protocol_eval(base.eq, label="base(d1)", seeds=SEEDS)
    out["base"] = {k: v for k, v in rb.items() if k != "rob"}
    out["base"]["rob"] = {str(s): v for s, v in rb["rob"].items()}

    # D1a. DD エピソードと 98 件の重なり
    k_emp, k_r0 = rb["emp_k"], rb["rob"][0]["k"]
    eq_emp = base.eq(k_emp)
    eps = dd_episode_list(eq_emp, n_top=3)
    ent = pool["entry"].to_numpy()
    exi = pool["exit"].to_numpy()
    print("\n--- D1a. emp_k での DD エピソード上位3 と l5 98件の建玉重なり ---")
    ep_rows = []
    for e in eps:
        pk = np.datetime64(e["peak"].tz_localize(None) if e["peak"].tzinfo else e["peak"])
        tr = np.datetime64(e["trough"].tz_localize(None) if e["trough"].tzinfo else e["trough"])
        ent_n = pool["entry"].dt.tz_localize(None).to_numpy()
        exi_n = pool["exit"].dt.tz_localize(None).to_numpy()
        ov = (ent_n <= tr) & (exi_n >= pk)
        n_all, n_l5 = int(ov.sum()), int((ov & l5).sum())
        exp_l5 = n_all * l5.sum() / len(pool)
        pnl_l5 = float(pool.loc[ov & l5, "ret"].sum())
        ep_rows.append({"peak": str(e["peak"])[:10], "trough": str(e["trough"])[:10],
                        "depth": e["depth"], "n_overlap_all": n_all, "n_overlap_l5": n_l5,
                        "expected_l5_if_uniform": exp_l5, "l5_pnl_in_episode": pnl_l5})
        print(f"  {str(e['peak'])[:10]} -> {str(e['trough'])[:10]}  depth {e['depth']:+.1%}  "
              f"重なり全体 {n_all} / l5 {n_l5} (一様期待 {exp_l5:.1f})  l5寄与PnL {pnl_l5:+.4f}")
    out["dd_episodes"] = ep_rows

    # D1b. 固定 k での DD・p95 変化(機構の直接テスト)
    excl_pool = pool[~l5].reset_index(drop=True)
    excl = Cfg("excl_l5", excl_pool, closes)
    print("\n--- D1b. 固定 k 機構テスト(除外で DD/p95 が浅くならねば k は動かない) ---")
    dd_b = max_dd(base.eq(k_emp))
    dd_x = max_dd(excl.eq(k_emp))
    p95_b = boot_dd(base.eq(k_r0), n_boot=1500, seed=0)["p95"]
    p95_x = boot_dd(excl.eq(k_r0), n_boot=1500, seed=0)["p95"]
    print(f"  emp_k={k_emp:.3f}: maxDD base {dd_b:+.2%} -> excl {dd_x:+.2%} (Δ {(dd_x-dd_b)*100:+.2f}pp)")
    print(f"  rob_s0 k={k_r0:.3f}: p95 base {p95_b:+.2%} -> excl {p95_x:+.2%} (Δ {(p95_x-p95_b)*100:+.2f}pp)")
    out["fixed_k"] = {"k_emp": k_emp, "maxdd_base": dd_b, "maxdd_excl": dd_x,
                      "k_rob0": k_r0, "p95_base": p95_b, "p95_excl": p95_x}
    dd_softens = (abs(dd_x) < abs(dd_b) - 0.002) or (abs(p95_x) < abs(p95_b) - 0.002)
    out["fixed_k"]["dd_softens"] = bool(dd_softens)
    print(f"  -> 除外が DD 形状を実質的に軟化: {dd_softens}")

    # D2. 除外プールのフル再シミュレート(ペアシード)
    print("\n--- D2. 除外プール(n=%d)ペアシード robust 較正 ---" % len(excl_pool))
    rx = protocol_eval(excl.eq, label="excl_l5(98件除外)", seeds=SEEDS)
    out["excl_l5"] = {k: v for k, v in rx.items() if k != "rob"}
    out["excl_l5"]["rob"] = {str(s): v for s, v in rx["rob"].items()}
    print("  seed |   base    |  excl_l5  (diff)")
    diffs = []
    for s in SEEDS:
        d = rx["rob"][s]["cagr"] - rb["rob"][s]["cagr"]
        diffs.append(d)
        print(f"   s{s}  | {rb['rob'][s]['cagr']:+.2%}  | {rx['rob'][s]['cagr']:+.2%}  ({d*100:+.2f}pp)")
    gain = rx["rob_cagr_mean"] - rb["rob_cagr_mean"]
    print(f"  mean | {rb['rob_cagr_mean']:+.2%}  | {rx['rob_cagr_mean']:+.2%}  ({gain*100:+.2f}pp)")
    sig = (rx["emp_cagr"] > rb["emp_cagr"]) and (abs(rx["emp_p95"]) > abs(rb["emp_p95"]) + 0.005)
    print(f"  empirical: base {rb['emp_cagr']:+.2%} (p95 {rb['emp_p95']:+.1%}) -> "
          f"excl {rx['emp_cagr']:+.2%} (p95 {rx['emp_p95']:+.1%})  レバ偽装署名: {'あり' if sig else 'なし'}")
    out["paired_rob_diff_pp"] = [d * 100 for d in diffs]
    out["rob_mean_gain_pp"] = gain * 100
    out["leverage_disguise"] = bool(sig)

    # 年次(rob_s0)・負け年
    yb = yearly_returns(base.eq(k_r0))
    yx = yearly_returns(excl.eq(rx["rob"][0]["k"]))
    dy = (yx - yb).dropna()
    print("  年次差分(rob_s0): " + "  ".join(f"{int(y)}:{v*100:+.1f}pp" for y, v in dy.items()))
    out["yearly_diff_rob0"] = {int(y): float(v) for y, v in dy.items()}
    out["neg_years"] = {"base": int((yb < 0).sum()), "excl": int((yx < 0).sum())}

    # G3: IS 較正 → OOS
    print("\n--- D2-G3. IS(-2021) 較正 -> OOS(2022-) 素検証 ---")
    g3b = is_oos_eval(base)
    g3x = is_oos_eval(excl)
    out["g3"] = {"base": g3b, "excl_l5": g3x}
    print(f"  base: IS rob {g3b['is_rob_cagr']:+.2%} -> OOS {g3b['oos_rob_cagr']:+.2%} "
          f"(DD {g3b['oos_rob_dd']:+.1%}) | emp -> OOS {g3b['oos_emp_cagr']:+.2%}")
    print(f"  excl: IS rob {g3x['is_rob_cagr']:+.2%} -> OOS {g3x['oos_rob_cagr']:+.2%} "
          f"(DD {g3x['oos_rob_dd']:+.1%}) | emp -> OOS {g3x['oos_emp_cagr']:+.2%}")
    g3_pass = (g3x["oos_rob_cagr"] > g3b["oos_rob_cagr"]) and \
              (g3x["oos_emp_cagr"] > g3b["oos_emp_cagr"])
    out["g3_pass"] = bool(g3_pass)
    print(f"  G3(OOS rob & emp 両改善): {g3_pass}")

    # D3. プラセボ除外(m 最小98件)との比較 = 言い換え検定の口座版
    print("\n--- D3. プラセボ除外(m 最小98件)ペアシード ---")
    pl = Cfg("excl_placebo", pool[~placebo].reset_index(drop=True), closes)
    rp = protocol_eval(pl.eq, label="excl_placebo(m最小98)", seeds=SEEDS)
    out["excl_placebo"] = {k: v for k, v in rp.items() if k != "rob"}
    out["excl_placebo"]["rob"] = {str(s): v for s, v in rp["rob"].items()}
    gp = rp["rob_cagr_mean"] - rb["rob_cagr_mean"]
    print(f"  rob_mean: placebo除外 {rp['rob_cagr_mean']:+.2%} ({gp*100:+.2f}pp) vs "
          f"l5除外 {rx['rob_cagr_mean']:+.2%} ({gain*100:+.2f}pp)")
    out["placebo_excl_gain_pp"] = gp * 100
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = pd.read_parquet(POOL_PATH)
    ok = len(pool) == EXPECT_N and abs(pool["ret"].sum() - EXPECT_SUM) < 1e-3
    print("=== edge09: 末尾5分限界クロス拒否(veto候補2)の敵対検証 ===")
    print(f"pool: n={len(pool)} sum={pool['ret'].sum():+.4f} 検算一致: {ok}")
    if not ok:
        return 1
    df = pd.read_csv(E01_CSV, parse_dates=["entry", "exit"])
    align = ((df["instr"].to_numpy() == pool["instr"].to_numpy()).all()
             and np.allclose(df["ret"].to_numpy(), pool["ret"].to_numpy()))
    df["l5"] = df["spike_made_l5"].astype(str).str.lower() == "true"
    print(f"edge01_trades.csv 行整列一致: {align}  l5 n={int(df['l5'].sum())}")
    if not align or int(df["l5"].sum()) != EXPECT_L5_N:
        return 1
    RESULT["checks"] = {"pool_ok": ok, "csv_aligned": bool(align),
                        "l5_n": int(df["l5"].sum())}

    # A. プラセボ
    RESULT["A_placebo"] = attack_A(df)

    # M1 基盤(B と摂動で使用)
    sec("M1 ロード + 反実仮想基盤の構築")
    cfb = build_cf_base(pool)
    print(f"surro 充足率: " + " ".join(
        f"τ{t}:{np.isfinite(cfb['surro'][t]).mean()*100:.1f}%" for t in TAUS) +
        f"  ({time.time()-t0:.0f}s)")

    # B. dose-response(τ=5 の検算込み)
    B = attack_B(df, cfb)
    RESULT["B_dose_response"] = {k: v for k, v in B.items() if k != "members5"}
    if not B["tau5_matches_edge01"]:
        print("!! τ=5 再現不一致 — メンバーシップは edge01 の列を正とし、攻撃は継続")
    base_mem = df["l5"].to_numpy()

    # C. L/S 非対称
    RESULT["C_ls_asymmetry"] = attack_C(df)

    # 4. 実装ノイズ
    RESULT["D_noise"] = attack_noise(df, cfb, base_mem)

    # D. 口座レベル
    m = np.abs(df["z_sig"].to_numpy()) - ENTRY_Z
    order = np.argsort(m)
    placebo = np.zeros(len(df), dtype=bool)
    placebo[order[:EXPECT_L5_N]] = True
    RESULT["E_account"] = attack_D(pool, df, base_mem, placebo)

    # --- 最終判定 -----------------------------------------------------------
    sec("最終判定")
    A = RESULT["A_placebo"]
    Dn = RESULT["D_noise"]
    E = RESULT["E_account"]
    kill_placebo = A["indistinguishable_from_placebo"]
    kill_fragile = Dn["fragile"]
    kill_mech = (E["rob_mean_gain_pp"] < NOISE_BAND_PP) or (not E["fixed_k"]["dd_softens"]) \
        or (not E["g3_pass"]) or E["leverage_disguise"]
    survives = not (kill_placebo or kill_fragile or kill_mech)
    print(f"攻撃1 プラセボと区別不能(言い換え)      : {'KILL' if kill_placebo else 'pass'}")
    print(f"攻撃4 メンバーシップ脆弱(±0.5pip)        : {'KILL' if kill_fragile else 'pass'}")
    print(f"攻撃5 k較正利得の機構不在/G3/署名        : {'KILL' if kill_mech else 'pass'} "
          f"(rob_mean_gain {E['rob_mean_gain_pp']:+.2f}pp, DD軟化 {E['fixed_k']['dd_softens']}, "
          f"G3 {E['g3_pass']}, 署名 {E['leverage_disguise']})")
    print(f"\n>>> veto候補「末尾5分限界クロス拒否」 survives = {survives}")
    RESULT["verdict"] = {"kill_placebo": bool(kill_placebo), "kill_fragile": bool(kill_fragile),
                         "kill_mechanism": bool(kill_mech), "survives": bool(survives)}

    OUT_JSON.write_text(json.dumps(RESULT, indent=2, default=_json_default))
    print(f"\nsaved -> {OUT_JSON}\n      -> {OUT_DOSE}\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
