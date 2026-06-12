"""exp52: d1(エントリー1バー遅延, exp47 全ゲート通過)の細粒度DD監査 — mp11 を殺した
exp44 の M1 ゲートを d1 に適用する(採用前の最終敵対検証)。

仮説(懐疑側): exp47 の d1 優位 +2.22pp(robust 5シード平均)の主因は「H4グリッドMtMの
DD形状改善→較正k上振れ」。遅延で H4 close 時点の MtM は浅く見えるが、バー内では同じ
逆行(第1波)を食らっており、M1 粒度の真の谷では優位が消える可能性がある
(mp11 は H4 で +0.99pp だったが M1谷比 1.161 で掛け目込み実効CAGRが逆転死 = exp44)。

手順:
  0. ベース(d0)/d1 プールを exp47 と同一方式で再構成(検算: rob/emp 較正値の一致)。
  1. robust較正k(seed0)+ empirical較正k を再計算し exp47 値と照合。
  2. M1粒度MtMリプレイ(exp44 の方法論をそのまま踏襲: メジャー=実M1 close、
     クロス=脚M1 close の inner-join 合成、H4バーb の close 時刻 = label+4h)。
     監査点: (d0,d1) × (rob_s0 k, emp k, rob 5シード平均k=exp47値)= 6 リプレイ。
  3. 谷比(M1 DD / H4 DD)を d0 vs d1 で比較。M1 DD が H4較正DDを超える構成は
     掛け目 k_adj(M1 DD = H4較正DD となる k)を反復算出し、掛け目込み実効CAGRで
     d1 優位が残るか判定。参考: exp44 確定値 = d0(mp8) 比率 1.036 / 掛け目 x0.965。
  4. per-trade MAE 監査: 各トレードの最大逆行(MAE)を H4 グリッド(=シミュが見る世界)と
     M1 グリッド(=真の世界)の両方で測り、d0/d1 のマッチドペアで比較。
     「遅延は MAE の観測を H4 close から外しただけ」なら、H4 MAE の改善幅 >> M1 MAE の
     改善幅 + 細粒度DDの谷比悪化として露呈するはず。

実行: PYTHONPATH=. uv run python research/experiments/exp52_d1_m1audit.py
出力: research/outputs/exp52_audit.csv / exp52_episodes.csv / exp52_mae.csv /
      exp52_result.json

結論(2026-06-12 実行): **confirmed — d1 の優位は M1 粒度でも生存(ただし掛け目 x0.955)**。
  検算: プール(n=1207, sum=+1.9622)・較正(emp k=8.895/+27.50%, rob_s0 k=6.084/+18.24%)
  とも exp47 と完全一致。d0 監査は exp44 確定値を再現(谷比 1.036 / 掛け目 x0.965 /
  実効 +15.80% / p95_M1≈-20.7%)= 方法論の複製は正。
  ・谷比: d1 = 1.047(rob_s0)/ 1.048(rob 5シード平均k)/ 1.053(emp)。d0(1.036)より
    僅かに悪いが mp11 を殺した 1.161 とは別物。ゲート 1.15 を大差で PASS。
  ・d1 の M1 谷は 2022-03-28 08:50(JPY ショック週の月曜早朝)へ移動(H4 谷は両構成とも
    2022-05-12)。深掘り局面の顔ぶれ(2019 JPYフラッシュクラッシュ -9.7pp、2020 COVID、
    2016 GBPフラッシュクラッシュ、2022-09 週末ギャップ)は d0 と同一=d1 固有の
    intrabar テール構造化は無い。
  ・掛け目: d1 = x0.955(rob_s0)/ x0.954(m5)/ x0.950(emp)vs d0 = x0.965/0.966。
  ・掛け目込み実効優位: rob_s0 +1.84→+1.57pp / rob_m5 +2.22→**+1.92pp** /
    emp +2.86→+2.25pp。全較正で正のまま(優位の 85-87% が残存)。
  ・MAE 監査が「観測外し」仮説を直接棄却: マッチドペア 1207 組で M1 真値の MAE 改善
    (+0.035pp)≧ H4 視点の改善(+0.032pp)、隠れ逆行(M1−H4)は d0 -0.149pp /
    d1 -0.146pp と同等、最初8hの M1 MAE も d1 が浅い(-0.203% vs -0.217%)。
    遅延は逆行第1波を本当に(僅かに)回避しており、H4 close の死角に隠したのではない。
  → 採用時は d1 の正直化掛け目として k x0.95(d0 の x0.965 より 1pp 深い)を明記すること。
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
    cagr_of,
    calibrate_empirical,
    calibrate_robust_seeded,
)
from fxlab.data import load_m1  # noqa: E402
from fxlab import universe as uni  # noqa: E402
from fxlab.universe import CROSS_DEFS  # noqa: E402

pd.set_option("display.width", 240)

BASE_NET = 1.9086
MAX_POS = 8
INIT = 10_000.0
EPISODE_TH = 0.02
GATE_RATIO = 1.15
OUT_DIR = ROOT / "research" / "outputs"
OUT_AUDIT = OUT_DIR / "exp52_audit.csv"
OUT_EPI = OUT_DIR / "exp52_episodes.csv"
OUT_MAE = OUT_DIR / "exp52_mae.csv"
OUT_JSON = OUT_DIR / "exp52_result.json"
EXP47_JSON = OUT_DIR / "exp47_result.json"


def sec(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# exp47 から複製: 価格再構成 + 遅延プール生成(1ファイル完結の規約のため)
# ---------------------------------------------------------------------------
def reconstruct(pool: pd.DataFrame) -> dict:
    closes_by = {i: uni.instrument_close(i, "H4") for i in sorted(pool["instr"].unique())}
    n = len(pool)
    idx_e = np.full(n, -1)
    idx_x = np.full(n, -1)
    entry_close = np.full(n, np.nan)
    exit_close = np.full(n, np.nan)
    for instr, g in pool.groupby("instr"):
        s = closes_by[instr]
        ie = s.index.get_indexer(g["entry"])
        ix = s.index.get_indexer(g["exit"])
        assert (ie >= 0).all() and (ix >= 0).all(), f"{instr}: timestamp miss"
        rows = g.index.to_numpy()
        idx_e[rows] = ie
        idx_x[rows] = ix
        entry_close[rows] = s.to_numpy()[ie]
        exit_close[rows] = s.to_numpy()[ix]
    d = pool["dir"].to_numpy().astype(float)
    gross = d * (exit_close / entry_close - 1.0)
    cost = gross - pool["ret"].to_numpy()
    slip = pool["entry_price"].to_numpy() / entry_close
    return {"closes_by": closes_by, "idx_e": idx_e, "idx_x": idx_x,
            "entry_close": entry_close, "exit_close": exit_close,
            "gross": gross, "cost": cost, "slip": slip}


def delayed_pool(pool: pd.DataFrame, rc: dict, dly: int):
    n = len(pool)
    dclose = np.full(n, np.nan)
    dts = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    ie_d_full = np.full(n, -1)
    for instr, g in pool.groupby("instr"):
        rows = g.index.to_numpy()
        s = rc["closes_by"][instr]
        ie_d = rc["idx_e"][rows] + dly
        ie_cl = np.minimum(ie_d, len(s) - 1)
        dclose[rows] = s.to_numpy()[ie_cl]
        dts[rows] = s.index.values[ie_cl]
        ie_d_full[rows] = ie_d
    kept = ie_d_full < rc["idx_x"]
    d = pool["dir"].to_numpy().astype(float)
    ret_new = d * (rc["exit_close"] / dclose - 1.0) - rc["cost"]
    mod = pool.copy()
    mod["entry"] = pd.DatetimeIndex(dts).tz_localize("UTC")
    mod["entry_price"] = dclose * rc["slip"]
    mod["ret"] = ret_new
    mod["bars_held"] = np.maximum(pool["bars_held"].to_numpy() - dly, 1)
    mod = mod[kept].sort_values("entry").reset_index(drop=True)
    return mod, kept, ret_new


# ---------------------------------------------------------------------------
# exp44 から複製: 建玉ログ付きシミュ + M1 リプレイ(方法論をそのまま踏襲)
# ---------------------------------------------------------------------------
def simulate_with_log(pool, closes, sizing, *, init=INIT, max_pos=MAX_POS):
    """mm_lab.simulate と同一ロジック + 建玉ログ(alloc/entry/exit)を返す(exp24/44 方式)。"""
    grid = closes.index
    col_of = {c: i for i, c in enumerate(closes.columns)}
    carr = closes.to_numpy()
    n = len(grid)
    gi = grid.to_numpy()
    entry_pos = np.clip(np.searchsorted(gi, pool["entry"].to_numpy(), side="left"), 0, n - 1)
    exit_pos = np.clip(np.searchsorted(gi, pool["exit"].to_numpy(), side="left"), 0, n - 1)

    by_entry = {}
    for ti in range(len(pool)):
        by_entry.setdefault(int(entry_pos[ti]), []).append(ti)

    instr_arr = pool["instr"].to_numpy()
    dir_arr = pool["dir"].to_numpy().astype(float)
    eprice_arr = pool["entry_price"].to_numpy()
    ret_arr = pool["ret"].to_numpy()
    z_arr = pool["z_entry"].to_numpy()
    bars_arr = pool["bars_held"].to_numpy()

    equity = init
    peak_mtm = init
    open_pos = []
    eq_mtm = np.empty(n)
    log = []
    skipped = 0

    for b in range(n):
        if open_pos:
            still = []
            for p in open_pos:
                if p["exit_pos"] <= b:
                    equity += p["alloc"] * p["ret"]
                else:
                    still.append(p)
            open_pos = still
        unreal = 0.0
        for p in open_pos:
            px = carr[b, p["col"]]
            unreal += p["alloc"] * (p["dir"] * (px / p["eprice"] - 1.0))
        mtm = equity + unreal
        eq_mtm[b] = mtm
        peak_mtm = max(peak_mtm, mtm)
        dd_mtm = mtm / peak_mtm - 1.0
        if b in by_entry:
            for ti in by_entry[b]:
                if len(open_pos) >= max_pos:
                    skipped += 1
                    continue
                ctx = {"equity_real": equity, "equity_mtm": mtm, "peak_mtm": peak_mtm,
                       "dd_mtm": dd_mtm, "n_open": len(open_pos), "max_pos": max_pos,
                       "recent_vol": float("nan"), "z": float(z_arr[ti]),
                       "instr": instr_arr[ti], "ret": float(ret_arr[ti]),
                       "bars_held": int(bars_arr[ti])}
                alloc = float(sizing(ctx))
                if alloc <= 0:
                    skipped += 1
                    continue
                open_pos.append({"ti": ti, "col": col_of[instr_arr[ti]], "dir": dir_arr[ti],
                                 "eprice": eprice_arr[ti], "alloc": alloc,
                                 "exit_pos": int(exit_pos[ti]), "ret": float(ret_arr[ti])})
                log.append({"ti": ti, "instr": instr_arr[ti], "dir": dir_arr[ti],
                            "eprice": eprice_arr[ti], "alloc": alloc,
                            "entry_pos": b, "exit_pos": int(exit_pos[ti]),
                            "ret": float(ret_arr[ti])})
    return pd.Series(eq_mtm, index=grid), pd.DataFrame(log), {"skipped": skipped}


_M1_RAW: dict[str, pd.Series] = {}
_PX: dict[str, np.ndarray] = {}


def m1_close_naive(name: str) -> pd.Series:
    """メジャー=実 M1 close、合成クロス=脚 M1 close の inner-join 合成(exp44/anatomy_time 方式)。"""
    if name in _M1_RAW:
        return _M1_RAW[name]
    if name in CROSS_DEFS:
        a, op, b = CROSS_DEFS[name]
        ca, cb = m1_close_naive(a), m1_close_naive(b)
        df = pd.concat([ca.rename("a"), cb.rename("b")], axis=1, join="inner").dropna()
        s = df["a"] / df["b"] if op == "/" else df["a"] * df["b"]
    else:
        c = load_m1(name)["close"]
        s = pd.Series(c.to_numpy(), index=c.index.tz_localize(None))  # 非破壊 tz-naive 化
    _M1_RAW[name] = s
    return s


def px_grid(name: str, grid_idx: pd.DatetimeIndex) -> np.ndarray:
    if name not in _PX:
        _PX[name] = m1_close_naive(name).reindex(grid_idx, method="ffill").to_numpy()
    return _PX[name]


def m1_replay(log: pd.DataFrame, closes: pd.DataFrame, grid_idx: pd.DatetimeIndex,
              init=INIT):
    """建玉ログから M1 グリッド上の MtM equity を再構成(exp44 と同一規約)。

    H4 バー b の close 時刻 = label+4h。含み損益は [エントリーバーclose, エグジットバーclose)
    の M1 close で評価、実現損益はエグジットバー close 時刻にステップ。
    """
    gi = closes.index.tz_localize(None).to_numpy()
    gridv = grid_idx.to_numpy()
    n_m1 = len(gridv)
    unreal = np.zeros(n_m1)

    real = np.full(n_m1, init)
    exit_times = gi[log["exit_pos"].to_numpy()] + np.timedelta64(4, "h")
    pnl = (log["alloc"] * log["ret"]).to_numpy()
    order = np.argsort(exit_times)
    step_pos = np.searchsorted(gridv, exit_times[order], side="left")
    cum = 0.0
    last = 0
    for sp, v in zip(step_pos, pnl[order]):
        sp = min(sp, n_m1)
        if sp > last:
            real[last:sp] = init + cum
            last = sp
        cum += v
    real[last:] = init + cum

    for instr, gl in log.groupby("instr"):
        px = px_grid(instr, grid_idx)
        for _, p in gl.iterrows():
            t_in = gi[int(p["entry_pos"])] + np.timedelta64(4, "h")
            t_out = gi[int(p["exit_pos"])] + np.timedelta64(4, "h")
            a = int(np.searchsorted(gridv, t_in, side="left"))
            b = int(np.searchsorted(gridv, t_out, side="left"))
            if b <= a:
                continue
            unreal[a:b] += p["alloc"] * (p["dir"] * (px[a:b] / p["eprice"] - 1.0))
    return pd.Series(real + unreal, index=grid_idx)


def dd_of(arr: np.ndarray) -> np.ndarray:
    return arr / np.maximum.accumulate(arr) - 1.0


def trough_contributions(log, closes, grid_idx, j, eq_at, top=3):
    gi = closes.index.tz_localize(None).to_numpy()
    t = grid_idx.to_numpy()[j]
    rows = []
    for _, p in log.iterrows():
        t_in = gi[int(p["entry_pos"])] + np.timedelta64(4, "h")
        t_out = gi[int(p["exit_pos"])] + np.timedelta64(4, "h")
        if not (t_in <= t < t_out):
            continue
        px = _PX[p["instr"]][j]
        ur = p["alloc"] * p["dir"] * (px / p["eprice"] - 1.0)
        rows.append((p["instr"], "L" if p["dir"] > 0 else "S", float(ur / eq_at * 100)))
    rows.sort(key=lambda x: x[2])
    return [f"{i}{d}:{v:+.1f}%" for i, d, v in rows[:top]]


def m1_audit_one(label, pool, closes, mk, k, grid_idx, *, max_contrib_epi=6):
    """較正 k での M1 監査: DD 比較 + 深掘り局面 + 掛け目 k_adj(exp44 方式)。"""
    t0 = time.time()
    eqm, log, info = simulate_with_log(pool, closes, mk(k), max_pos=MAX_POS)
    dd_h4_series = dd_of(eqm.to_numpy())
    dd_h4 = float(dd_h4_series.min())
    t_h4_trough = eqm.index[int(np.argmin(dd_h4_series))]

    eq_m1 = m1_replay(log, closes, grid_idx)
    em1 = eq_m1.to_numpy()
    dd_m1_series = dd_of(em1)
    dd_m1 = float(dd_m1_series.min())
    i_trough = int(np.argmin(dd_m1_series))
    t_m1_trough = grid_idx[i_trough]
    ratio = abs(dd_m1) / abs(dd_h4)
    print(f"  [{label}] k={k:.3f}  DD(H4)={dd_h4:+.2%} @{str(t_h4_trough)[:16]}  "
          f"DD(M1)={dd_m1:+.2%} @{str(t_m1_trough)[:16]}  比率={ratio:.3f}  "
          f"({time.time()-t0:.0f}s)")

    # H4 DD を M1 グリッドへ整列し、2pp 以上深い局面を抽出
    h4_times = closes.index.tz_localize(None).to_numpy() + np.timedelta64(4, "h")
    pos = np.searchsorted(h4_times, grid_idx.to_numpy(), side="right") - 1
    dd_h4_on = np.where(pos >= 0, dd_h4_series[np.clip(pos, 0, None)], 0.0)
    gap = dd_m1_series - dd_h4_on
    mask = gap <= -EPISODE_TH

    episodes = []
    if mask.any():
        idx = np.flatnonzero(mask)
        brk = np.flatnonzero(np.diff(idx) > 1440)
        starts = np.concatenate([[0], brk + 1])
        ends = np.concatenate([brk, [len(idx) - 1]])
        gridv = grid_idx.to_numpy()
        for s_, e_ in zip(starts, ends):
            a, b = idx[s_], idx[e_]
            seg = slice(a, b + 1)
            j = a + int(np.argmin(gap[seg]))
            tt = pd.Timestamp(gridv[j])
            w0 = max(0, j - 2880)
            diffs = np.diff(gridv[w0:j + 1]).astype("timedelta64[m]").astype(int)
            max_gap_min = int(diffs.max()) if len(diffs) else 0
            episodes.append({
                "start": str(pd.Timestamp(gridv[a]))[:16], "end": str(pd.Timestamp(gridv[b]))[:16],
                "trough": str(tt)[:16], "dow": tt.day_name()[:3], "hour_utc": tt.hour,
                "dd_m1": float(dd_m1_series[j]), "dd_h4_aligned": float(dd_h4_on[j]),
                "gap_pp": float(gap[j] * 100), "max_m1gap_min_2d": max_gap_min,
                "weekend_gap": max_gap_min >= 120, "_j": j,
            })
        episodes.sort(key=lambda x: x["gap_pp"])
        for e in episodes[:max_contrib_epi]:
            e["top_unreal"] = trough_contributions(log, closes, grid_idx, e["_j"],
                                                   em1[e["_j"]])
        for e in episodes:
            e.pop("_j", None)
            e.setdefault("top_unreal", [])

    # 掛け目: M1 DD が H4 較正レベル(dd_h4@k)に一致する k_adj を反復(exp44 と同一)
    k_adj, dd_m1_adj, cagr_adj = k, dd_m1, cagr_of(eqm)
    target = abs(dd_h4)
    if abs(dd_m1) > target * 1.005:
        for _ in range(4):
            k_adj = k_adj * (target / abs(dd_m1_adj))
            eqm_a, log_a, _ = simulate_with_log(pool, closes, mk(k_adj), max_pos=MAX_POS)
            eq_m1_a = m1_replay(log_a, closes, grid_idx)
            dd_m1_adj = float(dd_of(eq_m1_a.to_numpy()).min())
            cagr_adj = cagr_of(eqm_a)
            if abs(abs(dd_m1_adj) - target) < 0.002:
                break

    return {"label": label, "k": k, "cagr_h4": cagr_of(eqm),
            "dd_h4": dd_h4, "dd_m1": dd_m1, "ratio": ratio,
            "trough_h4": str(t_h4_trough)[:16], "trough_m1": str(t_m1_trough)[:16],
            "p95_m1_approx": -0.20 * ratio,
            "k_adj": k_adj, "haircut": k_adj / k, "dd_m1_adj": dd_m1_adj,
            "cagr_adj": cagr_adj, "episodes": episodes, "skipped": info["skipped"]}


# ---------------------------------------------------------------------------
# per-trade MAE(H4グリッド=シミュ視点 vs M1グリッド=真の世界)
# ---------------------------------------------------------------------------
def mae_table(pool_, closes, grid_idx) -> pd.DataFrame:
    """各トレードの最大逆行。mae_h4 = H4 MtM が評価するバー close(entry+1..exit-1)、
    mae_m1 = [entryバーclose, exitバーclose) の M1 close。mae_m1_8h = 最初の8時間。"""
    gi_utc = closes.index.to_numpy()
    gi_nv = closes.index.tz_localize(None).to_numpy()
    gridv = grid_idx.to_numpy()
    carr = closes.to_numpy()
    col_of = {c: i for i, c in enumerate(closes.columns)}
    nb = len(gi_utc)
    e_pos = np.clip(np.searchsorted(gi_utc, pool_["entry"].to_numpy(), side="left"), 0, nb - 1)
    x_pos = np.clip(np.searchsorted(gi_utc, pool_["exit"].to_numpy(), side="left"), 0, nb - 1)
    dirs = pool_["dir"].to_numpy().astype(float)
    eps = pool_["entry_price"].to_numpy()
    out = np.zeros((len(pool_), 3))
    for instr, g in pool_.groupby("instr"):
        px_m1 = px_grid(instr, grid_idx)
        col = col_of[instr]
        for i in g.index:
            d, ep = dirs[i], eps[i]
            e_, x_ = int(e_pos[i]), int(x_pos[i])
            if x_ - e_ >= 2:  # H4 MtM が建玉を評価するバーが存在する
                seg = carr[e_ + 1:x_, col]
                out[i, 0] = float(np.min(d * (seg / ep - 1.0)))
            t_in = gi_nv[e_] + np.timedelta64(4, "h")
            t_out = gi_nv[x_] + np.timedelta64(4, "h")
            a = int(np.searchsorted(gridv, t_in, side="left"))
            b = int(np.searchsorted(gridv, t_out, side="left"))
            if b > a:
                r = d * (px_m1[a:b] / ep - 1.0)
                out[i, 1] = float(np.nanmin(r))
                b8 = int(np.searchsorted(gridv, t_in + np.timedelta64(8, "h"), side="left"))
                out[i, 2] = float(np.nanmin(r[:max(b8 - a, 1)]))
    return pd.DataFrame({"mae_h4": out[:, 0], "mae_m1": out[:, 1], "mae_m1_8h": out[:, 2]},
                        index=pool_.index)


def mae_stats(df: pd.DataFrame, col: str) -> str:
    v = df[col] * 100
    return (f"mean {v.mean():+.3f}% | p50 {v.median():+.3f}% | p90 {v.quantile(0.10):+.3f}% | "
            f"p95 {v.quantile(0.05):+.3f}% | worst {v.min():+.2f}%")


# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = mm.build_pool()
    pool["orig"] = np.arange(len(pool))
    closes = mm.load_closes()
    exp47 = json.loads(EXP47_JSON.read_text())
    b47 = exp47["baseline"]
    v47 = exp47["variants"]["1"]["account"]
    p47 = exp47["variants"]["1"]["pool"]

    sec("0. プール再構成(exp47 方式)と検算")
    net = pool["ret"].sum()
    rc = reconstruct(pool)
    ok0 = abs((rc["gross"] - rc["cost"]).sum() - net) < 1e-9 and abs(net - BASE_NET) < 1e-3
    d1pool, kept, ret_new = delayed_pool(pool, rc, 1)
    ok1 = (len(d1pool) == p47["n"]) and abs(d1pool["ret"].sum() - p47["sum_ret"]) < 1e-6
    print(f"d0: n={len(pool)} sum={net:+.4f} (基準 {BASE_NET:+.4f})  再構成一致: {ok0}")
    print(f"d1: n={len(d1pool)} (exp47: {p47['n']})  sum={d1pool['ret'].sum():+.4f} "
          f"(exp47: {p47['sum_ret']:+.4f})  一致: {ok1}  消滅 {int((~kept).sum())}件")
    if not (ok0 and ok1):
        print("!! プール再構成が exp47 と不一致。以降の比較は無効。")
        return 1

    sec("1. 較正の再計算(robust seed0 + empirical)と exp47 照合")
    mk0 = champion_sizing(pool, max_pos=MAX_POS)
    mk1 = champion_sizing(d1pool, max_pos=MAX_POS)
    cals = {}
    for tag, pl, mk, ref in [("d0", pool, mk0, b47), ("d1", d1pool, mk1, v47)]:
        cache = {}

        def eq_of_k(k, pl=pl, mk=mk, cache=cache):
            kk = round(float(k), 10)
            if kk not in cache:
                eqm, _, _ = mm.simulate(pl, closes, mk(kk), max_pos=MAX_POS)
                cache[kk] = eqm
            return cache[kk]

        k_emp = calibrate_empirical(eq_of_k, 0.20)
        k_r0 = calibrate_robust_seeded(eq_of_k, 0.20, seed=0)
        c_emp, c_r0 = cagr_of(eq_of_k(k_emp)), cagr_of(eq_of_k(k_r0))
        ref_r0 = ref["rob"]["0"]
        m_emp = abs(k_emp - ref["emp_k"]) < 0.02 and abs(c_emp - ref["emp_cagr"]) < 5e-4
        m_r0 = abs(k_r0 - ref_r0["k"]) < 0.02 and abs(c_r0 - ref_r0["cagr"]) < 5e-4
        k_m5 = float(np.mean([ref["rob"][str(s)]["k"] for s in range(5)]))
        cals[tag] = {"k_emp": k_emp, "c_emp": c_emp, "k_r0": k_r0, "c_r0": c_r0,
                     "k_mean5": k_m5, "match_emp": m_emp, "match_r0": m_r0}
        print(f"  {tag}: emp k={k_emp:.3f} CAGR={c_emp:+.2%} (exp47 {ref['emp_k']:.3f}/"
              f"{ref['emp_cagr']:+.2%} 一致:{m_emp}) | rob_s0 k={k_r0:.3f} CAGR={c_r0:+.2%} "
              f"(exp47 {ref_r0['k']:.3f}/{ref_r0['cagr']:+.2%} 一致:{m_r0}) | "
              f"rob 5シード平均k={k_m5:.3f}  [{time.time()-t0:.0f}s]")
    if not all(c["match_emp"] and c["match_r0"] for c in cals.values()):
        print("!! 較正が exp47 と不一致 — 中断")
        return 1

    sec("2. M1粒度 MtM リプレイ(6 監査点)")
    grid_idx = pd.DatetimeIndex(load_m1("EURUSD").index.tz_localize(None))
    print(f"M1 grid: {len(grid_idx):,} bars  ({grid_idx[0]} .. {grid_idx[-1]})")
    audits = {}
    for key, pl, mk, k in [
        ("d0_rob_s0", pool, mk0, cals["d0"]["k_r0"]),
        ("d1_rob_s0", d1pool, mk1, cals["d1"]["k_r0"]),
        ("d0_rob_m5", pool, mk0, cals["d0"]["k_mean5"]),
        ("d1_rob_m5", d1pool, mk1, cals["d1"]["k_mean5"]),
        ("d0_emp", pool, mk0, cals["d0"]["k_emp"]),
        ("d1_emp", d1pool, mk1, cals["d1"]["k_emp"]),
    ]:
        audits[key] = m1_audit_one(key, pl, closes, mk, k, grid_idx)

    print("\n  -- 深掘り局面(M1 が H4整列DDより2pp以上深い, 谷の深い順, 上位5) --")
    epi_rows = []
    for key, a in audits.items():
        print(f"  [{key}] episodes={len(a['episodes'])}  掛け目 k_adj={a['k_adj']:.3f} "
              f"(haircut x{a['haircut']:.3f})  M1 DD(adj)={a['dd_m1_adj']:+.2%}  "
              f"実効CAGR={a['cagr_adj']:+.2%}  p95_M1近似={a['p95_m1_approx']:+.1%}")
        for e in a["episodes"][:5]:
            print(f"      {e['trough']} ({e['dow']} {e['hour_utc']:02d}h)  M1={e['dd_m1']:+.1%} "
                  f"vs H4={e['dd_h4_aligned']:+.1%} (gap {e['gap_pp']:+.1f}pp)  "
                  f"週末G{'有' if e['weekend_gap'] else '無'}  "
                  f"建玉: {', '.join(e['top_unreal'])}")
        for e in a["episodes"]:
            epi_rows.append({"config": key, **{k_: v for k_, v in e.items()
                                               if k_ != "top_unreal"},
                             "top_unreal": "; ".join(e["top_unreal"])})
    pd.DataFrame(epi_rows).to_csv(OUT_EPI, index=False)

    sec("3. 谷比 d0 vs d1 と掛け目込み実効CAGR")
    rows = []
    for key, a in audits.items():
        rows.append({k_: a[k_] for k_ in ("label", "k", "cagr_h4", "dd_h4", "dd_m1", "ratio",
                                          "trough_h4", "trough_m1", "p95_m1_approx",
                                          "k_adj", "haircut", "dd_m1_adj", "cagr_adj")})
    adf = pd.DataFrame(rows)
    print(adf.round(4).to_string(index=False))
    adf.to_csv(OUT_AUDIT, index=False)

    eff = {}
    for basis in ("rob_s0", "rob_m5", "emp"):
        a0, a1 = audits[f"d0_{basis}"], audits[f"d1_{basis}"]
        adv_h4 = (a1["cagr_h4"] - a0["cagr_h4"]) * 100
        adv_adj = (a1["cagr_adj"] - a0["cagr_adj"]) * 100
        eff[basis] = {"adv_h4_pp": adv_h4, "adv_adj_pp": adv_adj,
                      "ratio_d0": a0["ratio"], "ratio_d1": a1["ratio"],
                      "haircut_d0": a0["haircut"], "haircut_d1": a1["haircut"]}
        print(f"\n  [{basis}] 谷比 d0={a0['ratio']:.3f} / d1={a1['ratio']:.3f}  "
              f"掛け目 d0=x{a0['haircut']:.3f} / d1=x{a1['haircut']:.3f}")
        print(f"        H4優位 {adv_h4:+.2f}pp → 掛け目込み実効優位 {adv_adj:+.2f}pp  "
              f"(d0 {a0['cagr_adj']:+.2%} vs d1 {a1['cagr_adj']:+.2%})")

    sec("4. per-trade MAE 監査(H4視点 vs M1真値, マッチドペア)")
    mae0 = mae_table(pool, closes, grid_idx)
    mae1 = mae_table(d1pool, closes, grid_idx)
    print(f"  d0 (n={len(pool)}):")
    print(f"    MAE_H4 : {mae_stats(mae0, 'mae_h4')}")
    print(f"    MAE_M1 : {mae_stats(mae0, 'mae_m1')}")
    print(f"  d1 (n={len(d1pool)}):")
    print(f"    MAE_H4 : {mae_stats(mae1, 'mae_h4')}")
    print(f"    MAE_M1 : {mae_stats(mae1, 'mae_m1')}")

    # マッチドペア(orig で対応付け)
    t0_ = pool.set_index("orig")
    m0 = mae0.set_axis(pool["orig"].to_numpy(), axis=0)
    m1_ = mae1.set_axis(d1pool["orig"].to_numpy(), axis=0)
    common = m1_.index
    pair = pd.DataFrame({
        "orig": common,
        "instr": t0_.loc[common, "instr"].to_numpy(),
        "dir": t0_.loc[common, "dir"].to_numpy(),
        "entry_d0": t0_.loc[common, "entry"].to_numpy(),
        "year": pd.DatetimeIndex(t0_.loc[common, "exit"]).year,
        "bars_held_d0": t0_.loc[common, "bars_held"].to_numpy(),
        "ret_d0": t0_.loc[common, "ret"].to_numpy(),
        "mae_h4_d0": m0.loc[common, "mae_h4"].to_numpy(),
        "mae_m1_d0": m0.loc[common, "mae_m1"].to_numpy(),
        "mae_m1_8h_d0": m0.loc[common, "mae_m1_8h"].to_numpy(),
        "mae_h4_d1": m1_["mae_h4"].to_numpy(),
        "mae_m1_d1": m1_["mae_m1"].to_numpy(),
        "mae_m1_8h_d1": m1_["mae_m1_8h"].to_numpy(),
    })
    pair["d_h4"] = pair["mae_h4_d1"] - pair["mae_h4_d0"]   # >0 = d1 の方が浅い(改善)
    pair["d_m1"] = pair["mae_m1_d1"] - pair["mae_m1_d0"]
    pair["hid_d0"] = pair["mae_m1_d0"] - pair["mae_h4_d0"]  # <0 = H4 が見ない逆行
    pair["hid_d1"] = pair["mae_m1_d1"] - pair["mae_h4_d1"]
    pair.to_csv(OUT_MAE, index=False)

    dh4, dm1 = pair["d_h4"].mean() * 100, pair["d_m1"].mean() * 100
    print(f"\n  マッチドペア n={len(pair)}:")
    print(f"    H4視点の MAE 改善(d1-d0): mean {dh4:+.3f}pp | median "
          f"{pair['d_h4'].median()*100:+.3f}pp | d1浅い率 {(pair['d_h4'] > 0).mean():.0%}")
    print(f"    M1真値の MAE 改善(d1-d0): mean {dm1:+.3f}pp | median "
          f"{pair['d_m1'].median()*100:+.3f}pp | d1浅い率 {(pair['d_m1'] > 0).mean():.0%}")
    print(f"    隠れ逆行(M1−H4): d0 mean {pair['hid_d0'].mean()*100:+.3f}pp / "
          f"d1 mean {pair['hid_d1'].mean()*100:+.3f}pp  "
          f"(d1 の方が{'大きい=H4錯覚の署名' if pair['hid_d1'].mean() < pair['hid_d0'].mean() else '小さい/同等'})")
    print(f"    最初8hの M1 MAE: d0 mean {pair['mae_m1_8h_d0'].mean()*100:+.3f}% / "
          f"d1 mean {pair['mae_m1_8h_d1'].mean()*100:+.3f}%")

    sec("5. 判定")
    r0, r1 = audits["d0_rob_s0"]["ratio"], audits["d1_rob_s0"]["ratio"]
    g_ratio = r1 <= GATE_RATIO
    adv_adj_s0 = eff["rob_s0"]["adv_adj_pp"]
    adv_adj_m5 = eff["rob_m5"]["adv_adj_pp"]
    print(f"  ゲート(exp44): d1 M1/H4 谷比 {r1:.3f} <= {GATE_RATIO}: {'PASS' if g_ratio else 'FAIL'} "
          f"(d0 = {r0:.3f}, mp11 は 1.161 で死亡)")
    print(f"  掛け目込み実効優位: rob_s0 {adv_adj_s0:+.2f}pp / rob 5シード平均k {adv_adj_m5:+.2f}pp / "
          f"emp {eff['emp']['adv_adj_pp']:+.2f}pp")
    survive = g_ratio and adv_adj_s0 > 0 and adv_adj_m5 > 0
    print(f"  → {'d1 の優位は M1 粒度でも生存' if survive else 'd1 の優位は M1 粒度で毀損/逆転'}")

    payload = {
        "cals": cals,
        "audits": {k_: {kk: vv for kk, vv in a.items() if kk != "episodes"}
                   | {"n_episodes": len(a["episodes"])} for k_, a in audits.items()},
        "effective": eff,
        "mae": {"pairs": int(len(pair)), "d_h4_mean_pp": float(dh4), "d_m1_mean_pp": float(dm1),
                "hid_d0_mean_pp": float(pair["hid_d0"].mean() * 100),
                "hid_d1_mean_pp": float(pair["hid_d1"].mean() * 100),
                "mae_m1_mean_d0": float(mae0["mae_m1"].mean()),
                "mae_m1_mean_d1": float(mae1["mae_m1"].mean()),
                "mae_h4_mean_d0": float(mae0["mae_h4"].mean()),
                "mae_h4_mean_d1": float(mae1["mae_h4"].mean())},
        "verdict": {"gate_ratio_d1": g_ratio, "ratio_d0_s0": r0, "ratio_d1_s0": r1,
                    "adv_adj_rob_s0_pp": adv_adj_s0, "adv_adj_rob_m5_pp": adv_adj_m5,
                    "survive": bool(survive)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nsaved -> {OUT_AUDIT}\n      -> {OUT_EPI}\n      -> {OUT_MAE}\n      -> {OUT_JSON}")
    print(f"総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
