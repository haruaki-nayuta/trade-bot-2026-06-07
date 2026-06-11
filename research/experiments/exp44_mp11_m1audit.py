"""exp44: max_pos 8→11 の本番昇格判定 — M1粒度MtM・証拠金監査。

背景: mp11 の robust +17.2〜17.4% vs mp8 +16.3%(+1.1pp)は reports/10/14 で実測済み。
容量挙動の懸念は reports/15 gap_0 で解消(溢れスキップ=純益の1.0%・経路依存ゼロ)。
残るゲートは実装現実性のみ:
  (a) H4 close 粒度の MtM DD が M1 粒度の真の谷を過小評価していないか
      (特に合成クロスは close 代用で intrabar 逆行が不可視)
  (b) 証拠金・レバ規制(国内25倍/海外100倍)内に収まるか

手順:
  A. ベースライン再計算: champion_sizing(P=4) × mp8/mp11 をペアシード(0-4)で
     protocol_eval(robust p95=20% / empirical 20%)。+1.1pp級の再確認。
  B. プロトコル残項目: IS(<2022)単独で max_pos を選んだら何が選ばれるか + OOS成績 /
     改善幅の年次分解(2022・最良年除外) / 全年プラス維持。
  C. M1粒度MtMリプレイ: robust k(5シード平均)で建玉ログを再生し、
     メジャー=実M1 close・合成クロス=脚M1 close の inner-join 合成(anatomy_time 方式)で
     M1 グリッド上の MtM equity を再構成。M1 vs H4 の maxDD 比率/差/谷日付、
     H4比 2pp 以上深い局面のリストと原因(週末ギャップ/イベント)。
     ブートp95のM1再推定は重いので「p95_M1 ≈ 20% × (DD_M1/DD_H4)」の掛け目近似(規約)。
     M1 DD が H4 較正レベルを超える場合は k_adj(M1 DD = H4較正DD となる掛け目)と
     その実効CAGRを再計算。
  D. 証拠金監査: M1 グリッド上の 総建玉(gross notional)/MtM equity の時系列。
     最大レバ、国内25倍・海外100倍での証拠金維持率の最低値、z-power加重の1玉最大レバ。
  E. 判定: M1粒度DD ≤ 1.15×H4 かつ 最大レバ ≤ 25 なら mp11 昇格(必要な掛け目を明記)。
     超えるなら掛け目込み実効CAGRで mp8 継続と比較。

実行: PYTHONPATH=. uv run python research/experiments/exp44_mp11_m1audit.py

結論(2026-06-11 実行): **reject — mp11 昇格見送り、mp8 継続**。
  A. ペアシード(0-4): mp8 robust mean +16.41% / mp11 +17.40% = +0.99pp(全シード正、
     per-seed +0.73〜+1.28pp)。reports/10/14 の +1.1pp 級を再確認。
     ただし empirical 較正では emp CAGR +24.6%→+27.6% かつ p95 -27.8%→-29.2% =
     レバ偽装署名あり(tail_ok=False)。robust が p95=20% に縛るため H4 粒度では成立して
     いたが、M1 監査がこの署名の実体を暴いた(下記)。
  B. IS argmax = mp11(IS robust +15.0% vs +14.5%、OOS も +22.5% vs +21.0%)= H4 粒度
     プロトコル自体は mp11 有利のまま。2022 除外でも改善 +0.56pp で符号維持。
  C. M1 粒度: mp8 比率 1.036(谷 2022-05-12)に対し **mp11 は 1.161(谷 2022-03-07 01:51
     =ウクライナ侵攻直後の月曜早朝)> ゲート 1.15 で FAIL**。機構: 同時刻に mp8 は 8玉
     gross 4.5x で M1 DD -13.5%、mp11 は 11玉 gross 7.1x で -15.4%。mp11 の追加 3玉
     (EURUSD L/NZDUSD S/EURCAD L)はボラクラスタでちょうど増える=intrabar テールを構造的に
     深くする(まぐれでない: 第2-5谷の比率は両構成 1.00-1.11 で並ぶが、mp11 だけ最悪谷が
     高インフレ局面に当たる)。掛け目込み実効CAGR: mp8 +15.80%(k×0.965) vs
     mp11 +14.85%(k×0.861)= **+0.99pp の優位が -0.95pp に逆転**。同一絶対 M1 DD に
     揃える代替規約でも mp11≈+15.8% の同着止まりで優位なし。
  D. 証拠金: 両構成 PASS(最大総レバ 9.2-9.4x ≪ 25x、維持率最低 国内265%/海外1062%、
     1玉最大 1.84-1.88x)。証拠金は昇格の障害ではない。
  副産物: 2019-01-02 22:39 JPY フラッシュクラッシュで M1 DD -9.8% vs H4 -0.7%
     (gap -9.1pp、GBPJPY L が -8.6%)。H4 で不可視の口座 -10% 級イベントは実在する。
     mp8 の正直化には k×0.965 の掛け目を推奨(exp24 の k×0.95 と整合)。
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
from mm_production import _fz, champion_sizing, Z0, P, CLIP_HI  # noqa: E402
from tail_protocol import (  # noqa: E402
    boot_dd,
    cagr_of,
    calibrate_empirical,
    calibrate_robust_seeded,
    max_dd,
    protocol_eval,
    yearly_returns,
)
from fxlab.data import load_m1  # noqa: E402
from fxlab.universe import CROSS_DEFS  # noqa: E402

pd.set_option("display.width", 240)

OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
SEEDS = (0, 1, 2, 3, 4)
EPISODE_TH = 0.02          # M1 が H4 より 2pp 以上深い局面
GATE_RATIO = 1.15          # M1/H4 DD 比率の昇格上限
GATE_LEV = 25.0            # 国内レバ規制
INIT = 10_000.0
OUT_JSON = ROOT / "research" / "outputs" / "exp44_mp11_m1audit.json"
OUT_EPI = ROOT / "research" / "outputs" / "exp44_m1_episodes.csv"


# ---------------------------------------------------------------- A. protocol
def eval_protocol(label, pool, closes, mk, max_pos, seeds=SEEDS):
    cache = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            eqm, _, _ = mm.simulate(pool, closes, mk(kk), max_pos=max_pos)
            cache[kk] = eqm
        return cache[kk]

    res = protocol_eval(eq_of_k, label=label, seeds=seeds)
    res["eq_of_k"] = eq_of_k
    return res


# ---------------------------------------------------------------- C. M1 replay
def simulate_with_log(pool, closes, sizing, *, init=INIT, max_pos=8):
    """mm_lab.simulate と同一ロジック + 建玉ログ(alloc/entry/exit)を返す(exp24 方式)。"""
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
    eq_real = np.empty(n)
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
        eq_real[b] = equity
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
    return (pd.Series(eq_mtm, index=grid), pd.Series(eq_real, index=grid),
            pd.DataFrame(log), {"skipped": skipped})


_M1_RAW: dict[str, pd.Series] = {}
_PX: dict[str, np.ndarray] = {}


def m1_close_naive(name: str) -> pd.Series:
    """メジャー=実 M1 close、合成クロス=脚 M1 close の inner-join 合成(anatomy_time 方式)。"""
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
    """建玉ログから M1 グリッド上の MtM equity と gross notional を再構成。

    規約(exp24 と同一): H4 バー b の close 時刻 = gi[b]+4h。建玉の含み損益は
    [エントリーバー close, エグジットバー close) の M1 close で評価、実現損益は
    エグジットバー close 時刻にステップ(H4 シミュの决済規約と整合)。
    """
    gi = closes.index.tz_localize(None).to_numpy()
    gridv = grid_idx.to_numpy()
    n_m1 = len(gridv)
    unreal = np.zeros(n_m1)
    gross = np.zeros(n_m1)

    # 実現 equity: exit バー終端時刻にステップ
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
            gross[a:b] += p["alloc"]
    eq = real + unreal
    return pd.Series(eq, index=grid_idx), gross


def dd_of(arr: np.ndarray) -> np.ndarray:
    return arr / np.maximum.accumulate(arr) - 1.0


def m1_audit_one(label, pool, closes, mk, max_pos, k, grid_idx):
    """較正 k での M1 監査: DD 比較 + 深掘り局面 + 掛け目 k_adj。"""
    t0 = time.time()
    eqm, eqr, log, info = simulate_with_log(pool, closes, mk(k), max_pos=max_pos)
    dd_h4_series = dd_of(eqm.to_numpy())
    dd_h4 = float(dd_h4_series.min())
    t_h4_trough = eqm.index[int(np.argmin(dd_h4_series))]

    eq_m1, gross = m1_replay(log, closes, grid_idx)
    em1 = eq_m1.to_numpy()
    dd_m1_series = dd_of(em1)
    dd_m1 = float(dd_m1_series.min())
    i_trough = int(np.argmin(dd_m1_series))
    t_m1_trough = grid_idx[i_trough]
    ratio = abs(dd_m1) / abs(dd_h4)
    print(f"  [{label}] k={k:.3f}  DD(H4)={dd_h4:+.2%} @{str(t_h4_trough)[:16]}  "
          f"DD(M1)={dd_m1:+.2%} @{str(t_m1_trough)[:16]}  比率={ratio:.3f}  "
          f"({time.time()-t0:.0f}s)")

    # --- H4 DD をM1グリッドへ整列(ffill ステップ)し、2pp以上深い局面を抽出 ---
    h4_times = closes.index.tz_localize(None).to_numpy() + np.timedelta64(4, "h")
    pos = np.searchsorted(h4_times, grid_idx.to_numpy(), side="right") - 1
    dd_h4_on = np.where(pos >= 0, dd_h4_series[np.clip(pos, 0, None)], 0.0)
    gap = dd_m1_series - dd_h4_on
    mask = gap <= -EPISODE_TH

    episodes = []
    if mask.any():
        idx = np.flatnonzero(mask)
        # 1日(1440グリッド本)未満の途切れはまとめる
        brk = np.flatnonzero(np.diff(idx) > 1440)
        starts = np.concatenate([[0], brk + 1])
        ends = np.concatenate([brk, [len(idx) - 1]])
        gridv = grid_idx.to_numpy()
        for s_, e_ in zip(starts, ends):
            a, b = idx[s_], idx[e_]
            seg = slice(a, b + 1)
            j = a + int(np.argmin(gap[seg]))
            tt = pd.Timestamp(gridv[j])
            # 週末ギャップ判定: トラフ前2日以内に2時間超のM1グリッド欠落があるか
            w0 = max(0, j - 2880)
            diffs = np.diff(gridv[w0:j + 1]).astype("timedelta64[m]").astype(int)
            max_gap_min = int(diffs.max()) if len(diffs) else 0
            # トラフ時点の建玉内訳(最大含み損3件)
            contrib = trough_contributions(log, closes, grid_idx, j, em1[j])
            episodes.append({
                "start": str(pd.Timestamp(gridv[a]))[:16], "end": str(pd.Timestamp(gridv[b]))[:16],
                "trough": str(tt)[:16], "dow": tt.day_name()[:3], "hour_utc": tt.hour,
                "dd_m1": float(dd_m1_series[j]), "dd_h4_aligned": float(dd_h4_on[j]),
                "gap_pp": float(gap[j] * 100), "max_m1gap_min_2d": max_gap_min,
                "weekend_gap": max_gap_min >= 120, "top_unreal": contrib,
            })
        episodes.sort(key=lambda x: x["gap_pp"])

    # --- 掛け目: M1 DD が H4 較正レベル(dd_h4@k)に一致する k_adj を反復 ---
    k_adj, dd_m1_adj, cagr_adj = k, dd_m1, cagr_of(eqm)
    target = abs(dd_h4)
    if abs(dd_m1) > target * 1.005:
        for _ in range(4):
            k_adj = k_adj * (target / abs(dd_m1_adj))
            eqm_a, _, log_a, _ = simulate_with_log(pool, closes, mk(k_adj), max_pos=max_pos)
            eq_m1_a, _ = m1_replay(log_a, closes, grid_idx)
            dd_m1_adj = float(dd_of(eq_m1_a.to_numpy()).min())
            cagr_adj = cagr_of(eqm_a)
            if abs(abs(dd_m1_adj) - target) < 0.002:
                break

    return {
        "label": label, "k": k, "cagr_h4": cagr_of(eqm),
        "dd_h4": dd_h4, "dd_m1": dd_m1, "ratio": ratio,
        "trough_h4": str(t_h4_trough)[:16], "trough_m1": str(t_m1_trough)[:16],
        "p95_m1_approx": -0.20 * ratio,   # 規約: p95_M1 ≈ 20% × (DD_M1/DD_H4)
        "k_adj": k_adj, "haircut": k_adj / k, "dd_m1_adj": dd_m1_adj, "cagr_adj": cagr_adj,
        "episodes": episodes, "eq_m1": eq_m1, "gross": gross, "log": log,
        "skipped": info["skipped"],
    }


def trough_contributions(log, closes, grid_idx, j, eq_at, top=3):
    """トラフ分(grid index j)時点の建玉ごとの含み損益(対equity%)上位 top 件。"""
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


# ---------------------------------------------------------------- main
def main() -> int:
    pool = mm.build_pool()
    closes = mm.load_closes()
    print(f"pool {len(pool)} trades / H4 grid {len(closes)} bars x {len(closes.columns)} instr")
    fbar = float(np.mean([_fz(z) for z in pool["z_entry"].to_numpy()]))
    print(f"f(z) 正規化定数 fbar={fbar:.4f}  (z0={Z0}, P={P}, clip_hi={CLIP_HI})")

    results: dict = {"meta": {"seeds": list(SEEDS), "fbar": fbar,
                              "p95_m1_rule": "p95_M1 ~= 20% x (DD_M1/DD_H4) 掛け目近似"}}

    # ====== A. ペアシード較正(mp8 vs mp11) ==================================
    print("\n=== A. ペアシード較正 protocol_eval seeds 0-4 (robust p95=20% / empirical 20%) ===")
    mk8 = champion_sizing(pool, max_pos=8)
    mk11 = champion_sizing(pool, max_pos=11)
    r8 = eval_protocol("mp8 (baseline)", pool, closes, mk8, 8)
    r11 = eval_protocol("mp11 (candidate)", pool, closes, mk11, 11)

    rows = []
    for sd in SEEDS:
        rows.append({"seed": sd,
                     "mp8_k": r8["rob"][sd]["k"], "mp8_cagr": r8["rob"][sd]["cagr"],
                     "mp11_k": r11["rob"][sd]["k"], "mp11_cagr": r11["rob"][sd]["cagr"],
                     "diff_pp": (r11["rob"][sd]["cagr"] - r8["rob"][sd]["cagr"]) * 100})
    tab = pd.DataFrame(rows)
    print("\nper-seed robust 比較:")
    print(tab.round(4).to_string(index=False))
    mean_diff = float(tab["diff_pp"].mean())
    print(f"robust 平均差(mp11-mp8) = {mean_diff:+.2f}pp  "
          f"(mp8 mean {r8['rob_cagr_mean']:+.2%} / mp11 mean {r11['rob_cagr_mean']:+.2%})")
    print(f"empirical: mp8 k={r8['emp_k']:.2f} CAGR={r8['emp_cagr']:+.2%} p95={r8['emp_p95']:+.1%} | "
          f"mp11 k={r11['emp_k']:.2f} CAGR={r11['emp_cagr']:+.2%} p95={r11['emp_p95']:+.1%}")
    tail_ok = not (r11["emp_cagr"] > r8["emp_cagr"] and r11["emp_p95"] < r8["emp_p95"] - 0.005)
    print(f"レバ偽装署名(emp CAGR↑ & p95悪化>0.5pp): {'なし → OK' if tail_ok else 'あり → NG'}")

    k8 = float(np.mean([r8["rob"][sd]["k"] for sd in SEEDS]))
    k11 = float(np.mean([r11["rob"][sd]["k"] for sd in SEEDS]))
    print(f"robust k(5シード平均): mp8={k8:.3f} / mp11={k11:.3f}  ← M1監査・証拠金監査の較正値")

    # 年次(robust k平均)
    eq8, _, _ = mm.simulate(pool, closes, mk8(k8), max_pos=8)
    eq11, _, _ = mm.simulate(pool, closes, mk11(k11), max_pos=11)
    y8, y11 = yearly_returns(eq8), yearly_returns(eq11)
    ydiff = (y11 - y8) * 100
    ytab = pd.DataFrame({"mp8_%": y8 * 100, "mp11_%": y11 * 100, "diff_pp": ydiff}).round(2)
    print("\n年次リターン(robust k平均):")
    print(ytab.to_string())
    neg8, neg11 = int((y8 < 0).sum()), int((y11 < 0).sum())
    best_y = int(ydiff.idxmax())
    d_all = float(ydiff.mean())
    d_ex_best = float(ydiff.drop(best_y).mean())
    d_ex_2022 = float(ydiff.drop(2022).mean()) if 2022 in ydiff.index else float("nan")
    print(f"負け年: mp8={neg8} / mp11={neg11}  改善幅平均={d_all:+.2f}pp  "
          f"最良年({best_y})除外={d_ex_best:+.2f}pp  2022除外={d_ex_2022:+.2f}pp")

    results["A_protocol"] = {
        "per_seed": rows, "mean_diff_pp": mean_diff,
        "mp8": {k: v for k, v in r8.items() if k not in ("rob", "eq_of_k")},
        "mp11": {k: v for k, v in r11.items() if k not in ("rob", "eq_of_k")},
        "rob8": {str(s): r8["rob"][s] for s in SEEDS}, "rob11": {str(s): r11["rob"][s] for s in SEEDS},
        "k8_mean": k8, "k11_mean": k11, "tail_ok": tail_ok,
        "yearly": {str(y): {"mp8": float(y8[y]), "mp11": float(y11[y])} for y in y8.index},
        "neg_years": {"mp8": neg8, "mp11": neg11},
        "diff_mean_pp": d_all, "diff_ex_best_pp": d_ex_best, "best_year": best_y,
        "diff_ex_2022_pp": d_ex_2022,
    }

    # ====== B. IS-argmax 監査 ================================================
    print("\n=== B. IS(<2022) 単独で max_pos を選んだら? (robust seed0 較正 → OOS 素検証) ===")
    is_pool = pool[pool["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]
    oos_cl = closes[closes.index >= OOS_START]
    is_res = {}
    for label, mk, mp in [("mp8", mk8, 8), ("mp11", mk11, 11)]:
        cache = {}

        def eq_is(k, mk=mk, mp=mp, cache=cache):
            kk = round(float(k), 10)
            if kk not in cache:
                eqm, _, _ = mm.simulate(is_pool, is_cl, mk(kk), max_pos=mp)
                cache[kk] = eqm
            return cache[kk]

        k_is_rob = calibrate_robust_seeded(eq_is, 0.20, seed=0)
        cagr_is = cagr_of(eq_is(k_is_rob))
        eqo, _, _ = mm.simulate(oos_pool, oos_cl, mk(k_is_rob), max_pos=mp)
        k_is_emp = calibrate_empirical(eq_is, 0.20)
        eqo_e, _, _ = mm.simulate(oos_pool, oos_cl, mk(k_is_emp), max_pos=mp)
        is_res[label] = {
            "k_is_rob": k_is_rob, "is_rob_cagr": cagr_is,
            "oos_cagr": cagr_of(eqo), "oos_dd": max_dd(eqo),
            "k_is_emp": k_is_emp, "oos_cagr_emp": cagr_of(eqo_e), "oos_dd_emp": max_dd(eqo_e),
        }
        print(f"  {label:5s} IS robust k={k_is_rob:5.2f} IS CAGR={cagr_is:+.2%} -> "
              f"OOS CAGR={cagr_of(eqo):+.2%} DD={max_dd(eqo):+.1%} | "
              f"emp k={k_is_emp:5.2f} -> OOS {cagr_of(eqo_e):+.2%} DD={max_dd(eqo_e):+.1%}")
    is_pick = max(is_res, key=lambda x: is_res[x]["is_rob_cagr"])
    print(f"IS argmax = {is_pick}(IS robust CAGR基準)")
    results["B_is_argmax"] = {"pick": is_pick, **{k: v for k, v in is_res.items()}}

    # ====== C. M1 粒度 MtM 監査 ==============================================
    print("\n=== C. M1粒度 MtM リプレイ(robust k 5シード平均で較正した構成) ===")
    grid_idx = pd.DatetimeIndex(load_m1("EURUSD").index.tz_localize(None))
    print(f"M1 grid: {len(grid_idx):,} bars  ({grid_idx[0]} .. {grid_idx[-1]})")
    a8 = m1_audit_one("mp8 ", pool, closes, mk8, 8, k8, grid_idx)
    a11 = m1_audit_one("mp11", pool, closes, mk11, 11, k11, grid_idx)

    for a in (a8, a11):
        print(f"\n  -- {a['label']} 深掘り局面(M1がH4整列DDより{EPISODE_TH:.0%}pp以上深い, 谷の深い順) --")
        if not a["episodes"]:
            print("    なし")
        for e in a["episodes"][:8]:
            print(f"    {e['trough']} ({e['dow']} {e['hour_utc']:02d}h UTC)  "
                  f"M1={e['dd_m1']:+.1%} vs H4={e['dd_h4_aligned']:+.1%} (gap {e['gap_pp']:+.1f}pp)  "
                  f"週末ギャップ{'有' if e['weekend_gap'] else '無'}(max欠落{e['max_m1gap_min_2d']}分)  "
                  f"建玉: {', '.join(e['top_unreal'])}")
        print(f"    掛け目: k_adj={a['k_adj']:.3f} (haircut x{a['haircut']:.3f})  "
              f"M1 DD(adj)={a['dd_m1_adj']:+.2%}  実効CAGR={a['cagr_adj']:+.2%}  "
              f"p95_M1近似={a['p95_m1_approx']:+.1%}")

    epi_rows = []
    for a in (a8, a11):
        for e in a["episodes"]:
            epi_rows.append({"config": a["label"].strip(), **{k: v for k, v in e.items()
                                                              if k != "top_unreal"},
                             "top_unreal": "; ".join(e["top_unreal"])})
    OUT_EPI.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(epi_rows).to_csv(OUT_EPI, index=False)

    results["C_m1"] = {a["label"].strip(): {k: v for k, v in a.items()
                                            if k not in ("eq_m1", "gross", "log", "episodes")}
                       | {"n_episodes": len(a["episodes"]),
                          "episodes_top5": [{k: v for k, v in e.items() if k != "top_unreal"}
                                            for e in a["episodes"][:5]]}
                       for a in (a8, a11)}

    # ====== D. 証拠金監査 ====================================================
    print("\n=== D. 証拠金監査(M1グリッド: 総建玉/MtM equity) ===")
    marg = {}
    for a, mp in ((a8, 8), (a11, 11)):
        em1 = a["eq_m1"].to_numpy()
        gross = a["gross"]
        with np.errstate(divide="ignore", invalid="ignore"):
            lev = np.where(em1 > 0, gross / em1, np.inf)
        i_max = int(np.argmax(lev))
        max_lev = float(lev[i_max])
        t_max = pd.Timestamp(grid_idx.to_numpy()[i_max])
        live = lev[gross > 0]
        # 1玉最大レバ(実測): エントリー時 alloc / M1 equity
        gi = closes.index.tz_localize(None).to_numpy()
        t_ins = gi[a["log"]["entry_pos"].to_numpy()] + np.timedelta64(4, "h")
        e_at = em1[np.clip(np.searchsorted(grid_idx.to_numpy(), t_ins, side="left"),
                           0, len(em1) - 1)]
        single = a["log"]["alloc"].to_numpy() / e_at
        m = {"max_lev": max_lev, "t_max_lev": str(t_max)[:16],
             "mean_lev_when_open": float(np.mean(live[np.isfinite(live)])),
             "p99_lev": float(np.percentile(live[np.isfinite(live)], 99)),
             "mm25_min_pct": 25.0 / max_lev * 100, "mm100_min_pct": 100.0 / max_lev * 100,
             "single_max_lev": float(single.max()),
             "single_theory_max": a["k"] / mp * CLIP_HI / fbar}
        marg[a["label"].strip()] = m
        print(f"  {a['label']} k={a['k']:.2f}: 最大総レバ={m['max_lev']:.2f}x @{m['t_max_lev']} "
              f"(平均{m['mean_lev_when_open']:.2f}x / p99 {m['p99_lev']:.2f}x)\n"
              f"        証拠金維持率最低: 国内25倍={m['mm25_min_pct']:.0f}% / 海外100倍={m['mm100_min_pct']:.0f}%\n"
              f"        1玉最大レバ: 実測{m['single_max_lev']:.2f}x / 理論上限 {m['single_theory_max']:.2f}x")
    results["D_margin"] = marg

    # ====== E. 判定 ==========================================================
    print("\n=== E. 判定 ===")
    g_ratio = a11["ratio"] <= GATE_RATIO
    g_lev = marg["mp11"]["max_lev"] <= GATE_LEV
    print(f"  ゲート(a) M1/H4 DD比率 {a11['ratio']:.3f} <= {GATE_RATIO}: {'PASS' if g_ratio else 'FAIL'}")
    print(f"  ゲート(b) 最大総レバ {marg['mp11']['max_lev']:.2f}x <= {GATE_LEV}: {'PASS' if g_lev else 'FAIL'}")
    print(f"  robust平均差 {mean_diff:+.2f}pp / tail_ok={tail_ok} / IS argmax={is_pick} / "
          f"負け年 mp8={neg8} mp11={neg11}")
    honest11, honest8 = a11["cagr_adj"], a8["cagr_adj"]
    print(f"  掛け目込み実効CAGR: mp8={honest8:+.2%} (k_adj={a8['k_adj']:.2f}) vs "
          f"mp11={honest11:+.2%} (k_adj={a11['k_adj']:.2f})  差={100*(honest11-honest8):+.2f}pp")
    adopt = g_ratio and g_lev and tail_ok and mean_diff > 0 and neg11 <= neg8 \
        and d_ex_best > 0 and honest11 > honest8
    verdict = "adopt" if adopt else "reject"
    print(f"  → {verdict}: mp11 "
          f"{'昇格(掛け目 k x%.3f を明記)' % a11['haircut'] if adopt else '昇格見送り'}")
    results["E_verdict"] = {"gate_ratio": g_ratio, "gate_lev": g_lev, "verdict": verdict,
                            "honest_cagr_mp8": honest8, "honest_cagr_mp11": honest11}

    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nsaved -> {OUT_JSON}\n        -> {OUT_EPI}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
