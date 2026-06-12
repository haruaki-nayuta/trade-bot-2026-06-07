"""exp73: 時間トレール(注文6分割×10バーごと段階決済)の検証 — ユーザー発案。

問い: z出口は維持したまま(先にz出口が来たら残量全決済)、来なければ bar10,20,30,40,50,60 で
1/6 ずつ閉じる「時間トレール」はチャンピオン(confluence_meanrev_v2_d1 + P4.0 mp8)を改善するか。

事前知識(本実験の座標系。必読の引用):
  - **全量時間ストップは閉鎖済み**(exp13/exp63, reports/21 §5: 時間ストップ30本は救済+0.401に
    対し長持ち勝者の誤殺 -0.715 = 約3倍が上回る。60本でもプール差 -0.014)。
  - **z基準の分割決済・早期利確も閉鎖済み**(exp26: 18構成全てベースライン未満。reports/25 §1)。
  - **reports/21(機構の定量は reports/15 出口層)**: 建玉の残り期待値 E[最終−現在] は
    **どの時点・どの含み損深度でもプラス(+15〜+79bps)** → トランシェを閉じるたびにその正の
    期待値を放棄する = **プール段は構造的にマイナスのはず(これは織り込み済み)**。
    プール段マイナスだけでは reject にしない。
  - **本判定は口座段**:「プール期待値を払って DD形状改善 → 較正k上昇を買う」交換レートが
    正味プラスか(d1 が勝った機構=パスの谷を浅くして同一テール制約下の k を上げる、と同じ経路で
    勝てるか)。
  - **勝ち側コストの定量化**: 勝ちトレード(保有中央値12本/p90 24本, reports/25)は z出口が先に
    来てトランシェがほぼ発火しない一方、bar10 で 1/6 だけ早出しするケースは「リバーサルバーの
    オーバーシュート」(exp71, reports/25 §3: 出口の機械的前倒しは勝ちの尻尾切り)を部分的に
    切る — 勝ちトレードに付着するコストを bar10 トランシェ単位で分解して報告する。

設計(事前登録。グリッド漁り禁止。構成は4つだけ):
  A) 6分割×10バー(bar10..60で1/6ずつ)          … ユーザー提案=主構成(判定対象)
  B) 6分割×10バー・開始をbar30に遅延(bar30..80) … 勝ちゾーン(中央12/p90 24本)を保護した変種
  C) 3分割×20バー(bar20,40,60で1/3ずつ)         … 粗い用量
  D) 参照: 全量時間ストップ60本                   … 既知の死(比較座標。判定対象外)
  いずれも z出口が先に来たら残量を全決済(現行どおり)。

コスト規約(按分でコスト中立であることの明示):
  トランシェ約定価格 = 該当バー close ∓ 半スプレッド按分。半スプレッドはノーション比例なので、
  エントリー半スプレッド(全量1回)+ 決済半スプレッド(1/6 × 6回)= 往復ちょうど1スプレッド
  = **分割しても合計の相対コストは1回決済と同一(按分でコスト中立)**。
  実装では per-unit ret に親トレードの往復コスト rc["cost"](exp47 reconstruct の恒等式
  cost = gross − ret)をそのまま含め、口座段で alloc×w を掛けることで自動的に按分される。

口座段(本判定):
  mm_lab.simulate はトレード=1行(単一exit)前提のため、simulate を写経した薄い変種
  **simulate_weighted** を本ファイル内に実装する。equity・MtM・実現のセマンティクスは
  mm_lab.simulate と同一で、alloc×w のみ追加。slot は parent(親トレード)単位で1カウント、
  エントリーは親単位で all-or-nothing スキップ。サイジングは champion_sizing と同一の
  f(z)(親の z_entry, f̄ も親プールで正規化=全構成共通)を使う。
  **検証必須**: 全行 w=1・トランシェなしで mm_lab.simulate とバー単位一致(equity 完全一致)+
  ベースライン(rob_s0 k≈6.084 / CAGR≈+18.24%、emp k≈8.895 / +27.5%)の再現。
  較正: tail_protocol ペアシード seeds 0-4 robust(p95 DD=20%)+ empirical をベースと
  ペア較正。ゲート: レバ偽装署名(emp CAGR↑かつ emp p95 0.5pp超悪化)・
  G3(IS<2022較正→OOS素検証 rob/emp両改善)・全年プラス維持。

判定基準(事前登録):
  gain_pp(5シード平均ΔCAGR)が較正ノイズ帯(±0.4-0.8pp)未満 → 'noise'。
  署名あり or 明確な劣化 → 'reject'。ノイズ帯超(>+0.8pp)+全ゲート通過 → 'adopt-candidate'。
  D(参照)は判定対象外。

実行: PYTHONPATH=. uv run python research/experiments/exp73_time_trail.py
出力: research/outputs/exp73_result.json
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
sys.path.insert(0, str(ROOT / "research" / "experiments"))

import mm_lab as mm  # noqa: E402
from mm_production import build_pool_d1, champion_sizing  # noqa: E402
from tail_protocol import (  # noqa: E402
    boot_dd, cagr_of, calibrate_empirical, calibrate_robust_seeded, max_dd,
    yearly_returns,
)
from exp47_entry_delay import reconstruct  # noqa: E402
from fxlab import universe as uni  # noqa: E402

OUT_DIR = ROOT / "research" / "outputs"
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
MAX_POS = 8
SEEDS = (0, 1, 2, 3, 4)
POOL_N_EXPECT = 1207
POOL_SUM_EXPECT = 1.9622098170096924   # exp52/71/72 の d1 確定値

CONFIGS = {
    "A_6x10": [10, 20, 30, 40, 50, 60],
    "B_6x10_start30": [30, 40, 50, 60, 70, 80],
    "C_3x20": [20, 40, 60],
    "D_ts60_ref": [60],
}
JUDGED = ("A_6x10", "B_6x10_start30", "C_3x20")   # D は参照のみ

T0 = time.time()


def el():
    return f"[{time.time()-T0:.0f}s]"


# --- プール段: トランシェ分解とブレンド再価格 ----------------------------
def build_rows(pool, rc, offsets):
    """各親トレードを「同一exitバーごとに重みを束ねた行」に分解する。

    トランシェ j(オフセット off): e+off < x なら該当バー close で決済(per-unit ret =
    dir·(c[e+off]/c[e]−1) − cost で往復コスト全額を含み、重みは alloc×w 側で按分=コスト中立)。
    e+off >= x なら z出口に合流(per-unit ret = 親の ret そのもの)。
    返り値: rows(DataFrame), diag(per-trade Δ・発火・放棄期待値の診断)。
    """
    m = len(offsets)
    w = 1.0 / m
    n = len(pool)
    delta = np.zeros(n)
    fired_tr = np.zeros(n, dtype=int)        # 親ごとの発火トランシェ数
    abandoned_w = 0.0                         # Σ w·(放棄した前向きリターン)
    ab_fwd_list = []                          # 発火トランシェの前向き(放棄)リターン per-unit
    first_off_delta = np.zeros(n)             # 最初のオフセットのトランシェΔ(勝ち側コスト分解用)
    first_off_fired = np.zeros(n, dtype=bool)
    recs = []
    for instr, g in pool.groupby("instr"):
        s = rc["closes_by"][instr]
        carr = s.to_numpy()
        tarr = s.index.values
        for ti in g.index.to_numpy():
            e, x = int(rc["idx_e"][ti]), int(rc["idx_x"][ti])
            d = float(pool.at[ti, "dir"])
            cost = float(rc["cost"][ti])
            ret0 = float(pool.at[ti, "ret"])
            # exitバー位置 -> 重み(z出口合流分は x に束ねる)
            exits: dict[int, float] = {}
            for off in offsets:
                p = e + off
                key = p if p < x else x
                exits[key] = exits.get(key, 0.0) + w
            blended = 0.0
            for p in sorted(exits):
                wsum = exits[p]
                if p == x:
                    retj = ret0
                    ex_ts = pool.at[ti, "exit"]
                else:
                    retj = d * (carr[p] / carr[e] - 1.0) - cost
                    ex_ts = pd.Timestamp(tarr[p]).tz_localize("UTC")
                    n_fire = int(round(wsum * m))
                    fired_tr[ti] += n_fire
                    fwd = d * (rc["exit_close"][ti] / carr[p] - 1.0)  # 放棄した残り(gross, コスト中立)
                    abandoned_w += wsum * fwd
                    ab_fwd_list.extend([fwd] * n_fire)
                    if p == e + offsets[0]:
                        first_off_fired[ti] = True
                        first_off_delta[ti] = w * (retj - ret0)
                blended += wsum * retj
                recs.append({
                    "instr": instr, "parent": int(ti),
                    "entry": pool.at[ti, "entry"], "exit": ex_ts,
                    "dir": int(d), "entry_price": float(pool.at[ti, "entry_price"]),
                    "ret": float(retj), "w": float(wsum),
                    "z_entry": float(pool.at[ti, "z_entry"]),
                    "bars_held": int(p - e),
                })
            delta[ti] = blended - ret0
    rows = pd.DataFrame(recs).sort_values(["parent", "bars_held"]).reset_index(drop=True)
    diag = {"delta": delta, "fired_tr": fired_tr, "m": m,
            "abandoned_w": abandoned_w, "ab_fwd": np.array(ab_fwd_list),
            "first_off_delta": first_off_delta, "first_off_fired": first_off_fired}
    return rows, diag


def pool_stage_report(cfg, pool, diag):
    n = len(pool)
    ret0 = pool["ret"].to_numpy()
    win = ret0 > 0
    worst10 = ret0 < np.quantile(ret0, 0.10)
    d = diag["delta"]
    m = diag["m"]
    yr = pd.Series(d, index=pd.DatetimeIndex(pool["exit"]).year).groupby(level=0).sum()
    ab = diag["ab_fwd"]
    fo = diag["first_off_delta"]
    fof = diag["first_off_fired"]
    out = {
        "config": cfg,
        "n_tranches": int(n * m),
        "n_fired": int(diag["fired_tr"].sum()),
        "tranche_fire_rate": float(diag["fired_tr"].sum() / (n * m)),
        "parent_any_fire_rate": float((diag["fired_tr"] > 0).mean()),
        "delta_sum": float(d.sum()),
        "delta_bps_per_trade": float(d.mean() * 1e4),
        "delta_win_sum": float(d[win].sum()),
        "delta_win_bps": float(d[win].mean() * 1e4),
        "delta_loss_sum": float(d[~win].sum()),
        "delta_loss_bps": float(d[~win].mean() * 1e4),
        "delta_worst10_sum": float(d[worst10].sum()),
        "abandoned_fwd_mean_bps": float(ab.mean() * 1e4) if len(ab) else float("nan"),
        "abandoned_w_sum": float(diag["abandoned_w"]),
        "identity_ratio": float(-d.sum() / diag["abandoned_w"]) if diag["abandoned_w"] != 0 else float("nan"),
        "first_off_win_cost_bps": float(fo[win & fof].mean() * 1e4) if (win & fof).any() else float("nan"),
        "first_off_win_fired_n": int((win & fof).sum()),
        "first_off_win_cost_sum": float(fo[win & fof].sum()),
        "yearly_delta": {int(y): float(v) for y, v in yr.items()},
        "neg_delta_years": int((yr < 0).sum()),
    }
    print(f"  [{cfg}] Δ {out['delta_bps_per_trade']:+.2f}bps/件 (sum {out['delta_sum']:+.4f})  "
          f"発火 {out['n_fired']}/{out['n_tranches']}トランシェ({out['tranche_fire_rate']:.1%}) "
          f"親≥1発火 {out['parent_any_fire_rate']:.1%}")
    print(f"      付着先: 勝ち {out['delta_win_sum']:+.4f}({out['delta_win_bps']:+.2f}bps/件) / "
          f"負け {out['delta_loss_sum']:+.4f}({out['delta_loss_bps']:+.2f}bps/件) / "
          f"ワースト10% {out['delta_worst10_sum']:+.4f}")
    print(f"      放棄した残り期待値: mean {out['abandoned_fwd_mean_bps']:+.1f}bps/トランシェ "
          f"(reports/15の+15〜+79bps帯と照合)  恒等比 -Δsum/Σw·fwd = {out['identity_ratio']:.3f}")
    if np.isfinite(out["first_off_win_cost_bps"]):
        print(f"      勝ち側コスト(最初のオフセットbar{CONFIGS[cfg][0]}): "
              f"{out['first_off_win_fired_n']}勝ちトレードで発火, "
              f"{out['first_off_win_cost_bps']:+.2f}bps/件 (sum {out['first_off_win_cost_sum']:+.4f}) "
              f"= exp71のオーバーシュート切り")
    return out


# --- 口座段: 重み付き複数行シミュレータ(mm_lab.simulate の写経+w) -------
def simulate_weighted(rows: pd.DataFrame, closes: pd.DataFrame, sizing, *, init=10_000.0,
                      max_pos=8, vol_win=120):
    """mm_lab.simulate を写経した薄い変種。equity・MtM・実現のセマンティクスは同一で、
    (1) 行の alloc に w を乗じる (2) slot は parent 単位で1カウント
    (3) エントリーは親単位で all-or-nothing(満枠 or sizing<=0 で親ごとスキップ)のみ追加。
    全行 w=1・1親1行なら mm_lab.simulate とバー単位で完全一致する(main で検証)。"""
    grid = closes.index
    col_of = {c: i for i, c in enumerate(closes.columns)}
    carr = closes.to_numpy()
    n = len(grid)

    gi = grid.to_numpy()
    entry_pos = np.clip(np.searchsorted(gi, rows["entry"].to_numpy(), side="left"), 0, n - 1)
    exit_pos = np.clip(np.searchsorted(gi, rows["exit"].to_numpy(), side="left"), 0, n - 1)

    parent_arr = rows["parent"].to_numpy()
    instr_arr = rows["instr"].to_numpy()
    dir_arr = rows["dir"].to_numpy().astype(float)
    eprice_arr = rows["entry_price"].to_numpy()
    ret_arr = rows["ret"].to_numpy()
    w_arr = rows["w"].to_numpy()
    z_arr = rows["z_entry"].to_numpy()
    bars_arr = rows["bars_held"].to_numpy()

    # 親 -> 行リスト / エントリーバー -> 親リスト(親の出現順=プール順を保持)
    rows_of: dict[int, list[int]] = {}
    by_entry: dict[int, list[int]] = {}
    for ri in range(len(rows)):
        pa = int(parent_arr[ri])
        if pa not in rows_of:
            rows_of[pa] = []
            by_entry.setdefault(int(entry_pos[ri]), []).append(pa)
        rows_of[pa].append(ri)

    equity = init
    peak_mtm = init
    open_pos = []                 # dict(pa, col, dir, eprice, alloc, exit_pos, ret)
    open_parents: dict[int, int] = {}   # 親 -> 未決済行数
    eq_mtm = np.empty(n)
    eq_real = np.empty(n)
    mtm_ret_hist = np.empty(n)
    prev_mtm = init
    conc = []
    skipped = 0

    for b in range(n):
        # ① 決済
        if open_pos:
            still = []
            for p in open_pos:
                if p["exit_pos"] <= b:
                    equity += p["alloc"] * p["ret"]
                    open_parents[p["pa"]] -= 1
                    if open_parents[p["pa"]] == 0:
                        del open_parents[p["pa"]]
                else:
                    still.append(p)
            open_pos = still

        # ② MtM
        unreal = 0.0
        for p in open_pos:
            px = carr[b, p["col"]]
            run_ret = p["dir"] * (px / p["eprice"] - 1.0)
            unreal += p["alloc"] * run_ret
        mtm = equity + unreal
        eq_mtm[b] = mtm
        eq_real[b] = equity
        peak_mtm = max(peak_mtm, mtm)
        mtm_ret_hist[b] = (mtm / prev_mtm - 1.0) if prev_mtm > 0 else 0.0
        prev_mtm = mtm

        if b >= vol_win:
            rv = mtm_ret_hist[b - vol_win + 1:b + 1]
            recent_vol = float(np.std(rv) * np.sqrt(mm.BARS_PER_YEAR.get("H4", 1512)))
        else:
            recent_vol = float("nan")
        dd_mtm = mtm / peak_mtm - 1.0

        # ③ 新規エントリー(親単位 all-or-nothing)
        if b in by_entry:
            for pa in by_entry[b]:
                if len(open_parents) >= max_pos:
                    skipped += 1
                    continue
                ri0 = rows_of[pa][0]
                ctx = {
                    "equity_real": equity, "equity_mtm": mtm, "peak_mtm": peak_mtm,
                    "dd_mtm": dd_mtm, "n_open": len(open_parents), "max_pos": max_pos,
                    "recent_vol": recent_vol, "z": float(z_arr[ri0]),
                    "instr": instr_arr[ri0], "ret": float(ret_arr[ri0]),
                    "bars_held": int(bars_arr[ri0]),
                }
                alloc = float(sizing(ctx))
                if alloc <= 0:
                    skipped += 1
                    continue
                for ri in rows_of[pa]:
                    open_pos.append({
                        "pa": pa, "col": col_of[instr_arr[ri]], "dir": dir_arr[ri],
                        "eprice": eprice_arr[ri], "alloc": alloc * float(w_arr[ri]),
                        "exit_pos": int(exit_pos[ri]), "ret": float(ret_arr[ri]),
                    })
                open_parents[pa] = len(rows_of[pa])
                conc.append(len(open_parents))

    eq_mtm = pd.Series(eq_mtm, index=grid)
    eq_real = pd.Series(eq_real, index=grid)
    info = {"final": equity, "skipped": skipped, "n_taken": len(conc),
            "max_conc": max(conc) if conc else 0,
            "avg_conc": float(np.mean(conc)) if conc else 0.0}
    return eq_mtm, eq_real, info


def full_eval(tag, rows, closes, mk, seeds=SEEDS):
    """tail_protocol 流フル評価: empirical + robust seeds0-4 + G3(IS<2022較正→OOS素検証)。"""
    cache = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            cache[kk] = simulate_weighted(rows, closes, mk(kk), max_pos=MAX_POS)
        return cache[kk][0]

    out = {"label": tag}
    k_emp = calibrate_empirical(eq_of_k, 0.20)
    eq_e = eq_of_k(k_emp)
    bs = boot_dd(eq_e, n_boot=1500, seed=0)
    out.update(emp_k=k_emp, emp_cagr=cagr_of(eq_e), emp_dd=max_dd(eq_e),
               emp_p95=bs["p95"], emp_p99=bs["p99"])
    rob = {}
    for sd in seeds:
        kr = calibrate_robust_seeded(eq_of_k, 0.20, n_boot=600, seed=sd)
        rob[sd] = {"k": kr, "cagr": cagr_of(eq_of_k(kr))}
    out["rob"] = rob
    out["rob_cagr_mean"] = float(np.mean([v["cagr"] for v in rob.values()]))
    out["rob_k_mean"] = float(np.mean([v["k"] for v in rob.values()]))
    yr_e = yearly_returns(eq_of_k(k_emp))
    out["yr_emp"] = {int(y): float(v) for y, v in yr_e.items()}
    out["neg_years_emp"] = int((yr_e < 0).sum())
    out["worst_year_emp"] = float(yr_e.min())
    yr0 = yearly_returns(eq_of_k(rob[seeds[0]]["k"]))
    out["yr_rob0"] = {int(y): float(v) for y, v in yr0.items()}
    out["neg_years_rob0"] = int((yr0 < 0).sum())
    info = cache[round(float(k_emp), 10)][2]
    out["avg_conc"], out["max_conc"], out["skipped"] = info["avg_conc"], info["max_conc"], info["skipped"]

    # G3: IS<2022 較正 -> OOS 素検証(rob seed0 + emp の両方)
    is_rows = rows[rows["entry"] < OOS_START].reset_index(drop=True)
    oos_rows = rows[rows["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]
    oos_cl = closes[closes.index >= OOS_START]

    def eq_fn(rr, cc):
        c = {}

        def f(k):
            kk = round(float(k), 10)
            if kk not in c:
                c[kk] = simulate_weighted(rr, cc, mk(kk), max_pos=MAX_POS)[0]
            return c[kk]
        return f
    fi, fo = eq_fn(is_rows, is_cl), eq_fn(oos_rows, oos_cl)
    k_ir = calibrate_robust_seeded(fi, 0.20, n_boot=600, seed=0)
    out["k_is_rob"] = k_ir
    out["is_rob_cagr"], out["oos_rob_cagr"] = cagr_of(fi(k_ir)), cagr_of(fo(k_ir))
    out["oos_rob_dd"] = max_dd(fo(k_ir))
    k_ie = calibrate_empirical(fi, 0.20)
    out["k_is_emp"] = k_ie
    out["is_emp_cagr"], out["oos_emp_cagr"] = cagr_of(fi(k_ie)), cagr_of(fo(k_ie))
    out["oos_emp_dd"] = max_dd(fo(k_ie))
    print(f"  {tag:16s} emp k={k_emp:5.2f} CAGR={out['emp_cagr']:+7.2%} p95={out['emp_p95']:+6.1%} | "
          f"rob5 mean={out['rob_cagr_mean']:+7.2%} k̄={out['rob_k_mean']:.2f} | "
          f"OOS rob/emp {out['oos_rob_cagr']:+.2%}/{out['oos_emp_cagr']:+.2%}  {el()}")
    return out


def main() -> int:
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1().copy().reset_index(drop=True)
    closes = mm.load_closes()
    rc = reconstruct(pool)
    n = len(pool)
    s_ret = float(pool["ret"].sum())
    print(f"=== exp73: 時間トレール(分割×時間段階決済) (d1 pool n={n}, sum={s_ret:+.4f}) {el()} ===")
    assert n == POOL_N_EXPECT, f"pool n={n} != {POOL_N_EXPECT}"
    assert abs(s_ret - POOL_SUM_EXPECT) < 1e-6, f"pool sum {s_ret} != {POOL_SUM_EXPECT}"
    print(f"  検算OK: n={POOL_N_EXPECT} / sum(ret)={POOL_SUM_EXPECT:+.4f} と一致")

    sec = lambda t: print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

    # ---------------- 1. プール段(価格付け) ----------------
    sec("1. プール段: トランシェ分解のブレンド再価格(Δ・付着先・発火率・機構照合)")
    rows_by, pool_rows = {}, []
    for cfg, offs in CONFIGS.items():
        rows, diag = build_rows(pool, rc, offs)
        rows_by[cfg] = rows
        pool_rows.append(pool_stage_report(cfg, pool, diag))
    print("\n  参照照合: D_ts60_ref のΔsumは exp63 時間ストップ60本(-0.014, 介入36件)と"
          "一致するはず(同一規約)")

    # ---------------- 2. simulate_weighted の検証 ----------------
    sec("2. simulate_weighted 検証: w=1・トランシェなしで mm_lab.simulate とバー単位一致")
    base_rows = pool[["instr", "entry", "exit", "dir", "entry_price", "ret",
                      "bars_held", "z_entry"]].copy()
    base_rows["w"] = 1.0
    base_rows["parent"] = np.arange(n)
    mk = champion_sizing(pool, max_pos=MAX_POS)
    max_diff = 0.0
    for ktest in (2.0, 6.084):
        eq_a, eqr_a, info_a = mm.simulate(pool, closes, mk(ktest), max_pos=MAX_POS)
        eq_b, eqr_b, info_b = simulate_weighted(base_rows, closes, mk(ktest), max_pos=MAX_POS)
        md = float(np.max(np.abs(eq_a.to_numpy() - eq_b.to_numpy())))
        mdr = float(np.max(np.abs(eqr_a.to_numpy() - eqr_b.to_numpy())))
        max_diff = max(max_diff, md, mdr)
        print(f"  k={ktest}: |Δeq_mtm|max={md:.3e} |Δeq_real|max={mdr:.3e} "
              f"skipped {info_a['skipped']}=={info_b['skipped']} "
              f"n_taken {info_a['n_taken']}=={info_b['n_taken']}")
        assert info_a["skipped"] == info_b["skipped"] and info_a["n_taken"] == info_b["n_taken"]
    sim_identical = max_diff < 1e-6

    # ---------------- 3. 口座段(本判定) ----------------
    sec("3. 口座段: ペアシード seeds0-4 robust + empirical + G3(base vs A/B/C/D)")
    base = full_eval("base", base_rows, closes, mk)
    base_repro = (sim_identical
                  and abs(base["rob"][0]["k"] - 6.084) < 0.05
                  and abs(base["rob"][0]["cagr"] - 0.1824) < 0.005
                  and abs(base["emp_k"] - 8.895) < 0.05
                  and abs(base["emp_cagr"] - 0.2750) < 0.005)
    print(f"  ベースライン再現: rob_s0 k={base['rob'][0]['k']:.3f}/{base['rob'][0]['cagr']:+.2%} "
          f"(目標 6.084/+18.24%) emp k={base['emp_k']:.3f}/{base['emp_cagr']:+.2%} "
          f"(目標 8.895/+27.50%) -> {'OK' if base_repro else 'FAIL'}")
    if not base_repro:
        print("  !! ベースライン再現に失敗。simulate_weighted のバグの可能性 — 判定は無効。")

    results = {"base": base}
    acc_rows = []
    for cfg in CONFIGS:
        r = full_eval(cfg, rows_by[cfg], closes, mk)
        results[cfg] = r
        per_seed = {sd: r["rob"][sd]["cagr"] - base["rob"][sd]["cagr"] for sd in SEEDS}
        gain_pp = (r["rob_cagr_mean"] - base["rob_cagr_mean"]) * 100
        sig = (r["emp_cagr"] > base["emp_cagr"]) and \
              (abs(r["emp_p95"]) > abs(base["emp_p95"]) + 0.005)
        g3_rob = r["oos_rob_cagr"] - base["oos_rob_cagr"]
        g3_emp = r["oos_emp_cagr"] - base["oos_emp_cagr"]
        g3 = (g3_rob > 0) and (g3_emp > 0)
        all_pos = (r["neg_years_emp"] == 0) and (r["neg_years_rob0"] == 0)
        gates = {
            "gain_pp": gain_pp,
            "per_seed_pp": {sd: v * 100 for sd, v in per_seed.items()},
            "all_seeds_pos": all(v > 0 for v in per_seed.values()),
            "signature": bool(sig),
            "g3_oos_rob_pp": g3_rob * 100, "g3_oos_emp_pp": g3_emp * 100,
            "g3_pass": bool(g3),
            "all_years_pos": bool(all_pos),
            "k_emp_change": f"{base['emp_k']:.2f}->{r['emp_k']:.2f}",
            "k_rob_change": f"{base['rob_k_mean']:.2f}->{r['rob_k_mean']:.2f}",
        }
        r["gates"] = gates
        acc_rows.append({"config": cfg, **gates})
        print(f"      gain {gain_pp:+.2f}pp  seeds " +
              " ".join(f"s{sd}:{v*100:+.2f}" for sd, v in per_seed.items()) +
              f"  署名={'あり' if sig else 'なし'}  G3 rob/emp {g3_rob*100:+.2f}/{g3_emp*100:+.2f}pp"
              f"={'PASS' if g3 else 'FAIL'}  全年+={'+' if all_pos else 'x'}  "
              f"k(rob) {base['rob_k_mean']:.2f}->{r['rob_k_mean']:.2f}")

    # ---------------- 4. 判定 ----------------
    sec("4. 判定(事前登録基準: noise=|gain|<0.8pp / reject=署名or明確な劣化 / adopt=+0.8pp超+全ゲート)")
    verdicts = {}
    for cfg in JUDGED:
        g = results[cfg]["gates"]
        if g["signature"] or g["gain_pp"] < -0.8:
            v = "reject"
        elif g["gain_pp"] > 0.8 and g["all_seeds_pos"] and g["g3_pass"] and g["all_years_pos"]:
            v = "adopt-candidate"
        else:
            v = "noise"
        verdicts[cfg] = v
        print(f"  {cfg}: {v} (gain {g['gain_pp']:+.2f}pp, 署名={g['signature']}, "
              f"G3={g['g3_pass']}, 全年+={g['all_years_pos']})")
    order = {"adopt-candidate": 2, "noise": 1, "reject": 0}
    # 総合判定: 主構成Aを軸に、変種B/Cがadoptならそれを反映
    overall = verdicts["A_6x10"]
    best_variant = max(JUDGED, key=lambda c: order[verdicts[c]])
    if order[verdicts[best_variant]] > order[overall]:
        overall = verdicts[best_variant]
    print(f"  総合: {overall} (主構成A={verdicts['A_6x10']}, 最良変種={best_variant}="
          f"{verdicts[best_variant]})")

    payload = {
        "meta": {"date": "2026-06-13", "pool_n": n, "pool_sum": s_ret,
                 "max_pos": MAX_POS, "seeds": list(SEEDS),
                 "configs": {c: o for c, o in CONFIGS.items()},
                 "judged": list(JUDGED)},
        "verification": {"sim_identical_max_diff": max_diff,
                         "sim_identical": bool(sim_identical),
                         "baseline_reproduced": bool(base_repro),
                         "base_rob_s0_k": base["rob"][0]["k"],
                         "base_rob_s0_cagr": base["rob"][0]["cagr"],
                         "base_emp_k": base["emp_k"], "base_emp_cagr": base["emp_cagr"]},
        "pool_stage": pool_rows,
        "account_stage": {c: {k2: ({str(s): vv for s, vv in v2.items()} if k2 == "rob" else v2)
                              for k2, v2 in results[c].items()
                              if k2 not in ("yr_emp", "yr_rob0")}
                          for c in results},
        "yearly": {c: {"emp": results[c]["yr_emp"], "rob0": results[c]["yr_rob0"]}
                   for c in results},
        "verdicts": verdicts,
        "overall_verdict": overall,
    }
    out_path = OUT_DIR / "exp73_result.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nsaved -> {out_path}\n総経過 {time.time()-T0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
