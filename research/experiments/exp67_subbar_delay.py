"""exp67: サブバー遅延用量曲線 — d1 を時間単位で細かく刻む(ユーザー案)。

これまでの遅延用量は H4 バー単位(d0/d1/d2/d3, exp47)でのみ測定され、d1(=シグナル
close から 4h 待ち)が採用された。本実験はシグナル(0h位相グリッドのまま=exp46で死んだ
位相オフセットとは別物)を固定し、**執行時刻だけ**を X∈{0,1,2,3,4,5,6,8}h で刻む。
約定は自銘柄 M1 close(クロスは脚M1 inner-join合成=exp52方式)。

規約(事前登録):
  - 約定時刻 = シグナルバー label + 3:59 + X。X=0 は d0 価格、X=4 は d1 価格を再現
    すること(検算アンカー)。M1 が休場(直近tickが30分超前)なら次の M1 で約定(週明け等)。
  - 因果ゲート: X∈[4,8) は「t+1 close 時点で exit 未発火」(=idx_x≥idx_e+2、d1と同一・因果)。
    X<4 は将来の z を知り得ないため全件保有(exit が t+1 close なら 1-4h の短期保有になる)。
    X=8 は d2 アンカー(参照のみ。d≥2 は禁止採用済み)。
  - 約定が UTC 20:00-22:59 の M1 になるトレードの share を監査し、除染変種
    (その場合 23:00 以降の最初の M1 へ送る)も測る(ロールオーバーBIDアーティファクト対策)。
  - エントリースリッページは元トレードと同じ片道比率、コストは元の往復を維持(exp47 方式)。
  - 判定: プール段断面(年次/IS-OOS/単年/時刻分布)→ 口座 seed0 全X → d1 超えのみ
    seeds 0-4+ゲート。採用には d1 比で意味のある優位+全ゲート+曲線の内部構造を要求。

実行: PYTHONPATH=. uv run python research/experiments/exp67_subbar_delay.py
出力: research/outputs/exp67_pool.csv / exp67_account.csv / exp67_result.json
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
from tail_protocol import protocol_eval, yearly_returns  # noqa: E402
from exp47_entry_delay import reconstruct  # noqa: E402
from exp52_d1_m1audit import m1_close_naive  # noqa: E402
from fxlab import universe as uni  # noqa: E402

OUT_DIR = ROOT / "research" / "outputs"
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
MAX_POS = 8
XS = (0, 1, 2, 3, 4, 5, 6, 8)  # 時間
STALE_MIN = 30                  # これより古いtickしか無ければ「休場」→次のM1で約定


def fill_lookup(m1_idx_ns, m1_px, t_ns):
    """t_ns 以前の直近 M1。30分超古ければ次の M1(週明け等)。(price, fill_ts_ns)"""
    i = np.searchsorted(m1_idx_ns, t_ns, side="right") - 1
    if i < 0:
        i = 0
    if t_ns - m1_idx_ns[i] > STALE_MIN * 60 * 1_000_000_000:
        j = np.searchsorted(m1_idx_ns, t_ns, side="right")
        i = min(j, len(m1_idx_ns) - 1)
    return m1_px[i], m1_idx_ns[i]


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool0 = mm.build_pool()
    closes = mm.load_closes()
    rc = reconstruct(pool0)
    d1ref = build_pool_d1()
    n = len(pool0)
    print(f"=== exp67: サブバー遅延 X={XS}h (d0 pool n={n}) ===")

    # M1 系列(tz-naive int64)
    m1 = {}
    for instr in sorted(pool0["instr"].unique()):
        s = m1_close_naive(instr)
        m1[instr] = (s.index.values.astype("datetime64[ns]").astype(np.int64),
                     s.to_numpy())
        print(f"  M1 {instr}: {len(s):,}本  [{time.time()-t0:.0f}s]", end="\r")
    print()

    dirs = pool0["dir"].to_numpy().astype(float)
    ret0 = pool0["ret"].to_numpy()
    label_ns = pool0["entry"].dt.tz_localize(None).to_numpy().astype("datetime64[ns]").astype(np.int64)
    H = 3_600 * 1_000_000_000
    base_close_ns = label_ns + 4 * H - 60 * 1_000_000_000  # label+3:59

    pools, prows = {}, []
    for X in XS:
        # ゲート(因果)
        if X < 4:
            kept = rc["idx_x"] >= rc["idx_e"] + 1
        elif X < 8:
            kept = rc["idx_x"] >= rc["idx_e"] + 2
        else:
            kept = rc["idx_x"] >= rc["idx_e"] + 3
        fillp = np.full(n, np.nan)
        fill_ts = np.full(n, 0, dtype=np.int64)
        for instr, g in pool0.groupby("instr"):
            idx_ns, px = m1[instr]
            rows = g.index.to_numpy()
            for ti in rows:
                p, ts = fill_lookup(idx_ns, px, base_close_ns[ti] + X * H)
                fillp[ti] = p
                fill_ts[ti] = ts
        # 約定が出口バーclose以降になったら消滅(保守)
        exit_close_ns = (pool0["exit"].dt.tz_localize(None).to_numpy()
                         .astype("datetime64[ns]").astype(np.int64) + 4 * H - 60 * 1_000_000_000)
        kept = kept & (fill_ts < exit_close_ns)
        ret_new = dirs * (rc["exit_close"] / fillp - 1.0) - rc["cost"]
        diff_tr = np.where(kept, ret_new - ret0, -ret0)
        yr = pd.Series(diff_tr).groupby(pool0["exit"].dt.year).sum()
        by = int(yr.idxmax())
        total = float(diff_tr.sum())
        is_m = (pool0["entry"] < OOS_START).to_numpy()
        hrs = pd.DatetimeIndex(fill_ts.view("datetime64[ns]")).hour
        roll_share = float(np.mean(((hrs >= 20) & (hrs < 23))[kept]))
        prows.append({"X_h": X, "n": int(kept.sum()), "dropped": int((~kept).sum()),
                      "sum_ret": float(ret_new[kept].sum()), "diff_vs_d0": total,
                      "diff_is": float(diff_tr[is_m].sum()),
                      "diff_oos": float(diff_tr[~is_m].sum()),
                      "best_year": by, "excl_best": float(yr.drop(by).sum()),
                      "excl_2022": float(yr.drop(2022).sum()) if 2022 in yr.index else total,
                      "rollover_fill_share": roll_share})
        print(f"  X={X}h: n={int(kept.sum())} diff_vs_d0={total:+.4f} "
              f"(IS {diff_tr[is_m].sum():+.3f}/OOS {diff_tr[~is_m].sum():+.3f}) "
              f"最良年{by}除外後 {yr.drop(by).sum():+.4f} ロールオーバー約定 {roll_share:.1%}"
              f"  [{time.time()-t0:.0f}s]")
        mod = pool0.copy()
        mod["entry"] = pd.DatetimeIndex(fill_ts.view("datetime64[ns]")).tz_localize("UTC")
        mod["entry_price"] = fillp * rc["slip"]
        mod["ret"] = ret_new
        mod = mod[kept].sort_values("entry").reset_index(drop=True)
        pools[X] = mod

    # 検算アンカー
    p0 = pools[0]
    d0_match = abs(p0["ret"].sum() - ret0.sum())
    p4 = pools[4]
    d1_match = (len(p4) == len(d1ref), abs(p4["ret"].sum() - d1ref["ret"].sum()))
    print(f"\n検算: X=0 sum差 vs d0 = {d0_match:+.4f}(M1終値≒H4終値の丸め差) / "
          f"X=4 n={len(p4)} vs d1 {len(d1ref)}, sum差 {d1_match[1]:+.4f}")

    pdf = pd.DataFrame(prows)
    pdf.to_csv(OUT_DIR / "exp67_pool.csv", index=False)

    print("\n--- 口座 seed0(全X) ---")
    arows, results = [], {}
    for X in XS:
        mk = champion_sizing(pools[X], max_pos=MAX_POS)
        cache = {}

        def eq_of_k(k, pl=pools[X], mk=mk, cache=cache):
            kk = round(float(k), 10)
            if kk not in cache:
                cache[kk] = mm.simulate(pl, closes, mk(kk), max_pos=MAX_POS)[0]
            return cache[kk]
        r = protocol_eval(eq_of_k, label=f"X={X}h", seeds=(0,))
        results[X] = r
        arows.append({"X_h": X, "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"],
                      "emp_p95": r["emp_p95"], "rob_s0": r["rob"][0]["cagr"]})
        print(f"    [{time.time()-t0:.0f}s]")

    adf = pd.DataFrame(arows)
    ref = adf.loc[adf["X_h"] == 4, "rob_s0"].iloc[0]
    adf["d_vs_d1_pp"] = (adf["rob_s0"] - ref) * 100
    adf.to_csv(OUT_DIR / "exp67_account.csv", index=False)
    payload = {"pool": prows, "account": adf.to_dict("records"),
               "anchors": {"d0_sum_gap": float(d0_match),
                           "d1_n_match": bool(d1_match[0]), "d1_sum_gap": float(d1_match[1])}}
    (OUT_DIR / "exp67_result.json").write_text(json.dumps(payload, indent=2, default=float))
    print("\n=== 用量曲線(口座 seed0, d1=X4h 比) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(adf.to_string(index=False))
    print(f"\nsaved -> {OUT_DIR / 'exp67_pool.csv'} / exp67_account.csv / exp67_result.json")
    print(f"総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
