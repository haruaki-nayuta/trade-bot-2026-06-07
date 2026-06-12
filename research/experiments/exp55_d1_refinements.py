"""exp55: d1 の精錬レバー(プール変更系)スカウト — CAGR20%越え第3ラウンド本命。

候補(全て exp47 の再構成方式: d0 プール + 自銘柄グリッドで entry/exit を変換):
  A) τ ゲート用量曲線: d1 の見送り規則は現行「z が exit 域(0.5)に戻ったら」。
     τ ∈ {0.75, 1.0, 1.25, 1.5} で「部分的に収束し始めた=残存プレミアムが薄い」エントリーを
     追加で刈る(τ=0.5 がベース。τ<0.5 は元トレードが既に消滅しているため定義不能)。
  B) Δz(深化)ゲート: exp54 の発見=待機中に |z| が深まったトレードへの厚張りは大幅悪化(-4.4pp)。
     診断(Δz バケット別 mean ret)で単調性を確認の上、|z_exec| ≥ |z_sig| + δ なら見送り
     (δ ∈ {0.3, 0.5, 0.75, 1.0})。落ちるナイフの加速を執行バーで観測する新しい因果フィルタ。
  C) 出口遅延 dx ∈ {1, 2}: 入口 d1 と対称の未検証執行レバー(z が exit 閾値へ戻った後の
     残存ドリフトを 1 本収穫できるか)。プール段で即死判定を先に行う。
  D) h20 除染 d1(exp47 §7 の任意装備, rob +18.86% 実測)の再現確認(スタック編入候補)。

検算: τ=0.5(無追加ゲート)の再構成 d1 がキャッシュ済み本番 d1 プール(1207件, sum +1.9622)と
一致することを比較の前提とする(exp51 で実証済みの同値性)。

判定(スカウト段): プール段断面(年次/IS-OOS/単年依存/h20) + 口座 empirical+robust seed0。
生き残りは exp56 でフルプロトコル(seeds 0-4 + 6ゲート)。

実行: PYTHONPATH=. uv run python research/experiments/exp55_d1_refinements.py
出力: research/outputs/exp55_pool.csv / exp55_account.csv
"""

from __future__ import annotations

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
from mm_production import champion_sizing, build_pool_d1  # noqa: E402
from tail_protocol import protocol_eval  # noqa: E402
from exp47_entry_delay import reconstruct, delayed_pool  # noqa: E402
from fxlab import universe as uni  # noqa: E402

MAX_POS = 8
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
OUT_DIR = ROOT / "research" / "outputs"
BASE_NET = 1.9086


def zsigned_at(pool_ts: pd.Series, instr_arr: np.ndarray, win: int = 50) -> np.ndarray:
    """各トレードの指定タイムスタンプにおける符号付き z(自銘柄 H4 グリッド・確定 close)。"""
    out = np.full(len(pool_ts), np.nan)
    df = pd.DataFrame({"ts": pool_ts.to_numpy(), "instr": instr_arr})
    for instr, g in df.groupby("instr"):
        s = uni.instrument_close(instr, "H4")
        z = (s - s.rolling(win).mean()) / s.rolling(win).std()
        out[g.index.to_numpy()] = z.reindex(pd.DatetimeIndex(g["ts"])).to_numpy()
    return out


def pool_audit(tag, base_pool, kept, ret_new, ret0):
    """プール段断面: 差分合計 / IS-OOS / 最良年除外 / 2022除外。diff は対 d0 ベースプール。"""
    diff_tr = np.where(kept, ret_new - ret0, -ret0)
    yr = pd.Series(diff_tr).groupby(base_pool["exit"].dt.year).sum()
    is_mask = (base_pool["entry"] < OOS_START).to_numpy()
    best_y = int(yr.idxmax())
    total = float(diff_tr.sum())
    row = {"cfg": tag, "n": int(kept.sum()), "dropped": int((~kept).sum()),
           "sum_ret": float(np.where(kept, ret_new, 0.0).sum()),
           "diff_vs_d0": total, "diff_is": float(diff_tr[is_mask].sum()),
           "diff_oos": float(diff_tr[~is_mask].sum()),
           "best_year": best_y, "best_year_diff": float(yr[best_y]),
           "excl_best": float(yr.drop(best_y).sum()),
           "excl_2022": float(yr.drop(2022).sum()) if 2022 in yr.index else total}
    return row, yr


def to_pool(base_pool, rc, kept, ret_new, entry_ts, entry_close):
    mod = base_pool.copy()
    mod["entry"] = entry_ts
    mod["entry_price"] = entry_close * rc["slip"]
    mod["ret"] = ret_new
    mod["bars_held"] = np.maximum(base_pool["bars_held"].to_numpy() - 1, 1)
    return mod[kept].sort_values("entry").reset_index(drop=True)


def account_eval(tag, pool, closes, seeds=(0,)):
    mk = champion_sizing(pool, max_pos=MAX_POS)
    cache = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            cache[kk] = mm.simulate(pool, closes, mk(kk), max_pos=MAX_POS)[0]
        return cache[kk]
    r = protocol_eval(eq_of_k, label=tag, seeds=seeds)
    return {"cfg": tag, "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"],
            "emp_p95": r["emp_p95"], "rob_s0": r["rob"][seeds[0]]["cagr"]}


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool0 = mm.build_pool()           # d0 ベース(1214件)
    closes = mm.load_closes()
    rc = reconstruct(pool0)
    ret0 = pool0["ret"].to_numpy()
    dirs = pool0["dir"].to_numpy().astype(float)
    print(f"=== exp55: d1 精錬レバー (d0 pool n={len(pool0)} sum={ret0.sum():+.4f}) ===")

    # --- d1 再構成(共通土台) -------------------------------------------------
    mod1, kept1, ret_new1, ex1 = delayed_pool(pool0, rc, 1)
    z_exec = zsigned_at(pd.Series(ex1["dts"]), pool0["instr"].to_numpy())   # 符号付き
    z_sig_signed = zsigned_at(pool0["entry"], pool0["instr"].to_numpy())    # シグナルバー
    # d1 プール(τ=0.5 相当)の検算
    d1_cached = build_pool_d1()
    print(f"d1 再構成: n={kept1.sum()} sum={ret_new1[kept1].sum():+.4f}  "
          f"(キャッシュ d1: n={len(d1_cached)} sum={d1_cached['ret'].sum():+.4f})")

    rows_pool, rows_acc = [], []

    # ベースライン(d1)
    row, _ = pool_audit("d1_base", pool0, kept1, ret_new1, ret0)
    rows_pool.append(row)

    # --- B 診断: Δz バケット別 mean ret(kept トレードのみ) -------------------
    print("\n--- Δz = |z_exec| - |z_sig| 診断(d1 kept トレード) ---")
    dz = np.abs(z_exec) - np.abs(z_sig_signed)
    k_idx = np.where(kept1)[0]
    dzk = dz[k_idx]
    retk = ret_new1[k_idx]
    buckets = [(-np.inf, -0.5), (-0.5, -0.25), (-0.25, 0.0), (0.0, 0.25),
               (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, np.inf)]
    for lo, hi in buckets:
        m = (dzk >= lo) & (dzk < hi)
        if m.sum() > 0:
            print(f"  Δz∈[{lo:+.2f},{hi:+.2f}): n={m.sum():4d}  mean_ret={retk[m].mean()*1e4:+7.1f}bps"
                  f"  sum={retk[m].sum():+.4f}  win={np.mean(retk[m]>0):.1%}")

    # --- A) τ ゲート / B) Δz ゲート / 組合せの候補生成 -----------------------
    variants = {}
    for tau in (0.75, 1.0, 1.25, 1.5):
        keep = kept1 & (dirs * z_exec <= -tau)
        variants[f"tau{tau}"] = keep
    for dlt in (0.3, 0.5, 0.75, 1.0):
        keep = kept1 & ~(np.abs(z_exec) >= np.abs(z_sig_signed) + dlt)
        variants[f"dz{dlt}"] = keep

    for tag, keep in variants.items():
        row, _ = pool_audit(tag, pool0, keep, ret_new1, ret0)
        rows_pool.append(row)

    # --- C) 出口遅延 dx(入口 d1 固定) ---------------------------------------
    for dx in (1, 2):
        exit_close_dx = np.full(len(pool0), np.nan)
        exit_ts_dx = np.full(len(pool0), np.datetime64("NaT"), dtype="datetime64[ns]")
        ix_dx_full = np.full(len(pool0), -1)
        for instr, g in pool0.groupby("instr"):
            rows_i = g.index.to_numpy()
            s = rc["closes_by"][instr]
            ix_d = np.minimum(rc["idx_x"][rows_i] + dx, len(s) - 1)
            exit_close_dx[rows_i] = s.to_numpy()[ix_d]
            exit_ts_dx[rows_i] = s.index.values[ix_d]
            ix_dx_full[rows_i] = ix_d
        # 入口は d1(ex1)。kept は entry(d1) < exit(dx)
        ie_d1 = np.full(len(pool0), -1)
        for instr, g in pool0.groupby("instr"):
            rows_i = g.index.to_numpy()
            ie_d1[rows_i] = rc["idx_e"][rows_i] + 1
        kept_dx = ie_d1 < ix_dx_full
        ret_dx = dirs * (exit_close_dx / ex1["dclose"] - 1.0) - rc["cost"]
        row, _ = pool_audit(f"exitdx{dx}", pool0, kept_dx, ret_dx, ret0)
        rows_pool.append(row)
        variants[f"exitdx{dx}"] = None  # プール段のみ(口座は生き残ったときだけ)
        if row["diff_vs_d0"] > rows_pool[0]["diff_vs_d0"]:
            mod = pool0.copy()
            mod["entry"] = ex1["dts"]
            mod["entry_price"] = ex1["dclose"] * rc["slip"]
            mod["exit"] = pd.DatetimeIndex(exit_ts_dx).tz_localize("UTC")
            mod["ret"] = ret_dx
            mod["bars_held"] = np.maximum(pool0["bars_held"].to_numpy() - 1 + dx, 1)
            variants[f"exitdx{dx}"] = ("pool", mod[kept_dx].sort_values("entry").reset_index(drop=True))

    # --- D) h20 除染 d1(exp47 再現) ----------------------------------------
    mod_h, kept_h, ret_h, ex_h = delayed_pool(pool0, rc, 1, skip_h20=True)
    row, _ = pool_audit("d1_h20decon", pool0, kept_h, ret_h, ret0)
    rows_pool.append(row)

    pdf = pd.DataFrame(rows_pool)
    print("\n=== プール段断面(diff は対 d0) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(pdf.to_string(index=False))
    pdf.to_csv(OUT_DIR / "exp55_pool.csv", index=False)

    # --- 口座 seed0(プール段で diff_vs_d0 > d1_base のもの + ベース) ----------
    d1_diff = rows_pool[0]["diff_vs_d0"]
    print(f"\n=== 口座 seed0(プール段 diff > d1_base {d1_diff:+.4f} の候補 + 参考) ===")
    rows_acc.append(account_eval("d1_base", mod1, closes))
    print(f"    [{time.time()-t0:.0f}s]")
    for tag, keep in variants.items():
        if keep is None:
            continue
        if isinstance(keep, tuple):
            cand = keep[1]
        else:
            cand = to_pool(pool0, rc, keep, ret_new1, ex1["dts"], ex1["dclose"])
        prow = next(r for r in rows_pool if r["cfg"] == tag)
        # 口座段は全候補走らせる(プール段マイナスでも DD 形状で勝つ可能性があるのが教訓)
        rows_acc.append(account_eval(tag, cand, closes))
        print(f"    [{time.time()-t0:.0f}s]")
    rows_acc.append(account_eval("d1_h20decon", mod_h, closes))

    adf = pd.DataFrame(rows_acc)
    base_rob = adf.loc[adf["cfg"] == "d1_base", "rob_s0"].iloc[0]
    adf["d_rob_s0_pp"] = (adf["rob_s0"] - base_rob) * 100
    print("\n=== 口座段(seed0, base 比 pp) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(adf.to_string(index=False))
    adf.to_csv(OUT_DIR / "exp55_account.csv", index=False)
    print(f"\nsaved -> {OUT_DIR / 'exp55_pool.csv'} / {OUT_DIR / 'exp55_account.csv'}")
    print(f"総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
