"""exp68: 待機バー(d0→d1)の値動きをサイジングに使う — 鮮度サイジング家族の最後のセル。

家族の検証状況:
  - 上げる方向(z_exec/z_max/z_mean で厚張り) = exp54 で -4.4pp の大惨事
    → reports/19 §5「サイジング入力の鮮度は上げない」(逆選択の増幅)
  - 二値ゲート(τ=戻り見送り / Δz=深化見送り) = exp55/56 で全滅(署名・1点スパイク)
  - **下げる方向(深化したトレードの初期サイズを連続的に絞る)= 未検証**(本実験)。
    機構仮説: 深化トレードはストレスクラスタに居る(exp54 の死因)。配分とテールの相関を
    切れば MtM 谷が浅くなり較正 k が上がる(d1 自身と同じ利得経路)。プール段では
    Δz バケット平均はフラット(exp55 診断)なので下げても収益をあまり失わない。

事前登録5変種(各 m は f(z_sig) に乗算、mean正規化で k 線形性維持。チューニング禁止):
  dzdown_c05 : m = clip(1 - 0.5×max(Δz,0), 0.5, 1)   Δz=|z_exec|-|z_sig|(深化が正)
  dzdown_c10 : m = clip(1 - 1.0×max(Δz,0), 0.25, 1)  用量曲線用
  rwdown     : m = clip(1 - 0.5×max(-r_wait/vol,0)/2, 0.5, 1)
               r_wait=dir×(執行close/シグナルclose-1)、vol=シグナル時20本ボラ(逆行σ数/2で線形減)
  tent       : m = clip(1 - 0.5×|Δz|, 0.5, 1)         深化も大戻りも絞る(対称)
  rangedown  : m = clip(1 - 0.5×max(range_σ-2,0)/2, 0.5, 1)
               range_σ=執行バー(high-low)/close ÷ vol。荒れた待機バー(>2σ)で絞る
判定: 口座 seed0 スカウト → base 超えのみ seeds 0-4 + ゲート(署名/G3生CAGR+分割年感度/G5/全年)。
reports/22 プロトコル「検証強度は効果量に比例」に従い、+0.3pp 級なら記録のみで不採用。

実行: PYTHONPATH=. uv run python research/experiments/exp68_waitbar_sizing.py
出力: research/outputs/exp68_result.csv / exp68_result.json
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
from mm_production import build_pool_d1, _fz  # noqa: E402
from tail_protocol import (  # noqa: E402
    boot_dd, cagr_of, calibrate_empirical, calibrate_robust_seeded, max_dd,
    protocol_eval, yearly_returns,
)
from exp47_entry_delay import year_diff_audit  # noqa: E402
from exp55_d1_refinements import zsigned_at  # noqa: E402
from fxlab import universe as uni  # noqa: E402

OUT_DIR = ROOT / "research" / "outputs"
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
MAX_POS = 8
SEEDS = (0, 1, 2, 3, 4)


def account_eval(tag, pool, closes, mvec, seeds):
    """f(z_sig)×m を z 列埋め込み(exp61 で検証済みの恒等手法)で評価。"""
    fz = np.array([_fz(z) for z in pool["z_entry"].to_numpy()])
    w = fz * mvec
    wn = w / (w.mean() or 1.0)
    pl = pool.copy()
    pl["z_entry"] = wn

    def make(k):
        base = k / MAX_POS
        return lambda ctx: ctx["equity_real"] * base * ctx["z"]
    cache = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            cache[kk] = mm.simulate(pl, closes, make(kk), max_pos=MAX_POS)[0]
        return cache[kk]
    r = protocol_eval(eq_of_k, label=tag, seeds=seeds)
    yr_e = yearly_returns(eq_of_k(r["emp_k"]))
    r["yr_emp"] = {int(y): float(v) for y, v in yr_e.items()}
    r["neg_years_emp"] = int((yr_e < 0).sum())
    r["worst_year"] = float(yr_e.min())
    yr0 = yearly_returns(eq_of_k(r["rob"][seeds[0]]["k"]))
    r["neg_years_rob0"] = int((yr0 < 0).sum())
    r["yr_rob0"] = {int(y): float(v) for y, v in yr0.items()}
    # IS較正→OOS(分割年感度つき: 2021/2022/2023)
    r["splits"] = {}
    for ystart in (2021, 2022, 2023):
        SP = pd.Timestamp(f"{ystart}-01-01", tz="UTC")
        is_pool = pl[pl["entry"] < SP].reset_index(drop=True)
        oos_pool = pl[pl["entry"] >= SP].reset_index(drop=True)
        is_cl, oos_cl = closes[closes.index < SP], closes[closes.index >= SP]

        def eq_fn(p2, c2):
            c = {}

            def f(k):
                kk = round(float(k), 10)
                if kk not in c:
                    c[kk] = mm.simulate(p2, c2, make(kk), max_pos=MAX_POS)[0]
                return c[kk]
            return f
        fi, fo = eq_fn(is_pool, is_cl), eq_fn(oos_pool, oos_cl)
        k_ir = calibrate_robust_seeded(fi, 0.20, seed=0)
        k_ie = calibrate_empirical(fi, 0.20)
        r["splits"][ystart] = {"oos_rob": cagr_of(fo(k_ir)), "oos_emp": cagr_of(fo(k_ie))}
    r["eq_of_k"] = eq_of_k
    return r


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1().copy().reset_index(drop=True)
    closes = mm.load_closes()
    n = len(pool)
    print(f"=== exp68: 待機バー値動きサイジング(下げ方向) n={n} ===")

    # 状態量(全て執行バー close 時点で既知=因果)
    z_exec_signed = zsigned_at(pool["entry"], pool["instr"].to_numpy())
    dirs = pool["dir"].to_numpy().astype(float)
    z_sig = pool["z_entry"].to_numpy()
    dz = np.abs(z_exec_signed) - z_sig                     # 深化が正
    dz = np.where(np.isfinite(dz), dz, 0.0)
    # r_wait と執行バーレンジ(シグナルclose=執行バーの1本前close)
    rw_sig = np.full(n, np.nan)
    rng_sig = np.full(n, np.nan)
    for instr, g in pool.groupby("instr"):
        d = uni.instrument_data(instr, "H4")
        close, high, low = d["close"], d["high"], d["low"]
        prev = close.shift(1)
        ts = pd.DatetimeIndex(g["entry"])
        rows = g.index.to_numpy()
        rw_sig[rows] = (close.reindex(ts).to_numpy() / prev.reindex(ts).to_numpy() - 1.0)
        rng_sig[rows] = ((high - low) / close).reindex(ts).to_numpy()
    vol = pool["vol_entry"].to_numpy()
    r_wait = dirs * rw_sig                                  # 負=逆行が続いた
    adv_sig = np.where(vol > 0, np.maximum(-r_wait, 0.0) / vol, 0.0)   # 逆行のσ数
    range_sig = np.where(vol > 0, rng_sig / vol, 0.0)

    print(f"Δz: median {np.median(dz):+.2f} / 深化(>0) {np.mean(dz>0):.0%} / "
          f"逆行σ中央値 {np.median(adv_sig):.2f} / レンジσ中央値 {np.median(range_sig):.2f}")

    variants = {
        "dzdown_c05": np.clip(1 - 0.5 * np.maximum(dz, 0), 0.5, 1.0),
        "dzdown_c10": np.clip(1 - 1.0 * np.maximum(dz, 0), 0.25, 1.0),
        "rwdown": np.clip(1 - 0.5 * adv_sig / 2.0, 0.5, 1.0),
        "tent": np.clip(1 - 0.5 * np.abs(dz), 0.5, 1.0),
        "rangedown": np.clip(1 - 0.5 * np.maximum(range_sig - 2.0, 0) / 2.0, 0.5, 1.0),
    }

    print("\n--- 口座 seed0 スカウト ---")
    results = {"base": account_eval("base", pool, closes, np.ones(n), seeds=(0,))}
    print(f"    [{time.time()-t0:.0f}s]")
    for tag, m in variants.items():
        share_down = float(np.mean(m < 0.999))
        results[tag] = account_eval(tag, pool, closes, m, seeds=(0,))
        print(f"      [{tag}] 絞り対象 {share_down:.0%} / m平均 {m.mean():.3f}  [{time.time()-t0:.0f}s]")
    base_s0 = results["base"]["rob"][0]["cagr"]
    finalists = [t for t in variants if results[t]["rob"][0]["cagr"] > base_s0]
    print(f"\nseed0 で base({base_s0:+.2%}) 超え: {finalists or 'なし'}")

    for tag in (["base"] + finalists if finalists else []):
        m = np.ones(n) if tag == "base" else variants[tag]
        results[tag] = account_eval(tag, pool, closes, m, seeds=SEEDS)
        print(f"    [{time.time()-t0:.0f}s]")

    base = results["base"]
    rows = []
    for tag, r in results.items():
        full = len(r["rob"]) == len(SEEDS)
        row = {"cfg": tag, "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"],
               "emp_p95": r["emp_p95"], "rob_s0": r["rob"][0]["cagr"],
               "rob_mean": r["rob_cagr_mean"] if full else np.nan,
               "worst_year": r["worst_year"], "neg_emp": r["neg_years_emp"]}
        if tag != "base" and full:
            per = {sd: r["rob"][sd]["cagr"] - base["rob"][sd]["cagr"] for sd in SEEDS}
            sig = (r["emp_cagr"] > base["emp_cagr"]) and \
                  (abs(r["emp_p95"]) > abs(base["emp_p95"]) + 0.005)
            row["gain_pp"] = (r["rob_cagr_mean"] - base["rob_cagr_mean"]) * 100
            row["all_seeds_pos"] = all(v > 0 for v in per.values())
            row["signature"] = bool(sig)
            # G3 生CAGR両側 + 分割年感度(2021/2022/2023 全てで両側改善か)
            g3 = {y: (r["splits"][y]["oos_rob"] > base["splits"][y]["oos_rob"]) and
                     (r["splits"][y]["oos_emp"] > base["splits"][y]["oos_emp"])
                  for y in (2021, 2022, 2023)}
            row["g3_2021"], row["g3_2022"], row["g3_2023"] = g3[2021], g3[2022], g3[2023]
            a_emp = year_diff_audit("emp", r["yr_emp"], base["yr_emp"])
            row["g5_keep_emp"] = a_emp["keep_share_excl_best"]
            print(f"  [{tag}] gain {row['gain_pp']:+.2f}pp seeds " +
                  " ".join(f"{v*100:+.2f}" for v in per.values()) +
                  f" 署名={'あり' if sig else 'なし'} G3(21/22/23)=" +
                  "".join("+" if g3[y] else "x" for y in (2021, 2022, 2023)))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "exp68_result.csv", index=False)
    payload = {"results": {t: {k: ({str(s): vv for s, vv in v.items()} if k == "rob"
                                   else ({str(y): vv for y, vv in v.items()} if k == "splits" else v))
                               for k, v in r.items()
                               if k not in ("eq_of_k", "yr_emp", "yr_rob0")}
                           for t, r in results.items()}}
    (OUT_DIR / "exp68_result.json").write_text(json.dumps(payload, indent=2, default=float))
    print("\n=== 最終表 ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(df.to_string(index=False))
    print(f"\nsaved -> {OUT_DIR / 'exp68_result.csv'}\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
