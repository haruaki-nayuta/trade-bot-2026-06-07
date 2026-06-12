"""exp64: 経路条件付き積み増し(scale-in)の検証 — 「入口で判定できないなら途中で積む」案。

ユーザーの問い: 入口でワーストを判定できないなら、一定の条件でポジションを積み増すのはどうか。

既知の近傍(reports/11 exp28): 深閾値(z=2.5)の並行ストリーム追加は empirical +26.5% に見えるが
robust 同一テールで完全同着・p95 -29.4→-32% に膨張=レバ偽装。ただし「保有中ポジの経路情報を
条件にした積み増し」は未測定。本実験は積み増しトランシェを「追加プールトレード」として
正確に価格付けする(トリガーバー close で建玉→元トレードと同時に決済、同一コストモデル)。

変種(事前登録・各トレード最大1回の積み増し=カスケード禁止):
  ナンピン系(注: ユーザー受入基準はナンピン禁止。記録のため測定):
    dz+0.5 : |z| がシグナル時より 0.5 深まったら積む
    dz+1.0 : 同 1.0
  ピラミッド系(未踏):
    prof+0.5% : 含み益 +0.5% に到達したら積む
    zrev1.25  : |z| が 1.25 まで戻ったら(収束開始確認)積む
  時間系(悪魔の代弁者):
    uw20 : 20本経過時点で水没していたら積む

判定:
  1) プール段 = トランシェの前向き純リターン(トリガー→出口、往復コスト控除)。
     E[forward|トリガー] ≤ コスト なら幾何学に関係なく死亡。
  2) 口座段 = 拡張プール(元1207+トランシェ)を champion_sizing(P4.0)×mp8 で
     同一テール較正(empirical + robust seeds 0-2)。レバ偽装署名・同時建玉数を監査。

実行: PYTHONPATH=. uv run python research/experiments/exp64_scale_in.py
出力: research/outputs/exp64_result.csv / exp64_result.json
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
from tail_protocol import boot_dd, protocol_eval, yearly_returns  # noqa: E402
from exp47_entry_delay import reconstruct  # noqa: E402
from fxlab import universe as uni  # noqa: E402

OUT_DIR = ROOT / "research" / "outputs"
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
MAX_POS = 8
SEEDS = (0, 1, 2)


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1().copy().reset_index(drop=True)
    closes = mm.load_closes()
    rc = reconstruct(pool)  # idx_e/idx_x/exit_close/cost/slip(タイムスタンプ→自銘柄グリッド)
    n = len(pool)
    print(f"=== exp64: 経路条件付き積み増し (d1 pool n={n}) ===")

    # 銘柄ごとの z(50)・close を前計算し、各トレードの経路を切り出す
    zseg_all, cseg_all = {}, {}
    for instr, g in pool.groupby("instr"):
        s = rc["closes_by"][instr]
        z = (s - s.rolling(50).mean()) / s.rolling(50).std()
        zarr, carr = z.to_numpy(), s.to_numpy()
        tarr = s.index.values
        for ti in g.index.to_numpy():
            e, x = int(rc["idx_e"][ti]), int(rc["idx_x"][ti])
            zseg_all[ti] = zarr[e:x + 1]
            cseg_all[ti] = (carr[e:x + 1], tarr[e:x + 1])

    dirs = pool["dir"].to_numpy().astype(float)
    z_sig = pool["z_entry"].to_numpy()           # シグナル時 |z|
    cost = rc["cost"]                             # 単位ノーションあたり往復コスト
    slip = rc["slip"]
    exit_close = rc["exit_close"]

    def find_trigger(ti, kind):
        """トリガーバーのパス内 index(1..len-2)。None=不発。"""
        zs = zseg_all[ti]
        cs, _ = cseg_all[ti]
        if len(zs) < 3:
            return None
        d = dirs[ti]
        path = d * (cs / cs[0] - 1.0)
        rng = range(1, len(zs) - 1)              # 出口バーでの積み増しは無意味
        if kind.startswith("dz"):
            dz = float(kind[2:])
            for j in rng:
                if -d * zs[j] >= z_sig[ti] + dz:   # ロング: z がさらに深い負
                    return j
        elif kind == "prof":
            for j in rng:
                if path[j] >= 0.005:
                    return j
        elif kind == "zrev":
            for j in rng:
                if -d * zs[j] <= 1.25:             # |z| が 1.25 まで収束
                    return j
        elif kind == "uw20":
            j = 20
            if j < len(zs) - 1 and path[j] < 0:
                return j
        return None

    variants = {"dz+0.5": "dz0.5", "dz+1.0": "dz1.0", "prof+0.5%": "prof",
                "zrev1.25": "zrev", "uw20": "uw20"}
    rows, addons = [], {}
    sec = lambda t: print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

    sec("1. プール段: トランシェの前向き純リターン(トリガー→出口、往復コスト控除)")
    for label, kind in variants.items():
        recs = []
        for ti in range(n):
            j = find_trigger(ti, kind)
            if j is None:
                continue
            cs, ts = cseg_all[ti]
            zs = zseg_all[ti]
            fwd = dirs[ti] * (exit_close[ti] / cs[j] - 1.0) - cost[ti]
            recs.append({"orig": ti, "instr": pool.at[ti, "instr"],
                         "entry": pd.Timestamp(ts[j]).tz_localize("UTC"),
                         "exit": pool.at[ti, "exit"], "dir": int(dirs[ti]),
                         "entry_price": cs[j] * slip[ti], "ret": fwd,
                         "bars_held": len(zs) - 1 - j,
                         "z_entry": abs(zs[j]), "vol_entry": np.nan,
                         "orig_ret": pool.at[ti, "ret"]})
        ad = pd.DataFrame(recs)
        addons[label] = ad
        if ad.empty:
            print(f"  [{label}] トリガー 0 件")
            continue
        r = ad["ret"]
        is_m = ad["entry"] < OOS_START
        yr = ad.groupby(ad["exit"].dt.year)["ret"].sum()
        by = yr.idxmax()
        worst_orig = ad["orig_ret"] < pool["ret"].quantile(0.10)
        rows.append({
            "cfg": label, "n_addon": len(ad), "fire_rate": len(ad) / n,
            "fwd_mean_bps": r.mean() * 1e4, "fwd_sum": r.sum(), "fwd_win": (r > 0).mean(),
            "fwd_is": r[is_m].sum(), "fwd_oos": r[~is_m].sum(),
            "best_year_share": float(yr[by] / r.sum()) if r.sum() != 0 else np.nan,
            "addon_on_worst10_share": float(worst_orig.mean()),
            "fwd_on_worst10": float(r[worst_orig.to_numpy()].sum()),
        })
        print(f"  [{label}] 発火 {len(ad)}件({len(ad)/n:.0%})  mean {r.mean()*1e4:+.1f}bps  "
              f"sum {r.sum():+.4f}  win {(r>0).mean():.1%}  IS {r[is_m].sum():+.3f}/OOS {r[~is_m].sum():+.3f}")
        print(f"      うち元トレードがワースト10%のもの: {worst_orig.sum()}件 "
              f"(そのトランシェ合計 {r[worst_orig.to_numpy()].sum():+.4f})")

    pdf = pd.DataFrame(rows)

    sec("2. 口座段: 拡張プールの同一テール較正(emp + robust seeds 0-2)")
    base_eval = None
    acc_rows = []

    def account(tag, pl):
        mk = champion_sizing(pl, max_pos=MAX_POS)
        cache = {}

        def eq_of_k(k):
            kk = round(float(k), 10)
            if kk not in cache:
                cache[kk] = mm.simulate(pl, closes, mk(kk), max_pos=MAX_POS)
            return cache[kk][0]
        r = protocol_eval(eq_of_k, label=tag, seeds=SEEDS)
        eqm, _, info = cache[round(float(r["emp_k"]), 10)]
        yr = yearly_returns(eq_of_k(r["emp_k"]))
        r["neg_years_emp"] = int((yr < 0).sum())
        r["worst_year"] = float(yr.min())
        r["avg_conc"] = info["avg_conc"]
        r["max_conc"] = info["max_conc"]
        r["skipped"] = info["skipped"]
        return r

    base_eval = account("base", pool)
    acc_rows.append({"cfg": "base", "emp_k": base_eval["emp_k"], "emp_cagr": base_eval["emp_cagr"],
                     "emp_p95": base_eval["emp_p95"], "rob_mean3": base_eval["rob_cagr_mean"],
                     "avg_conc": base_eval["avg_conc"], "skipped": base_eval["skipped"],
                     "neg_years": base_eval["neg_years_emp"]})
    for label, ad in addons.items():
        if ad.empty or ad["ret"].sum() <= 0:
            print(f"  [{label}] プール段で純損失 or 0件 → 口座段スキップ")
            continue
        aug = pd.concat([pool, ad.drop(columns=["orig", "orig_ret"])],
                        ignore_index=True).sort_values("entry").reset_index(drop=True)
        r = account(label, aug)
        gain = r["rob_cagr_mean"] - base_eval["rob_cagr_mean"]
        sig = (r["emp_cagr"] > base_eval["emp_cagr"]) and \
              (abs(r["emp_p95"]) > abs(base_eval["emp_p95"]) + 0.005)
        acc_rows.append({"cfg": label, "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"],
                         "emp_p95": r["emp_p95"], "rob_mean3": r["rob_cagr_mean"],
                         "gain_pp": gain * 100, "signature": sig,
                         "avg_conc": r["avg_conc"], "skipped": r["skipped"],
                         "neg_years": r["neg_years_emp"]})
        print(f"      rob3 {r['rob_cagr_mean']:+.2%} (base比 {gain*100:+.2f}pp)  "
              f"署名={'あり' if sig else 'なし'}  avg_conc {base_eval['avg_conc']:.2f}->{r['avg_conc']:.2f}  "
              f"skip {base_eval['skipped']}->{r['skipped']}")

    adf = pd.DataFrame(acc_rows)
    pdf.to_csv(OUT_DIR / "exp64_result.csv", index=False)
    payload = {"pool_stage": rows, "account_stage": acc_rows}
    (OUT_DIR / "exp64_result.json").write_text(json.dumps(payload, indent=2, default=float))
    print("\n=== プール段 ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(pdf.to_string(index=False))
        print("\n=== 口座段 ===")
        print(adf.to_string(index=False))
    print(f"\nsaved -> {OUT_DIR / 'exp64_result.csv'} / exp64_result.json\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
