"""exp73b: 時間トレール非線形変種(E凸後ろ加重 / F凹前加重 / Hハードカット軟化)— exp73 の続編。

exp73(線形トレール)の確定結果: A(6x10) -3.98pp / B(bar30開始) -1.57pp / C(3x20) -3.79pp /
D(ts60参照) -0.34pp。全構成が署名なしの正直な純劣化。死因は2系統:
  (i) 支払いの9割が勝ちトレードの bar10 発火 = オーバーシュート切り(exp71 の機械的前倒し)
  (ii) 支払いを絞っても k が買えない(DD谷はトランシェ発火前のブリードが作る — B の教訓)

事前登録する3変種(ユーザー指定。これ以外を追加しない):
  E) 凸・後ろ加重: bar10/20/30/40/50/60 で 5/10/15/20/25/25% ずつ決済
     (序盤を厚く持ち勝ちのオーバーシュートを守りつつ、塩漬けゾーンで加速)
  F) 凹・前加重: bar10/20/30/40/50/60 で 25/25/20/15/10/5% ずつ決済
     (対照極。A の死因(i)からは最悪が予測される = 用量曲線の向き確定用)
  H) ハードカット軟化版: bar60 で 50%、bar80 で残り全部
     (「60過ぎでざっくり」の段階版。D(全量60) と B(30開始) の中間点)
いずれも z出口が先に来たら残量全決済(exp73 と同一規約)。コスト按分(ノーション比例の
半スプレッド = 往復1スプレッドで分割してもコスト中立)・look-ahead 規約も exp73 と同一。

基盤は exp73 から import して再利用:
  - simulate_weighted / full_eval / pool_stage_report はそのまま import。
  - build_rows のみ等重み(w=1/m)前提のため、重み付き一般化 build_rows_w を本ファイルに実装。
    検証: 等重みで exp73.build_rows と rows/Δ がビット一致 + exp73_result.json のプール段数値と
    一致(写経一致検証)。さらに w=1 ベースラインの mm_lab.simulate 一致と
    ベースライン較正値(rob_s0 k=6.084/+18.18%, emp k=8.895/+27.41%)の再現を必須ゲートとする。

測定(exp73 と同一フォーマット):
  - プール段: Δbps/件、付着先分解(勝ち/負け)、発火率、年別。
  - 口座段: tail_protocol ペアシード seeds0-4 robust ペア較正で gain_pp・k変化・レバ偽装署名・
    G3(IS<2022→OOS rob/emp両改善)・全年プラス。
  - 総括: A〜H 8点(A,B,C,D=exp73 + E,F,H=本実験 + base=原点)の用量曲線
    (横軸=発火加重平均バー / 勝ちトレードの早期決済重み、縦軸=gain_pp)。

判定基準(exp73 と同一・事前登録): gain_pp が較正ノイズ帯(±0.8pp)未満 → 'noise'。
署名あり or gain<-0.8pp → 'reject'。+0.8pp超+全ゲート通過 → 'adopt-candidate'。

実行: PYTHONPATH=. uv run python research/experiments/exp73b_time_trail_nonlinear.py
出力: research/outputs/exp73b_result.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "experiments"))

import exp73_time_trail as e73  # noqa: E402  (mm_lab 等の sys.path は e73 が解決)
import mm_lab as mm  # noqa: E402
from mm_production import build_pool_d1, champion_sizing  # noqa: E402
from exp47_entry_delay import reconstruct  # noqa: E402
from fxlab import universe as uni  # noqa: E402

OUT_DIR = ROOT / "research" / "outputs"
SEEDS = e73.SEEDS
MAX_POS = e73.MAX_POS

# 事前登録 3 構成: (offsets, weights)。重みは決済ノーション比(合計1)。
NEW_CONFIGS: dict[str, tuple[list[int], list[float]]] = {
    "E_convex_back":   ([10, 20, 30, 40, 50, 60], [0.05, 0.10, 0.15, 0.20, 0.25, 0.25]),
    "F_concave_front": ([10, 20, 30, 40, 50, 60], [0.25, 0.25, 0.20, 0.15, 0.10, 0.05]),
    "H_hardcut_soft":  ([60, 80], [0.50, 0.50]),
}
JUDGED = tuple(NEW_CONFIGS)
# exp73 の旧構成(等重み)— 用量曲線の座標として再計測する
OLD_CONFIGS: dict[str, tuple[list[int], list[float]]] = {
    c: (offs, [1.0 / len(offs)] * len(offs)) for c, offs in e73.CONFIGS.items()
}

T0 = time.time()


def el():
    return f"[{time.time()-T0:.0f}s]"


# --- プール段: 重み付き一般化(exp73.build_rows の w=1/m を任意重みに拡張) ---
def build_rows_w(pool, rc, offsets, weights):
    """exp73.build_rows の重み一般化。トランシェ j は重み weights[j] で bar offsets[j] 決済。
    e+off >= x なら z出口に合流(per-unit ret = 親の ret)。等重みなら exp73.build_rows と
    完全一致する(main で検証)。diag に fired_w(親ごとの発火重み)と
    avg_fired_bar_w(発火加重平均バー=用量曲線の横軸)を追加。"""
    assert len(offsets) == len(weights)
    assert abs(sum(weights) - 1.0) < 1e-9, "重み合計は1"
    m = len(offsets)
    n = len(pool)
    delta = np.zeros(n)
    fired_tr = np.zeros(n, dtype=int)
    fired_w = np.zeros(n)                     # 親ごとの発火重み(早期決済されたノーション比)
    abandoned_w = 0.0                         # Σ w·(放棄した前向きリターン)
    ab_fwd_list = []                          # 発火トランシェの前向き(放棄)リターン per-unit
    first_off_delta = np.zeros(n)             # 最初のオフセットのトランシェΔ(勝ち側コスト分解用)
    first_off_fired = np.zeros(n, dtype=bool)
    wbar_num = 0.0                            # Σ w·bar(発火加重平均バーの分子)
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
            for off, wj in zip(offsets, weights):
                p = e + off
                key = p if p < x else x
                exits[key] = exits.get(key, 0.0) + wj
            blended = 0.0
            for p in sorted(exits):
                wsum = exits[p]
                if p == x:
                    retj = ret0
                    ex_ts = pool.at[ti, "exit"]
                else:
                    retj = d * (carr[p] / carr[e] - 1.0) - cost
                    ex_ts = pd.Timestamp(tarr[p]).tz_localize("UTC")
                    fired_tr[ti] += 1            # p<x のキーはオフセット1本に対応(e+offは互いに異なる)
                    fired_w[ti] += wsum
                    fwd = d * (rc["exit_close"][ti] / carr[p] - 1.0)  # 放棄した残り(gross, コスト中立)
                    abandoned_w += wsum * fwd
                    ab_fwd_list.append(fwd)
                    wbar_num += wsum * (p - e)
                    if p == e + offsets[0]:
                        first_off_fired[ti] = True
                        first_off_delta[ti] = wsum * (retj - ret0)
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
    fw_sum = float(fired_w.sum())
    diag = {"delta": delta, "fired_tr": fired_tr, "m": m,
            "abandoned_w": abandoned_w, "ab_fwd": np.array(ab_fwd_list),
            "first_off_delta": first_off_delta, "first_off_fired": first_off_fired,
            "fired_w": fired_w,
            "avg_fired_bar_w": (wbar_num / fw_sum) if fw_sum > 0 else float("nan")}
    return rows, diag


def dose_metrics(pool, diag):
    """用量曲線の座標: 発火加重平均バー / 勝ちトレードの早期決済重み(勝ちゾーン食い込み度)。"""
    win = pool["ret"].to_numpy() > 0
    fw = diag["fired_w"]
    return {
        "avg_fired_bar_w": float(diag["avg_fired_bar_w"]),
        "win_fired_w_mean": float(fw[win].mean()),    # 勝ち1件あたり早期決済されたノーション比
        "loss_fired_w_mean": float(fw[~win].mean()),
        "fired_w_total_rate": float(fw.mean()),       # 全体の早期決済ノーション比
    }


def main() -> int:
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1().copy().reset_index(drop=True)
    closes = mm.load_closes()
    rc = reconstruct(pool)
    n = len(pool)
    s_ret = float(pool["ret"].sum())
    print(f"=== exp73b: 時間トレール非線形変種 E/F/H (d1 pool n={n}, sum={s_ret:+.4f}) {el()} ===")
    assert n == e73.POOL_N_EXPECT, f"pool n={n} != {e73.POOL_N_EXPECT}"
    assert abs(s_ret - e73.POOL_SUM_EXPECT) < 1e-6, f"pool sum {s_ret} != {e73.POOL_SUM_EXPECT}"
    print(f"  検算OK: n={e73.POOL_N_EXPECT} / sum(ret)={e73.POOL_SUM_EXPECT:+.4f} と一致")

    exp73_json = json.loads((OUT_DIR / "exp73_result.json").read_text())
    sec = lambda t: print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

    # ---------------- 0. build_rows_w の一致検証(等重みで exp73.build_rows と同一) ----------------
    sec("0. build_rows_w 検証: 等重みで exp73.build_rows とビット一致 + exp73 JSON プール段一致")
    pool73 = {r["config"]: r for r in exp73_json["pool_stage"]}
    max_dd_rows = 0.0
    for cfg, (offs, ws) in OLD_CONFIGS.items():
        rows_a, diag_a = e73.build_rows(pool, rc, offs)
        rows_b, diag_b = build_rows_w(pool, rc, offs, ws)
        assert len(rows_a) == len(rows_b)
        dd_ret = float(np.max(np.abs(rows_a["ret"].to_numpy() - rows_b["ret"].to_numpy())))
        dd_w = float(np.max(np.abs(rows_a["w"].to_numpy() - rows_b["w"].to_numpy())))
        dd_delta = float(np.max(np.abs(diag_a["delta"] - diag_b["delta"])))
        assert (rows_a["exit"].to_numpy() == rows_b["exit"].to_numpy()).all()
        assert (rows_a["bars_held"].to_numpy() == rows_b["bars_held"].to_numpy()).all()
        assert (diag_a["fired_tr"] == diag_b["fired_tr"]).all()
        max_dd_rows = max(max_dd_rows, dd_ret, dd_w, dd_delta)
        d_json = abs(float(diag_b["delta"].sum()) - pool73[cfg]["delta_sum"])
        assert d_json < 1e-9, f"{cfg}: delta_sum が exp73 JSON と不一致 ({d_json})"
        print(f"  {cfg:16s} |Δret|max={dd_ret:.1e} |Δw|max={dd_w:.1e} |ΔΔ|max={dd_delta:.1e} "
              f"JSON delta_sum 一致(±1e-9)")
    rows_identical = max_dd_rows < 1e-12
    print(f"  -> 写経一致検証 {'OK' if rows_identical else 'FAIL'} (max diff {max_dd_rows:.2e})")

    # ---------------- 1. プール段(E/F/H の価格付け + A-D の用量座標再計測) ----------------
    sec("1. プール段: E/F/H トランシェ分解のブレンド再価格(Δ・付着先・発火率・機構照合)")
    e73.CONFIGS.update({c: offs for c, (offs, _) in NEW_CONFIGS.items()})  # report 内の bar 表示用
    rows_by, pool_rows, dose_by = {}, [], {}
    for cfg, (offs, ws) in NEW_CONFIGS.items():
        print(f"  -- {cfg}: bars={offs} weights={ws}")
        rows, diag = build_rows_w(pool, rc, offs, ws)
        rows_by[cfg] = rows
        rep = e73.pool_stage_report(cfg, pool, diag)
        dm = dose_metrics(pool, diag)
        rep.update(offsets=offs, weights=ws, **dm)
        dose_by[cfg] = dm
        pool_rows.append(rep)
        print(f"      用量座標: 発火加重平均bar={dm['avg_fired_bar_w']:.1f} "
              f"勝ち早期決済重み={dm['win_fired_w_mean']:.3f} 負け={dm['loss_fired_w_mean']:.3f}")
    for cfg, (offs, ws) in OLD_CONFIGS.items():   # A-D は座標のみ(口座段は exp73 JSON を引用)
        _, diag = build_rows_w(pool, rc, offs, ws)
        dose_by[cfg] = dose_metrics(pool, diag)

    # ---------------- 2. simulate_weighted の検証(w=1 で mm_lab.simulate と一致) ----------------
    sec("2. simulate_weighted 検証: w=1・トランシェなしで mm_lab.simulate とバー単位一致")
    base_rows = pool[["instr", "entry", "exit", "dir", "entry_price", "ret",
                      "bars_held", "z_entry"]].copy()
    base_rows["w"] = 1.0
    base_rows["parent"] = np.arange(n)
    mk = champion_sizing(pool, max_pos=MAX_POS)
    max_diff = 0.0
    for ktest in (2.0, 6.084):
        eq_a, eqr_a, info_a = mm.simulate(pool, closes, mk(ktest), max_pos=MAX_POS)
        eq_b, eqr_b, info_b = e73.simulate_weighted(base_rows, closes, mk(ktest), max_pos=MAX_POS)
        md = float(np.max(np.abs(eq_a.to_numpy() - eq_b.to_numpy())))
        mdr = float(np.max(np.abs(eqr_a.to_numpy() - eqr_b.to_numpy())))
        max_diff = max(max_diff, md, mdr)
        print(f"  k={ktest}: |Δeq_mtm|max={md:.3e} |Δeq_real|max={mdr:.3e} "
              f"skipped {info_a['skipped']}=={info_b['skipped']} "
              f"n_taken {info_a['n_taken']}=={info_b['n_taken']}")
        assert info_a["skipped"] == info_b["skipped"] and info_a["n_taken"] == info_b["n_taken"]
    sim_identical = max_diff < 1e-6

    # ---------------- 3. 口座段(本判定) ----------------
    sec("3. 口座段: ペアシード seeds0-4 robust + empirical + G3(base vs E/F/H)")
    base = e73.full_eval("base", base_rows, closes, mk)
    base_repro = (sim_identical and rows_identical
                  and abs(base["rob"][0]["k"] - 6.084) < 0.05
                  and abs(base["rob"][0]["cagr"] - 0.1818) < 0.005
                  and abs(base["emp_k"] - 8.895) < 0.05
                  and abs(base["emp_cagr"] - 0.2741) < 0.005)
    print(f"  ベースライン再現: rob_s0 k={base['rob'][0]['k']:.3f}/{base['rob'][0]['cagr']:+.2%} "
          f"(目標 6.084/+18.18%) emp k={base['emp_k']:.3f}/{base['emp_cagr']:+.2%} "
          f"(目標 8.895/+27.41%) -> {'OK' if base_repro else 'FAIL'}")
    if not base_repro:
        print("  !! ベースライン再現に失敗。判定は無効。")

    results = {"base": base}
    for cfg in NEW_CONFIGS:
        r = e73.full_eval(cfg, rows_by[cfg], closes, mk)
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
        print(f"      gain {gain_pp:+.2f}pp  seeds " +
              " ".join(f"s{sd}:{v*100:+.2f}" for sd, v in per_seed.items()) +
              f"  署名={'あり' if sig else 'なし'}  G3 rob/emp {g3_rob*100:+.2f}/{g3_emp*100:+.2f}pp"
              f"={'PASS' if g3 else 'FAIL'}  全年+={'+' if all_pos else 'x'}  "
              f"k(rob) {base['rob_k_mean']:.2f}->{r['rob_k_mean']:.2f}")

    # ---------------- 4. 判定 + 用量曲線(A〜H 8点) ----------------
    sec("4. 判定 + 用量曲線(base + A,B,C,D[exp73] + E,F,H[本実験] = 8点)")
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
    overall = max((verdicts[c] for c in JUDGED), key=lambda v: order[v])
    print(f"  総合: {overall}")

    acct73 = exp73_json["account_stage"]
    dose_curve = [{"config": "base", "avg_fired_bar_w": float("nan"),
                   "win_fired_w_mean": 0.0, "loss_fired_w_mean": 0.0,
                   "fired_w_total_rate": 0.0, "gain_pp": 0.0, "source": "anchor"}]
    for cfg in list(OLD_CONFIGS) + list(NEW_CONFIGS):
        src = "exp73" if cfg in OLD_CONFIGS else "exp73b"
        gp = (acct73[cfg]["gates"]["gain_pp"] if cfg in OLD_CONFIGS
              else results[cfg]["gates"]["gain_pp"])
        dose_curve.append({"config": cfg, **dose_by[cfg], "gain_pp": float(gp), "source": src})
    print("\n  用量曲線(横軸=勝ちトレードの早期決済重み, 縦軸=gain_pp):")
    print(f"  {'config':18s} {'win早期決済w':>10s} {'負け早期w':>9s} {'発火加重bar':>10s} {'gain_pp':>8s}")
    for p in sorted(dose_curve, key=lambda x: x["win_fired_w_mean"]):
        bar = f"{p['avg_fired_bar_w']:10.1f}" if np.isfinite(p["avg_fired_bar_w"]) else "         -"
        print(f"  {p['config']:18s} {p['win_fired_w_mean']:10.3f} {p['loss_fired_w_mean']:9.3f} "
              f"{bar} {p['gain_pp']:+8.2f}")
    any_pos = any(p["gain_pp"] > 0 for p in dose_curve if p["config"] != "base")
    any_above_noise = any(p["gain_pp"] > 0.8 for p in dose_curve if p["config"] != "base")
    print(f"  -> 正の gain_pp を持つ構成: {'あり' if any_pos else 'なし'} / "
          f"ノイズ帯(+0.8pp)超: {'あり' if any_above_noise else 'なし'}")

    payload = {
        "meta": {"date": "2026-06-13", "pool_n": n, "pool_sum": s_ret,
                 "max_pos": MAX_POS, "seeds": list(SEEDS),
                 "configs": {c: {"offsets": o, "weights": w}
                             for c, (o, w) in NEW_CONFIGS.items()},
                 "judged": list(JUDGED),
                 "reused_from_exp73": list(OLD_CONFIGS)},
        "verification": {"build_rows_w_max_diff": max_dd_rows,
                         "build_rows_w_identical": bool(rows_identical),
                         "sim_identical_max_diff": max_diff,
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
        "dose_curve": dose_curve,
        "verdicts": verdicts,
        "overall_verdict": overall,
    }
    out_path = OUT_DIR / "exp73b_result.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nsaved -> {out_path}\n総経過 {time.time()-T0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
