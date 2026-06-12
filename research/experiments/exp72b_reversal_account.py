"""exp72b: 勝ち決済後・反対方向リエントリーの口座統合検証(同一テール較正)。

exp72(プール段)の結果: 事前登録 h=5 は net +2.27bps/件(t=+1.51)でプール段ゲートは
形式通過したが、どの h も週末跨ぎ除外に耐えず「採用推奨構成は無し」。本実験は最終段=
口座段: 反対方向トレードをプール行として d1 プールに追加した複合プールを、
mm_lab.simulate + champion_sizing(P=4.0, mp8)で同一テール較正し、純増を判定する。

構成(事前登録):
  ・主判定: h=5(勝ち決済バー close ± 半スプレッドで反対方向に建玉 → 5バー後 close 決済)
  ・参考:   h=15(数値上の最良だが週末跨ぎ53%・t=1.23)
  ・反転行: instr / entry(=親の決済時刻) / exit(=+hバー) / dir(=-親dir) /
    entry_price(コスト込み= close ± 半スプレッド) / ret(コスト後, exp72 と同一規約) /
    bars_held=h / z_entry=エントリーバーの |z|(window=50, 因果=決済バー close で判定・執行)/
    vol_entry=同バーの20本ボラ。
  ・champion_sizing は f(z) 配分のため、浅い z の反転トレードは自動的に最小クリップ
    (0.3)で小さく張られる=リスク契約は不変。f̄ は複合プール全体で再計算(規約)。

判定(exp64/65 の規約):
  ・ベースライン(d1単独)と複合プールを tail_protocol のペアシード seeds 0-4 で
    robust(p95 DD=20%)較正 → ΔCAGR(gain_pp)・全シード符号。
  ・レバ偽装署名: emp CAGR↑ かつ emp p95 が 0.5pp 超悪化(ブートシード0-2でも監査)。
  ・G3: IS(<2022)較正 → OOS 素検証で rob/emp 両方の生CAGRがベース超え。
  ・全年プラス維持(emp / rob_s0 の年次)。
  ・gain_pp が較正ノイズ帯(±0.4-0.8pp)未満なら noise、署名あり/劣化なら reject、
    ノイズ帯超+全ゲート通過なら adopt-candidate(reports/22: +0.3pp級は採用しない前例)。

検算(進行前提・不一致なら中止):
  ・プール n=1207 / sum(ret)=+1.9622
  ・反転 h5 コスト前平均 +4.4433bps ±0.1(exp70/72)・コスト後 +2.2701bps ±0.05(exp72)
  ・ベースライン rob_s0 k=6.084 / CAGR +18.24%(exp52)

実行: PYTHONPATH=. uv run python research/experiments/exp72b_reversal_account.py
出力: research/outputs/exp72b_result.json / exp72b_result.csv
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
from mm_production import build_pool_d1, champion_sizing  # noqa: E402
from tail_protocol import (  # noqa: E402
    boot_dd, cagr_of, calibrate_empirical, calibrate_robust_seeded, max_dd,
    protocol_eval, yearly_returns,
)
from fxlab import config  # noqa: E402
from fxlab import universe as uni  # noqa: E402

OUT_DIR = ROOT / "research" / "outputs"
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
MAX_POS = 8
SEEDS = (0, 1, 2, 3, 4)
HS = (5, 15)                      # 主判定 h=5(事前登録)、h=15 は参考
H_MAIN = 5
# 検算ターゲット
REF_POOL_N, REF_POOL_SUM = 1207, 1.9622
REF_H5_GROSS, REF_H5_NET = 4.4433, 2.2701
REF_ROB0_K, REF_ROB0_CAGR = 6.084, 0.1824
NOISE_BAND_PP = 0.8               # 較正ノイズ帯の上限(±0.4-0.8pp)


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def build_reversal_pools(pool: pd.DataFrame, hs=HS) -> dict:
    """勝ち決済(ret>0)後の反対方向トレードを mm_lab プール行形式で構築(h 別)。

    コスト規約は exp72 と同一: entry = 決済バー close ± 半スプレッド、
    exit = +h バー close ∓ 半スプレッド(ret = gross - hs/c[x] - hs/c[x+h])。
    z_entry / vol_entry はエントリーバー(=親の決済バー)close 時点 = 因果。
    """
    recs = {h: [] for h in hs}
    win = pool[pool["ret"] > 0]
    for instr, g in win.groupby("instr"):
        d = uni.instrument_data(instr, "H4")
        close = d["close"]
        idx = close.index
        carr = close.to_numpy()
        zarr = ((close - close.rolling(50).mean()) / close.rolling(50).std()).to_numpy()
        varr = close.pct_change().rolling(20).std().to_numpy()
        pos_of = pd.Series(np.arange(len(idx)), index=idx)
        x_pos = pos_of.reindex(g["exit"]).to_numpy()
        hs_price = config.spread_pips(instr) * config.pip_size(instr) / 2.0
        for ti, x in zip(g.index.to_numpy(), x_pos):
            if not np.isfinite(x):
                continue
            x = int(x)
            rev_dir = -int(pool.at[ti, "dir"])
            for h in hs:
                if x + h >= len(carr):
                    continue
                gross = rev_dir * (carr[x + h] / carr[x] - 1.0)
                cost = hs_price / carr[x] + hs_price / carr[x + h]
                recs[h].append({
                    "instr": instr,
                    "entry": idx[x],
                    "exit": idx[x + h],
                    "dir": rev_dir,
                    "entry_price": carr[x] + rev_dir * hs_price,
                    "ret": gross - cost,
                    "bars_held": h,
                    "z_entry": abs(zarr[x]),
                    "vol_entry": varr[x],
                })
    return {h: pd.DataFrame(recs[h]) for h in hs}


def full_eval(tag, pl, closes, seeds=SEEDS):
    """exp65 と同一規約のフル評価(emp + robust seeds + 年次 + IS較正→OOS素検証)。"""
    mk = champion_sizing(pl, max_pos=MAX_POS)
    cache = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            cache[kk] = mm.simulate(pl, closes, mk(kk), max_pos=MAX_POS)
        return cache[kk][0]

    r = protocol_eval(eq_of_k, label=tag, seeds=seeds)
    _, _, info_e = cache[round(float(r["emp_k"]), 10)]
    r["avg_conc"], r["max_conc"], r["skipped"] = (
        info_e["avg_conc"], info_e["max_conc"], info_e["skipped"])
    yr_e = yearly_returns(eq_of_k(r["emp_k"]))
    r["yr_emp"] = {int(y): float(v) for y, v in yr_e.items()}
    r["neg_years_emp"] = int((yr_e < 0).sum())
    r["worst_year_emp"] = float(yr_e.min())
    yr0 = yearly_returns(eq_of_k(r["rob"][seeds[0]]["k"]))
    r["yr_rob0"] = {int(y): float(v) for y, v in yr0.items()}
    r["neg_years_rob0"] = int((yr0 < 0).sum())
    r["worst_year_rob0"] = float(yr0.min())

    # G3: IS(<2022)較正 → OOS 素検証(rob seed0 / emp の両基準)
    is_pool = pl[pl["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = pl[pl["entry"] >= OOS_START].reset_index(drop=True)
    is_cl, oos_cl = closes[closes.index < OOS_START], closes[closes.index >= OOS_START]

    def eq_fn(p2, c2):
        c = {}

        def f(k):
            kk = round(float(k), 10)
            if kk not in c:
                c[kk] = mm.simulate(p2, c2, mk(kk), max_pos=MAX_POS)[0]
            return c[kk]
        return f

    fi, fo = eq_fn(is_pool, is_cl), eq_fn(oos_pool, oos_cl)
    k_ir = calibrate_robust_seeded(fi, 0.20, seed=0)
    r["is_rob_cagr"], r["oos_rob_cagr"] = cagr_of(fi(k_ir)), cagr_of(fo(k_ir))
    r["oos_rob_dd"] = max_dd(fo(k_ir))
    k_ie = calibrate_empirical(fi, 0.20)
    r["is_emp_cagr"], r["oos_emp_cagr"] = cagr_of(fi(k_ie)), cagr_of(fo(k_ie))
    r["oos_emp_dd"] = max_dd(fo(k_ie))
    r["eq_of_k"] = eq_of_k
    return r


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1().copy().reset_index(drop=True)
    closes = mm.load_closes()
    res: dict = {}

    # --- 0. 検算(プール + 反転 h5 の exp70/72 一致) -------------------------
    sec("0. 検算: d1 プール再現 + 反転 h5 の exp70/72 一致")
    n_pool, sum_ret = len(pool), float(pool["ret"].sum())
    print(f"プール n={n_pool} (期待 {REF_POOL_N}) / sum(ret)={sum_ret:+.4f} (期待 +{REF_POOL_SUM})")
    assert n_pool == REF_POOL_N, f"プール件数不一致: {n_pool}"
    assert abs(sum_ret - REF_POOL_SUM) < 1e-3, f"sum(ret) 不一致: {sum_ret}"

    revs = build_reversal_pools(pool)
    rv5 = revs[H_MAIN]
    net5 = float(rv5["ret"].mean() * 1e4)  # exp72 の net 平均 +2.2701bps と直接照合
    print(f"反転 h5: n={len(rv5)} (期待 865) / net平均 {net5:+.4f}bps (期待 +{REF_H5_NET}±0.05)")
    assert len(rv5) == 865, f"反転件数不一致: {len(rv5)}"
    assert abs(net5 - REF_H5_NET) < 0.05, f"exp72 net 検算不一致: {net5:.4f}"
    # gross も照合(コストを足し戻す)
    gross5 = []
    for instr, g in rv5.groupby("instr"):
        hs_price = config.spread_pips(instr) * config.pip_size(instr) / 2.0
        d = uni.instrument_data(instr, "H4")["close"]
        c_e = d.reindex(g["entry"]).to_numpy()
        c_x = d.reindex(g["exit"]).to_numpy()
        gross5.append(g["ret"].to_numpy() + hs_price / c_e + hs_price / c_x)
    gross5 = float(np.concatenate(gross5).mean() * 1e4)
    print(f"反転 h5: gross平均 {gross5:+.4f}bps (期待 +{REF_H5_GROSS}±0.1)")
    assert abs(gross5 - REF_H5_GROSS) < 0.1, f"exp70 gross 検算不一致: {gross5:.4f}"
    print("検算 OK")
    res["verification"] = {"n_pool": n_pool, "sum_ret": sum_ret, "n_rev5": int(len(rv5)),
                           "rev5_net_bps": net5, "rev5_gross_bps": gross5}
    fz_small = float((rv5["z_entry"].fillna(0) < 2.2 * 0.3 ** 0.25).mean())
    print(f"反転トレードの z_entry: 中央値 {rv5['z_entry'].median():.2f} / "
          f"f(z)=最小クリップ(0.3)該当率 {fz_small:.1%}")

    # --- 1. ベースライン(d1単独) ------------------------------------------
    sec("1. ベースライン: d1 単独(seeds 0-4 robust + emp + G3 素材)")
    base = full_eval("base(d1)", pool, closes)
    k0, c0 = base["rob"][0]["k"], base["rob"][0]["cagr"]
    print(f"rob_s0 k={k0:.3f} (期待 {REF_ROB0_K}) / CAGR={c0:+.2%} (期待 +{REF_ROB0_CAGR:.2%})")
    assert abs(k0 - REF_ROB0_K) < 0.05, f"ベースライン k 不一致: {k0:.3f}"
    assert abs(c0 - REF_ROB0_CAGR) < 0.002, f"ベースライン CAGR 不一致: {c0:+.4f}"
    print(f"ベースライン検算 OK  [{time.time()-t0:.0f}s]")

    # --- 2. 複合プール(base + 反転 h) -------------------------------------
    results = {"base": base}
    gates_all = {}
    for h in HS:
        sec(f"2-h{h}. 複合プール: d1 + 反転h{h} (n={len(pool)}+{len(revs[h])})")
        aug = pd.concat([pool, revs[h]], ignore_index=True
                        ).sort_values("entry").reset_index(drop=True)
        r = full_eval(f"d1+rev_h{h}", aug, closes)
        results[h] = r
        per_seed = {sd: r["rob"][sd]["cagr"] - base["rob"][sd]["cagr"] for sd in SEEDS}
        gain_pp = (r["rob_cagr_mean"] - base["rob_cagr_mean"]) * 100
        sig = (r["emp_cagr"] > base["emp_cagr"]) and \
              (abs(r["emp_p95"]) > abs(base["emp_p95"]) + 0.005)
        g3 = (r["oos_rob_cagr"] > base["oos_rob_cagr"]) and \
             (r["oos_emp_cagr"] > base["oos_emp_cagr"])
        g4 = (r["neg_years_emp"] == 0) and (r["neg_years_rob0"] == 0)
        # 署名のブートシード監査(emp_k, seeds 0-2)
        p95b = {sd: boot_dd(base["eq_of_k"](base["emp_k"]), n_boot=1500, seed=sd)["p95"]
                for sd in (0, 1, 2)}
        p95c = {sd: boot_dd(r["eq_of_k"](r["emp_k"]), n_boot=1500, seed=sd)["p95"]
                for sd in (0, 1, 2)}
        n_sig = sum((r["emp_cagr"] > base["emp_cagr"]) and
                    (abs(p95c[sd]) > abs(p95b[sd]) + 0.005) for sd in (0, 1, 2))
        gates = {
            "gain_pp": gain_pp,
            "per_seed_pp": {sd: v * 100 for sd, v in per_seed.items()},
            "all_seeds_pos": bool(all(v > 0 for v in per_seed.values())),
            "signature": bool(sig), "sig_seeds_3boot": int(n_sig),
            "g3_oos_raw_both": bool(g3),
            "g3_detail": {"oos_rob": r["oos_rob_cagr"], "oos_rob_base": base["oos_rob_cagr"],
                          "oos_emp": r["oos_emp_cagr"], "oos_emp_base": base["oos_emp_cagr"]},
            "g4_all_years_pos": bool(g4),
            "neg_years_emp": r["neg_years_emp"], "neg_years_rob0": r["neg_years_rob0"],
            "worst_year_emp": r["worst_year_emp"], "worst_year_rob0": r["worst_year_rob0"],
        }
        gates_all[h] = gates
        print(f"  gain {gain_pp:+.2f}pp  seeds " +
              " ".join(f"s{sd}:{v*100:+.2f}" for sd, v in per_seed.items()) +
              f"\n  emp {base['emp_cagr']:+.2%}→{r['emp_cagr']:+.2%}  "
              f"p95 {base['emp_p95']:+.1%}→{r['emp_p95']:+.1%}  署名={'あり' if sig else 'なし'}"
              f"(ブート3シード {n_sig}/3)"
              f"\n  G3(IS較正→OOS素) rob {base['oos_rob_cagr']:+.2%}→{r['oos_rob_cagr']:+.2%} / "
              f"emp {base['oos_emp_cagr']:+.2%}→{r['oos_emp_cagr']:+.2%}  {'PASS' if g3 else 'FAIL'}"
              f"\n  全年プラス: emp 負年{r['neg_years_emp']} rob0 負年{r['neg_years_rob0']}  "
              f"{'PASS' if g4 else 'FAIL'} (最悪年 emp {r['worst_year_emp']:+.1%})"
              f"\n  同時建玉 avg {base['avg_conc']:.2f}→{r['avg_conc']:.2f}  "
              f"max {base['max_conc']}→{r['max_conc']}  skip {base['skipped']}→{r['skipped']}"
              f"  [{time.time()-t0:.0f}s]")

    # --- 3. 判定 -------------------------------------------------------------
    sec("3. 判定(主構成 h=5)")
    g = gates_all[H_MAIN]
    if g["signature"] or (g["gain_pp"] < -NOISE_BAND_PP) or \
            (g["gain_pp"] > 0 and not g["g4_all_years_pos"]):
        verdict = "reject"
    elif abs(g["gain_pp"]) < NOISE_BAND_PP or not g["all_seeds_pos"] or not g["g3_oos_raw_both"]:
        verdict = "noise" if not g["signature"] and g["gain_pp"] > -NOISE_BAND_PP else "reject"
    else:
        verdict = "adopt-candidate"
    print(f"  gain {g['gain_pp']:+.2f}pp / 全シード符号 {g['all_seeds_pos']} / "
          f"署名 {g['signature']} / G3 {g['g3_oos_raw_both']} / 全年+ {g['g4_all_years_pos']}")
    print(f"  → verdict = {verdict}")
    print("  (注: ノイズ帯 ±0.4-0.8pp。reports/22 前例: +0.3pp級は採用しない。"
          "プール段で週末跨ぎ除外に全構成不耐 = 採用には h5 でも18%が週末ギャップ持ち越し)")

    # --- 保存 ------------------------------------------------------------------
    rows = []
    for c in ["base"] + list(HS):
        r = results[c]
        row = {"cfg": str(c), "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"],
               "emp_p95": r["emp_p95"], "rob_mean5": r["rob_cagr_mean"],
               **{f"rob_s{sd}": r["rob"][sd]["cagr"] for sd in SEEDS},
               **{f"rob_k{sd}": r["rob"][sd]["k"] for sd in SEEDS},
               "is_rob": r["is_rob_cagr"], "oos_rob": r["oos_rob_cagr"],
               "is_emp": r["is_emp_cagr"], "oos_emp": r["oos_emp_cagr"],
               "neg_years_emp": r["neg_years_emp"], "neg_years_rob0": r["neg_years_rob0"],
               "worst_year_emp": r["worst_year_emp"],
               "avg_conc": r["avg_conc"], "max_conc": r["max_conc"], "skipped": r["skipped"]}
        if c != "base":
            row.update({f"gate_{k}": v for k, v in gates_all[c].items()
                        if k not in ("per_seed_pp", "g3_detail")})
        rows.append(row)
    df = pd.DataFrame(rows)
    res["configs"] = {str(c): {k: ({str(s): vv for s, vv in v.items()} if k == "rob" else v)
                               for k, v in results[c].items()
                               if k not in ("eq_of_k",)}
                      for c in ["base"] + list(HS)}
    res["gates"] = {str(h): {k: ({str(s): vv for s, vv in v.items()}
                                 if isinstance(v, dict) else v)
                             for k, v in gg.items()} for h, gg in gates_all.items()}
    res["verdict_h5"] = verdict
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "exp72b_result.csv", index=False)
    (OUT_DIR / "exp72b_result.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print("\n=== 最終表 ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(df.to_string(index=False))
    print(f"\nsaved -> {OUT_DIR / 'exp72b_result.csv'} / exp72b_result.json"
          f"\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
