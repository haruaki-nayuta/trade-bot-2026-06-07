"""exp72: 勝ちトレード決済後の同銘柄・反対方向リエントリー(プール段検証)。

仮説(ユーザー発案):「チャンピオンが勝った直後に、その銘柄へ反対方向で入り直せば
逆戻りを取れる」。根拠は exp70-D(勝ちトレード865件の決済後、トレード方向ドリフトが
h1 -1.3 / h2 -1.7 / h5 -4.44(t=-2.96) / h10 -3.8 bps コスト前。負け決済後はゼロ n.s.)。

反実仮想トレードの構築(事前登録設計):
  ・対象: d1 プール(build_pool_d1, n=1207)の各決済。主構成は「直前トレードが勝ち」
    (ret>0, n=865)の決済のみ。**勝敗は決済時点で確定済みの情報 = 条件付けは因果**
    (look-ahead なし)。
  ・エントリー: 決済バー close ± 半スプレッド。決済: エントリーから h バー後の
    close ∓ 半スプレッド。コスト規約は現行プールと同一
    (fxlab.backtest._slippage_series: half_spread_price/close をエントリー・
    エグジット各バーで計上、half_spread_price = spread_pips × pip_size / 2)。
    銘柄別スプレッドは fxlab.config.SPREADS_PIPS、クロスは
    uni.register_cross_spreads(3.0) = 3.0pips を踏襲。
  ・用量曲線: h ∈ {1,2,3,5,8,10,15}。**事前登録の主判定は h=5**。
    ※ 注意: h=5 は exp70(同一データ)で観測されたドリフトのピークであり、
    「同一データでの選択」を含む。判定は h=5 単点でなく曲線の形(肩の有無)も併記する。
  ・close 系列は exp70 と同一の uni.instrument_data(instr, "H4")["close"]
    (= mm_lab.load_closes の列と同源。検算を exp70 と厳密一致させるため
    銘柄固有グリッドを使用)。z 系列の再計算は不要。

対照と感度(事前登録):
  (a) 負け決済後の反対方向(exp70 予測: エッジ無し)
  (b) 全決済後(条件なし)
  (c) 決済バー(=反転エントリーバー)が 20-23時UTC のものを除外(ロールオーバー, reports/12)
  (d) 反対方向保有中(エントリーバー→hバー後)に週末ギャップ(>6h)を跨ぐものを除外
  (e) メジャー7 vs クロス12
  (f) 年別 + IS(<2022)/OOS(>=2022)
  (g) 衝突率: 反対方向保有中にチャンピオン本体が同銘柄で新規エントリーする率
      (本体の新エントリー方向が反転トレードと同方向/逆方向の内訳も併記)

プール段ゲート(事前登録, h=5 主構成):
  ①コスト後 net_bps > 0  ②OOS(>=2022)でも net > 0  ③単年依存なし
  (net 寄与最大年を除外しても平均の符号維持)。3条件全成立で pass_pool_gate=True。

検算(進行前提): プール n=1207・sum(ret)=+1.9622、対象勝ち n=865、
  h5 のコスト前平均が exp70 の -4.4433bps の符号反転 +4.4433bps と ±0.1bps で一致。

統計上の注意: 同一銘柄で h バー以内に連続する勝ち決済があると反転トレード同士の
観測が重複(オーバーラップ)し、t 値はやや楽観的になる(exp70-D と同じ性質)。

実行: PYTHONPATH=. uv run python research/experiments/exp72_postwin_reversal.py
出力: research/outputs/exp72_result.json
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

from mm_production import build_pool_d1  # noqa: E402
from fxlab import config  # noqa: E402
from fxlab import universe as uni  # noqa: E402

OUT_DIR = ROOT / "research" / "outputs"
HS = (1, 2, 3, 5, 8, 10, 15)   # 用量曲線
H_MAIN = 5                      # 事前登録の主判定点(exp70 ピーク。同一データ選択に注意)
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
EXP70_H5_BPS = 4.4433           # 検算ターゲット(符号反転後)


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def build_reversal_table(pool: pd.DataFrame) -> pd.DataFrame:
    """各決済について反転トレードの gross/net(h別)+感度フラグを列挙。"""
    n = len(pool)
    out = {}
    for h in HS:
        out[f"gross_{h}"] = np.full(n, np.nan)   # コスト前(反転方向)
        out[f"net_{h}"] = np.full(n, np.nan)     # コスト後
        out[f"wknd_{h}"] = np.full(n, np.nan)    # 保有中に週末ギャップを跨ぐか
        out[f"conf_{h}"] = np.full(n, np.nan)    # 保有中に本体の新規エントリーがあるか
        out[f"conf_same_{h}"] = np.full(n, np.nan)  # その本体方向が反転と同方向か
    out["exit_hour"] = np.full(n, np.nan)
    out["cost_bps"] = np.full(n, np.nan)         # 往復コスト(h=H_MAIN 時, bps)

    for instr, g in pool.groupby("instr"):
        d = uni.instrument_data(instr, "H4")
        close = d["close"]
        idx = close.index
        carr = close.to_numpy()
        tarr = idx.to_numpy()
        gap_hours = np.diff(tarr) / np.timedelta64(1, "h")  # gap[i] = ts[i]→ts[i+1]
        pos_of = pd.Series(np.arange(len(idx)), index=idx)
        x_pos = pos_of.reindex(g["exit"]).to_numpy()
        half_spread = config.spread_pips(instr) * config.pip_size(instr) / 2.0
        # 本体の同銘柄エントリー(衝突検出用): 時刻昇順の (entry_ts, dir)
        ee = g.sort_values("entry")
        e_ts = ee["entry"].to_numpy()
        e_dir = ee["dir"].to_numpy().astype(float)

        for ti, x in zip(g.index.to_numpy(), x_pos):
            if not np.isfinite(x):
                continue
            x = int(x)
            rev_dir = -float(pool.at[ti, "dir"])  # 反対方向
            out["exit_hour"][ti] = idx[x].hour
            for h in HS:
                if x + h >= len(carr):
                    continue
                gross = rev_dir * (carr[x + h] / carr[x] - 1.0)
                cost = half_spread / carr[x] + half_spread / carr[x + h]
                out[f"gross_{h}"][ti] = gross
                out[f"net_{h}"][ti] = gross - cost
                out[f"wknd_{h}"][ti] = float((gap_hours[x:x + h] > 6.0).max())
                if h == H_MAIN:
                    out["cost_bps"][ti] = cost * 1e4
                # 衝突: ts[x] < entry <= ts[x+h] の本体新規エントリー
                lo = np.searchsorted(e_ts, tarr[x], side="right")
                hi = np.searchsorted(e_ts, tarr[x + h], side="right")
                out[f"conf_{h}"][ti] = float(hi > lo)
                if hi > lo:
                    out[f"conf_same_{h}"][ti] = float((e_dir[lo:hi] == rev_dir).any())
    return pd.DataFrame(out, index=pool.index)


def curve_stats(sub: pd.DataFrame, hs=HS) -> dict:
    """サブセットの h 別 gross/net 平均(bps)と net の t 値。"""
    rows = {}
    for h in hs:
        gv = sub[f"gross_{h}"].dropna() * 1e4
        nv = sub[f"net_{h}"].dropna() * 1e4
        if len(nv) < 2:
            rows[f"h{h}"] = {"n": int(len(nv))}
            continue
        rows[f"h{h}"] = {
            "n": int(len(nv)),
            "gross_bps": float(gv.mean()),
            "net_bps": float(nv.mean()),
            "t_net": float(nv.mean() / (nv.std() / np.sqrt(len(nv)))),
            "net_median_bps": float(nv.median()),
            "net_win_rate": float((nv > 0).mean()),
        }
    return rows


def print_curve(label: str, rows: dict):
    print(f"\n[{label}]")
    print("   h |    n  | gross(bps) |  net(bps) |  t(net) | net勝率")
    for h in HS:
        r = rows.get(f"h{h}", {})
        if "net_bps" not in r:
            print(f"  {h:>2} | {r.get('n', 0):>5} |     (n<2)")
            continue
        print(f"  {h:>2} | {r['n']:>5} | {r['gross_bps']:>+9.2f} | {r['net_bps']:>+8.2f} | "
              f"{r['t_net']:>+6.2f} | {r['net_win_rate']:.1%}")


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1().copy()
    res: dict = {}

    # --- 検算ゲート(不一致なら中止) ---------------------------------------
    sec("0. 検算(d1 プール再現 + exp70 h5 ドリフト一致)")
    n_pool, sum_ret = len(pool), float(pool["ret"].sum())
    print(f"プール n={n_pool} (期待 1207) / sum(ret)={sum_ret:+.4f} (期待 +1.9622)")
    assert n_pool == 1207, f"プール件数不一致: {n_pool}"
    assert abs(sum_ret - 1.9622) < 1e-3, f"プール sum(ret) 不一致: {sum_ret}"

    rt = build_reversal_table(pool)
    df = pd.concat([pool, rt], axis=1)
    win_m = df["ret"] > 0
    W, L = df[win_m], df[~win_m]
    n_win = int(win_m.sum())
    g5 = W[f"gross_{H_MAIN}"].dropna() * 1e4
    print(f"勝ち決済 n={n_win} (期待 865) / 反転 h5 コスト前平均 {g5.mean():+.4f}bps "
          f"(期待 +{EXP70_H5_BPS}±0.1, n={len(g5)})")
    assert n_win == 865, f"勝ち件数不一致: {n_win}"
    assert abs(g5.mean() - EXP70_H5_BPS) < 0.1, f"exp70 検算不一致: {g5.mean():.4f}"
    print("検算 OK")
    res["verification"] = {"n_pool": n_pool, "sum_ret": sum_ret, "n_wins": n_win,
                           "h5_gross_bps": float(g5.mean()), "exp70_target": EXP70_H5_BPS}

    # --- 1. 主構成: 勝ち決済後の反対方向(用量曲線) -------------------------
    sec("1. 主構成: 勝ち決済後・反対方向(コスト前/後の用量曲線)")
    cost5 = W["cost_bps"].dropna()
    print(f"往復コスト(h5時点, bps): 平均 {cost5.mean():.2f} / 中央値 {cost5.median():.2f} "
          f"/ p10 {cost5.quantile(0.10):.2f} / p90 {cost5.quantile(0.90):.2f}")
    main_curve = curve_stats(W)
    print_curve("after-WIN reversal (n=865)", main_curve)
    res["main_after_win"] = main_curve
    res["cost_bps_h5"] = {"mean": float(cost5.mean()), "median": float(cost5.median()),
                          "p10": float(cost5.quantile(0.10)), "p90": float(cost5.quantile(0.90))}

    # --- 2. 対照 (a) 負け決済後 / (b) 全決済 ---------------------------------
    sec("2. 対照: (a) 負け決済後 / (b) 全決済(条件なし)")
    loss_curve = curve_stats(L)
    all_curve = curve_stats(df)
    print_curve("(a) after-LOSS reversal (n=342)", loss_curve)
    print_curve("(b) ALL exits reversal (n=1207)", all_curve)
    res["control_after_loss"] = loss_curve
    res["control_all_exits"] = all_curve

    # --- 3. 感度 (c) ロールオーバー / (d) 週末跨ぎ ---------------------------
    sec("3. 感度: (c) 決済バー20-23時UTC除外 / (d) 週末跨ぎ除外(勝ち決済後)")
    roll_m = W["exit_hour"].isin([20, 21, 22, 23])
    print(f"(c) 決済バー 20-23時UTC: {int(roll_m.sum())}/{len(W)} 件 "
          f"({roll_m.mean():.1%}) を除外")
    no_roll = curve_stats(W[~roll_m])
    print_curve("(c) ロールオーバー除外", no_roll)
    res["sens_no_rollover"] = no_roll
    res["sens_no_rollover_excluded_n"] = int(roll_m.sum())

    wk_rates = {h: float(W[f"wknd_{h}"].dropna().mean()) for h in HS}
    print("\n(d) 週末跨ぎ率: " + "  ".join(f"h{h}:{wk_rates[h]:.0%}" for h in HS))
    nowk = {}
    for h in HS:
        sub = W[W[f"wknd_{h}"] == 0]
        nowk.update(curve_stats(sub, hs=(h,)))
    print_curve("(d) 週末跨ぎ除外(各h)", nowk)
    res["sens_no_weekend"] = nowk
    res["weekend_cross_rate"] = {f"h{h}": wk_rates[h] for h in HS}

    # --- 4. 感度 (e) メジャー vs クロス -------------------------------------
    sec("4. 感度: (e) メジャー7 vs クロス12(勝ち決済後)")
    majors = set(uni.universe(crosses=False))
    maj_m = W["instr"].isin(majors)
    maj_curve = curve_stats(W[maj_m])
    crs_curve = curve_stats(W[~maj_m])
    print_curve(f"メジャー (n={int(maj_m.sum())})", maj_curve)
    print_curve(f"クロス (n={int((~maj_m).sum())})", crs_curve)
    res["sens_majors"] = maj_curve
    res["sens_crosses"] = crs_curve

    # --- 5. (f) 年別 + IS/OOS(h=5 主判定点) -------------------------------
    sec(f"5. (f) 年別 + IS(<2022)/OOS(>=2022)  [h={H_MAIN}]")
    Wn = W.dropna(subset=[f"net_{H_MAIN}"]).copy()
    Wn["year"] = Wn["exit"].dt.year
    ynet = Wn.groupby("year")[f"net_{H_MAIN}"]
    year_tbl = pd.DataFrame({"n": ynet.size(), "net_bps": ynet.mean() * 1e4,
                             "net_sum_bps": ynet.sum() * 1e4})
    print(year_tbl.to_string(float_format=lambda x: f"{x:+.2f}"))
    pos_years = int((year_tbl["net_sum_bps"] > 0).sum())
    n_years = len(year_tbl)
    print(f"プラス年 {pos_years}/{n_years}")
    res["yearly_h5"] = {int(y): {k: float(v) for k, v in r.items()}
                        for y, r in year_tbl.iterrows()}
    res["pos_years_h5"] = f"{pos_years}/{n_years}"

    is_m = Wn["exit"] < OOS_START
    isv = Wn.loc[is_m, f"net_{H_MAIN}"] * 1e4
    oosv = Wn.loc[~is_m, f"net_{H_MAIN}"] * 1e4
    t_is = float(isv.mean() / (isv.std() / np.sqrt(len(isv))))
    t_oos = float(oosv.mean() / (oosv.std() / np.sqrt(len(oosv))))
    print(f"IS  (<2022) : n={len(isv):>4}  net {isv.mean():+.2f}bps (t={t_is:+.2f})")
    print(f"OOS (>=2022): n={len(oosv):>4}  net {oosv.mean():+.2f}bps (t={t_oos:+.2f})")
    res["is_oos_h5"] = {"is_n": int(len(isv)), "is_net_bps": float(isv.mean()), "is_t": t_is,
                        "oos_n": int(len(oosv)), "oos_net_bps": float(oosv.mean()), "oos_t": t_oos}
    # 用量曲線への OOS 付記
    oos_by_h = {}
    for h in HS:
        sub = W.dropna(subset=[f"net_{h}"])
        v = sub.loc[sub["exit"] >= OOS_START, f"net_{h}"] * 1e4
        oos_by_h[f"h{h}"] = float(v.mean()) if len(v) else float("nan")
    res["oos_net_by_h"] = oos_by_h
    print("OOS net by h: " + "  ".join(f"h{h}:{oos_by_h[f'h{h}']:+.2f}" for h in HS))

    # 単年依存チェック: net 寄与最大年を除外して符号維持か
    best_year = int(year_tbl["net_sum_bps"].idxmax())
    rest = Wn[Wn["year"] != best_year][f"net_{H_MAIN}"] * 1e4
    t_rest = float(rest.mean() / (rest.std() / np.sqrt(len(rest))))
    print(f"単年依存: 寄与最大年 {best_year} "
          f"({year_tbl.loc[best_year, 'net_sum_bps']:+.0f}bps累計) を除外 → "
          f"net {rest.mean():+.2f}bps (t={t_rest:+.2f}, n={len(rest)})")
    res["best_year_excl_h5"] = {"best_year": best_year, "net_bps_excl": float(rest.mean()),
                                "t_excl": t_rest, "n": int(len(rest))}

    # --- 6. (g) 衝突率 --------------------------------------------------------
    sec("6. (g) 衝突率: 反転保有中に本体が同銘柄で新規エントリー(勝ち決済後)")
    conf = {}
    for h in HS:
        c = W[f"conf_{h}"].dropna()
        cs = W[f"conf_same_{h}"].dropna()  # 衝突した件のみ 0/1
        conf[f"h{h}"] = {
            "conflict_rate": float(c.mean()), "n": int(len(c)),
            "same_dir_share": float(cs.mean()) if len(cs) else float("nan"),
            "n_conflicts": int(c.sum()),
        }
        print(f"  h{h:>2}: 衝突率 {conf[f'h{h}']['conflict_rate']:.1%} "
              f"(n_conf={conf[f'h{h}']['n_conflicts']}) | "
              f"衝突時に本体が反転と同方向 {conf[f'h{h}']['same_dir_share']:.1%}")
    res["conflict"] = conf

    # --- 7. 事前登録ゲート -----------------------------------------------------
    sec("7. プール段ゲート(事前登録, h=5 主構成)")
    g1 = main_curve[f"h{H_MAIN}"]["net_bps"] > 0
    g2 = res["is_oos_h5"]["oos_net_bps"] > 0
    g3 = res["best_year_excl_h5"]["net_bps_excl"] > 0
    gate = bool(g1 and g2 and g3)
    print(f"  ①コスト後 net>0      : {'PASS' if g1 else 'FAIL'} "
          f"({main_curve[f'h{H_MAIN}']['net_bps']:+.2f}bps)")
    print(f"  ②OOS net>0           : {'PASS' if g2 else 'FAIL'} "
          f"({res['is_oos_h5']['oos_net_bps']:+.2f}bps)")
    print(f"  ③最良年除外で符号維持: {'PASS' if g3 else 'FAIL'} "
          f"({res['best_year_excl_h5']['net_bps_excl']:+.2f}bps)")
    print(f"  → pass_pool_gate = {gate}")
    res["gate"] = {"g1_net_pos": bool(g1), "g2_oos_pos": bool(g2),
                   "g3_no_single_year": bool(g3), "pass_pool_gate": gate}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "exp72_result.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print(f"\nsaved -> {OUT_DIR / 'exp72_result.json'}\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
