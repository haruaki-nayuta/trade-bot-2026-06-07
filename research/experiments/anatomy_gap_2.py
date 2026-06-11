"""anatomy_gap_2: チャンピオン confluence_meanrev_v2 最終コンフィグの閾値ロバストネス・マップ。

解剖6視点の盲点を埋める: 既存の解剖はすべて「選ばれた1点のコンフィグが生んだプール」を
条件とした事後分析であり、本番の閾値の組(z=2.0 / RSI=35 / volq=0.70 / z250=1.75 / ER=0.55 /
出口|z|≤0.5、ルックバック 50/250/40/100/20)が 5+次元空間の高原(plateau)上にあるのか
1点突出(spike)なのかを、閾値を動かしてシグナルを再生成する形では一度も検証していない。

手順:
  1. ベースライン再現検算: パラメータ化したシグナル生成で19銘柄H4のプールを再構築し、
     results/mm_pool_v2_H4_19.parquet (n=1214 / sum=+1.9086) と一致することを確認。
  2. one-at-a-time 摂動 (22構成) + 全閾値同時摂動 (loose/tight 2構成) でプール再生成。
  3. 各構成の sum(ret)/PF/平均bps/勝率/プラス年数/時代分割(2016-21 vs 2022-26) を表化し、
     (a) ベースラインの全構成内パーセンタイル (中央付近=高原 / 最上位=尖点疑い)
     (b) パラメータごとの単調悪化の崖の有無
     (c) ベースライン超の近傍の時代分割頑健性
     を報告する。

実行: uv run python -m research.experiments.anatomy_gap_2
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import vectorbt as vbt

from fxlab import config, universe as uni
from fxlab.backtest import run
from fxlab.trades import trade_table

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

TF = "H4"
POOL_PATH = config.RESULTS_DIR / "mm_pool_v2_H4_19.parquet"

# 本番ベースライン(strategies/confluence_meanrev_v2.PARAMS と同値。
# vol_sd_win=20 は generate_signals 内にハードコードされている実現ボラ推定窓を明示化したもの)
BASE = {
    "window": 50, "entry_z": 2.0, "exit_z": 0.5,
    "rsi_p": 14, "rsi_low": 35, "rsi_high": 65,
    "vol_sd_win": 20, "vol_win": 100, "vol_pct": 0.70,
    "slow_win": 250, "slow_z": 1.75,
    "er_win": 40, "er_max": 0.55,
}

# --- one-at-a-time 摂動の定義(パラメータ名, 表示名, 摂動値リスト[基準含む昇順]) ---
# RSI は本番が low=35/high=65 の対称なので low と high を連動して動かす(rsi_low で代表)。
OAT = [
    ("entry_z",   "エントリーz",      [1.8, 2.0, 2.2]),
    ("rsi_low",   "RSI閾値(対称)",    [30, 35, 40]),
    ("vol_pct",   "ボラ分位",         [0.60, 0.70, 0.80]),
    ("slow_z",    "長期z250閾値",     [1.50, 1.75, 2.00]),
    ("er_max",    "ER上限",           [0.45, 0.55, 0.65]),
    ("exit_z",    "出口|z|",          [0.3, 0.5, 0.7]),
    ("window",    "短期z窓",          [40, 50, 60]),
    ("slow_win",  "長期z窓",          [200, 250, 300]),
    ("er_win",    "ER窓",             [30, 40, 50]),
    ("vol_win",   "ボラ分位窓",       [80, 100, 120]),
    ("vol_sd_win","実現ボラ窓",       [15, 20, 25]),
]


def perturbed_params(pname: str, val) -> dict:
    p = dict(BASE)
    if pname == "rsi_low":
        p["rsi_low"] = val
        p["rsi_high"] = 100 - val  # 対称維持 (35/65, 30/70, 40/60)
    else:
        p[pname] = val
    return p


def joint_params(kind: str) -> dict:
    """全エントリー閾値を同時に1ステップ動かすストレス構成(ルックバック・出口は本番)。"""
    p = dict(BASE)
    if kind == "loose":   # 全条件を緩める=コンフルエンス弱体化
        p.update({"entry_z": 1.8, "rsi_low": 40, "rsi_high": 60,
                  "vol_pct": 0.80, "slow_z": 1.50, "er_max": 0.65})
    elif kind == "tight":  # 全条件を絞る=取引数減
        p.update({"entry_z": 2.2, "rsi_low": 30, "rsi_high": 70,
                  "vol_pct": 0.60, "slow_z": 2.00, "er_max": 0.45})
    return p


# --- パラメータ化シグナル生成(confluence_meanrev_v2 と同一ロジック+指標メモ化) ---
def _zscore(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def _er(close: pd.Series, w: int) -> pd.Series:
    direction = (close - close.shift(w)).abs()
    volatility = close.diff().abs().rolling(w).sum()
    return (direction / volatility).replace([np.inf, -np.inf], np.nan)


class IndicatorCache:
    """銘柄ごとに指標を窓パラメータ別にメモ化(25構成×19銘柄の再計算を省く)。"""

    def __init__(self, close: pd.Series):
        self.close = close
        self._z, self._rsi, self._vol, self._calm, self._erc = {}, {}, {}, {}, {}

    def z(self, w):
        if w not in self._z:
            self._z[w] = _zscore(self.close, w)
        return self._z[w]

    def rsi(self, p):
        if p not in self._rsi:
            self._rsi[p] = vbt.RSI.run(self.close, p).rsi
        return self._rsi[p]

    def vol(self, sd_win):
        if sd_win not in self._vol:
            self._vol[sd_win] = self.close.pct_change().rolling(sd_win).std()
        return self._vol[sd_win]

    def calm(self, sd_win, vol_win, vol_pct):
        key = (sd_win, vol_win, vol_pct)
        if key not in self._calm:
            v = self.vol(sd_win)
            self._calm[key] = v <= v.rolling(vol_win).quantile(vol_pct)
        return self._calm[key]

    def er(self, w):
        if w not in self._erc:
            self._erc[w] = _er(self.close, w)
        return self._erc[w]


def make_gen(cache: IndicatorCache):
    """fxlab.backtest.run に渡す generate_signals(data, **params)。本家 v2 と同一ロジック。"""

    def gen(data: pd.DataFrame, *, window, entry_z, exit_z, rsi_p, rsi_low, rsi_high,
            vol_sd_win, vol_win, vol_pct, slow_win, slow_z, er_win, er_max):
        z = cache.z(window)
        rsi = cache.rsi(rsi_p)
        calm = cache.calm(vol_sd_win, vol_win, vol_pct)
        zs = cache.z(slow_win)
        long_ok = (zs < -slow_z).fillna(False)
        short_ok = (zs > slow_z).fillna(False)
        er_ok = (cache.er(er_win) <= er_max).fillna(False)
        le = (z < -entry_z) & (z.shift() >= -entry_z) & (rsi < rsi_low) & calm & long_ok & er_ok
        se = (z > entry_z) & (z.shift() <= entry_z) & (rsi > rsi_high) & calm & short_ok & er_ok
        lx = z > -exit_z
        sx = z < exit_z
        return le.fillna(False), lx.fillna(False), se.fillna(False), sx.fillna(False)

    return gen


# --- プール構築と集計 -----------------------------------------------------
def build_pool_for_params(datas: dict[str, pd.DataFrame], caches: dict[str, IndicatorCache],
                          params: dict) -> pd.DataFrame:
    frames = []
    for nm, data in datas.items():
        pf = run(nm, TF, make_gen(caches[nm]), params, data=data, size_mode="value")
        tt = trade_table(pf, data)
        if tt.empty:
            continue
        frames.append(pd.DataFrame({
            "instr": nm,
            "entry": tt["entry"].to_numpy(),
            "exit": tt["exit"].to_numpy(),
            "ret": tt["return_pct"].to_numpy() / 100.0,
            "bars_held": tt["bars_held"].to_numpy(),
        }))
    if not frames:
        return pd.DataFrame(columns=["instr", "entry", "exit", "ret", "bars_held"])
    return pd.concat(frames, ignore_index=True).sort_values("entry").reset_index(drop=True)


def pool_stats(pool: pd.DataFrame) -> dict:
    r = pool["ret"]
    pos, neg = r[r > 0].sum(), r[r < 0].sum()
    yearly = pool.groupby(pd.to_datetime(pool["exit"]).dt.year)["ret"].sum()
    era1 = pool.loc[pd.to_datetime(pool["exit"]) < "2022-01-01", "ret"].sum()
    era2 = pool.loc[pd.to_datetime(pool["exit"]) >= "2022-01-01", "ret"].sum()
    return {
        "n": int(len(pool)),
        "sum_ret": float(r.sum()),
        "mean_bps": float(r.mean() * 1e4) if len(r) else np.nan,
        "win": float((r > 0).mean()) if len(r) else np.nan,
        "pf": float(pos / abs(neg)) if neg < 0 else np.inf,
        "pos_years": int((yearly > 0).sum()),
        "n_years": int(len(yearly)),
        "worst_year": float(yearly.min()) if len(yearly) else np.nan,
        "era_16_21": float(era1),
        "era_22_26": float(era2),
        "med_hold": float(pool["bars_held"].median()) if len(pool) else np.nan,
    }


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    instruments = [x for x in uni.universe(crosses=True) if x != "AUDJPY"]
    print(f"=== anatomy_gap_2: v2閾値ロバストネス・マップ ({TF}, {len(instruments)}銘柄) ===\n")

    # 価格データと指標キャッシュを先読み
    datas = {nm: uni.instrument_data(nm, TF) for nm in instruments}
    caches = {nm: IndicatorCache(datas[nm]["close"]) for nm in instruments}

    # --- 1. ベースライン再現検算 ---
    ref = pd.read_parquet(POOL_PATH)
    base_pool = build_pool_for_params(datas, caches, BASE)
    bs = pool_stats(base_pool)
    print("[検算] ベースライン構成の再現:")
    print(f"  参照プール : n={len(ref)}  sum(ret)={ref['ret'].sum():+.4f}")
    print(f"  再生成     : n={bs['n']}  sum(ret)={bs['sum_ret']:+.4f}")
    n_ok = bs["n"] == len(ref)
    s_ok = abs(bs["sum_ret"] - ref["ret"].sum()) < 1e-6
    print(f"  -> n一致: {n_ok} / sum一致(<1e-6): {s_ok}")
    if not (n_ok and s_ok):
        # 突合せ(どこがずれたか)
        m = base_pool.merge(ref[["instr", "entry", "ret"]], on=["instr", "entry"],
                            how="outer", suffixes=("_new", "_ref"), indicator=True)
        print(m[m["_merge"] != "both"].head(20))
        print("!! ベースライン再現に失敗。以降の比較は無効。")
        return 1
    print()

    # --- 2. 摂動構成の評価 ---
    rows = []
    rows.append({"config": "BASELINE", "param": "-", "value": "-", **bs})
    for pname, label, vals in OAT:
        for v in vals:
            if v == BASE.get(pname):
                continue  # 基準値はBASELINE行で代表
            p = perturbed_params(pname, v)
            st = pool_stats(build_pool_for_params(datas, caches, p))
            rows.append({"config": f"{pname}={v}", "param": label, "value": v, **st})
            print(f"  done {pname}={v}  (n={st['n']}, sum={st['sum_ret']:+.3f})  "
                  f"[{time.time()-t0:.0f}s]")
    for kind in ["loose", "tight"]:
        st = pool_stats(build_pool_for_params(datas, caches, joint_params(kind)))
        rows.append({"config": f"JOINT_{kind}", "param": "全閾値同時", "value": kind, **st})
        print(f"  done JOINT_{kind}  (n={st['n']}, sum={st['sum_ret']:+.3f})")

    df = pd.DataFrame(rows)
    base_sum = bs["sum_ret"]
    df["d_sum_%"] = (df["sum_ret"] - base_sum) / base_sum * 100  # 総純益比の増減%

    # --- 3. 出力 ---
    print("\n=== 全構成の成績表(sum_ret 降順) ===")
    out = df[["config", "n", "sum_ret", "d_sum_%", "mean_bps", "win", "pf",
              "pos_years", "n_years", "worst_year", "era_16_21", "era_22_26", "med_hold"]]
    print(out.sort_values("sum_ret", ascending=False).to_string(
        index=False, float_format=lambda x: f"{x:.4g}"))

    # (a) ベースラインのパーセンタイル
    sums = df["sum_ret"].to_numpy()
    pct_all = float((sums < base_sum).mean() * 100)
    oat_mask = ~df["config"].str.startswith(("JOINT", "BASELINE"))
    sums_oat = df.loc[oat_mask, "sum_ret"].to_numpy()
    pct_oat = float((sums_oat < base_sum).mean() * 100)
    print("\n=== (a) ベースラインの位置 ===")
    print(f"  全{len(df)}構成内パーセンタイル: {pct_all:.0f}%tile "
          f"(OAT {len(sums_oat)}近傍のみ: {pct_oat:.0f}%tile)")
    print(f"  近傍分布: min={sums.min():+.3f} / median={np.median(sums):+.3f} / "
          f"max={sums.max():+.3f} / base={base_sum:+.3f}")
    print(f"  ベースライン超の近傍: {int((sums > base_sum).sum())}構成 / "
          f"全近傍プラス維持: {bool((sums > 0).all())} / "
          f"全構成の最小プラス年数: {int(df['pos_years'].min())}/{int(df['n_years'].max())}")

    # (b) パラメータごとの断面(単調崖チェック)
    print("\n=== (b) パラメータ断面(値の昇順, sum_ret / PF / n) ===")
    for pname, label, vals in OAT:
        line = []
        for v in vals:
            if v == BASE.get(pname):
                r = df[df["config"] == "BASELINE"].iloc[0]
            else:
                r = df[df["config"] == f"{pname}={v}"].iloc[0]
            star = "*" if v == BASE.get(pname) else " "
            line.append(f"{v}{star}: {r['sum_ret']:+.3f}/PF{r['pf']:.2f}/n{r['n']}")
        s_seq = []
        for v in vals:
            rr = df[df["config"] == ("BASELINE" if v == BASE.get(pname)
                                     else f"{pname}={v}")].iloc[0]
            s_seq.append(rr["sum_ret"])
        mono = "単調増" if all(np.diff(s_seq) > 0) else (
            "単調減" if all(np.diff(s_seq) < 0) else "山/谷")
        print(f"  {label:<12} [{mono}]  " + "  |  ".join(line))

    # (c) ベースライン超近傍の時代分割
    print("\n=== (c) ベースライン超の近傍: 時代分割頑健性 ===")
    better = df[(df["sum_ret"] > base_sum) & (df["config"] != "BASELINE")]
    if better.empty:
        print("  ベースラインを上回る近傍なし(=基準点が局所最大)。")
    else:
        b16, b22 = bs["era_16_21"], bs["era_22_26"]
        print(f"  ベースライン: 2016-21 {b16:+.3f} / 2022-26 {b22:+.3f}")
        for _, r in better.sort_values("sum_ret", ascending=False).iterrows():
            both = (r["era_16_21"] > b16) and (r["era_22_26"] > b22)
            print(f"  {r['config']:<16} sum={r['sum_ret']:+.3f} ({r['d_sum_%']:+.1f}%)  "
                  f"2016-21 {r['era_16_21']:+.3f} / 2022-26 {r['era_22_26']:+.3f}  "
                  f"両時代改善={'YES' if both else 'NO'}  PF{r['pf']:.2f} n{r['n']} "
                  f"プラス年{r['pos_years']}/{r['n_years']}")

    csv_path = config.RESULTS_DIR / "anatomy_gap_2_threshold_map.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n保存: {csv_path}  (経過 {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
