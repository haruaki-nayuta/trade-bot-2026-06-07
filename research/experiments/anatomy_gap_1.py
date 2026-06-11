"""anatomy_gap_1: ユニバース選択バイアスの監査(解剖の盲点埋め)。

問い: チャンピオン confluence_meanrev_v2 の19銘柄ユニバース(7メジャー+12クロス)は、
8通貨から作れる28ペアのうち9ペア(AUDJPY/NZDJPY/CADJPY/CHFJPY/NZDCHF/CADCHF/
GBPCAD/GBPNZD/EURNZD)を除外している。この除外が「事前基準(流動性)」なのか
「事後成績による選別(=隠れカーブフィット)」なのかを、除外9ペアに同一ルール・
同一コスト規約(往復3pips)でv2を適用した実測で判定する。

手順:
  1) 既存19銘柄のプールを同一コードパスで再生成し、results/mm_pool_v2_H4_19.parquet
     と一致することを検算(分解の土台の正当性確認)。
  2) 除外9ペアをメジャーM1→H4合成(uni.CROSS_DEFS と同じ規約: close合成・OHLC=close代用、
     JPYクロスはpip=0.01、その他0.0001、スプレッド3pips)してv2トレードを生成。
  3) グループ比較(メジャー7 / 採用12クロス / 除外9 / 除外8=AUDJPY以外 / AUDJPY単独):
     mean bps / PF / 勝率 / 年次プラス数。ブートストラップで平均差のCI。
  4) 全28ペア希釈プールの総純益・年次(全暦年プラス維持か)。z-power加重(本番サイジング)でも確認。
  5) コスト感応度: 除外9のスプレッドを2倍(6pips)にしても結論が変わらないか。

実行: uv run python -m research.experiments.anatomy_gap_1
注意: 既存ファイルは一切変更しない。uni.CROSS_DEFS / config.SPREADS_PIPS の拡張は
      本プロセス内のみ(ディスクに書かない)。results/ への書き込みもしない。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from fxlab import config, universe as uni  # noqa: E402
from fxlab.backtest import run  # noqa: E402
from fxlab.trades import trade_table  # noqa: E402
from strategies.confluence_meanrev_v2 import PARAMS as V2_PARAMS, generate_signals  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

TF = "H4"
CROSS_SPREAD = 3.0
POOL_PATH = config.RESULTS_DIR / "mm_pool_v2_H4_19.parquet"

MAJORS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]

# 既存 CROSS_DEFS のスナップショット(拡張前)。AUDJPY は定義済みだがプール除外。
INCLUDED_CROSSES = [c for c in uni.CROSS_DEFS if c != "AUDJPY"]  # 12本

# 除外9ペアのうち AUDJPY 以外の8本をメジャーから合成(慣習的なクォート方向=
# JPYが終値通貨、報告書05の JPYCHF/JPYCAD 逆建てによるコスト単位アーティファクトを回避)
EXTRA_CROSS_DEFS: dict[str, tuple[str, str, str]] = {
    "NZDJPY": ("NZDUSD", "*", "USDJPY"),
    "CADJPY": ("USDJPY", "/", "USDCAD"),
    "CHFJPY": ("USDJPY", "/", "USDCHF"),
    "NZDCHF": ("NZDUSD", "*", "USDCHF"),
    "CADCHF": ("USDCHF", "/", "USDCAD"),
    "GBPCAD": ("GBPUSD", "*", "USDCAD"),
    "GBPNZD": ("GBPUSD", "/", "NZDUSD"),
    "EURNZD": ("EURUSD", "/", "NZDUSD"),
}
EXCLUDED9 = ["AUDJPY"] + list(EXTRA_CROSS_DEFS)


def _zscore(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def build_trades(name: str) -> pd.DataFrame:
    """mm_lab.build_pool と同一規約で1銘柄のv2トレード表を生成(キャッシュ不使用)。"""
    params = dict(V2_PARAMS)
    win = params.get("window", 50)
    data = uni.instrument_data(name, TF)
    pf = run(name, TF, generate_signals, params, data=data, size_mode="value")
    tt = trade_table(pf, data)
    if tt.empty:
        return pd.DataFrame()
    close = data["close"]
    z = _zscore(close, win)
    vol = close.pct_change().rolling(20).std()
    return pd.DataFrame({
        "instr": name,
        "entry": tt["entry"].to_numpy(),
        "exit": tt["exit"].to_numpy(),
        "dir": np.where(tt["dir"].to_numpy() == "Long", 1, -1),
        "entry_price": tt["entry_price"].to_numpy(),
        "ret": tt["return_pct"].to_numpy() / 100.0,
        "bars_held": tt["bars_held"].to_numpy(),
        "z_entry": np.abs(z.reindex(tt["entry"]).to_numpy()),
        "vol_entry": vol.reindex(tt["entry"]).to_numpy(),
    })


def build_pool(instruments: list[str]) -> pd.DataFrame:
    frames = [build_trades(nm) for nm in instruments]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("entry").reset_index(drop=True)


def zpow(z: np.ndarray) -> np.ndarray:
    """本番 z-power サイジング指数(P=4.0)。トレードウェイト ∝ f(z_entry)。"""
    return np.clip((np.asarray(z, dtype=float) / 2.2) ** 4.0, 0.3, 3.0)


def yearly_sums(pool: pd.DataFrame, col: str = "ret") -> pd.Series:
    return pool.groupby(pd.to_datetime(pool["exit"]).dt.year)[col].sum()


def stats(pool: pd.DataFrame, label: str) -> dict:
    r = pool["ret"]
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    y = yearly_sums(pool)
    return {
        "group": label,
        "n": len(r),
        "sum_ret": round(float(r.sum()), 4),
        "mean_bps": round(float(r.mean() * 1e4), 1),
        "win_%": round(float((r > 0).mean() * 100), 1),
        "PF": round(float(gp / gl), 3) if gl > 0 else np.inf,
        "med_bars": float(pool["bars_held"].median()),
        "pos_years": f"{int((y > 0).sum())}/{len(y)}",
    }


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 20000, seed: int = 7):
    """mean(a)-mean(b) のブートストラップ 95%CI と両側p値(差=0に対する)。"""
    rng = np.random.default_rng(seed)
    da = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    db = rng.choice(b, size=(n_boot, len(b)), replace=True).mean(axis=1)
    diff = da - db
    lo, hi = np.percentile(diff, [2.5, 97.5])
    obs = a.mean() - b.mean()
    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    return obs, lo, hi, p


def main() -> None:
    # --- セットアップ: 合成クロス定義の拡張(プロセス内のみ)+スプレッド登録 ---
    uni.CROSS_DEFS.update(EXTRA_CROSS_DEFS)
    uni.register_cross_spreads(CROSS_SPREAD)  # 21クロス全部に3pips

    # ========== 1) 既存19銘柄プールの再生成と検算 ==========
    print("=" * 90)
    print("[1] 既存19銘柄プールの再生成 vs results/mm_pool_v2_H4_19.parquet(検算)")
    print("=" * 90)
    ref = pd.read_parquet(POOL_PATH)
    pool19 = build_pool(MAJORS + INCLUDED_CROSSES)

    print(f"参照: n={len(ref)}  sum_ret={ref['ret'].sum():+.4f}")
    print(f"再生: n={len(pool19)}  sum_ret={pool19['ret'].sum():+.4f}")
    m = ref.merge(pool19, on=["instr", "entry"], suffixes=("_ref", "_new"), how="outer",
                  indicator=True)
    n_unmatched = int((m["_merge"] != "both").sum())
    max_dret = float((m.loc[m._merge == "both", "ret_ref"]
                      - m.loc[m._merge == "both", "ret_new"]).abs().max())
    print(f"(instr,entry) 不一致行: {n_unmatched} / ret の最大絶対差(マッチ行): {max_dret:.2e}")
    ok = (len(ref) == len(pool19)) and n_unmatched == 0 and max_dret < 1e-9
    print(f"検算: {'PASS(完全一致)' if ok else 'FAIL または近似一致 — 下記グループ集計は再生成プール基準'}")

    # ========== 2) 除外9ペアに同一ルールを適用 ==========
    print()
    print("=" * 90)
    print("[2] 除外9ペア(同一v2ルール・同一コスト規約3pips)の成績")
    print("=" * 90)
    pool_ex = build_pool(EXCLUDED9)
    rows = []
    for nm in EXCLUDED9:
        sub = pool_ex[pool_ex["instr"] == nm]
        if sub.empty:
            rows.append({"group": nm, "n": 0})
            continue
        rows.append(stats(sub, nm))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n[2b] 除外9ペア × 決済年 sum(ret) マトリクス:")
    ymat = pool_ex.assign(year=pd.to_datetime(pool_ex["exit"]).dt.year) \
        .pivot_table(index="year", columns="instr", values="ret", aggfunc="sum").round(4)
    print(ymat.reindex(columns=EXCLUDED9).to_string())

    # ========== 3) グループ比較 ==========
    print()
    print("=" * 90)
    print("[3] グループ比較(再生成プール基準・コスト規約同一)")
    print("=" * 90)
    g_major = pool19[pool19["instr"].isin(MAJORS)]
    g_inc12 = pool19[pool19["instr"].isin(INCLUDED_CROSSES)]
    g_audjpy = pool_ex[pool_ex["instr"] == "AUDJPY"]
    g_ex8 = pool_ex[pool_ex["instr"] != "AUDJPY"]
    tbl = pd.DataFrame([
        stats(pool19, "採用19(=現行プール)"),
        stats(g_major, "  └ メジャー7"),
        stats(g_inc12, "  └ 採用クロス12"),
        stats(pool_ex, "除外9(監査対象)"),
        stats(g_audjpy, "  └ AUDJPY(事後除外の疑い)"),
        stats(g_ex8, "  └ 除外8(定義すら無し)"),
    ])
    print(tbl.to_string(index=False))

    obs, lo, hi, p = bootstrap_mean_diff(g_ex8["ret"].to_numpy(), g_inc12["ret"].to_numpy())
    print(f"\n平均リターン差 [除外8 - 採用12クロス]: {obs*1e4:+.1f}bps  "
          f"95%CI [{lo*1e4:+.1f}, {hi*1e4:+.1f}]bps  boot-p={p:.3f}")
    obs, lo, hi, p = bootstrap_mean_diff(pool_ex["ret"].to_numpy(), g_inc12["ret"].to_numpy())
    print(f"平均リターン差 [除外9 - 採用12クロス]: {obs*1e4:+.1f}bps  "
          f"95%CI [{lo*1e4:+.1f}, {hi*1e4:+.1f}]bps  boot-p={p:.3f}")

    # --- 構造仮説の域外検証: AUDJPY除外の根拠「キャリー/リスクオンJPYクロスは
    #     強トレンドし平均回帰に不適」(reports/04, AUDJPYの事後観察から形成)が、
    #     一度も検証されていない NZDJPY/CADJPY にも当てはまるか ---
    print("\n[3b] 構造仮説(キャリーJPYクロス不適)の域外検証 + 採用境界の重なり")
    CARRY_JPY = ["AUDJPY", "NZDJPY", "CADJPY"]
    g_cjpy = pool_ex[pool_ex["instr"].isin(CARRY_JPY)]
    g_ex_other6 = pool_ex[~pool_ex["instr"].isin(CARRY_JPY)]
    print(pd.DataFrame([
        stats(g_cjpy, "除外キャリーJPY3(AUD/NZD/CADJPY)"),
        stats(g_ex_other6, "除外その他6"),
    ]).to_string(index=False))
    obs, lo, hi, p = bootstrap_mean_diff(g_ex_other6["ret"].to_numpy(), g_inc12["ret"].to_numpy())
    print(f"平均リターン差 [除外その他6 - 採用12クロス]: {obs*1e4:+.1f}bps  "
          f"95%CI [{lo*1e4:+.1f}, {hi*1e4:+.1f}]bps  boot-p={p:.3f}")

    print("\n[3c] 銘柄単位の分布(採用と除外の境界が成績で切れているか):")
    per = []
    for nm in MAJORS + INCLUDED_CROSSES:
        per.append(stats(pool19[pool19["instr"] == nm], nm) | {"side": "採用"})
    for nm in EXCLUDED9:
        per.append(stats(pool_ex[pool_ex["instr"] == nm], nm) | {"side": "除外"})
    per_df = pd.DataFrame(per).sort_values("mean_bps", ascending=False)
    print(per_df[["group", "side", "n", "mean_bps", "PF", "pos_years"]].to_string(index=False))

    # ========== 4) 全28ペア希釈プール ==========
    print()
    print("=" * 90)
    print("[4] 全28ペア希釈プール(採用19+除外9)— 等加重・年次")
    print("=" * 90)
    pool28 = pd.concat([pool19, pool_ex], ignore_index=True).sort_values("entry").reset_index(drop=True)
    print(pd.DataFrame([stats(pool19, "採用19"), stats(pool28, "全28")]).to_string(index=False))
    y19, yex, y28 = yearly_sums(pool19), yearly_sums(pool_ex), yearly_sums(pool28)
    ytab = pd.DataFrame({"採用19": y19, "除外9": yex, "全28": y28}).round(4)
    ytab["28プラス?"] = np.where(ytab["全28"] > 0, "+", "-")
    print("\n決済年ベースの sum(ret):")
    print(ytab.to_string())
    print(f"\n全28: 全暦年プラス維持 = {bool((y28 > 0).all())} "
          f"({int((y28 > 0).sum())}/{len(y28)}) / 除外9単独: {int((yex > 0).sum())}/{len(yex)}年プラス")

    # --- 25銘柄プール: 唯一の構造ルール「キャリーJPY除外」だけ適用した場合 ---
    print("\n[4c] 25銘柄プール(19 + 除外その他6 / キャリーJPY3のみ除外)")
    CARRY_JPY = ["AUDJPY", "NZDJPY", "CADJPY"]
    pool25 = pd.concat([pool19, pool_ex[~pool_ex["instr"].isin(CARRY_JPY)]],
                       ignore_index=True).sort_values("entry").reset_index(drop=True)
    print(pd.DataFrame([stats(pool25, "25銘柄")]).to_string(index=False))
    y25 = yearly_sums(pool25)
    print("年次 sum(ret):", {int(k): round(float(v), 4) for k, v in y25.items()})
    print(f"25銘柄: 全暦年プラス = {bool((y25 > 0).all())} ({int((y25 > 0).sum())}/{len(y25)})")

    # --- z-power 加重(本番サイジングの簡易版): ret_w = ret * f(z)/mean(f) ---
    print("\n[4b] z-power加重(P=4.0, 本番ウェイト比例)での確認")
    for label, pl in [("採用19", pool19), ("除外9", pool_ex), ("全28", pool28),
                      ("25銘柄", pool25)]:
        f = zpow(pl["z_entry"].to_numpy())
        rw = pl["ret"].to_numpy() * f / f.mean()
        pw = pl.assign(ret=rw)
        s = stats(pw, label)
        print(f"  {label:7s}: sum_w={s['sum_ret']:+.4f}  mean_w={s['mean_bps']:+.1f}bps  "
              f"PF_w={s['PF']}  pos_years={s['pos_years']}")

    # ========== 5) コスト感応度: 除外9のスプレッド2倍(6pips) ==========
    print()
    print("=" * 90)
    print("[5] コスト感応度 — 除外9ペアをスプレッド6pips(2倍)で再生成")
    print("=" * 90)
    for nm in EXCLUDED9:
        config.SPREADS_PIPS[nm] = 6.0
    pool_ex6 = build_pool(EXCLUDED9)
    for nm in EXCLUDED9:
        config.SPREADS_PIPS[nm] = CROSS_SPREAD  # 戻す
    print(pd.DataFrame([stats(pool_ex, "除外9 @3pips"),
                        stats(pool_ex6, "除外9 @6pips")]).to_string(index=False))
    yex6 = yearly_sums(pool_ex6)
    print(f"除外9 @6pips: sum={pool_ex6['ret'].sum():+.4f} / "
          f"プラス年 {int((yex6 > 0).sum())}/{len(yex6)}")

    # ========== 6) 検算サマリ ==========
    print()
    print("=" * 90)
    print("[6] 検算: 分解の合計=全体")
    print("=" * 90)
    parts = g_major["ret"].sum() + g_inc12["ret"].sum()
    print(f"メジャー7({g_major['ret'].sum():+.4f}) + 採用12クロス({g_inc12['ret'].sum():+.4f}) "
          f"= {parts:+.4f}  vs 採用19合計 {pool19['ret'].sum():+.4f}  "
          f"差={abs(parts - pool19['ret'].sum()):.2e}")
    parts28 = pool19["ret"].sum() + pool_ex["ret"].sum()
    print(f"採用19({pool19['ret'].sum():+.4f}) + 除外9({pool_ex['ret'].sum():+.4f}) "
          f"= {parts28:+.4f}  vs 全28合計 {pool28['ret'].sum():+.4f}  "
          f"差={abs(parts28 - pool28['ret'].sum()):.2e}")


if __name__ == "__main__":
    main()
