"""nb_weekend_gap — 週末ギャップ下方向の埋め(イベント戦略)の敵対的検証

実行: uv run python -m research.experiments.nb_weekend_gap

定義:
 - 週明け最初の M5 バー = 直前バーとの時間差 > 12h
 - gap = (week_open_bar.open - 金曜最終close) / pip
 - down イベント: gap <= そのペアの train(<2023) gap 25% 分位 → バー close でロング
 - up イベント:   gap >= train 75% 分位 → fade ショート(対称性チェック)
 - 保有: h ∈ {1,5,10,20} バー(close-to-close)

敵対的チェック:
 1. 7ペア個別 train/test 再現(イベント数・平均pips・t)
 2. 7ペア合算 test(raw pips + ペア別 train イベント標準偏差で正規化した z)
 3. スプレッドストレス: 日曜オープン実効往復 2/3/5 pips での純期待値
    (注意: イベントは全て UTC 21時前後=ロールオーバー帯なので「20-23時除外」は
     イベント全滅と同義。代わりに↓のエントリー遅延感度+コストストレスを主軸にする)
 4. エントリー遅延感度: d ∈ {0,1,2,3,6,12} バー遅らせて入る。BID スプレッド回復
    アーティファクト(日曜オープンの BID 安値が見かけのギャップ/埋めを水増し)なら
    エッジは最初の 1-2 バーに集中して消えるはず
 5. 上方向ギャップ fade が test で死んでいるか(対称性)
 6. 年別分解・外れ値耐性(median / win率 / 5%トリム)

結論(2026-06-11 実行):
 - 表面上は超強力: down-gap fill は test で 7/7 ペアプラス、プール test n=412
   h10 +6.7p (t=15.2) / h20 +11.7p (t=15.9)、z 正規化でも同等、全年プラス、
   外れ値耐性あり(median +12.1p、5%トリムでも不変)
 - しかし敵対的コントロール [7] で過半が崩れる:
   (a) ギャップなし(mid バケット)でも test h20 +5.8p (t=15.7)。7ペア全部・
       USD買い側/売り側の両方が同時にプラス = mid 価格の現象としては成立不能。
       既知の「ロールオーバー帯 BID スプレッド拡大→回復」アーティファクトの
       日曜拡大版。イベントは全件 UTC 21-22 時オープンなので 20-23 時除外は
       イベント全滅と同義=本プロトコルの主判定基準では生存不能
   (b) train→test で gap 分布自体が負方向にシフト: USDJPY median -1.3→-6.9p、
       USDCHF -1.4→-6.6p。USDJPY が急騰した 2023-25 年に週明け BID が毎週
       下ギャップ=価格でなくスプレッド(データ品質レジーム)起源。
       train q25 閾値の test 該当率が 25%→48-49% に崩壊(較正が壊れている)
   (c) 用量反応ほぼフラット(down 内 corr(gap, r20) プール rho=-0.07)
 - ギャップ固有の増分 = down − mid: test h10 +3.6p / h20 +5.9p(train +4.0p)。
   30分遅延後も down d6→h20 +5.2p vs mid +3.1p で増分 +2.1p は残存
   → 完全な偽物ではないが小さい
 - 実行(ASK買い→BID売り)で取れるのは増分側のみ。ユニバーサルドリフト分は
   エントリー時の拡大スプレッド支払いでほぼ相殺される。日曜実効往復 2-3p を
   引いた現実的な残差は +1〜+4p/イベント。BID-only データではこれ以上の
   解像度なし(ask 系列がないため mid 補正不能)
 - up-gap fade は test 死亡(プール h20 ショート -0.8p t=-1.2、各ペア |t|<2)
   = 対称性なし。「ギャップは埋まる」一般則ではない
 - 判定: 不採用(現データでは採用根拠を作れない)。数字の派手さは既知の
   BID アーティファクトと不可分。復活条件 = デモ口座で日曜オープンの実 ask/bid
   約定を数十イベント実測し、down−mid 増分が実コスト後も正と確認できた場合のみ
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.lab.nextbar_common import SPLIT, load_xy

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
COST_RT = {  # 通常時の往復スプレッド(pips)
    "EURUSD": 0.6, "USDJPY": 0.7, "GBPUSD": 0.9, "AUDUSD": 0.8,
    "USDCHF": 1.0, "USDCAD": 1.2, "NZDUSD": 1.4,
}
HORIZONS = (1, 5, 10, 20)
DELAYS = (0, 1, 2, 3, 6, 12)
STRESS = (2.0, 3.0, 5.0)


def tstat(y: np.ndarray) -> float:
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) < 3 or y.std(ddof=1) == 0:
        return np.nan
    return float(y.mean() / (y.std(ddof=1) / np.sqrt(len(y))))


def stat_str(y) -> str:
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) < 3:
        return f"n={len(y)}<3"
    return f"{y.mean():+6.2f}p (t={tstat(y):+4.1f}, n={len(y)})"


def pair_events(pair: str) -> pd.DataFrame:
    """週明けオープンイベント表: gap(pips) + 各 horizon/遅延のリターン(pips)。"""
    df, _tgt, pip = load_xy(pair, "M5")
    c = df["close"]
    ts = pd.Series(df.index, index=df.index)
    is_open = (ts - ts.shift(1)) > pd.Timedelta(hours=12)
    gap = ((df["open"] - c.shift(1)) / pip).where(is_open)
    ev = pd.DataFrame({"gap": gap.dropna()})
    for h in HORIZONS:
        ev[f"r{h}"] = ((c.shift(-h) - c) / pip).reindex(ev.index)
    for d in DELAYS:
        # 遅延エントリー: t+d の close で建てて 10 バー保有
        ev[f"d{d}_h10"] = ((c.shift(-(d + 10)) - c.shift(-d)) / pip).reindex(ev.index)
        # 遅延エントリーから h20 アンカーまでの残存(埋めの何割が初動に集中するか)
        ev[f"d{d}_to20"] = ((c.shift(-20) - c.shift(-d)) / pip).reindex(ev.index)
    ev["hour"] = ev.index.hour
    ev["pair"] = pair
    return ev


def main() -> None:
    allev: dict[str, pd.DataFrame] = {}
    thr_dn: dict[str, float] = {}
    thr_up: dict[str, float] = {}
    sd_train: dict[str, dict[str, float]] = {}

    print("=" * 110)
    print("[0] イベント抽出(週明け最初の M5 バー、直前バーと >12h)")
    for p in PAIRS:
        ev = pair_events(p)
        tr = ev.index < SPLIT
        thr_dn[p] = float(ev.loc[tr, "gap"].quantile(0.25))
        thr_up[p] = float(ev.loc[tr, "gap"].quantile(0.75))
        # 正規化用: train 全イベントの horizon リターン標準偏差(ペア別)
        sd_train[p] = {
            f"r{h}": float(ev.loc[tr, f"r{h}"].std(ddof=1)) for h in HORIZONS
        }
        allev[p] = ev
        hrs = ev["hour"].value_counts().to_dict()
        print(
            f"  {p}: events total {len(ev)} (train {tr.sum()} / test {(~tr).sum()}), "
            f"train q25={thr_dn[p]:+.1f}p q75={thr_up[p]:+.1f}p, open-hour dist {hrs}"
        )

    print("\n" + "=" * 110)
    print("[1] ペア別: down-gap(<=train q25)ロング — train / test 平均pips と t")
    header = f"  {'pair':<8} {'split':<6} {'n':>4} | " + " | ".join(f"{'h'+str(h):>22}" for h in HORIZONS)
    print(header)
    test_dn: dict[str, pd.DataFrame] = {}
    for p in PAIRS:
        ev = allev[p]
        dn = ev[ev["gap"] <= thr_dn[p]]
        test_dn[p] = dn[dn.index >= SPLIT]
        for st, sub in [("train", dn[dn.index < SPLIT]), ("test", dn[dn.index >= SPLIT])]:
            cells = " | ".join(f"{stat_str(sub[f'r{h}']):>22}" for h in HORIZONS)
            print(f"  {p:<8} {st:<6} {len(sub):>4} | {cells}")

    print("\n" + "=" * 110)
    print("[2] 7ペア合算プール(test, down-gap ロング)")
    pool = pd.concat(test_dn.values())
    for h in HORIZONS:
        raw = pool[f"r{h}"].to_numpy(float)
        z = np.concatenate(
            [test_dn[p][f"r{h}"].to_numpy(float) / sd_train[p][f"r{h}"] for p in PAIRS]
        )
        raw_f = raw[np.isfinite(raw)]
        win = float((raw_f > 0).mean()) if len(raw_f) else np.nan
        print(
            f"  h{h:>2}: raw {stat_str(raw)} | z(train-sd正規化) mean {np.nanmean(z):+.3f} "
            f"(t={tstat(z):+.1f}) | win率 {win:.2f} | median {np.nanmedian(raw):+.2f}p"
        )
    # 外れ値耐性: h20 の 5% トリム平均
    r20 = np.sort(pool["r20"].dropna().to_numpy(float))
    k = max(1, int(len(r20) * 0.05))
    print(
        f"  h20 5%トリム平均 {r20[k:-k].mean():+.2f}p (全体 {r20.mean():+.2f}p) | "
        f"ベスト1件除外 {r20[:-1].mean():+.2f}p"
    )
    # 年別
    yr = pool.assign(y=pool.index.year).groupby("y")
    print("  年別 (h10 / h20):")
    for y, g in yr:
        print(f"    {y}: n={len(g):3d} h10 {stat_str(g['r10'])} | h20 {stat_str(g['r20'])}")

    print("\n" + "=" * 110)
    print("[3] スプレッドストレス: 日曜オープン実効往復コスト別の純期待値(test, 1イベント平均)")
    print(f"  {'pair':<8} {'n':>4} | " + " | ".join(
        [f"{'h10 gross':>10}"] + [f"{'net@'+str(s)+'p':>9}" for s in STRESS]
        + [f"{'h20 gross':>10}"] + [f"{'net@'+str(s)+'p':>9}" for s in STRESS]))
    for p in PAIRS + ["POOL"]:
        sub = pool if p == "POOL" else test_dn[p]
        n = sub["r10"].notna().sum()
        cells = []
        for h in (10, 20):
            g = float(sub[f"r{h}"].mean())
            cells.append(f"{g:>+10.2f}")
            cells += [f"{g - s:>+9.2f}" for s in STRESS]
        print(f"  {p:<8} {n:>4} | " + " | ".join(cells))
    print(f"  (参考) 通常時往復コスト: {COST_RT}")
    # 正規化プールでのストレス t 値(コストをペア毎に z 換算して引く)
    for s in STRESS:
        z = np.concatenate(
            [(test_dn[p]["r20"].to_numpy(float) - s) / sd_train[p]["r20"] for p in PAIRS]
        )
        print(f"  プール h20 net@{s}p: z mean {np.nanmean(z):+.3f} (t={tstat(z):+.1f})")

    print("\n" + "=" * 110)
    print("[4] エントリー遅延感度(test, down-gap, プール) — BID回復アーティファクト切り分け")
    print("  d バー遅らせて建て、10 バー保有 / および d から h20 アンカーまでの残存")
    for d in DELAYS:
        h10 = pool[f"d{d}_h10"]
        rem = pool[f"d{d}_to20"]
        print(
            f"  d={d:>2} ({d*5:>3}分後): hold10 {stat_str(h10)} | to-h20 残存 {stat_str(rem)}"
        )

    print("\n" + "=" * 110)
    print("[5] 対称性: up-gap(>=train q75)fade ショート(test) — ショート損益 = -ret")
    rows = []
    for p in PAIRS:
        ev = allev[p]
        up = ev[(ev["gap"] >= thr_up[p]) & (ev.index >= SPLIT)]
        rows.append(up)
        cells = " | ".join(f"{stat_str(-up[f'r{h}']):>22}" for h in HORIZONS)
        print(f"  {p:<8} n={len(up):>3} | {cells}")
    upool = pd.concat(rows)
    cells = " | ".join(f"{stat_str(-upool[f'r{h}']):>22}" for h in HORIZONS)
    print(f"  {'POOL':<8} n={len(upool):>3} | {cells}")

    print("\n" + "=" * 110)
    print("[6] 参考: down-gap の深さ分布(test 該当イベントの gap pips)と埋め率")
    for p in PAIRS:
        g = test_dn[p]["gap"]
        fill = (test_dn[p]["r20"] / (-test_dn[p]["gap"])).replace([np.inf, -np.inf], np.nan)
        print(
            f"  {p:<8} gap mean {g.mean():+7.1f}p median {g.median():+7.1f}p "
            f"min {g.min():+8.1f}p | h20/|gap| 埋め率 median {fill.median():+.2f}"
        )

    print("\n" + "=" * 110)
    print("[7] 敵対的コントロール: エッジは『ギャップ』由来か『日曜オープン一般』由来か")
    print("  7a. gap 分布の train→test シフト(BID スプレッド構造変化の疑い)")
    print(f"  {'pair':<8} {'tr_mean':>8} {'tr_med':>8} {'tr_%<0':>7} | {'te_mean':>8} {'te_med':>8} {'te_%<0':>7} {'te該当率':>8}")
    for p in PAIRS:
        ev = allev[p]
        g_tr = ev.loc[ev.index < SPLIT, "gap"]
        g_te = ev.loc[ev.index >= SPLIT, "gap"]
        hit = float((g_te <= thr_dn[p]).mean())
        print(
            f"  {p:<8} {g_tr.mean():>+8.2f} {g_tr.median():>+8.2f} {(g_tr < 0).mean():>7.2f} | "
            f"{g_te.mean():>+8.2f} {g_te.median():>+8.2f} {(g_te < 0).mean():>7.2f} {hit:>8.2f}"
        )
    print("  7b. test 週明けバケット別ロング (down=gap<=q25 / mid / up=gap>=q75 / 無条件)")
    print(f"  {'pair':<8} | {'down h10':>22} | {'mid h10':>22} | {'up h10':>22} | {'all h20':>22}")
    mids, alls = [], []
    for p in PAIRS:
        ev = allev[p]
        te = ev[ev.index >= SPLIT]
        dn = te[te["gap"] <= thr_dn[p]]
        md = te[(te["gap"] > thr_dn[p]) & (te["gap"] < thr_up[p])]
        up = te[te["gap"] >= thr_up[p]]
        mids.append(md)
        alls.append(te)
        print(
            f"  {p:<8} | {stat_str(dn['r10']):>22} | {stat_str(md['r10']):>22} | "
            f"{stat_str(up['r10']):>22} | {stat_str(te['r20']):>22}"
        )
    mpool, apool, upool2 = pd.concat(mids), pd.concat(alls), pd.concat(
        [allev[p][(allev[p]["gap"] >= thr_up[p]) & (allev[p].index >= SPLIT)] for p in PAIRS]
    )
    print(
        f"  {'POOL':<8} | {stat_str(pool['r10']):>22} | {stat_str(mpool['r10']):>22} | "
        f"{stat_str(upool2['r10']):>22} | {stat_str(apool['r20']):>22}"
    )
    print(
        f"  POOL h20: down {stat_str(pool['r20'])} | mid {stat_str(mpool['r20'])} | "
        f"up {stat_str(upool2['r20'])}"
    )
    # train 期の同バケット(test>train の regime シフト確認)
    tr_dn = pd.concat([allev[p][(allev[p]["gap"] <= thr_dn[p]) & (allev[p].index < SPLIT)] for p in PAIRS])
    tr_md = pd.concat(
        [allev[p][(allev[p]["gap"] > thr_dn[p]) & (allev[p]["gap"] < thr_up[p]) & (allev[p].index < SPLIT)] for p in PAIRS]
    )
    tr_all = pd.concat([allev[p][allev[p].index < SPLIT] for p in PAIRS])
    print(
        f"  (train) h20: down {stat_str(tr_dn['r20'])} | mid {stat_str(tr_md['r20'])} | "
        f"all {stat_str(tr_all['r20'])}"
    )
    print("  7b' 遅延後の残存ドリフト比較(test): down バケット vs mid バケット")
    for d in (3, 6, 12):
        print(
            f"    d={d:>2}→h20: down {stat_str(pool[f'd{d}_to20'])} | mid {stat_str(mpool[f'd{d}_to20'])}"
        )
    print("  7c. 用量反応: test down イベント内 Spearman corr(gap, r20)(深いほど埋めが大きいなら負)")
    for p in PAIRS + ["POOL"]:
        sub = pool if p == "POOL" else test_dn[p]
        rho = sub["gap"].rank().corr(sub["r20"].rank())
        print(f"  {p:<8} rho={rho:+.3f} (n={len(sub)})")
    print("  7d. down イベントを 21時オープンのみ / 22時オープンのみに分けた test h10/h20")
    for hsel in (21, 22):
        sub = pool[pool["hour"] == hsel]
        print(f"  open@{hsel}時: h10 {stat_str(sub['r10'])} | h20 {stat_str(sub['r20'])}")


if __name__ == "__main__":
    main()
