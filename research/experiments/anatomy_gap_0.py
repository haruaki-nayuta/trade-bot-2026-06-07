"""anatomy_gap_0: max_pos=8 オーバーフロー・スキップの実シミュレーション。

盲点: 本番規約 max_pos=8（溢れはスキップ）の下で、トレードプール(1214件)の
どのトレードが実際に「採用されない」かを一度も計測していない。
プールの最大同時オープンは件数12本 / z-power加重15.0ユニットで上限8を超える。

3つの採用規約で entry 昇順走査の逐次シミュレーションを行う:
  (a) count    : open件数 < 8 なら採用（同時刻はinstr名順=恣意性最小の固定順）
  (b) weight   : Σf(z) + f(z_new) <= 8 なら採用（z-power加重ベース）
  (c) zpriority: countと同じ容量だが、同時刻バーの複数シグナルは|z_entry|降順で優先

イグジット解放の規約: exit <= t のポジションを時刻tのエントリー処理前に解放
（H4バー終値で決済・同バー終値で新規、決済処理が先 = 本番botの自然な実装）。
感度確認として exit < t（同バー解放なし=厳しい側）も count 規約で併測する。

実行: uv run python -m research.experiments.anatomy_gap_0
価格データ不要・pandasのみ。
"""

from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "results" / "mm_pool_v2_H4_19.parquet"

MAX_POS = 8.0
BASE_SUM = 1.9086  # 検算用ベースライン


def fz(z: np.ndarray | float) -> np.ndarray | float:
    """z-power サイジング f(z) = clip((|z|/2.2)^4.0, 0.3, 3.0)"""
    return np.clip((np.abs(z) / 2.2) ** 4.0, 0.3, 3.0)


def pf(ret: pd.Series) -> float:
    gains = ret[ret > 0].sum()
    losses = -ret[ret <= 0].sum()
    return gains / losses if losses > 0 else np.inf


def max_concurrency(df: pd.DataFrame) -> tuple[int, float]:
    """無制約プールの最大同時オープン（件数, z-power加重ユニット）"""
    events = []
    for r in df.itertuples():
        w = float(fz(r.z_entry))
        events.append((r.entry, 0, +1, +w))  # entry。同時刻はexit(コード1)が先
        events.append((r.exit, 1, -1, -w))
    # 同時刻: exit(sort_key=0)を先に処理して解放 → entry(1)
    events.sort(key=lambda e: (e[0], -e[1]))
    cnt = wgt = 0.0
    max_cnt, max_wgt = 0, 0.0
    for _, _, dc, dw in events:
        cnt += dc
        wgt += dw
        max_cnt = max(max_cnt, int(cnt))
        max_wgt = max(max_wgt, wgt)
    return max_cnt, max_wgt


def simulate(df: pd.DataFrame, rule: str, free_same_bar: bool = True) -> pd.Series:
    """entry昇順走査で採用/スキップを決める。戻り値: bool Series(index=df.index, True=採用)"""
    d = df.sort_values(["entry", "instr"])
    admitted = pd.Series(False, index=df.index)
    open_heap: list[tuple[pd.Timestamp, int, float]] = []  # (exit, tiebreak, weight)
    n_open = 0
    w_open = 0.0
    tie = 0
    for t, grp in d.groupby("entry", sort=True):
        # 解放: exit <= t（free_same_bar=False なら exit < t）
        while open_heap and (
            open_heap[0][0] <= t if free_same_bar else open_heap[0][0] < t
        ):
            _, _, w = heapq.heappop(open_heap)
            n_open -= 1
            w_open -= w
        if rule == "zpriority":
            grp = grp.sort_values("z_entry", key=np.abs, ascending=False)
        for idx, row in grp.iterrows():
            w = float(fz(row["z_entry"]))
            if rule == "weight":
                ok = w_open + w <= MAX_POS + 1e-9
            else:  # count / zpriority
                ok = n_open < int(MAX_POS)
            if ok:
                admitted.loc[idx] = True
                tie += 1
                heapq.heappush(open_heap, (row["exit"], tie, w))
                n_open += 1
                w_open += w
    return admitted


def describe(df: pd.DataFrame, mask: pd.Series, mean_f_pool: float) -> dict:
    adm, skp = df[mask], df[~mask]
    yearly = adm.groupby(adm["exit"].dt.year)["ret"].sum()
    d = {
        "n_admitted": len(adm),
        "n_skipped": len(skp),
        "sum_admitted": adm["ret"].sum(),
        "sum_skipped": skp["ret"].sum(),
        "skipped_pct_of_total": 100.0 * skp["ret"].sum() / df["ret"].sum(),
        "mean_bps_admitted": 1e4 * adm["ret"].mean(),
        "winrate_admitted": 100.0 * (adm["ret"] > 0).mean(),
        "pf_admitted": pf(adm["ret"]),
        "min_year_sum": yearly.min(),
        "min_year": int(yearly.idxmin()),
        "n_neg_years": int((yearly < 0).sum()),
        "yearly": yearly,
        # z-power加重（プール平均f(z)で正規化=本番加重近似、admitted部分和）
        "wsum_admitted": (adm["ret"] * fz(adm["z_entry"])).sum() / mean_f_pool,
        "wsum_skipped": (skp["ret"] * fz(skp["z_entry"])).sum() / mean_f_pool,
    }
    return d


def main() -> None:
    df = pd.read_parquet(POOL)
    assert len(df) == 1214
    total = df["ret"].sum()
    f_all = fz(df["z_entry"].to_numpy())
    mean_f = f_all.mean()
    wsum_total = (df["ret"].to_numpy() * f_all).sum() / mean_f

    print(f"pool: n={len(df)} sum(ret)={total:+.4f} (baseline {BASE_SUM:+.4f})")
    print(f"z-power加重 ret_w 合計={wsum_total:+.4f} (mean f(z)={mean_f:.4f})")
    mc, mw = max_concurrency(df)
    print(f"無制約の最大同時オープン: {mc}本 / 加重 {mw:.2f}ユニット\n")

    # vol四分位（プール全体で算出）
    df = df.copy()
    df["vol_q"] = pd.qcut(df["vol_entry"], 4, labels=["Q1", "Q2", "Q3", "Q4"])

    rules = [
        ("a_count", "count", True),
        ("b_weight", "weight", True),
        ("c_zpriority", "zpriority", True),
        ("a_count_strict", "count", False),  # 同バー解放なし（感度確認）
    ]
    results = {}
    for name, rule, free in rules:
        mask = simulate(df, rule, free_same_bar=free)
        # 検算: admitted + skipped = 全体
        chk = df.loc[mask, "ret"].sum() + df.loc[~mask, "ret"].sum()
        assert abs(chk - total) < 1e-12, name
        results[name] = (mask, describe(df, mask, mean_f))

    print("=" * 100)
    print("採用シミュレーション結果（検算: admitted+skipped sum = 全体 +1.9086 を全規約で確認済み）")
    print("=" * 100)
    hdr = (
        f"{'規約':<16}{'skip件数':>8}{'skip sum':>10}{'総純益比':>9}"
        f"{'採用sum':>10}{'採用bps':>8}{'採用勝率':>9}{'採用PF':>8}{'最悪年':>14}{'負け年数':>8}"
    )
    print(hdr)
    for name, (mask, d) in results.items():
        print(
            f"{name:<16}{d['n_skipped']:>8}{d['sum_skipped']:>+10.4f}"
            f"{d['skipped_pct_of_total']:>8.1f}%{d['sum_admitted']:>+10.4f}"
            f"{d['mean_bps_admitted']:>8.1f}{d['winrate_admitted']:>8.1f}%"
            f"{d['pf_admitted']:>8.3f}"
            f"{d['min_year']:>7}:{d['min_year_sum']:>+6.3f}{d['n_neg_years']:>8}"
        )

    print()
    print("z-power加重（本番加重近似 ret_w、プール平均f(z)正規化）:")
    print(f"  プール全体 ret_w 合計 = {wsum_total:+.4f}")
    for name, (mask, d) in results.items():
        loss = 100.0 * d["wsum_skipped"] / wsum_total
        print(
            f"  {name:<16} 採用={d['wsum_admitted']:+.4f}  "
            f"スキップ={d['wsum_skipped']:+.4f} (加重合計の{loss:.1f}%)"
        )

    # ---- スキップの分布（主規約 a_count と b_weight を詳細に）----
    for name in ["a_count", "b_weight", "c_zpriority"]:
        mask, _ = results[name]
        skp = df[~mask]
        if len(skp) == 0:
            print(f"\n[{name}] スキップ 0 件")
            continue
        print("\n" + "-" * 100)
        print(f"[{name}] スキップ {len(skp)}件 sum={skp['ret'].sum():+.4f} の分布")
        print("-" * 100)
        by_year = skp.groupby(skp["entry"].dt.year).agg(
            n=("ret", "size"), sum_ret=("ret", "sum")
        )
        print("エントリー年別:")
        for y, r in by_year.iterrows():
            print(f"  {y}: n={int(r['n']):>3}  sum={r['sum_ret']:+.4f}")
        by_q = skp.groupby("vol_q", observed=False).agg(
            n=("ret", "size"), sum_ret=("ret", "sum")
        )
        pool_q = df.groupby("vol_q", observed=False).agg(
            n_pool=("ret", "size"), sum_pool=("ret", "sum")
        )
        print("vol_entry四分位別（skip率 = skip件数/プール件数）:")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            n_s = int(by_q.loc[q, "n"]) if q in by_q.index else 0
            s_s = by_q.loc[q, "sum_ret"] if q in by_q.index else 0.0
            n_p = int(pool_q.loc[q, "n_pool"])
            s_p = pool_q.loc[q, "sum_pool"]
            print(
                f"  {q}: skip n={n_s:>3} ({100*n_s/n_p:4.1f}%)  "
                f"skip sum={s_s:+.4f}  (プール{q}: n={n_p}, sum={s_p:+.4f})"
            )
        # 月別クラスタ（skip件数トップ8の年月）
        ym = skp.groupby(skp["entry"].dt.to_period("M")).agg(
            n=("ret", "size"), sum_ret=("ret", "sum")
        ).sort_values("n", ascending=False)
        print("skip集中の年月トップ8:")
        for p, r in ym.head(8).iterrows():
            print(f"  {p}: n={int(r['n']):>2}  sum={r['sum_ret']:+.4f}")
        # COVID窓
        cov = skp[(skp["entry"] >= "2020-02-01") & (skp["entry"] < "2020-06-01")]
        cov_pool = df[(df["entry"] >= "2020-02-01") & (df["entry"] < "2020-06-01")]
        print(
            f"COVID窓(2020-02〜05 entry): skip n={len(cov)} sum={cov['ret'].sum():+.4f} "
            f"/ プール同窓 n={len(cov_pool)} sum={cov_pool['ret'].sum():+.4f}"
        )
        # スキップトレード自体の質
        print(
            f"スキップトレードの質: 平均{1e4*skp['ret'].mean():+.1f}bps "
            f"勝率{100*(skp['ret']>0).mean():.1f}% "
            f"平均|z|={skp['z_entry'].abs().mean():.2f} "
            f"(採用側 平均|z|={df.loc[mask,'z_entry'].abs().mean():.2f})"
        )

    # ---- 年次テーブル（採用プール、決済年）----
    print("\n" + "=" * 100)
    print("採用プールの年次 sum(ret)（決済年集計） vs プール全体")
    print("=" * 100)
    base_y = df.groupby(df["exit"].dt.year)["ret"].sum()
    cols = ["a_count", "b_weight", "c_zpriority", "a_count_strict"]
    print(f"{'年':<6}{'プール':>9}" + "".join(f"{c:>16}" for c in cols))
    for y in base_y.index:
        row = f"{y:<6}{base_y[y]:>+9.3f}"
        for c in cols:
            v = results[c][1]["yearly"].get(y, 0.0)
            row += f"{v:>+16.3f}"
        print(row)
    print(f"{'合計':<6}{base_y.sum():>+9.3f}" + "".join(
        f"{results[c][1]['sum_admitted']:>+16.3f}" for c in cols))


if __name__ == "__main__":
    main()
