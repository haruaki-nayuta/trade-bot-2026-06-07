"""critic_audit — トレンドフォロー全滅判定への敵対的監査(方法論クリティック)。

監査項目:
  1. フリップ会計: tsmom D1 lb60 EURUSD を vectorbt 抜きの手書きループで再計算し、
     tl.build_pool(=vectorbt from_signals + trade_table)と突き合わせる。
     ドテン(同バー exit+反対 entry)の二重計上・取りこぼし検出。
     ついでに「次バー寄り付き約定」変種も計算し、同バー終値約定の有利/不利を定量。
  2. 集計マスキング: donch_e55x20_H4 と tsmom_lb60_D1 の銘柄×年 PF 行列・方向分解で
     「一貫してプラスのポケット」を探す。
  3. ボラ正規化重み: 1/vol_entry 加重平均リターン(ATRリスク均等相当)が等加重と符号逆転するか。
  4. コスト妥当性: 手書きループで往復=1フルスプレッドを数値確認 + グロス(コスト0)プールで
     コストドラッグを定量。
  5. プロトコル外バイアス: 銘柄×年のトレンド強度(ドリフトの|t値|)で期間・ユニバースの
     トレンド貧弱さを定量。

実行: PYTHONPATH=. uv run python research/experiments/trend2/critic_audit.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c/research/lab")
import trend_lab as tl  # noqa: E402

from fxlab import config  # noqa: E402
from strategies.tsmom import generate_signals as tsmom_signals  # noqa: E402

OUT = "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c/research/outputs/trend2"


def donchian_close(data: pd.DataFrame, entry_window: int = 55, exit_window: int = 20):
    """research/experiments/trend/exp_donchian.py と同一ロジック(再掲・固定)。"""
    close = data["close"]
    upper = close.rolling(entry_window).max().shift(1)
    lower = close.rolling(entry_window).min().shift(1)
    exit_upper = close.rolling(exit_window).max().shift(1)
    exit_lower = close.rolling(exit_window).min().shift(1)
    return close > upper, close < exit_lower, close < lower, close > exit_upper


# ====================================================================
# 1. フリップ会計の手書き再計算(tsmom D1 lb60, EURUSD)
# ====================================================================
def hand_loop_tsmom(data: pd.DataFrame, pair: str, lookback: int = 60,
                    fill: str = "same_close") -> pd.DataFrame:
    """vectorbt を使わない逐次シミュレーション。
    fill='same_close' : シグナルバーの終値で約定(vectorbt from_signals と同じ想定)
    fill='next_open'  : 翌バー寄り付きで約定(現実寄りの変種)
    買い注文 = price + 半スプレッド / 売り注文 = price - 半スプレッド。
    """
    close = data["close"].to_numpy()
    open_ = data["open"].to_numpy()
    idx = data.index
    half = config.spread_pips(pair) * config.pip_size(pair) / 2.0

    c = data["close"]
    mom = c / c.shift(lookback) - 1.0
    long_state = (mom > 0)
    short_state = (mom < 0)
    le = (long_state & ~long_state.shift(fill_value=False)).to_numpy()
    se = (short_state & ~short_state.shift(fill_value=False)).to_numpy()

    n = len(close)
    trades = []
    pos = 0
    entry_i = -1
    entry_fill = np.nan

    def price_at(i: int, is_buy: bool) -> float | None:
        if fill == "same_close":
            p = close[i]
        else:  # next_open
            if i + 1 >= n:
                return None
            p = open_[i + 1]
        return p + half if is_buy else p - half

    for i in range(n):
        if pos == 1 and se[i]:
            xp = price_at(i, is_buy=False)          # ロング手仕舞い=売り
            if xp is None:
                break
            trades.append((idx[entry_i], idx[i], 1, entry_fill, xp,
                           xp / entry_fill - 1.0))
            pos, entry_i, entry_fill = -1, i, xp    # 同価格で新規ショート(売り)
        elif pos == -1 and le[i]:
            xp = price_at(i, is_buy=True)           # ショート手仕舞い=買い
            if xp is None:
                break
            trades.append((idx[entry_i], idx[i], -1, entry_fill, xp,
                           (entry_fill - xp) / entry_fill))
            pos, entry_i, entry_fill = 1, i, xp     # 同価格で新規ロング(買い)
        elif pos == 0:
            if le[i]:
                p = price_at(i, is_buy=True)
                if p is not None:
                    pos, entry_i, entry_fill = 1, i, p
            elif se[i]:
                p = price_at(i, is_buy=False)
                if p is not None:
                    pos, entry_i, entry_fill = -1, i, p

    if pos != 0:  # 建玉中のまま終了 → vectorbt 同様、最終バー終値(スリッページなし)で評価
        last = close[-1]
        if pos == 1:
            trades.append((idx[entry_i], idx[-1], 1, entry_fill, last,
                           last / entry_fill - 1.0))
        else:
            trades.append((idx[entry_i], idx[-1], -1, entry_fill, last,
                           (entry_fill - last) / entry_fill))

    return pd.DataFrame(trades, columns=["entry", "exit", "dir", "entry_fill",
                                         "exit_fill", "ret"])


def audit_flip_accounting() -> dict:
    print("=" * 70)
    print("1. フリップ会計の手書き再計算 (tsmom D1 lb60, EURUSD)")
    print("=" * 70)
    data = tl.load_tf("EURUSD", "D1")
    pool = tl.build_pool(tsmom_signals, {"lookback": 60, "band": 0.0},
                         tf="D1", side="both", instruments=["EURUSD"])
    hand = hand_loop_tsmom(data, "EURUSD", 60, fill="same_close")
    hand_no = hand_loop_tsmom(data, "EURUSD", 60, fill="next_open")

    res = {
        "vbt_n": len(pool), "hand_n": len(hand),
        "vbt_sum_ret": float(pool["ret"].sum()), "hand_sum_ret": float(hand["ret"].sum()),
        "nextopen_n": len(hand_no), "nextopen_sum_ret": float(hand_no["ret"].sum()),
    }
    # トレード単位の突き合わせ(entry タイムスタンプで結合)
    m = pool.merge(hand, on="entry", suffixes=("_vbt", "_hand"))
    res["matched"] = len(m)
    if len(m):
        res["max_abs_ret_diff"] = float((m["ret_vbt"] - m["ret_hand"]).abs().max())
        res["dir_mismatch"] = int((m["dir_vbt"] != m["dir_hand"]).sum())
        res["exit_mismatch"] = int((pd.to_datetime(m["exit_vbt"]) != pd.to_datetime(m["exit_hand"])).sum())
        res["max_abs_entryfill_diff"] = float(
            (m["entry_price"] - m["entry_fill"]).abs().max())
    # ドテンバーの会計: exit と次トレード entry が同一バーになっている割合
    flips_hand = int((hand["exit"].iloc[:-1].to_numpy() == hand["entry"].iloc[1:].to_numpy()).sum())
    flips_vbt = int((pool["exit"].iloc[:-1].to_numpy() == pool["entry"].iloc[1:].to_numpy()).sum())
    res["flip_bars_hand"] = flips_hand
    res["flip_bars_vbt"] = flips_vbt
    # 同一エントリーバーの重複(二重計上)チェック
    res["dup_entries_vbt"] = int(pool["entry"].duplicated().sum())
    res["dup_entries_hand"] = int(hand["entry"].duplicated().sum())
    # コスト: 往復 = entry半 + exit半 = 1フルスプレッドか(価格で数値確認)
    half = config.spread_pips("EURUSD") * config.pip_size("EURUSD") / 2.0
    closed = hand.iloc[:-1]  # 最終オープントレード除く
    ce = data["close"].reindex(closed["entry"]).to_numpy()
    cx = data["close"].reindex(closed["exit"]).to_numpy()
    cost_entry = np.abs(closed["entry_fill"].to_numpy() - ce)
    cost_exit = np.abs(closed["exit_fill"].to_numpy() - cx)
    res["entry_slip_const"] = bool(np.allclose(cost_entry, half))
    res["exit_slip_const"] = bool(np.allclose(cost_exit, half))
    res["roundtrip_spread_pips"] = float((cost_entry.mean() + cost_exit.mean())
                                         / config.pip_size("EURUSD"))
    for k, v in res.items():
        print(f"  {k}: {v}")
    return res


# ====================================================================
# 2+3+4. プール構築(ネット+グロス)→ 行列・方向分解・ボラ加重
# ====================================================================
def pf_ratio(x: pd.Series) -> float:
    g = x[x > 0].sum()
    l = -x[x < 0].sum()
    if l > 0:
        return float(g / l)
    return float("inf") if g > 0 else float("nan")


def build_all_pools() -> dict:
    fams = {
        "tsmom_lb60_D1": (tsmom_signals, {"lookback": 60, "band": 0.0}, "D1"),
        "donch_e55x20_H4": (donchian_close, {"entry_window": 55, "exit_window": 20}, "H4"),
    }
    pools = {}
    for name, (gen, params, tf) in fams.items():
        pools[name] = tl.build_pool(gen, params, tf=tf, side="both")
        print(f"  pool {name}: n={len(pools[name])}")

    # --- グロス(コスト0)診断 ---
    tl.register_spreads()
    saved = dict(config.SPREADS_PIPS)
    for k in list(config.SPREADS_PIPS):
        config.SPREADS_PIPS[k] = 0.0
    _orig = tl.register_spreads
    tl.register_spreads = lambda: None
    gross = {}
    try:
        for name, (gen, params, tf) in fams.items():
            gross[name] = tl.build_pool(gen, params, tf=tf, side="both")
            print(f"  gross pool {name}: n={len(gross[name])}")
    finally:
        tl.register_spreads = _orig
        config.SPREADS_PIPS.update(saved)
        tl.register_spreads()
    return {"net": pools, "gross": gross}


def audit_masking_and_weights(pools: dict) -> None:
    for name, pool in pools["net"].items():
        gpool = pools["gross"][name]
        print("\n" + "=" * 70)
        print(f"2. 集計マスキング監査: {name}  (n={len(pool)})")
        print("=" * 70)
        p = pool.copy()
        p["year"] = pd.to_datetime(p["exit"]).dt.year

        st = tl.pool_stats(pool)
        print(f"  全体: {st}")

        # --- 銘柄×年 PF 行列 ---
        mat_pf = p.pivot_table(index="instr", columns="year", values="ret", aggfunc=pf_ratio)
        mat_sum = p.pivot_table(index="instr", columns="year", values="ret", aggfunc="sum")
        mat_pf.round(2).to_csv(f"{OUT}/{name}_instr_year_pf.csv")
        mat_sum.round(4).to_csv(f"{OUT}/{name}_instr_year_sum.csv")
        print("\n  銘柄×年 PF 行列 (>1 がプラスポケット):")
        print(mat_pf.round(2).to_string())

        pos_share = (mat_sum > 0).sum(axis=1)
        n_years = mat_sum.notna().sum(axis=1)
        instr_tot = p.groupby("instr")["ret"].agg(["sum", "count", pf_ratio])
        instr_tot.columns = ["sum_ret", "n", "pf"]
        instr_tot["pos_years"] = pos_share
        instr_tot["years"] = n_years
        print("\n  銘柄別トータル (sum_ret 降順):")
        print(instr_tot.sort_values("sum_ret", ascending=False).round(4).to_string())

        # --- 年別(全銘柄) ---
        yr = p.groupby("year")["ret"].agg(["sum", "count", pf_ratio])
        yr.columns = ["sum_ret", "n", "pf"]
        print("\n  年別(全銘柄プール):")
        print(yr.round(4).to_string())

        # --- 方向分解 ---
        print("\n  方向分解(side=long / short のみ):")
        for d, tag in ((1, "long"), (-1, "short")):
            sub = p[p["dir"] == d]
            sst = tl.pool_stats(sub)
            print(f"   [{tag}] {sst}")
        # 方向×年
        dy = p.pivot_table(index="year", columns="dir", values="ret", aggfunc="sum")
        dy.columns = [{1: "long_sum", -1: "short_sum"}.get(c, c) for c in dy.columns]
        print("\n  方向×年 sum_ret:")
        print(dy.round(4).to_string())

        # 銘柄×方向のプラスポケット
        id_ = p.groupby(["instr", "dir"])["ret"].agg(["sum", "count", pf_ratio])
        id_.columns = ["sum_ret", "n", "pf"]
        pockets = id_[(id_["pf"] > 1.0) & (id_["n"] >= 30)].sort_values("pf", ascending=False)
        print("\n  銘柄×方向 PF>1 (n>=30) のポケット:")
        print(pockets.round(4).to_string() if len(pockets) else "   (なし)")
        id_.round(4).to_csv(f"{OUT}/{name}_instr_dir.csv")

        # 銘柄×年×方向で「全年プラス」みたいな一貫ポケットがないか
        consist = instr_tot[(instr_tot["pf"] > 1.0)]
        print("\n  銘柄トータル PF>1:")
        print(consist.round(4).to_string() if len(consist) else "   (なし)")

        # --- 3. ボラ正規化重み付け ---
        print(f"\n3. ボラ正規化重み付け: {name}")
        q = p.dropna(subset=["vol_entry"])
        q = q[q["vol_entry"] > 0]
        w = 1.0 / q["vol_entry"]
        ew = float(q["ret"].mean() * 1e4)
        ww = float((q["ret"] * w).sum() / w.sum() * 1e4)
        # 銘柄内で先に平均→銘柄間等加重(銘柄偏在の補正)も
        instr_mean = q.groupby("instr")["ret"].mean()
        print(f"   等加重 mean_bps           : {ew:8.2f}  (n={len(q)})")
        print(f"   1/vol 加重 mean_bps       : {ww:8.2f}")
        print(f"   銘柄等加重(銘柄内平均の平均): {float(instr_mean.mean()*1e4):8.2f}")
        # 方向別でも
        for d, tag in ((1, "long"), (-1, "short")):
            qq = q[q["dir"] == d]
            if len(qq) == 0:
                continue
            wd = 1.0 / qq["vol_entry"]
            print(f"   [{tag}] 等加重 {float(qq['ret'].mean()*1e4):7.2f} bps"
                  f" / 1/vol加重 {float((qq['ret']*wd).sum()/wd.sum()*1e4):7.2f} bps")

        # --- 4. コストドラッグ(グロス比較) ---
        gst = tl.pool_stats(gpool)
        print(f"\n4. コストドラッグ: {name}")
        print(f"   net : mean_bps={st['mean_bps']:7.2f}  pool_pf={st['pool_pf']}  sum_ret={st['sum_ret']}")
        print(f"   gross: mean_bps={gst['mean_bps']:7.2f}  pool_pf={gst['pool_pf']}  sum_ret={gst['sum_ret']}  (n={gst['n']})")
        print(f"   → コスト drag ≈ {gst['mean_bps'] - st['mean_bps']:.2f} bps/trade")
        # グロスでの銘柄×方向ポケット(エッジの有無はグロスで判定すべき)
        gp = gpool.copy()
        gid = gp.groupby(["instr", "dir"])["ret"].agg(["sum", "count", pf_ratio])
        gid.columns = ["sum_ret", "n", "pf"]
        gpockets = gid[(gid["pf"] > 1.0) & (gid["n"] >= 30)].sort_values("pf", ascending=False)
        print("   グロスでの銘柄×方向 PF>1 (n>=30):")
        print(gpockets.round(4).to_string() if len(gpockets) else "   (なし)")
        # グロスの方向別合計
        for d, tag in ((1, "long"), (-1, "short")):
            sub = gp[gp["dir"] == d]
            print(f"   [gross {tag}] n={len(sub)} sum_ret={sub['ret'].sum():.4f} "
                  f"pf={pf_ratio(sub['ret']):.3f} mean_bps={sub['ret'].mean()*1e4:.2f}")


# ====================================================================
# 5. 期間・ユニバースのトレンド強度(ドリフト |t値|)
# ====================================================================
def audit_universe_trendiness() -> None:
    print("\n" + "=" * 70)
    print("5. ユニバース×期間のトレンド強度 (年次ドリフトの |t値| = |mean|/se of D1 log-ret)")
    print("=" * 70)
    rows = []
    for nm in tl.default_instruments():
        d1 = tl.load_tf(nm, "D1")
        lr = np.log(d1["close"]).diff().dropna()
        for y, g in lr.groupby(lr.index.year):
            if len(g) < 100:
                continue
            t = abs(g.mean()) / (g.std() / np.sqrt(len(g)))
            rows.append({"instr": nm, "year": int(y), "tstat": float(t)})
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="instr", columns="year", values="tstat")
    piv.round(2).to_csv(f"{OUT}/trendiness_tstat.csv")
    yr_mean = df.groupby("year")["tstat"].agg(["mean", lambda s: (s > 2).mean()])
    yr_mean.columns = ["mean_tstat", "share_t>2"]
    print("  年別平均トレンド強度(全20銘柄):")
    print(yr_mean.round(3).to_string())
    fx = df[df["instr"] != "XAUUSD"]
    print(f"\n  FX19銘柄: 平均|t|={fx['tstat'].mean():.2f}, |t|>2 の銘柄年比率={ (fx['tstat']>2).mean():.1%}")
    g = df[df["instr"] == "XAUUSD"]
    if len(g):
        print(f"  XAUUSD : 平均|t|={g['tstat'].mean():.2f}, |t|>2 の年比率={(g['tstat']>2).mean():.1%}")
        print("  XAUUSD 年別 |t|:", {int(r['year']): round(r['tstat'], 2) for _, r in g.iterrows()})


def main() -> None:
    audit_flip_accounting()
    print("\n" + "=" * 70)
    print("プール構築 (net + gross)")
    print("=" * 70)
    pools = build_all_pools()
    audit_masking_and_weights(pools)
    audit_universe_trendiness()
    # StructuredOutput 用に side 分解の要約も CSV 保存
    rows = []
    for kind in ("net", "gross"):
        for name, pool in pools[kind].items():
            for side, sub in (("both", pool),
                              ("long", pool[pool["dir"] == 1]),
                              ("short", pool[pool["dir"] == -1])):
                st = tl.pool_stats(sub)
                st.update({"family": name, "kind": kind, "side": side})
                rows.append(st)
    pd.DataFrame(rows).to_csv(f"{OUT}/critic_side_summary.csv", index=False)
    print(f"\nsaved CSVs under {OUT}/")


if __name__ == "__main__":
    main()
