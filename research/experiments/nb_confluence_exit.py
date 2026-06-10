"""nb_confluence_exit — 合流(confluence)の限界価値と出口設計 (EURUSD M5)。

敵対的検証プロトコル:
- train < 2023-01-01 <= test。閾値・分位は全て train のみから。
- 主軸評価は「UTC 20-23 時エントリー除外」(ロールオーバーの bid スプレッド
  アーティファクト対策)。フル評価も併記する。
- コストは EURUSD 往復 0.6 pips を 1 トレードごとに控除。
- 出口比較・経済性は「非重複トレードシミュレーション」(同時 1 ポジション)で行う。
  シグナル単位の平均×頻度は重複を無視するため、達成可能な日次 pips を過大評価する。

ベース部品(定義は nb_momrev / nb_vol_state / nb_candle_anatomy / nb_volume_activity):
- L0: z10 <= train q02(ただし hiER10 = ER10 > train 2/3 分位の足のみ) … 買い
- S0: z50 >= train q98 … 売り
- S1: ret3_norm = (close.diff(3)/pip) / (std100*sqrt(3)) >= train q98 … 売り
フィルタ候補:
- a: CLV=(close-low)/(high-low) がゾーン内 train q25 以下(ショートは q75 以上)
- b: vr = std(diff,10)/std(diff,100) >= train q75
- c: vrank100 = volume.rolling(100).rank(pct=True) >= 0.8
- d: 直近100本レンジ幅(pips) >= train 2/3 分位
- e: UTC 7-16 時のみ
出口:
- 時間出口 N ∈ {1,3,5,10,20}
- z クロス出口: ロングは z10 >= 0、ショートは z50 <= 0 で決済(最大 40 本で強制)

実行: uv run python -m research.experiments.nb_confluence_exit
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from research.lab.nextbar_common import SPLIT, horizon_targets, load_xy

PAIR = "EURUSD"
COST = 0.6  # 往復 pips
EXCL_HOURS = [20, 21, 22, 23]
ACTIVE_HOURS = list(range(7, 17))
ZCAP = 40


def tstat(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)
    if n < 3 or y.std(ddof=1) == 0:
        return float("nan")
    return float(y.mean() / (y.std(ddof=1) / np.sqrt(n)))


def build_feats(df: pd.DataFrame, pip: float) -> dict[str, pd.Series]:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    dpips = c.diff() / pip
    std100 = dpips.rolling(100).std()
    feats = {
        "z10": (c - c.rolling(10).mean()) / c.rolling(10).std(),
        "z50": (c - c.rolling(50).mean()) / c.rolling(50).std(),
        "er10": c.diff(10).abs() / (dpips.abs().rolling(10).sum() * pip),
        "ret3n": (c.diff(3) / pip) / (std100 * np.sqrt(3)),
        "vr": c.diff().rolling(10).std() / c.diff().rolling(100).std(),
        "vrank": v.rolling(100).rank(pct=True),
        "range100": (h.rolling(100).max() - l.rolling(100).min()) / pip,
    }
    rng = h - l
    feats["clv"] = (c - l) / rng.where(rng > 0)
    return feats


# ---------------------------------------------------------------- evaluation
class Ctx:
    """全評価で共有する文脈(マスク・ターゲット・シミュレータ入力)。"""

    def __init__(self) -> None:
        df, tgt, pip = load_xy(PAIR, "M5")
        self.df, self.tgt, self.pip = df, tgt, pip
        self.F = build_feats(df, pip)
        idx = df.index
        self.tr = pd.Series(idx < SPLIT, index=idx)
        self.te = ~self.tr
        self.ex_roll = pd.Series(~np.isin(idx.hour, EXCL_HOURS), index=idx)
        self.act = pd.Series(np.isin(idx.hour, ACTIVE_HOURS), index=idx)
        self.valid = tgt.notna()
        self.days_te = max((idx[self.te.values][-1] - idx[self.te.values][0]).days, 1)
        self.days_tr = max((idx[self.tr.values][-1] - idx[self.tr.values][0]).days, 1)
        self.hts = horizon_targets(df, PAIR)
        self.closev = df["close"].to_numpy()
        self.z10v = self.F["z10"].to_numpy()
        self.z50v = self.F["z50"].to_numpy()
        self.n = len(df)
        step = idx.to_series().diff()
        med = step.median()
        # gap_flag[i] = bar i と i+1 の間に大ギャップ(週末等)
        self.gap_flag = (step.shift(-1) > med * 3).to_numpy()
        self.gap_cum = np.concatenate([[0], np.cumsum(self.gap_flag)])

    # ---- 非重複トレードシミュレーション(同時1ポジション) ----
    def simulate(self, sig: pd.Series, sign: int, period: pd.Series,
                 exit_mode: str, N: int = 10, return_trades: bool = False) -> dict:
        mask = (sig & self.valid & self.ex_roll & period).to_numpy()
        entries = np.flatnonzero(mask)
        entries = entries[entries < self.n - 1]
        zarr = self.z10v if sign > 0 else self.z50v
        nets, holds, pos, spans = [], [], [], 0
        free = -1
        for p in entries:
            if p < free:
                continue
            if exit_mode == "time":
                q = min(p + N, self.n - 1)
            else:  # zcross
                q = min(p + ZCAP, self.n - 1)
                hi = min(p + ZCAP, self.n - 1)
                for s in range(p + 1, hi + 1):
                    zs = zarr[s]
                    if not np.isnan(zs) and ((sign > 0 and zs >= 0) or (sign < 0 and zs <= 0)):
                        q = s
                        break
            gross = (self.closev[q] - self.closev[p]) / self.pip * sign
            nets.append(gross - COST)
            holds.append(q - p)
            pos.append(p)
            if self.gap_cum[q] - self.gap_cum[p] > 0:
                spans += 1
            free = q
        nets = np.array(nets)
        holds = np.array(holds)
        days = self.days_te if period is self.te else self.days_tr
        if len(nets) == 0:
            return {"n": 0}
        if return_trades:
            return {"n": len(nets), "nets": nets, "holds": holds,
                    "pos": np.array(pos)}
        return {
            "n": len(nets),
            "net_mean": float(nets.mean()),
            "net_t": tstat(nets),
            "trades_day": len(nets) / days,
            "daily_net": float(nets.sum()) / days,
            "win": float((nets > 0).mean()),
            "hold_mean": float(holds.mean()),
            "hold_p": [float(np.percentile(holds, p)) for p in (25, 50, 75, 90)],
            "hold_max": int(holds.max()),
            "capped": float((holds >= (N if exit_mode == "time" else ZCAP)).mean()),
            "gap_span": spans,
        }

    # ---- シグナル単位の標準統計(重複許容、次足+ホライズン) ----
    def sig_stats(self, sig: pd.Series, sign: int) -> dict:
        s = sig & self.valid
        out = {}
        for tag, m in [("tr_ex", s & self.tr & self.ex_roll),
                       ("te_full", s & self.te),
                       ("te_ex", s & self.te & self.ex_roll)]:
            y = (self.tgt[m] * sign).dropna().to_numpy()
            out[tag] = {"n": len(y), "mean": float(y.mean()) if len(y) else np.nan,
                        "t": tstat(y)}
        m_ex = (s & self.te & self.ex_roll)
        idx = self.df.index[m_ex.values]
        for h in (5, 10, 20):
            out[f"h{h}"] = float((self.hts[h].reindex(idx) * sign).mean()) if len(idx) else np.nan
        out["sig_day"] = out["te_ex"]["n"] / self.days_te
        return out


EXITS = [("N=1", "time", 1), ("N=3", "time", 3), ("N=5", "time", 5),
         ("N=10", "time", 10), ("N=20", "time", 20), ("zX", "zcross", 0)]


def marginal_table(ctx: Ctx, base_name: str, base: pd.Series, sign: int,
                   filters: dict[str, pd.Series]) -> list[dict]:
    """ベース単体 + 各フィルタ(AND) + 2フィルタ AND の限界価値表。"""
    rows = []
    combos = [()] + [(k,) for k in filters] + list(itertools.combinations(filters, 2))
    for combo in combos:
        sig = base.copy()
        for k in combo:
            sig = sig & filters[k]
        name = base_name + ("" if not combo else "+" + "+".join(combo))
        st = ctx.sig_stats(sig, sign)
        rows.append({"name": name, "combo": combo, "sig": sig, "st": st})
    return rows


def print_table(ctx: Ctx, rows: list[dict]) -> None:
    hdr = (f"{'signal':<28} {'te_ex mean':>10} {'t':>5} {'n':>6} {'/day':>5} "
           f"{'netN1/day':>9} {'h5':>6} {'h10':>6} {'h20':>6} {'tr_ex mean':>10} "
           f"{'tr netN1/d':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        st = r["st"]
        te, trn = st["te_ex"], st["tr_ex"]
        net_day = (te["mean"] - COST) * st["sig_day"] if te["n"] else float("nan")
        tr_net_day = (trn["mean"] - COST) * trn["n"] / ctx.days_tr if trn["n"] else float("nan")
        print(f"{r['name']:<28} {te['mean']:>+10.3f} {te['t']:>5.1f} {te['n']:>6} "
              f"{st['sig_day']:>5.2f} {net_day:>+9.2f} {st['h5']:>+6.2f} "
              f"{st['h10']:>+6.2f} {st['h20']:>+6.2f} {trn['mean']:>+10.3f} "
              f"{tr_net_day:>+10.2f}")


def exit_grid(ctx: Ctx, rows: list[dict], sign: int) -> None:
    """全 combo × 全出口の「日次net (train/test)」マトリクス(非重複・コスト後)。"""
    cells = []
    hdr = f"{'signal':<28}" + "".join(f"{lbl:>16}" for lbl, _, _ in EXITS)
    print(hdr + "   (各セル: train日次net/test日次net, n<100 は -)")
    print("-" * len(hdr))
    for r in rows:
        line = f"{r['name']:<28}"
        for lbl, mode, N in EXITS:
            rt = ctx.simulate(r["sig"], sign, ctx.tr, mode, N)
            rte = ctx.simulate(r["sig"], sign, ctx.te, mode, N)
            r.setdefault("grid", {})[lbl] = (rt, rte)
            if rt.get("n", 0) < 100 or rte.get("n", 0) < 100:
                line += f"{'-':>16}"
            else:
                line += f"{rt['daily_net']:>+8.2f}/{rte['daily_net']:>+6.2f} "
            cells.append((r["name"], lbl, rt, rte))
        print(line)


def argmax_report(rows: list[dict], tag: str) -> tuple[dict, str]:
    """train 日次net argmax の組合せ×出口 → その test 成績(選択バイアス無し推定)。"""
    best, best_key, best_val = None, None, -1e9
    tbest, tbest_key, tbest_val = None, None, -1e9
    for r in rows:
        for lbl, (rt, rte) in r.get("grid", {}).items():
            if rt.get("n", 0) < 100 or rte.get("n", 0) < 100:
                continue
            if rt["daily_net"] > best_val:
                best, best_key, best_val = r, lbl, rt["daily_net"]
            if rte["daily_net"] > tbest_val:
                tbest, tbest_key, tbest_val = r, lbl, rte["daily_net"]
    rt, rte = best["grid"][best_key]
    print(f"  [{tag}] train-argmax: {best['name']} {best_key} "
          f"(train {rt['daily_net']:+.2f}p/日, net/t {rt['net_mean']:+.2f}) -> "
          f"test {rte['daily_net']:+.2f}p/日 (net/t {rte['net_mean']:+.2f}, "
          f"t={rte['net_t']:+.1f}, n={rte['n']})")
    rt2, rte2 = tbest["grid"][tbest_key]
    print(f"  [{tag}] test-argmax (楽観上限): {tbest['name']} {tbest_key} "
          f"test {rte2['daily_net']:+.2f}p/日 (net/t {rte2['net_mean']:+.2f}, "
          f"t={rte2['net_t']:+.1f}, n={rte2['n']}) | train {rt2['daily_net']:+.2f}")
    return best, best_key


def exit_study(ctx: Ctx, name: str, sig: pd.Series, sign: int) -> None:
    print(f"\n--- exit study: {name} ({'long' if sign > 0 else 'short'}) "
          f"— 非重複・UTC20-23除外・コスト{COST}p控除 ---")
    hdr = (f"{'exit':<10} {'n':>5} {'net/trade':>9} {'t':>5} {'win%':>5} {'tr/day':>6} "
           f"{'daily net':>9} {'hold mean':>9} {'p25/50/75/90':>16} {'cap%':>5} "
           f"{'gapN':>4} | {'train net/t':>11} {'daily':>6}")
    print(hdr)
    print("-" * len(hdr))
    for label, mode, N in [("N=1", "time", 1), ("N=3", "time", 3), ("N=5", "time", 5),
                           ("N=10", "time", 10), ("N=20", "time", 20),
                           ("zcross", "zcross", 0)]:
        r = ctx.simulate(sig, sign, ctx.te, mode, N)
        rt = ctx.simulate(sig, sign, ctx.tr, mode, N)
        if r["n"] == 0:
            print(f"{label:<10} n=0")
            continue
        hp = "/".join(f"{int(x)}" for x in r["hold_p"])
        print(f"{label:<10} {r['n']:>5} {r['net_mean']:>+9.3f} {r['net_t']:>5.1f} "
              f"{r['win'] * 100:>5.1f} {r['trades_day']:>6.2f} {r['daily_net']:>+9.2f} "
              f"{r['hold_mean']:>9.1f} {hp:>16} {r['capped'] * 100:>5.1f} "
              f"{r['gap_span']:>4} | {rt.get('net_mean', float('nan')):>+11.3f} "
              f"{rt.get('daily_net', float('nan')):>6.2f}")


def yearly_net(ctx: Ctx, sig: pd.Series, sign: int, mode: str, N: int) -> None:
    """test 期間の年別 net(非重複シム、UTC20-23除外)。"""
    for yr in sorted(set(ctx.df.index[ctx.te.values].year)):
        period = pd.Series(ctx.df.index.year == yr, index=ctx.df.index) & ctx.te
        mask = (sig & ctx.valid & ctx.ex_roll & period).to_numpy()
        entries = np.flatnonzero(mask)
        entries = entries[entries < ctx.n - 1]
        zarr = ctx.z10v if sign > 0 else ctx.z50v
        nets = []
        free = -1
        for p in entries:
            if p < free:
                continue
            if mode == "time":
                q = min(p + N, ctx.n - 1)
            else:
                q = min(p + ZCAP, ctx.n - 1)
                for s in range(p + 1, min(p + ZCAP, ctx.n - 1) + 1):
                    zs = zarr[s]
                    if not np.isnan(zs) and ((sign > 0 and zs >= 0) or (sign < 0 and zs <= 0)):
                        q = s
                        break
            nets.append((ctx.closev[q] - ctx.closev[p]) / ctx.pip * sign - COST)
            free = q
        nets = np.array(nets)
        if len(nets):
            print(f"    {yr}: net/trade {nets.mean():+.3f}p t={tstat(nets):+.1f} "
                  f"n={len(nets)} total {nets.sum():+.0f}p")


def main() -> None:
    ctx = Ctx()
    df, tgt = ctx.df, ctx.tgt
    F, tr, valid = ctx.F, ctx.tr, ctx.valid

    print(f"EURUSD M5 rows={ctx.n}  {df.index[0]} -> {df.index[-1]}")
    print(f"train days={ctx.days_tr} test days={ctx.days_te} "
          f"(test {df.index[ctx.te.values][0].date()} -> {df.index[ctx.te.values][-1].date()})")

    # ---------------- [0] base signals & filter constants (train only) ----
    er_thr = F["er10"][tr].quantile(2 / 3)
    hiER = F["er10"] > er_thr
    z10er = F["z10"].where(hiER)
    m_l0 = z10er.notna() & np.isfinite(z10er) & valid
    l0_thr = z10er[m_l0 & tr].quantile(0.02)
    L0 = z10er <= l0_thr

    m_s0 = F["z50"].notna() & np.isfinite(F["z50"]) & valid
    s0_thr = F["z50"][m_s0 & tr].quantile(0.98)
    S0 = F["z50"] >= s0_thr
    m_s1 = F["ret3n"].notna() & np.isfinite(F["ret3n"]) & valid
    s1_thr = F["ret3n"][m_s1 & tr].quantile(0.98)
    S1 = F["ret3n"] >= s1_thr
    S01 = S0 | S1

    vr_thr = F["vr"][tr].quantile(0.75)
    rng_thr = F["range100"][tr].quantile(2 / 3)
    fb = F["vr"] >= vr_thr
    fc = F["vrank"] >= 0.8
    fd = F["range100"] >= rng_thr
    fe = ctx.act

    print("\n[0] train 定数:")
    print(f"  hiER10: er10 > {er_thr:.4f} | L0: z10 <= {l0_thr:.4f} (hiER内)")
    print(f"  S0: z50 >= {s0_thr:.4f} | S1: ret3_norm >= {s1_thr:.4f}")
    print(f"  b: vr >= {vr_thr:.4f} | c: vrank100 >= 0.8 | d: range100p >= {rng_thr:.1f}p"
          f" | e: UTC 7-16")

    clv_l0 = F["clv"][L0 & tr].quantile(0.25)
    clv_s0 = F["clv"][S0 & tr].quantile(0.75)
    clv_s1 = F["clv"][S1 & tr].quantile(0.75)
    clv_s01 = F["clv"][S01 & tr].quantile(0.75)
    print(f"  a(ゾーン内CLV): L0 clv<={clv_l0:.3f} | S0 clv>={clv_s0:.3f} "
          f"| S1 clv>={clv_s1:.3f} | S0|S1 clv>={clv_s01:.3f}")

    # ---------------- [1] long marginal value ----------------------------
    print("\n[1] ロング限界価値表 (te_ex = test/UTC20-23除外。netN1/day = (次足平均-0.6)×頻度)")
    fl = {"a": F["clv"] <= clv_l0, "b": fb, "c": fc, "d": fd, "e": fe}
    rows_l = marginal_table(ctx, "L0", L0, +1, fl)
    print_table(ctx, rows_l)

    # ---------------- [2] short marginal value ---------------------------
    print("\n[2] ショート限界価値表 (符号は売り方向利益が+)")
    all_short_rows: list[dict] = []
    for bname, bsig, clv_thr in [("S0", S0, clv_s0), ("S1", S1, clv_s1),
                                 ("S0|S1", S01, clv_s01)]:
        fs = {"a": F["clv"] >= clv_thr, "b": fb, "c": fc, "d": fd, "e": fe}
        rows = marginal_table(ctx, bname, bsig, -1, fs)
        print_table(ctx, rows)
        all_short_rows.extend(rows)
        print()

    # ---------------- [3] exit grid: combo × exit, train/test ------------
    print("[3] 出口グリッド: 非重複・コスト0.6p後の日次net (train/test)。"
          "閾値は全て train、エントリーは UTC20-23 除外")
    print("\n-- long --")
    exit_grid(ctx, rows_l, +1)
    print("\n-- short --")
    exit_grid(ctx, all_short_rows, -1)

    print("\n[3b] 選択バイアスを排した推定 (train argmax -> test):")
    L_REC, l_exit = argmax_report(rows_l, "long")
    S_REC, s_exit = argmax_report(all_short_rows, "short")

    print("\n[3c] 出口詳細 (主要候補)")
    exit_study(ctx, "L0 (素)", L0, +1)
    exit_study(ctx, "L0+d", rows_l[4]["sig"], +1)
    exit_study(ctx, L_REC["name"], L_REC["sig"], +1)
    exit_study(ctx, "S0+d", [r for r in all_short_rows if r["name"] == "S0+d"][0]["sig"], -1)
    exit_study(ctx, "S0|S1+a+d",
               [r for r in all_short_rows if r["name"] == "S0|S1+a+d"][0]["sig"], -1)
    exit_study(ctx, S_REC["name"], S_REC["sig"], -1)

    # ---------------- [4] contradiction rate -----------------------------
    print("\n[4] ロング/ショート同時成立 (test, UTC20-23除外)")
    base_m = (ctx.te & ctx.ex_roll & valid)
    for ln, ls in [("L0", L0), (L_REC["name"], L_REC["sig"])]:
        for sn, ss in [("S0|S1", S01), (S_REC["name"], S_REC["sig"])]:
            both = int((ls & ss & base_m).sum())
            nl = int((ls & base_m).sum())
            print(f"  {ln} ∩ {sn}: {both} bars (ロング側の {both / max(nl, 1) * 100:.2f}%)")

    # ---------------- [5] final spec + economics + robustness ------------
    def exit_args(lbl: str) -> tuple[str, int]:
        return ("zcross", 0) if lbl == "zX" else ("time", int(lbl.split("=")[1]))

    l0d_sig = rows_l[4]["sig"]  # L0+d
    s0d_sig = [r for r in all_short_rows if r["name"] == "S0+d"][0]["sig"]
    finals = [
        (f"long train-argmax {L_REC['name']} {l_exit}", L_REC["sig"], +1, *exit_args(l_exit)),
        ("long judgment L0+d N=1", l0d_sig, +1, "time", 1),
        (f"short train-argmax {S_REC['name']} {s_exit}", S_REC["sig"], -1, *exit_args(s_exit)),
        ("short judgment S0+d N=5", s0d_sig, -1, "time", 5),
    ]
    print("\n[5] 最終候補の年別安定性 (test, 非重複, UTC20-23除外, コスト後)")
    for nm, sg, sign, mode, N in finals:
        print(f"  {nm}:")
        yearly_net(ctx, sg, sign, mode, N)

    # 日曜オープン感度: ギャップ直後 6 本(30分)のエントリー除外
    after_gap = np.zeros(ctx.n, dtype=bool)
    gpos = np.flatnonzero(ctx.gap_flag)
    for g in gpos:
        after_gap[g + 1: g + 7] = True
    no_open = pd.Series(~after_gap, index=df.index)
    print("\n  日曜オープン感度 (週明けギャップ直後6本のエントリー除外):")
    for nm, sg, sign, mode, N in finals:
        r0 = ctx.simulate(sg, sign, ctx.te, mode, N)
        r1 = ctx.simulate(sg & no_open, sign, ctx.te, mode, N)
        print(f"  {nm}: net/t {r0['net_mean']:+.3f} (n={r0['n']}) -> "
              f"除外後 {r1['net_mean']:+.3f} (n={r1['n']})")

    # ---------------- [6] finalist deep-dive ------------------------------
    print("\n[6] ファイナリスト深掘り")
    print("  -- S0+a+d (train/test 双方プラスの唯一の短期出口ブロック) --")
    s0ad_sig = [r for r in all_short_rows if r["name"] == "S0+a+d"][0]["sig"]
    exit_study(ctx, "S0+a+d", s0ad_sig, -1)
    print("  S0+a+d N=5 年別:")
    yearly_net(ctx, s0ad_sig, -1, "time", 5)
    s01ad_sig = [r for r in all_short_rows if r["name"] == "S0|S1+a+d"][0]["sig"]
    print("  S0|S1+a+d N=5 年別:")
    yearly_net(ctx, s01ad_sig, -1, "time", 5)

    print("\n  -- エントリー時間帯への利益集中チェック (test) --")
    for nm, sg, sign, mode, N in [("L0+d N=1", l0d_sig, +1, "time", 1),
                                  ("S0|S1+a+d N=5", s01ad_sig, -1, "time", 5),
                                  ("S0+a+d N=5", s0ad_sig, -1, "time", 5)]:
        r = ctx.simulate(sg, sign, ctx.te, mode, N, return_trades=True)
        hrs = ctx.df.index.hour.to_numpy()[r["pos"]]
        buckets = [("0-6", (hrs >= 0) & (hrs <= 6)), ("7-12", (hrs >= 7) & (hrs <= 12)),
                   ("13-16", (hrs >= 13) & (hrs <= 16)), ("17-19", (hrs >= 17) & (hrs <= 19))]
        parts = []
        for bn, bm in buckets:
            y = r["nets"][bm]
            parts.append(f"{bn}: {y.mean():+.2f}p×{len(y)}={y.sum():+.0f}p" if len(y)
                         else f"{bn}: -")
        print(f"  {nm}: " + " | ".join(parts))

    print("\n[7] 経済性まとめ (test 1250日 ≈ 3.42年, 年換算 = 日次net×365)")
    summary = [("long L0+d N=1", l0d_sig, +1, "time", 1),
               ("short S0|S1+a+d N=5", s01ad_sig, -1, "time", 5),
               ("short S0+a+d N=5", s0ad_sig, -1, "time", 5)]
    for nm, sg, sign, mode, N in summary:
        rte = ctx.simulate(sg, sign, ctx.te, mode, N)
        rtr = ctx.simulate(sg, sign, ctx.tr, mode, N)
        print(f"  {nm}: test net/t {rte['net_mean']:+.3f}p (t={rte['net_t']:+.1f}) "
              f"x {rte['trades_day']:.2f}回/日 = {rte['daily_net']:+.2f}p/日 "
              f"≈ {rte['daily_net'] * 365:+.0f}p/年 | train {rtr['daily_net']:+.2f}p/日")


if __name__ == "__main__":
    main()
