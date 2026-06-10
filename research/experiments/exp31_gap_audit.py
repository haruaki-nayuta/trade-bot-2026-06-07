"""exp31: 週末ギャップ・フェード統合の敵対監査 — M1粒度DD + 約定現実性。

exp30b(インライン)で champ+gap が empirical/robust 両方を大幅改善(robust +15.4→+30.1%)。
ただし2つの疑いがある:
  A. ギャップトレードは保有4h ≈ H4 1バー。H4終値グリッドでは建玉が「マークの間に現れて消える」
     ため、測定DDにそのリスクがほぼ写らない → 較正が k を過大評価している疑い。
     → 建玉ログから M1 終値グリッドで MtM equity を再構成し、真の最大DDで再較正する(exp24方式)。
  B. 日曜オープンの約定現実性: 週初の数分はスプレッドが数倍に開く。
     → (i) スプレッド感応度(2x→3x/4x) (ii) エントリー遅延(open→15/30分後のM1終値)で再評価。

実行: PYTHONPATH=. uv run python research/experiments/exp31_gap_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))

import mm_lab as mm  # noqa: E402
import ens_lab as ens  # noqa: E402
from fxlab import config  # noqa: E402
from fxlab.data import load, load_m1  # noqa: E402

pd.set_option("display.width", 220)

MAJORS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
COLS = ["instr", "entry", "exit", "dir", "entry_price", "ret", "z_entry", "stream", "w"]
OOS = pd.Timestamp("2022-01-01", tz="UTC")


def build_gap_pool(spread_mult=2.0, delay_min=0, min_gap_pips=10.0, hold_h=4) -> pd.DataFrame:
    """週末ギャップ・フェードのプール。delay_min>0 なら週初オープンから遅延した M1 終値で建玉。"""
    rows = []
    for pair in MAJORS:
        h1 = load(pair, "H1")
        c, o, idx = h1["close"], h1["open"], h1.index
        gap_mask = idx.to_series().diff() >= pd.Timedelta("24h")
        pos = np.where(gap_mask.to_numpy())[0]
        cost = spread_mult * config.spread_pips(pair) * config.pip_size(pair)
        m1c = load_m1(pair)["close"] if delay_min > 0 else None
        for i in pos:
            prev_close = c.iloc[i - 1]
            open_now = o.iloc[i]
            gp = abs(open_now - prev_close) / config.pip_size(pair)
            if gp < min_gap_pips:
                continue
            d = -np.sign(open_now - prev_close)
            t0 = idx[i]
            if delay_min > 0:
                t_in = t0 + pd.Timedelta(minutes=delay_min)
                k = m1c.index.searchsorted(t_in, side="left")
                if k >= len(m1c):
                    continue
                entry_px = float(m1c.iloc[k])
                entry_t = m1c.index[k]
            else:
                entry_px = float(open_now)
                entry_t = t0
            j = min(i + hold_h, len(c) - 1)
            ret = d * (c.iloc[j] / entry_px - 1.0) - cost / entry_px
            rows.append({"instr": pair, "entry": entry_t, "exit": idx[j], "dir": int(d),
                         "entry_price": entry_px, "ret": float(ret), "z_entry": 2.2})
    g = pd.DataFrame(rows)
    g["stream"] = "gap"
    g["w"] = 1.0
    return g.sort_values("entry").reset_index(drop=True)


def simulate_streams_log(pool, closes, k, budgets, fbars, init=10_000.0):
    """ens.simulate_streams + 建玉ログ(M1監査用に実時刻も保持)。"""
    grid = closes.index
    col_of = {c: i for i, c in enumerate(closes.columns)}
    carr = closes.to_numpy()
    n = len(grid)
    gi = grid.to_numpy()
    e_pos = np.clip(np.searchsorted(gi, pool["entry"].to_numpy(), side="left"), 0, n - 1)
    x_pos = np.clip(np.searchsorted(gi, pool["exit"].to_numpy(), side="left"), 0, n - 1)
    by_entry = {}
    for ti in range(len(pool)):
        by_entry.setdefault(int(e_pos[ti]), []).append(ti)
    instr_arr = pool["instr"].to_numpy(); dir_arr = pool["dir"].to_numpy().astype(float)
    ep_arr = pool["entry_price"].to_numpy(); ret_arr = pool["ret"].to_numpy()
    z_arr = pool["z_entry"].to_numpy(); w_arr = pool["w"].to_numpy().astype(float)
    s_arr = pool["stream"].to_numpy()
    ent_t = pool["entry"].to_numpy(); ext_t = pool["exit"].to_numpy()
    slots_total = int(sum(budgets.values()))
    base = k / max(slots_total, 1)
    equity = init
    open_pos = []
    n_open_by = {s: 0 for s in budgets}
    eq_mtm = np.empty(n); eq_real = np.empty(n)
    log = []
    skipped = 0
    for b in range(n):
        if open_pos:
            still = []
            for p in open_pos:
                if p["exit_pos"] <= b:
                    equity += p["alloc"] * p["ret"]
                    n_open_by[p["stream"]] -= 1
                else:
                    still.append(p)
            open_pos = still
        unreal = 0.0
        for p in open_pos:
            px = carr[b, p["col"]]
            unreal += p["alloc"] * (p["dir"] * (px / p["eprice"] - 1.0))
        eq_mtm[b] = equity + unreal
        eq_real[b] = equity
        if b in by_entry:
            for ti in by_entry[b]:
                s = s_arr[ti]
                if s not in budgets or n_open_by[s] >= budgets[s]:
                    skipped += 1
                    continue
                alloc = equity * base * w_arr[ti] * (ens.fz(float(z_arr[ti])) / fbars[s])
                if alloc <= 0:
                    skipped += 1
                    continue
                open_pos.append({"col": col_of[instr_arr[ti]], "dir": dir_arr[ti],
                                 "eprice": ep_arr[ti], "alloc": alloc,
                                 "exit_pos": int(x_pos[ti]), "ret": float(ret_arr[ti]),
                                 "stream": s})
                n_open_by[s] += 1
                log.append({"instr": instr_arr[ti], "dir": float(dir_arr[ti]),
                            "eprice": float(ep_arr[ti]), "alloc": float(alloc),
                            "stream": s, "t_in": ent_t[ti], "t_out": ext_t[ti],
                            "ret": float(ret_arr[ti])})
    info = {"skipped": skipped}
    return pd.Series(eq_mtm, index=grid), pd.Series(eq_real, index=grid), pd.DataFrame(log), info


_m1_naive_cache: dict[str, pd.Series] = {}


def m1_close_naive(instr: str) -> pd.Series:
    if instr not in _m1_naive_cache:
        c = load_m1(instr)["close"]
        _m1_naive_cache[instr] = pd.Series(c.to_numpy(), index=c.index.tz_localize(None))
    return _m1_naive_cache[instr]


def m1_dd(log: pd.DataFrame, base_m1: np.ndarray, init=10_000.0) -> float:
    """建玉ログ→M1 MtM equity→最大DD。t_in/t_out は実時刻(ストリームごとの流儀込み)。"""
    n_m1 = len(base_m1)
    unreal = np.zeros(n_m1)
    real = np.full(n_m1, init)
    # champ: 約定= H4ラベル+4h / gap: entry=実時刻, exit= H1ラベル+1h
    t_in = pd.to_datetime(log["t_in"]).dt.tz_localize(None).to_numpy()
    t_out = pd.to_datetime(log["t_out"]).dt.tz_localize(None).to_numpy()
    is_champ = (log["stream"] == "champ").to_numpy()
    t_in = np.where(is_champ, t_in + np.timedelta64(4, "h"), t_in)
    t_out = np.where(is_champ, t_out + np.timedelta64(4, "h"), t_out + np.timedelta64(1, "h"))

    pnl = (log["alloc"] * log["ret"]).to_numpy()
    order = np.argsort(t_out)
    step_pos = np.searchsorted(base_m1, t_out[order], side="left")
    cum = 0.0
    last = 0
    for sp, v in zip(step_pos, pnl[order]):
        sp = min(sp, n_m1)
        if sp > last:
            real[last:sp] = init + cum
            last = sp
        cum += v
    real[last:] = init + cum

    # 全行ループで M1 区間の含み損益を加算(数千行なので十分速い)
    for w in range(len(log)):
        r = log.iloc[w]
        c = m1_close_naive(r["instr"])
        px_idx = c.index.to_numpy()
        a = np.searchsorted(base_m1, t_in[w], side="left")
        b = np.searchsorted(base_m1, t_out[w], side="left")
        if b <= a:
            continue
        seg_t = base_m1[a:b]
        ai = np.searchsorted(px_idx, seg_t, side="right") - 1
        ai = np.clip(ai, 0, len(px_idx) - 1)
        seg_px = c.to_numpy()[ai]
        unreal[a:b] += r["alloc"] * (r["dir"] * (seg_px / r["eprice"] - 1.0))
    eq = real + unreal
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def main() -> int:
    closes = mm.load_closes()
    champ = mm.build_pool().copy()
    champ["stream"] = "champ"; champ["w"] = 1.0
    base_m1 = load_m1("EURUSD").index.tz_localize(None).to_numpy()

    print("=== B. 約定現実性: スプレッド倍率 × エントリー遅延 のプール感応度 ===")
    for sm, dl in [(2.0, 0), (3.0, 0), (4.0, 0), (2.0, 15), (2.0, 30), (3.0, 15)]:
        g = build_gap_pool(spread_mult=sm, delay_min=dl)
        is_ = g[g["entry"] < OOS]; oos = g[g["entry"] >= OOS]
        print(f"  spread x{sm} delay {dl:2d}min: n={len(g):4d} ΣR={g.ret.sum():+.3f} "
              f"平均={g.ret.mean()*1e4:+5.1f}bps | IS {is_.ret.mean()*1e4:+5.1f} OOS {oos.ret.mean()*1e4:+5.1f}")

    print("\n=== A. M1粒度 DD 監査(spread x3, delay 15min の保守プールで) ===")
    g = build_gap_pool(spread_mult=3.0, delay_min=15)
    g["w"] = 2.0
    pool = pd.concat([champ[COLS], g[COLS]], ignore_index=True).sort_values("entry").reset_index(drop=True)
    budgets = {"champ": 11, "gap": 2}
    fbars = ens.stream_fbars(pool)

    # H4グリッド較正(従来)→ M1監査 → M1粒度で再較正
    k, eqm, eqr, info = ens.calibrate_streams(pool, closes, budgets, fbars=fbars, target_dd=0.20)
    for it in range(4):
        eqm_h4, _, log, _ = simulate_streams_log(pool, closes, k, budgets, fbars)
        dd_h4 = float((eqm_h4 / eqm_h4.cummax() - 1.0).min())
        dd_m1 = m1_dd(log, base_m1)
        years = (eqm_h4.index[-1] - eqm_h4.index[0]).days / 365.25
        cagr = (eqm_h4.iloc[-1] / 10_000.0) ** (1 / years) - 1
        print(f"  [iter{it}] k={k:.2f} DD(H4)={dd_h4:+.1%} DD(M1)={dd_m1:+.1%} CAGR={cagr:+.2%}")
        if abs(abs(dd_m1) - 0.20) < 0.003:
            break
        k = k * (0.20 / abs(dd_m1))

    # 基準(champion単独)も M1 粒度で揃える(exp24: k=7.60, CAGR+20.52%)
    print("\n  基準(M1粒度): champion mp8 k=7.60 CAGR=+20.52%(exp24)/ mp11 は下で実測")
    ch = champ[COLS]
    fb = ens.stream_fbars(ch)
    k2, *_ = ens.calibrate_streams(ch, closes, {"champ": 11}, fbars=fb, target_dd=0.20)
    for it in range(4):
        eqm_h4, _, log, _ = simulate_streams_log(ch, closes, k2, {"champ": 11}, fb)
        dd_m1 = m1_dd(log, base_m1)
        years = (eqm_h4.index[-1] - eqm_h4.index[0]).days / 365.25
        cagr = (eqm_h4.iloc[-1] / 10_000.0) ** (1 / years) - 1
        print(f"  [champ mp11 iter{it}] k={k2:.2f} DD(M1)={dd_m1:+.1%} CAGR={cagr:+.2%}")
        if abs(abs(dd_m1) - 0.20) < 0.003:
            break
        k2 = k2 * (0.20 / abs(dd_m1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
