"""trend2/exp_stop_entry — タートル式ストップ注文エントリーの M1 再現 vs 終値確認版。

デバイアス課題: これまでのブレイクアウト検証は「終値確認後に終値で約定」だったため、
レベル突破から終値までの行き過ぎ(adverse drift)を全部払う構造的不利があった。
本実験は「チャネル水準に置いたストップ注文が M1 高安に触れた瞬間その価格で約定」を再現し、
同一構成の終値確認版(tl.build_pool)と比較する。

仕様:
  - H4/D1 の close ベース・チャネル: level = close.rolling(n).max()/min() を .shift(1)
    (前バー確定値のみ使用 = 先読みなし)。
  - 各 TF バー内の M1 high/low がレベルに触れたらそのレベル価格で約定。
    バー始値が既にレベルを越えてギャップしている場合は M1 open で約定(現実のストップ挙動)。
  - 建玉中は exit チャネル(同じく .shift(1))への M1 タッチで決済。ドテン無し
    (決済後、同一バー内でも次のタッチで再エントリー可)。
  - コスト: 約定価格に片側 spread/2 を加減(往復で 1 スプレッド)。
    XAUUSD は config.SPREADS_PIPS["XAUUSD"] = 0.40/0.0001(フル $0.40)。
  - 対象: メジャー7 + XAUUSD(実 M1 がある 8 銘柄のみ。クロス12は M1 が無いので除外)。
  - グリッド固定: (entry, exit) ∈ {(55,20), (100,50)} × tf ∈ {H4, D1}。追い込み禁止。

実行: リポジトリ直下で PYTHONPATH=. uv run python research/experiments/trend2/exp_stop_entry.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "lab"))

import trend_lab as tl  # noqa: E402
from fxlab import config  # noqa: E402
from fxlab.data import clear_cache, load_m1  # noqa: E402

MAJORS = list(tl.MAJORS)  # EURUSD..NZDUSD (7)
INSTRUMENTS = MAJORS + ["XAUUSD"]
GRID = [(55, 20), (100, 50)]
TFS = ["H4", "D1"]
BAR_MIN = {"H4": 240, "D1": 1440}
OUT = ROOT / "research" / "outputs"


# --- 終値確認版のシグナル(close ベース・チャネル、shift(1) で先読みなし) ----
def donchian_close_gen(data: pd.DataFrame, entry: int = 55, exit: int = 20):
    close = data["close"]
    upper = close.rolling(entry).max().shift(1)
    lower = close.rolling(entry).min().shift(1)
    x_up = close.rolling(exit).max().shift(1)
    x_lo = close.rolling(exit).min().shift(1)
    le = close > upper
    se = close < lower
    lx = close < x_lo
    sx = close > x_up
    return le, lx, se, sx


# --- ストップ注文版: M1 タッチ約定シミュレーション --------------------------
def simulate_stop(instr: str, tf: str, e: int, x: int,
                  m1o, m1h, m1l, m1c, m1ns) -> pd.DataFrame:
    """1 銘柄 × 1 構成のストップ注文トレード列を返す。

    m1o/m1h/m1l/m1c: M1 open/high/low/close (np.float64), m1ns: M1 epoch ns (int64)。
    """
    df = tl.load_tf(instr, tf)
    closep = df["close"]
    lvl_hi = closep.rolling(e).max().shift(1).to_numpy()
    lvl_lo = closep.rolling(e).min().shift(1).to_numpy()
    ex_hi = closep.rolling(x).max().shift(1).to_numpy()
    ex_lo = closep.rolling(x).min().shift(1).to_numpy()
    bar_hi = df["high"].to_numpy()
    bar_lo = df["low"].to_numpy()
    tf_ns = df.index.asi8  # UTC tz-aware -> epoch ns(M1 と同基準)

    starts = np.searchsorted(m1ns, tf_ns, side="left")
    ends = np.empty(len(tf_ns), dtype=np.int64)
    ends[:-1] = starts[1:]
    ends[-1] = np.searchsorted(m1ns, tf_ns[-1] + BAR_MIN[tf] * 60 * 10**9, side="left")

    half = config.spread_pips(instr) * config.pip_size(instr) / 2.0

    pos = 0  # 0=flat, 1=long, -1=short
    ent_ns = 0
    ent_fill = np.nan
    rows: list[tuple] = []

    def record(exit_ns: int, exit_fill: float, forced: bool = False) -> None:
        if pos == 1:
            ee = ent_fill + half
            xe = exit_fill - half
            ret = xe / ee - 1.0
        else:
            ee = ent_fill - half
            xe = exit_fill + half
            ret = (ee - xe) / ee
        rows.append((instr, ent_ns, exit_ns, pos, ent_fill, exit_fill, ret,
                     (exit_ns - ent_ns) / (60 * 10**9) / BAR_MIN[tf], forced))

    n = len(tf_ns)
    for i in range(n):
        LH, LL, XH, XL = lvl_hi[i], lvl_lo[i], ex_hi[i], ex_lo[i]
        if np.isnan(LH) or np.isnan(LL) or np.isnan(XH) or np.isnan(XL):
            continue
        # TF バーの高安で「このバーで何か起きうるか」を事前判定(高速化。集約元が同じ M1 なので等価)
        if pos == 0:
            if not (bar_hi[i] >= LH or bar_lo[i] <= LL):
                continue
        elif pos == 1:
            if not bar_lo[i] <= XL:
                continue
        else:
            if not bar_hi[i] >= XH:
                continue
        s0, e0 = int(starts[i]), int(ends[i])
        if e0 <= s0:
            continue
        o = m1o[s0:e0]
        hh = m1h[s0:e0]
        ll = m1l[s0:e0]
        L = e0 - s0
        p = 0
        while p < L:
            if pos == 0:
                up = hh[p:] >= LH
                dn = ll[p:] <= LL
                iu = int(np.argmax(up)) if up.any() else -1
                idn = int(np.argmax(dn)) if dn.any() else -1
                if iu < 0 and idn < 0:
                    break
                if idn < 0 or (iu >= 0 and iu <= idn):  # 同分タイは long 優先(実質発生せず)
                    k = p + iu
                    ent_fill = max(LH, o[k])  # ギャップ上抜けは open 約定
                    pos = 1
                else:
                    k = p + idn
                    ent_fill = min(LL, o[k])
                    pos = -1
                ent_ns = int(m1ns[s0 + k])
                p = k + 1
            elif pos == 1:
                m = ll[p:] <= XL
                if not m.any():
                    break
                k = p + int(np.argmax(m))
                record(int(m1ns[s0 + k]), min(XL, o[k]))
                pos = 0
                p = k + 1
            else:
                m = hh[p:] >= XH
                if not m.any():
                    break
                k = p + int(np.argmax(m))
                record(int(m1ns[s0 + k]), max(XH, o[k]))
                pos = 0
                p = k + 1

    if pos != 0:  # データ末尾で建玉中 → 最終 M1 終値で強制クローズ
        k = int(ends[-1]) - 1
        record(int(m1ns[k]), float(m1c[k]), forced=True)

    out = pd.DataFrame(rows, columns=["instr", "entry_ns", "exit_ns", "dir",
                                      "entry_price", "exit_price", "ret",
                                      "bars_held", "forced"])
    out["entry"] = pd.to_datetime(out["entry_ns"], utc=True)
    out["exit"] = pd.to_datetime(out["exit_ns"], utc=True)
    return out.drop(columns=["entry_ns", "exit_ns"])


def main() -> None:
    tl.register_spreads()  # XAUUSD スプレッド等を登録
    OUT.mkdir(parents=True, exist_ok=True)

    # --- ストップ注文版: 銘柄ごとに M1 を読み、4構成を処理して解放 ----------
    stop_trades: dict[tuple, list[pd.DataFrame]] = {
        (tf, e, x): [] for tf in TFS for (e, x) in GRID
    }
    for instr in INSTRUMENTS:
        m1 = load_m1(instr)
        m1o = m1["open"].to_numpy(np.float64)
        m1h = m1["high"].to_numpy(np.float64)
        m1l = m1["low"].to_numpy(np.float64)
        m1c = m1["close"].to_numpy(np.float64)
        m1ns = m1.index.asi8
        del m1
        clear_cache()
        gc.collect()
        for tf in TFS:
            for (e, x) in GRID:
                tr = simulate_stop(instr, tf, e, x, m1o, m1h, m1l, m1c, m1ns)
                stop_trades[(tf, e, x)].append(tr)
                print(f"  stop {instr} {tf} {e}x{x}: {len(tr)} trades", flush=True)
        del m1o, m1h, m1l, m1c, m1ns
        gc.collect()

    # --- プール集計 + CSV 保存 ----------------------------------------------
    results = []

    def add_result(label: str, tf: str, e: int, x: int, pool: pd.DataFrame, tag: str | None):
        st = tl.pool_stats(pool)
        st.update({"label": label, "tf": tf, "params": f"entry={e},exit={x}"})
        results.append(st)
        if tag is not None:
            cols = [c for c in ["instr", "entry", "exit", "dir", "entry_price",
                                "exit_price", "ret", "bars_held", "forced"] if c in pool.columns]
            pool[cols].to_csv(OUT / f"trend2_stopentry_{tag}.csv", index=False)
        print(f"{label}: {st}", flush=True)

    for tf in TFS:
        for (e, x) in GRID:
            frames = stop_trades[(tf, e, x)]
            fx7 = pd.concat([f for f in frames if f["instr"].iloc[0] != "XAUUSD"
                             ] if frames else [], ignore_index=True)
            fx7 = fx7.sort_values("entry").reset_index(drop=True)
            xau = pd.concat([f for f in frames if len(f) and f["instr"].iloc[0] == "XAUUSD"],
                            ignore_index=True).sort_values("entry").reset_index(drop=True)
            add_result(f"stop_{tf}_e{e}x{x}_FX7", tf, e, x, fx7, f"{tf}_{e}x{x}_fx7")
            add_result(f"stop_{tf}_e{e}x{x}_XAU", tf, e, x, xau, f"{tf}_{e}x{x}_xau")

    # --- 終値確認版(同一構成、tl.build_pool) -------------------------------
    for tf in TFS:
        for (e, x) in GRID:
            params = {"entry": e, "exit": x}
            fx7 = tl.build_pool(donchian_close_gen, params, tf=tf, side="both",
                                instruments=MAJORS)
            xau = tl.build_pool(donchian_close_gen, params, tf=tf, side="both",
                                instruments=["XAUUSD"])
            fx7.to_csv(OUT / f"trend2_closeentry_{tf}_{e}x{x}_fx7.csv", index=False)
            xau.to_csv(OUT / f"trend2_closeentry_{tf}_{e}x{x}_xau.csv", index=False)
            add_result(f"close_{tf}_e{e}x{x}_FX7", tf, e, x, fx7, None)
            add_result(f"close_{tf}_e{e}x{x}_XAU", tf, e, x, xau, None)

    res = pd.DataFrame(results)
    front = ["label", "tf", "params", "n", "trades_per_year", "sum_ret", "pool_pf",
             "is_pf", "oos_pf", "is_sum", "oos_sum", "mean_bps", "win_rate",
             "avg_bars", "yearly_pos", "worst_year"]
    res = res[[c for c in front if c in res.columns]]
    res.to_csv(OUT / "trend2_stopentry_summary.csv", index=False)
    print("\n=== SUMMARY ===")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
