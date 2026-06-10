"""exp23: チャンピオンv2 トレードの「バー内(M1)価格挙動」徹底検証。

これまでの全検証は H4 終値約定(vectorbt from_signals)で、バー内の値動きは未活用だった。
本実験は M1 実データ(クロスは脚から合成)で全トレードのバー内パスを再構成し、3つを実測する:

  A. エントリー後のバー内 MAE 分布 — エントリー価格から δ·σ 深い水準に「いつ・どれだけの確率で」
     到達するか(勝ちトレード vs 負けトレード別)。指値の置き所の物理を知る。
  B. 指値エントリー置換シミュ — 成行(終値)の代わりに entry−δ·σ·P に指値(TTL バー有効、
     未約定なら見送り or TTL終値で成行)。トレード毎の純リターン変化を実測。
     ※決済はチャンピオンの出口シグナル(z 回帰)のままなので независ→エントリーだけの差が出る。
  C. 指値決済シミュ — 出口しきい値(z=−exit_z)の価格を前バー統計から因果計算し、
     バー内タッチで約定したら何 bps 改善するか(タッチ無しは従来どおり終値決済)。

慣例: 約定コストは spread/2 を両側に適用(既存エンジンと同一)。クロスの M1 は close 合成のみ
= バー内極値を過小評価 → 指値の約定率・改善幅は控えめに出る(保守的)。

実行: PYTHONPATH=. uv run python research/experiments/exp23_intrabar_paths.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))

import mm_lab as mm  # noqa: E402
from fxlab import config  # noqa: E402
from fxlab.data import load_m1  # noqa: E402
from fxlab.universe import CROSS_DEFS  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

OUT = ROOT / "research" / "outputs"
OUT.mkdir(exist_ok=True)

EXIT_Z = 0.5
WIN = 50  # 短期Z窓(チャンピオン)


# ---------------------------------------------------------------- M1 paths
def m1_path(instr: str):
    """M1 の (index, close, low, high) を返す。クロスは close 合成(low=high=close)。"""
    if instr in CROSS_DEFS:
        a, op, b = CROSS_DEFS[instr]
        ca = load_m1(a)["close"]
        cb = load_m1(b)["close"]
        df = pd.concat([ca, cb], axis=1, keys=["a", "b"]).ffill().dropna()
        c = (df["a"] / df["b"]) if op == "/" else (df["a"] * df["b"])
        v = c.to_numpy()
        return c.index.to_numpy(), v, v, v
    df = load_m1(instr)
    return df.index.to_numpy(), df["close"].to_numpy(), df["low"].to_numpy(), df["high"].to_numpy()


def main() -> int:
    pool = mm.build_pool()
    closes = mm.load_closes()
    grid = closes.index
    gi = grid.to_numpy()
    print(f"pool {len(pool)} trades / H4 grid {len(grid)}")

    # 前バー統計(因果)による出口しきい値: P_thr[t] = mean50[t-1] + dir側 z=−exit_z 相当
    mu_prev = {}
    sd_prev = {}
    for col in closes.columns:
        c = closes[col]
        mu_prev[col] = c.rolling(WIN).mean().shift(1)
        sd_prev[col] = c.rolling(WIN).std().shift(1)

    deltas = [0.25, 0.5, 0.75, 1.0, 1.5]
    ttls = [1, 2, 4, 8]

    rows_a = []      # MAE 分布
    rows_b = []      # 指値エントリー
    rows_c = []      # 指値決済
    base_rows = []   # ベースライン照合

    for instr, g in pool.groupby("instr"):
        mi, mc, ml, mh = m1_path(instr)
        half_spread = config.spread_pips(instr) * config.pip_size(instr) / 2.0
        e_pos = np.searchsorted(gi, g["entry"].to_numpy(), side="left")
        x_pos = np.searchsorted(gi, g["exit"].to_numpy(), side="left")
        carr = closes[instr].to_numpy()
        mu = mu_prev[instr].to_numpy()
        sd = sd_prev[instr].to_numpy()

        for (idx, tr), ep, xp in zip(g.iterrows(), e_pos, x_pos):
            d = int(tr["dir"])  # +1 long / -1 short
            entry_px = carr[ep]
            exit_px = carr[xp]
            sigma = float(tr["vol_entry"]) * entry_px  # 20本ボラ(比率)→価格幅
            if not np.isfinite(sigma) or sigma <= 0:
                continue
            # 往復コスト(価格比)
            cost_entry = half_spread / entry_px
            cost_exit = half_spread / exit_px
            ret_base = d * (exit_px / entry_px - 1.0) - cost_entry - cost_exit
            base_rows.append({"instr": instr, "entry": tr["entry"], "ret_pool": tr["ret"],
                              "ret_recalc": ret_base})

            # --- M1 ウィンドウ: エントリーバーの次バー開始〜出口バー終端
            if xp <= ep:
                continue
            t0 = gi[ep + 1] if ep + 1 < len(gi) else None
            if t0 is None:
                continue
            # 出口バーの終端時刻(出口バー区間 [grid[xp], grid[xp]+4h))
            t1 = gi[xp] + np.timedelta64(4, "h")
            i0 = np.searchsorted(mi, t0, side="left")
            i1 = np.searchsorted(mi, t1, side="left")
            if i1 <= i0:
                continue

            # --- A: バー内 MAE(エントリー価格比, σ単位)を horizon 別に
            adverse = (entry_px - ml[i0:i1]) if d > 0 else (mh[i0:i1] - entry_px)
            # horizon = H4 バー数ごとの累積最大逆行
            for hz in ttls:
                tz = gi[min(ep + hz, len(gi) - 1)] + np.timedelta64(4, "h")
                iz = np.searchsorted(mi, tz, side="left")
                seg = adverse[: max(0, iz - i0)]
                mae = float(seg.max()) if len(seg) else np.nan
                rows_a.append({"instr": instr, "entry": tr["entry"], "dir": d,
                               "horizon": hz, "mae_sigma": mae / sigma if np.isfinite(mae) else np.nan,
                               "win": tr["ret"] > 0, "ret": tr["ret"],
                               "z_entry": tr["z_entry"], "bars_held": tr["bars_held"]})

            # --- B: 指値エントリー置換(fill = バー内逆行が δσ に到達)
            for dl in deltas:
                limit_px = entry_px - d * dl * sigma
                for ttl in ttls:
                    tz = gi[min(ep + ttl, len(gi) - 1)] + np.timedelta64(4, "h")
                    iz = min(np.searchsorted(mi, tz, side="left"), i1)
                    seg_low = ml[i0:iz] if d > 0 else mh[i0:iz]
                    hit = (seg_low <= limit_px) if d > 0 else (seg_low >= limit_px)
                    if hit.any():
                        k = int(np.argmax(hit))
                        # 約定バー(H4)が出口バー以降なら不成立(出口が先に来る)
                        fill_t = mi[i0 + k]
                        fill_h4 = int(np.searchsorted(gi, fill_t, side="right") - 1)
                        if fill_h4 >= xp:
                            ret_new, filled = np.nan, False  # 出口が先=取れない
                        else:
                            ce = half_spread / limit_px
                            ret_new = d * (exit_px / limit_px - 1.0) - ce - cost_exit
                            filled = True
                    else:
                        ret_new, filled = np.nan, False
                    rows_b.append({"instr": instr, "entry": tr["entry"], "dir": d,
                                   "delta": dl, "ttl": ttl, "filled": filled,
                                   "ret_base": ret_base, "ret_limit": ret_new,
                                   "win": ret_base > 0, "z_entry": tr["z_entry"]})

            # --- C: 指値決済(因果しきい値のバー内タッチ)
            touched = False
            for b in range(ep + 1, xp + 1):
                if not (np.isfinite(mu[b]) and np.isfinite(sd[b])):
                    continue
                thr = mu[b] - d * EXIT_Z * sd[b]  # long: mean−0.5σ50 / short: mean+0.5σ50
                tb0 = np.searchsorted(mi, gi[b], side="left")
                tb1 = np.searchsorted(mi, gi[b] + np.timedelta64(4, "h"), side="left")
                if tb1 <= tb0:
                    continue
                seg = mh[tb0:tb1] if d > 0 else ml[tb0:tb1]
                ok = (seg >= thr) if d > 0 else (seg <= thr)
                if ok.any():
                    cx = half_spread / thr
                    ret_new = d * (thr / entry_px - 1.0) - cost_entry - cx
                    rows_c.append({"instr": instr, "entry": tr["entry"], "dir": d,
                                   "exit_kind": "limit", "bar_at": b - ep,
                                   "bars_base": xp - ep,
                                   "ret_base": ret_base, "ret_limit": ret_new})
                    touched = True
                    break
            if not touched:
                rows_c.append({"instr": instr, "entry": tr["entry"], "dir": d,
                               "exit_kind": "close", "bar_at": xp - ep, "bars_base": xp - ep,
                               "ret_base": ret_base, "ret_limit": ret_base})
        print(f"  {instr}: done")

    A = pd.DataFrame(rows_a)
    B = pd.DataFrame(rows_b)
    C = pd.DataFrame(rows_c)
    V = pd.DataFrame(base_rows)
    A.to_csv(OUT / "exp23_mae.csv", index=False)
    B.to_csv(OUT / "exp23_limit_entry.csv", index=False)
    C.to_csv(OUT / "exp23_limit_exit.csv", index=False)

    # ===== 照合: 再計算リターン vs プール(エンジン)
    diff = (V["ret_recalc"] - V["ret_pool"]).abs()
    print(f"\n[照合] 再計算 vs プール: 中央絶対差 {diff.median():.5f} / p95 {diff.quantile(0.95):.5f}")

    # ===== A: MAE 分布(σ単位)
    print("\n===== A. エントリー後のバー内 MAE(σ単位, horizon=H4バー数) =====")
    for hz in ttls:
        a = A[A["horizon"] == hz]
        for w, lbl in [(True, "勝ち"), (False, "負け")]:
            s = a[a["win"] == w]["mae_sigma"].dropna()
            if len(s):
                print(f"  hz={hz:>2} {lbl}: n={len(s):>4} 中央 {s.median():.2f}σ p25 {s.quantile(.25):.2f} "
                      f"p75 {s.quantile(.75):.2f} | ≥0.5σ {(s>=0.5).mean():.0%} ≥1.0σ {(s>=1.0).mean():.0%}")

    # ===== B: 指値エントリー
    print("\n===== B. 指値エントリー置換(skip=未約定は見送り) =====")
    print("  delta ttl   fill%   Σret_base(全) Σret_limit(約定分のみ+未約定0)  約定分Σbase  約定分Σlimit  改善/トレード(bps,約定分)")
    for dl in deltas:
        for ttl in ttls:
            b = B[(B["delta"] == dl) & (B["ttl"] == ttl)]
            if not len(b):
                continue
            f = b[b["filled"]]
            tot_base = b["ret_base"].sum()
            tot_limit = f["ret_limit"].sum()
            imp = ((f["ret_limit"] - f["ret_base"]).mean() * 1e4) if len(f) else np.nan
            print(f"  {dl:>5} {ttl:>3}  {b['filled'].mean():>5.0%}  {tot_base:>12.3f} {tot_limit:>12.3f}"
                  f"  {f['ret_base'].sum():>11.3f}  {f['ret_limit'].sum():>11.3f}  {imp:>8.1f}")

    # ===== C: 指値決済
    print("\n===== C. 指値決済(因果しきい値のバー内タッチ) =====")
    lim = C[C["exit_kind"] == "limit"]
    print(f"  タッチ率: {len(lim)/max(len(C),1):.0%}  平均改善 {(lim['ret_limit']-lim['ret_base']).mean()*1e4:.1f}bps"
          f"  Σbase {C['ret_base'].sum():.3f} → Σlimit {C['ret_limit'].sum():.3f}")
    early = lim[lim["bar_at"] < lim["bars_base"]]
    print(f"  早期決済(バー数短縮): {len(early)}/{len(lim)} 平均短縮 {(early['bars_base']-early['bar_at']).mean():.1f}バー")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
