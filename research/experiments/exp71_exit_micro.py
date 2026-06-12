"""exp71: 出口ミクロ構造解剖(M1段)— 「勝ちは H4 close 決済より前にバー内で決済水準に到達しているか」

仮説: チャンピオン d1 の出口は「z が exit 閾値帯へ戻った H4 バーの close で成行決済」。
バー内(M1)では close より有利な価格(目標 = rolling mean = z=0 水準)に先に到達している
可能性があり、因果的な指値出口に置き換えれば取り残しを回収できるかもしれない。

測定:
  1. 決済バー内の有利幅: 決済 H4 バー [label, label+4h) の M1 極値(ロング=高値/ショート=安値)
     と H4 close 決済価格の差(bps)。これは「事後に見える取り残しの総量」(非因果の上限)。
  2. 因果的指値出口の反実仮想: 各保有バー b の間、直前バー b-1 までの close で計算した
     rolling(50).mean = m[b-1](z=0 水準)に指値を置く(毎 H4 close で更新)。
     M1 が指値を「超えた」最初のバーで L=m[b-1] 約定。最後まで刺さらなければ従来の
     z 出口(挙動不変)。Δret = dir·(L − Cx)/Ce(コスト中立)。
  3. 集計: 改善/悪化の割合と bps、純効果、保有短縮、勝ち/負け別、年次、感度2種。

約定モデル(保守側, exp45 の規約を踏襲):
  ・メジャー7: M1 BID の high/low。ロング決済(売り指値)= M1 high が L を「超えた」(>)
    バーで L 約定。タッチ(=)は不成立。ショート決済(買い指値)= M1 low < L。
  ・合成クロス12: 脚 M1 close の inner-join 合成の分単位 min/max(分内極値を見ない=保守)。
  ・スプレッド: 現行出口も指値出口も同じ BID 系列の価格 + 同じ半スプレッド規約なので
    差分はコスト中立(Δret にスプレッド項は現れない)。
  ・感度A(strict-short): ショートの買い指値は ASK=BID+sp が L に達する必要
    → M1 low < L − sp に厳格化。
  ・感度B(rollover-strict): 指値判定バーが UTC20:00 の H4 バー(ロールオーバー帯
    20-24h を含む)のとき両方向とも 1 スプレッド分のクリアを要求
    ([[rollover-bid-artifact-m5]] の偽極値対策)。

look-ahead 自己監査(本実験の因果性):
  ・目標価格 L_b = m[b-1] は「バー b-1 の確定 close まで」の rolling mean のみで計算。
    バー b-1 の close 時刻(= バー b の開始時刻)に指値を発注/更新する運用に対応し、
    バー b の M1 パス([label_b, label_b+4h))の判定に未来情報は一切入らない。
  ・指値はエントリーバー e の「次のバー」(b = e+1)から有効。エントリー約定は e の close
    なので、発注時点で L_{e+1} = m[e] は確定済み。
  ・M1 約定判定はバー b 内の M1 高値/安値と「事前に固定された L_b」の比較のみ。
    fill 価格は L_b 固定(バー内の有利側極値を fill 価格にしない=後知恵の排除)。
  ・実行時アサート: 保有中の全判定バーで「発注時点の前バー close が指値の手前側」
    (ロング: close[b-1] < L_b)であることを検証(違反件数を報告。z≤−exit_z 保有条件から
    理論上 0 件のはず)。
  ・プール検算: n=1207 / sum(ret)=+1.9622(exp52 の d1 確定値)との一致を要求。

実行: PYTHONPATH=. uv run python research/experiments/exp71_exit_micro.py
出力: research/outputs/exp71_result.json / exp71_trades.csv

結論(2026-06-12 実行): **因果的指値出口(z=0 水準)は純劣化 — レバー閉鎖**。
  ・測定1(非因果の上限): 決済 H4 バー内の有利側極値は close 比 平均 +7.5bps /
    中央値 +4.9bps(>10bps は 24%)。極値で決済できればプール純益 +46.3% 相当だが後知恵。
    勝ち +7.4 / 負け +7.9bps と対称で「勝ちだけ取り残している」わけでもない。
  ・測定2(因果指値): fill率 30.6%(ほぼ全て決済バー内 28.3%、早期 fill は 2.3% のみ)。
    改善 13.3%(平均 +14.3bps)< 悪化 17.2%(平均 −16.7bps)で
    **純効果 −0.97bps/トレード = プール純益の −6.0%**。年次は 9/11 年マイナス
    (プラスは 2020 +2.0bps と 2022 +0.3bps のみ)= 安定して有害。
  ・機構: z 出口バーの約 7 割はそもそも前バー mean に M1 でも届かない(z>−exit_z は
    mean−0.5σ 越えで発火するため)。届く 3 割では close が mean をさらに平均 ~7bps
    突き抜けて確定する(決済バー内 fill n=341 の平均 −6.9bps)= z=0 指値は
    リバーサルバーの走り(オーバーシュート利益)を切り捨てる。早期 fill(n=28)は
    +42.5bps と良いが希少すぎて覆せない。
  ・付着先: 勝ちトレード −2.33bps(fill率 35.6%)/ 負け +2.47bps(fill率 17.8%)。
    「勝ちの尻尾を切って負けを僅かに繕う」改変で、勝ち増幅と真逆。
  ・感度: strict-short −1.11bps / rollover-strict −1.00bps、メジャー −0.89 /
    クロス −1.01bps = 規約・銘柄群に依らず符号不変。保有短縮も filled 平均 0.9 バーで無意味。
  ・口座換算(一次近似, w=f(z)/f̄ 加重): robust k=6.205 で ΔCAGR ≈ **−1.38pp/年**
    (haircut 込み −1.31pp)/ empirical k=8.895 で −1.97pp。
  → 「バー内に close より有利な価格が実在する」(+7.5bps)は事実だが、それは極値であって
    因果に置ける指値の置き場ではない。z=0 指値はオーバーシュート保有という既存エッジの
    一部を破壊する。指値出口レバーは閉鎖(より浅い目標=さらに早い利確はもっと悪く、
    より深い目標は fill 率が消える方向で、構造的に逃げ場がない)。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))

from mm_production import _fz, build_pool_d1  # noqa: E402
from fxlab import config, data  # noqa: E402
from fxlab import universe as uni  # noqa: E402
from fxlab.universe import CROSS_DEFS  # noqa: E402

pd.set_option("display.width", 240)

POOL_N_EXPECTED = 1207
POOL_SUM_EXPECTED = 1.9622  # exp52 の d1 確定値
WIN = 50                    # チャンピオン z の rolling window(PARAMS["window"])
MP = 8
ROB_K = 6.205               # robust 較正 k(5シード平均, reports/19)
EMP_K = 8.895               # empirical 較正 k(exp52)
HAIRCUT_D1 = 0.955          # M1粒度監査の掛け目(exp52)
OUT_JSON = ROOT / "research" / "outputs" / "exp71_result.json"
OUT_CSV = ROOT / "research" / "outputs" / "exp71_trades.csv"


def sec(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def q(v, p):
    return float(np.nanpercentile(v, p)) if len(v) else float("nan")


# ---------------------------------------------------------------------------
# M1 → H4 ビン極値(逐次ロード・処理後に解放)
# ---------------------------------------------------------------------------
def build_market(instruments: list[str]):
    """各銘柄の H4 close Series(戦略と同一系列)と H4 ビン M1 極値 {lo, hi} を構築。

    メジャー: 実 M1 high/low。クロス: 脚 M1 close の inner-join 合成の min/max(保守)。
    メジャーは 1 銘柄ずつロード → ビン化 → close だけ複製保持 → キャッシュ解放。
    """
    s_by: dict[str, pd.Series] = {}
    bins_by: dict[str, pd.DataFrame] = {}
    legs: dict[str, pd.Series] = {}

    majors = [nm for nm in instruments if nm not in CROSS_DEFS]
    crosses = [nm for nm in instruments if nm in CROSS_DEFS]

    for nm in majors:
        t0 = time.time()
        m1 = data.load_m1(nm)
        key = m1.index.floor("4h")
        bins_by[nm] = pd.DataFrame({"lo": m1["low"].groupby(key).min(),
                                    "hi": m1["high"].groupby(key).max()})
        s_by[nm] = data.resample(m1, "H4")["close"].copy()  # = load(nm,"H4")["close"] と同一
        legs[nm] = m1["close"].copy()
        del m1
        data.clear_cache()  # OHLCV フルフレームを解放(close 複製のみ保持)
        print(f"  {nm}: M1 bins {len(bins_by[nm]):,} / H4 {len(s_by[nm]):,}  "
              f"({time.time()-t0:.0f}s)")

    for nm in crosses:
        t0 = time.time()
        a, op, b = CROSS_DEFS[nm]
        df = pd.concat([legs[a].rename("a"), legs[b].rename("b")], axis=1,
                       join="inner").dropna()
        c = df["a"] / df["b"] if op == "/" else df["a"] * df["b"]
        del df
        key = c.index.floor("4h")
        bins_by[nm] = pd.DataFrame({"lo": c.groupby(key).min(), "hi": c.groupby(key).max()})
        del c
        dfh = pd.concat([s_by[a].rename("a"), s_by[b].rename("b")], axis=1).dropna()
        s_by[nm] = dfh["a"] / dfh["b"] if op == "/" else dfh["a"] * dfh["b"]
        del dfh
        print(f"  {nm}: synth bins {len(bins_by[nm]):,} / H4 {len(s_by[nm]):,}  "
              f"({time.time()-t0:.0f}s)")

    legs.clear()
    return s_by, bins_by


# ---------------------------------------------------------------------------
# 因果的指値出口の反実仮想スキャン
# ---------------------------------------------------------------------------
def scan_limit(pool: pd.DataFrame, s_by, bins_by, mode: str):
    """各トレードについて指値出口の最初の fill を探す。

    mode: "base"           long: hi > L            / short: lo < L
          "strict_short"   long: hi > L            / short: lo < L − sp
          "rollover"       判定バーが UTC20:00 の H4 バーなら両方向 +1sp のクリア要求
    返り値: filled(bool), fill_pos(int, H4バー位置), L(float), violations(int)
    """
    n = len(pool)
    filled = np.zeros(n, dtype=bool)
    fill_pos = np.full(n, -1)
    L_arr = np.full(n, np.nan)
    violations = 0

    for instr, g in pool.groupby("instr"):
        s = s_by[instr]
        sv = s.to_numpy()
        mv = s.rolling(WIN).mean().to_numpy()
        bins = bins_by[instr].reindex(s.index)
        hi = bins["hi"].to_numpy()
        lo = bins["lo"].to_numpy()
        hours = s.index.hour.to_numpy()
        sp = config.spread_pips(instr) * config.pip_size(instr)

        if mode == "base":
            cl_long = np.zeros(len(s))
            cl_short = np.zeros(len(s))
        elif mode == "strict_short":
            cl_long = np.zeros(len(s))
            cl_short = np.full(len(s), sp)
        elif mode == "rollover":
            roll = np.where(hours == 20, sp, 0.0)
            cl_long = roll.copy()
            cl_short = roll.copy()
        else:
            raise ValueError(mode)

        ie = s.index.get_indexer(g["entry"])
        ix = s.index.get_indexer(g["exit"])
        assert (ie >= 0).all() and (ix >= 0).all(), f"{instr}: timestamp miss"
        dirs = g["dir"].to_numpy()
        rows = g.index.to_numpy()

        for r, e, x, d in zip(rows, ie, ix, dirs):
            for b in range(e + 1, x + 1):
                L = mv[b - 1]
                if not np.isfinite(L):
                    continue
                # 因果監査: 発注時点(バー b-1 close)で指値は必ず有利側にあるはず
                if mode == "base":
                    if (d > 0 and sv[b - 1] >= L) or (d < 0 and sv[b - 1] <= L):
                        violations += 1
                hit = (hi[b] > L + cl_long[b]) if d > 0 else (lo[b] < L - cl_short[b])
                if hit:
                    filled[r] = True
                    fill_pos[r] = b
                    L_arr[r] = L
                    break
    return filled, fill_pos, L_arr, violations


def delta_of(pool, s_by, filled, L_arr):
    """Δret = dir·(L − Cx)/Ce(コスト中立: 半スプレッド規約は両出口で同一なので相殺)。"""
    n = len(pool)
    Ce = np.full(n, np.nan)
    Cx = np.full(n, np.nan)
    ix_arr = np.full(n, -1)
    for instr, g in pool.groupby("instr"):
        s = s_by[instr]
        sv = s.to_numpy()
        ie = s.index.get_indexer(g["entry"])
        ix = s.index.get_indexer(g["exit"])
        rows = g.index.to_numpy()
        Ce[rows] = sv[ie]
        Cx[rows] = sv[ix]
        ix_arr[rows] = ix
    d = pool["dir"].to_numpy().astype(float)
    delta = np.where(filled, d * (L_arr - Cx) / Ce, 0.0)
    return delta, Ce, Cx, ix_arr


def summarize(tag, pool, delta, filled, fill_pos, ix_arr, w):
    d_bps = delta * 1e4
    imp = filled & (delta > 0)
    wor = filled & (delta < 0)
    saved = (ix_arr - fill_pos)[filled]
    out = {
        "tag": tag,
        "fill_rate": float(filled.mean()),
        "fill_in_exitbar_rate": float((filled & (fill_pos == ix_arr)).mean()),
        "fill_early_rate": float((filled & (fill_pos < ix_arr)).mean()),
        "improved_rate": float(imp.mean()),
        "improved_mean_bps": float(d_bps[imp].mean()) if imp.any() else float("nan"),
        "worsened_rate": float(wor.mean()),
        "worsened_mean_bps": float(d_bps[wor].mean()) if wor.any() else float("nan"),
        "net_mean_bps": float(d_bps.mean()),
        "net_sum_ret": float(delta.sum()),
        "net_pct_of_pool": float(delta.sum() / POOL_SUM_EXPECTED * 100),
        "net_weighted_sum_ret": float((w * delta).sum()),
        "bars_saved_mean_filled": float(saved.mean()) if filled.any() else float("nan"),
        "bars_saved_median_filled": float(np.median(saved)) if filled.any() else float("nan"),
    }
    print(f"  [{tag}] fill率 {out['fill_rate']:.1%}"
          f"(決済バー内 {out['fill_in_exitbar_rate']:.1%} / それ以前 {out['fill_early_rate']:.1%})")
    print(f"        改善 {out['improved_rate']:.1%} (平均 {out['improved_mean_bps']:+.1f}bps) / "
          f"悪化 {out['worsened_rate']:.1%} (平均 {out['worsened_mean_bps']:+.1f}bps)")
    print(f"        純効果 {out['net_mean_bps']:+.2f}bps/トレード  ΣΔret {out['net_sum_ret']:+.4f} "
          f"(プール純益の {out['net_pct_of_pool']:+.1f}%)  "
          f"保有短縮(filled) 平均 {out['bars_saved_mean_filled']:.1f}バー")
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    t_start = time.time()
    uni.register_cross_spreads(3.0)

    sec("0. d1 プール(キャッシュ)と検算")
    pool = build_pool_d1()
    pool = pool.reset_index(drop=True)
    n, sret = len(pool), float(pool["ret"].sum())
    ok = (n == POOL_N_EXPECTED) and abs(sret - POOL_SUM_EXPECTED) < 1e-3
    print(f"pool n={n} sum(ret)={sret:+.4f}  (exp52 確定値 {POOL_N_EXPECTED}/"
          f"{POOL_SUM_EXPECTED:+.4f})  一致: {ok}")
    if not ok:
        print("!! プールが exp52 確定値と不一致 — 中断")
        return 1
    wins = (pool["ret"] > 0).to_numpy()
    year = pool["exit"].dt.year.to_numpy()
    years_span = (pool["exit"].max() - pool["entry"].min()).days / 365.25
    fz = np.array([_fz(z) for z in pool["z_entry"].to_numpy()])
    w = fz / fz.mean()  # champion_sizing と同じ正規化(f(z)/f̄)
    instruments = sorted(pool["instr"].unique())
    is_cross = pool["instr"].isin(CROSS_DEFS).to_numpy()

    sec("1. 市場データ構築(M1 ビン極値: 銘柄ごと逐次ロード・解放)")
    s_by, bins_by = build_market(instruments)

    # --- プール価格再構成の検算(exp45/52 流) -----------------------------
    delta0 = np.zeros(n)  # placeholder
    _, Ce, Cx, ix_arr = delta_of(pool, s_by, np.zeros(n, dtype=bool), delta0)
    d = pool["dir"].to_numpy().astype(float)
    gross = d * (Cx / Ce - 1.0)
    cost = gross - pool["ret"].to_numpy()
    sp_arr = np.array([config.spread_pips(i) * config.pip_size(i) for i in pool["instr"]])
    entry_eff = Ce + d * sp_arr / 2.0
    ep_err = np.abs(entry_eff - pool["entry_price"].to_numpy()) / pool["entry_price"].to_numpy()
    print(f"\n  検算: cost=gross−ret 全件正 {bool((cost > 0).all())}  "
          f"(median {np.median(cost)*1e4:.2f}bps)  "
          f"entry_price 再構成誤差 median {np.median(ep_err):.2e} / max {ep_err.max():.2e}")

    sec("2. 測定1: 決済 H4 バー内の有利幅(M1 極値 vs close 決済)— 非因果の上限")
    fav = np.full(n, np.nan)
    for instr, g in pool.groupby("instr"):
        s = s_by[instr]
        bins = bins_by[instr].reindex(s.index)
        hi = bins["hi"].to_numpy()
        lo = bins["lo"].to_numpy()
        ix = s.index.get_indexer(g["exit"])
        rows = g.index.to_numpy()
        dd = g["dir"].to_numpy().astype(float)
        ext = np.where(dd > 0, hi[ix], lo[ix])
        fav[rows] = dd * (ext - s.to_numpy()[ix]) / s.to_numpy()[ix] * 1e4
    n_nan = int(np.isnan(fav).sum())
    n_neg = int((fav < -0.01).sum())  # 合成クロスの末端分ズレ等
    fv = fav[np.isfinite(fav)]
    fav_stats = {
        "mean_bps": float(fv.mean()), "median_bps": q(fv, 50),
        "p25_bps": q(fv, 25), "p75_bps": q(fv, 75), "p90_bps": q(fv, 90),
        "p99_bps": q(fv, 99),
        "share_gt5bps": float((fv > 5).mean()), "share_gt10bps": float((fv > 10).mean()),
        "share_gt20bps": float((fv > 20).mean()),
        "mean_bps_win": float(fav[wins & np.isfinite(fav)].mean()),
        "mean_bps_loss": float(fav[~wins & np.isfinite(fav)].mean()),
        "mean_bps_major": float(fav[~is_cross & np.isfinite(fav)].mean()),
        "mean_bps_cross": float(fav[is_cross & np.isfinite(fav)].mean()),
        "n_nan": n_nan, "n_negative": n_neg,
    }
    # 上限 = バー内極値で決済できた場合の Δret(後知恵): dir·(ext−Cx)/Ce = (fav/1e4)·Cx/Ce
    ub = np.where(np.isfinite(fav), (fav / 1e4) * (Cx / Ce), 0.0)
    print(f"  有利幅: mean {fav_stats['mean_bps']:+.1f}bps / median {fav_stats['median_bps']:+.1f}bps"
          f" / p75 {fav_stats['p75_bps']:+.1f} / p90 {fav_stats['p90_bps']:+.1f}")
    print(f"  >5bps {fav_stats['share_gt5bps']:.0%} / >10bps {fav_stats['share_gt10bps']:.0%}"
          f" / >20bps {fav_stats['share_gt20bps']:.0%}   勝ち {fav_stats['mean_bps_win']:+.1f}"
          f" / 負け {fav_stats['mean_bps_loss']:+.1f}   メジャー {fav_stats['mean_bps_major']:+.1f}"
          f" / クロス {fav_stats['mean_bps_cross']:+.1f}")
    print(f"  (NaN {n_nan}件 / 負値 {n_neg}件=合成クロスの分末端ズレ)")
    print(f"  後知恵上限(決済バー極値で決済): ΣΔret {ub.sum():+.4f} = プール純益の "
          f"{ub.sum()/POOL_SUM_EXPECTED*100:+.1f}% — これは因果には取れない参考値")

    sec("3. 測定2: 因果的指値出口の反実仮想(L = 前バー rolling mean, z=0 水準)")
    filled, fill_pos, L_arr, viol = scan_limit(pool, s_by, bins_by, "base")
    print(f"  因果監査: 発注時点で指値が不利側にあった違反 = {viol}件(理論値 0)")
    delta, *_ = delta_of(pool, s_by, filled, L_arr)
    res_base = summarize("base", pool, delta, filled, fill_pos, ix_arr, w)

    # --- 勝ち/負け別 -------------------------------------------------------
    print("\n  -- 勝ち/負け別(元 ret 符号) --")
    by_wl = {}
    for lab, m in [("win", wins), ("loss", ~wins)]:
        dm = delta[m] * 1e4
        fm = filled[m]
        by_wl[lab] = {
            "n": int(m.sum()), "fill_rate": float(fm.mean()),
            "net_mean_bps": float(dm.mean()),
            "improved_rate": float(((delta > 0) & m).sum() / m.sum()),
            "worsened_rate": float(((delta < 0) & m).sum() / m.sum()),
            "net_sum_ret": float(delta[m].sum()),
        }
        print(f"  {lab:4s}: n={by_wl[lab]['n']:4d}  fill率 {by_wl[lab]['fill_rate']:.1%}  "
              f"純 {by_wl[lab]['net_mean_bps']:+.2f}bps  改善 {by_wl[lab]['improved_rate']:.1%}"
              f" / 悪化 {by_wl[lab]['worsened_rate']:.1%}  ΣΔ {by_wl[lab]['net_sum_ret']:+.4f}")

    # --- fill タイプ別(決済バー内 vs 早期) -------------------------------
    print("\n  -- fill タイプ別 --")
    in_exitbar = filled & (fill_pos == ix_arr)
    early = filled & (fill_pos < ix_arr)
    by_type = {}
    for key, lab, m in [("exit_bar", "決済バー内で先回り fill", in_exitbar),
                        ("early", "z出口より前のバーで fill", early)]:
        by_type[key] = {"n": int(m.sum()),
                        "mean_bps": float((delta[m] * 1e4).mean()) if m.any() else float("nan"),
                        "sum_ret": float(delta[m].sum())}
        print(f"  {lab}: n={by_type[key]['n']:4d}  平均 {by_type[key]['mean_bps']:+.2f}bps  "
              f"ΣΔ {by_type[key]['sum_ret']:+.4f}")

    # --- 年次 ---------------------------------------------------------------
    print("\n  -- 年次(決済年)純効果 --")
    ydf = pd.DataFrame({"year": year, "delta": delta, "wdelta": w * delta})
    yr_tab = ydf.groupby("year").agg(n=("delta", "size"), sum_ret=("delta", "sum"),
                                     mean_bps=("delta", lambda v: v.mean() * 1e4),
                                     wsum_ret=("wdelta", "sum"))
    print(yr_tab.round(4).to_string())
    n_pos_years = int((yr_tab["sum_ret"] > 0).sum())
    print(f"  プラス年 {n_pos_years}/{len(yr_tab)}")

    # --- メジャー/クロス別 --------------------------------------------------
    by_grp = {}
    for lab, m in [("major", ~is_cross), ("cross", is_cross)]:
        by_grp[lab] = {"n": int(m.sum()), "fill_rate": float(filled[m].mean()),
                       "net_mean_bps": float((delta[m] * 1e4).mean())}
    print(f"\n  メジャー: fill率 {by_grp['major']['fill_rate']:.1%} 純 "
          f"{by_grp['major']['net_mean_bps']:+.2f}bps / クロス: fill率 "
          f"{by_grp['cross']['fill_rate']:.1%} 純 {by_grp['cross']['net_mean_bps']:+.2f}bps")

    sec("4. 感度(約定規約)")
    sens = {}
    for mode, lab in [("strict_short", "strict-short(買指値は ASK 基準)"),
                      ("rollover", "rollover-strict(20:00バーは+1sp)")]:
        f2, fp2, L2, _ = scan_limit(pool, s_by, bins_by, mode)
        d2, *_ = delta_of(pool, s_by, f2, L2)
        sens[mode] = summarize(lab, pool, d2, f2, fp2, ix_arr, w)

    sec("5. 口座換算(概算)と判定")
    # ΔCAGR ≈ (k/max_pos) × Σ(w·Δret)/年数(複利・建玉スキップは無視した一次近似)
    wsum = float((w * delta).sum())
    dcagr_rob = ROB_K / MP * wsum / years_span
    dcagr_rob_hc = ROB_K * HAIRCUT_D1 / MP * wsum / years_span
    dcagr_emp = EMP_K / MP * wsum / years_span
    print(f"  Σ(w·Δret) = {wsum:+.4f} / {years_span:.1f}年")
    print(f"  ΔCAGR 概算: robust(k={ROB_K}) {dcagr_rob*100:+.2f}pp/年 "
          f"(haircut込み {dcagr_rob_hc*100:+.2f}pp) / empirical(k={EMP_K}) {dcagr_emp*100:+.2f}pp")
    pursue = (res_base["net_mean_bps"] > 0 and by_wl["win"]["net_mean_bps"] > 0
              and n_pos_years >= 7)
    print(f"  → 指値出口レバーは{'検証続行に値する' if pursue else '閉鎖(純効果が負/不安定)'}")

    # --- 保存 ---------------------------------------------------------------
    tr = pd.DataFrame({
        "instr": pool["instr"], "entry": pool["entry"], "exit": pool["exit"],
        "year": year, "dir": pool["dir"], "ret": pool["ret"], "win": wins,
        "bars_held": pool["bars_held"], "w_fz": w,
        "fav_exitbar_bps": fav,
        "filled": filled, "fill_pos": fill_pos, "exit_pos": ix_arr,
        "bars_saved": np.where(filled, ix_arr - fill_pos, 0),
        "L": L_arr, "delta_ret": delta, "delta_bps": delta * 1e4,
    })
    tr.to_csv(OUT_CSV, index=False)

    payload = {
        "pool": {"n": n, "sum_ret": sret, "wins": int(wins.sum()),
                 "years_span": years_span},
        "audit": {"placement_side_violations": int(viol),
                  "cost_all_positive": bool((cost > 0).all()),
                  "cost_median_bps": float(np.median(cost) * 1e4),
                  "entry_price_recon_err_max": float(ep_err.max())},
        "exit_bar_microstructure": fav_stats | {
            "hindsight_upper_bound_sum_ret": float(ub.sum()),
            "hindsight_upper_bound_pct_of_pool": float(ub.sum() / POOL_SUM_EXPECTED * 100)},
        "limit_exit_base": res_base,
        "by_winloss": by_wl,
        "by_fill_type": by_type,
        "by_group": by_grp,
        "yearly": {int(y): {"n": int(r["n"]), "sum_ret": float(r["sum_ret"]),
                            "mean_bps": float(r["mean_bps"]),
                            "wsum_ret": float(r["wsum_ret"])}
                   for y, r in yr_tab.iterrows()},
        "n_pos_years": n_pos_years,
        "sensitivity": {k: {kk: vv for kk, vv in v.items()} for k, v in sens.items()},
        "cagr_impact": {"w_delta_sum": wsum, "dcagr_rob_pp": dcagr_rob * 100,
                        "dcagr_rob_haircut_pp": dcagr_rob_hc * 100,
                        "dcagr_emp_pp": dcagr_emp * 100},
        "verdict": {"pursue": bool(pursue),
                    "net_mean_bps": res_base["net_mean_bps"],
                    "win_net_bps": by_wl["win"]["net_mean_bps"],
                    "loss_net_bps": by_wl["loss"]["net_mean_bps"]},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nsaved -> {OUT_JSON}\n      -> {OUT_CSV}")
    print(f"総経過 {time.time()-t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
