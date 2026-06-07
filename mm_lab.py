"""資金管理(マネーマネジメント)研究ラボ — チャンピオンv2の口座レベル・サイジング基盤。

目的: 「理論上の最大ドローダウン(DD)≤20% に収めつつ利益最大化」する資金管理を多角的に設計・実測する
ための共有インフラ。account_sim.py の発展版で、決定的に異なるのは **DD を含み損(MtM)込みで測る**こと。

  なぜ MtM か: チャンピオン(confluence_meanrev_v2)は無ストップの平均回帰で、戻るまで持ち続ける
  「塩漬け」=深い含み損ポジションを構造的に抱える([[champion-nostop-optimal]])。account_sim は
  決済時のみ equity を更新する(実現損益ベース)ので、口座が実際に経験するピーク→ボトムの真の DD を
  過小評価する。「理論上の最大DD」を厳密に縛るには、各バーで建玉中ポジションを時価評価した
  MtM equity 上で DD を測らねばならない。本ラボは realized / MtM 両方を出し、MtM を正とする。

主要 API:
  build_pool(tf, instruments, params)  : v2 の全トレードを (instr,entry,exit,dir,price,ret,z,vol) で収集(キャッシュ)
  load_closes(tf, instruments)         : 共通バーグリッド上の close 行列(MtM 評価用)
  simulate(pool, closes, sizing, ...)  : サイジング関数を差し替え可能なバー駆動・複利・同時建玉上限シミュレータ
  stats(eq_mtm, eq_real, ...)          : CAGR/最大DD(MtM&実現)/Sharpe/年次/プラス年率 等
  bootstrap_maxdd(mtm_returns, ...)    : ブロックブートストラップで最大DDの分布(理論DDの95/99%点)

サイジング関数の契約:
  sizing(ctx: dict) -> float   # この新規トレードに配分する「金額(ドル)」。0 を返すと見送り(建玉しない)。
  ctx のキー:
    equity_real   : 現時点の実現 equity
    equity_mtm    : 現時点の MtM equity(含み損込み=throttle/DD制御の基準に推奨)
    peak_mtm      : これまでの MtM equity ピーク
    dd_mtm        : 現在のMtMドローダウン(0以下。-0.1 = -10%)
    n_open        : 現在の建玉数
    max_pos       : 同時建玉上限
    recent_vol    : 直近 vol_win バーの MtM バーリターンの年率ボラ(vol ターゲット用, 立ち上がりは nan)
    z             : このトレードのエントリー時 |Z|(短期乖離の深さ, 乖離連動サイズ用)
    instr, ret, bars_held : 参考情報

実行: uv run python mm_lab.py            # プール生成 + ベースライン(固定比率)の DD-レバレッジ曲線
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fxlab import config, universe as uni
from fxlab.backtest import run
from fxlab.trades import trade_table
from strategies.confluence_meanrev_v2 import PARAMS as V2_PARAMS

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

TF_DEFAULT = "H4"
EXCLUDE_DEFAULT = ["AUDJPY"]
BARS_PER_YEAR = {"H4": 6 * 252, "H1": 24 * 252, "D1": 252}  # MtM Sharpe/年率ボラ換算


def default_instruments(exclude=EXCLUDE_DEFAULT):
    return [x for x in uni.universe(crosses=True) if x not in set(exclude)]


def _zscore(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


# --- トレードプール生成 -------------------------------------------------
def build_pool(tf=TF_DEFAULT, instruments=None, params=None, cross_spread=3.0,
               cache=True) -> pd.DataFrame:
    """v2 の全トレードを口座シミュ用の最小表に収集。

    各トレード: instr, entry, exit, dir(+1/-1), entry_price, ret(符号付き・往復コスト控除済),
    bars_held, z_entry(|短期Z|=乖離の深さ), vol_entry(エントリー時の20本ボラ)。
    """
    uni.register_cross_spreads(cross_spread)
    instruments = instruments or default_instruments()
    params = params or dict(V2_PARAMS)
    win = params.get("window", 50)

    cache_path = config.RESULTS_DIR / f"mm_pool_v2_{tf}_{len(instruments)}.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    frames = []
    for nm in instruments:
        data = uni.instrument_data(nm, tf)
        pf = run(nm, tf, __import__("strategies.confluence_meanrev_v2", fromlist=["generate_signals"]).generate_signals,
                 params, data=data, size_mode="value")
        tt = trade_table(pf, data)
        if tt.empty:
            continue
        close = data["close"]
        z = _zscore(close, win)
        vol = close.pct_change().rolling(20).std()
        zlu = z.reindex(tt["entry"]).to_numpy()
        vlu = vol.reindex(tt["entry"]).to_numpy()
        f = pd.DataFrame({
            "instr": nm,
            "entry": tt["entry"].to_numpy(),
            "exit": tt["exit"].to_numpy(),
            "dir": np.where(tt["dir"].to_numpy() == "Long", 1, -1),
            "entry_price": tt["entry_price"].to_numpy(),
            "ret": tt["return_pct"].to_numpy() / 100.0,
            "bars_held": tt["bars_held"].to_numpy(),
            "z_entry": np.abs(zlu),
            "vol_entry": vlu,
        })
        frames.append(f)
    pool = pd.concat(frames, ignore_index=True).sort_values("entry").reset_index(drop=True)
    if cache:
        pool.to_parquet(cache_path)
    return pool


def load_closes(tf=TF_DEFAULT, instruments=None, cross_spread=3.0, cache=True) -> pd.DataFrame:
    """共通バーグリッド上の close 行列(列=instr)。MtM 評価用に ffill。"""
    uni.register_cross_spreads(cross_spread)
    instruments = instruments or default_instruments()
    cache_path = config.RESULTS_DIR / f"mm_closes_{tf}_{len(instruments)}.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)
    closes = pd.DataFrame({nm: uni.instrument_close(nm, tf) for nm in instruments})
    closes = closes.sort_index().ffill()
    if cache:
        closes.to_parquet(cache_path)
    return closes


# --- バー駆動シミュレータ(サイジング差替可能, MtM 対応) ----------------
def simulate(pool: pd.DataFrame, closes: pd.DataFrame, sizing, *, init=10_000.0,
             max_pos=6, vol_win=120):
    """サイジング関数を差し替え可能な単一口座・複利・同時建玉上限シミュレーション。

    バー毎に: ①決済反映(実現) ②MtM equity 算出・記録 ③新規エントリーは sizing(ctx) の金額で建玉
    (room があり sizing>0 のときのみ)。返り値: eq_mtm(Series), eq_real(Series), info(dict)。
    """
    grid = closes.index
    col_of = {c: i for i, c in enumerate(closes.columns)}
    carr = closes.to_numpy()
    n = len(grid)

    # entry/exit をバー位置へ写像(exit はグリッド上の位置, 無ければ次の有効位置)
    pos_of = pd.Series(np.arange(n), index=grid)
    entry_pos = pos_of.reindex(pool["entry"]).to_numpy()
    exit_pos = pos_of.reindex(pool["exit"]).to_numpy()
    # entry/exit が厳密一致しない場合に備え searchsorted で補正
    gi = grid.to_numpy()
    e_raw = pool["entry"].to_numpy()
    x_raw = pool["exit"].to_numpy()
    entry_pos = np.searchsorted(gi, e_raw, side="left")
    exit_pos = np.searchsorted(gi, x_raw, side="left")
    entry_pos = np.clip(entry_pos, 0, n - 1)
    exit_pos = np.clip(exit_pos, 0, n - 1)

    # エントリーバー毎にトレード index をまとめる
    by_entry = {}
    for ti in range(len(pool)):
        by_entry.setdefault(int(entry_pos[ti]), []).append(ti)

    instr_arr = pool["instr"].to_numpy()
    dir_arr = pool["dir"].to_numpy().astype(float)
    eprice_arr = pool["entry_price"].to_numpy()
    ret_arr = pool["ret"].to_numpy()
    z_arr = pool["z_entry"].to_numpy()
    bars_arr = pool["bars_held"].to_numpy()

    equity = init                    # 実現 equity
    peak_mtm = init
    open_pos = []                    # dict(ti, col, dir, eprice, alloc, exit_pos, ret)
    eq_mtm = np.empty(n)
    eq_real = np.empty(n)
    mtm_ret_hist = np.empty(n)       # MtM バーリターン(vol ターゲット用)
    prev_mtm = init
    conc = []
    skipped = 0

    for b in range(n):
        # ① 決済(exit_pos == b のもの)
        if open_pos:
            still = []
            for p in open_pos:
                if p["exit_pos"] <= b:
                    equity += p["alloc"] * p["ret"]
                else:
                    still.append(p)
            open_pos = still

        # ② MtM equity = 実現 + 建玉の含み損益
        unreal = 0.0
        for p in open_pos:
            px = carr[b, p["col"]]
            run_ret = p["dir"] * (px / p["eprice"] - 1.0)
            unreal += p["alloc"] * run_ret
        mtm = equity + unreal
        eq_mtm[b] = mtm
        eq_real[b] = equity
        peak_mtm = max(peak_mtm, mtm)
        mtm_ret_hist[b] = (mtm / prev_mtm - 1.0) if prev_mtm > 0 else 0.0
        prev_mtm = mtm

        # 直近ボラ(年率)— vol ターゲット用
        if b >= vol_win:
            rv = mtm_ret_hist[b - vol_win + 1:b + 1]
            recent_vol = float(np.std(rv) * np.sqrt(BARS_PER_YEAR.get("H4", 1512)))
        else:
            recent_vol = float("nan")

        dd_mtm = mtm / peak_mtm - 1.0

        # ③ 新規エントリー
        if b in by_entry:
            for ti in by_entry[b]:
                if len(open_pos) >= max_pos:
                    skipped += 1
                    continue
                ctx = {
                    "equity_real": equity, "equity_mtm": mtm, "peak_mtm": peak_mtm,
                    "dd_mtm": dd_mtm, "n_open": len(open_pos), "max_pos": max_pos,
                    "recent_vol": recent_vol, "z": float(z_arr[ti]),
                    "instr": instr_arr[ti], "ret": float(ret_arr[ti]),
                    "bars_held": int(bars_arr[ti]),
                }
                alloc = float(sizing(ctx))
                if alloc <= 0:
                    skipped += 1
                    continue
                open_pos.append({
                    "ti": ti, "col": col_of[instr_arr[ti]], "dir": dir_arr[ti],
                    "eprice": eprice_arr[ti], "alloc": alloc,
                    "exit_pos": int(exit_pos[ti]), "ret": float(ret_arr[ti]),
                })
                conc.append(len(open_pos))

    eq_mtm = pd.Series(eq_mtm, index=grid)
    eq_real = pd.Series(eq_real, index=grid)
    info = {"final": equity, "skipped": skipped, "n_taken": len(conc),
            "max_conc": max(conc) if conc else 0,
            "avg_conc": float(np.mean(conc)) if conc else 0.0}
    return eq_mtm, eq_real, info


# --- 指標 ---------------------------------------------------------------
def _max_dd(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1.0).min())


def stats(eq_mtm: pd.Series, eq_real: pd.Series, info: dict, init=10_000.0, tf="H4") -> dict:
    years = (eq_mtm.index[-1] - eq_mtm.index[0]).days / 365.25
    final = eq_mtm.iloc[-1]
    cagr = (final / init) ** (1 / years) - 1 if final > 0 else -1.0
    dd_mtm = _max_dd(eq_mtm)
    dd_real = _max_dd(eq_real)
    rets = eq_mtm.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    bpy = BARS_PER_YEAR.get(tf, 1512)
    sharpe = rets.mean() / rets.std() * np.sqrt(bpy) if rets.std() > 0 else float("nan")
    downside = rets[rets < 0].std()
    sortino = rets.mean() / downside * np.sqrt(bpy) if downside and downside > 0 else float("nan")
    yearly = eq_mtm.groupby(eq_mtm.index.year).last()
    yr_ret = yearly.pct_change()
    yr_ret.iloc[0] = yearly.iloc[0] / init - 1
    return {
        "total_return": final / init - 1, "cagr": cagr,
        "maxdd_mtm": dd_mtm, "maxdd_real": dd_real,
        "sharpe": sharpe, "sortino": sortino,
        "pos_year_rate": float((yr_ret > 0).mean()), "n_years": len(yr_ret),
        "worst_year": float(yr_ret.min()),
        "final": final, "max_conc": info["max_conc"], "skipped": info["skipped"],
        "n_taken": info["n_taken"], "yr_ret": yr_ret,
    }


def bootstrap_maxdd(eq_mtm: pd.Series, n_boot=2000, block=63, seed=0) -> dict:
    """MtM バーリターンのブロックブートストラップで最大DDの分布(理論DD)を出す。

    ブロック長で塩漬け期のクラスタリング(自己相関)を保持。同じ長さの系列を合成し、
    各サンプルの最大DDを集計。p50/p95/p99 を返す(20%以内に収めるべき"理論上"の上振れ)。
    """
    r = eq_mtm.pct_change().replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    n = len(r)
    if n < block * 2:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan")}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_all = rng.integers(0, n - block, size=(n_boot, n_blocks))
    dds = np.empty(n_boot)
    for i in range(n_boot):
        idx = (starts_all[i][:, None] + np.arange(block)).ravel()[:n]
        path = np.cumprod(1.0 + r[idx])
        peak = np.maximum.accumulate(path)
        dds[i] = (path / peak - 1.0).min()
    return {"p50": float(np.percentile(dds, 50)), "p95": float(np.percentile(dds, 5)),
            "p99": float(np.percentile(dds, 1)), "mean": float(dds.mean()),
            "worst": float(dds.min())}


# --- 較正(全手法を同一の MtM最大DD に揃えて公平比較) ------------------
def maxdd_mtm_of(pool, closes, sizing, max_pos=6):
    eqm, eqr, info = simulate(pool, closes, sizing, max_pos=max_pos)
    return abs(_max_dd(eqm)), eqm, eqr, info


def calibrate(pool, closes, make_sizing, target_dd=0.20, max_pos=6,
              lo=0.02, hi=12.0, iters=24):
    """総エクスポージャ倍率 k を二分探索し、経験的 MtM 最大DD == target_dd に揃える。

    make_sizing(k) は「k 倍だけ総建玉を線形スケールする」サイジング関数を返すこと
    (=手法の"配分の形"は固定、全体の大きさだけ k で動く)。|MtM DD| は k に単調増加。
    返り値: (k, eqm, eqr, info)  ※ DD<=target を満たす最大の k。
    """
    # hi が target に届かない(=どれだけ張っても DD が浅い)場合は hi で頭打ち
    dd_hi, *_ = maxdd_mtm_of(pool, closes, make_sizing(hi), max_pos)
    if dd_hi <= target_dd:
        eqm, eqr, info = simulate(pool, closes, make_sizing(hi), max_pos=max_pos)
        return hi, eqm, eqr, info
    for _ in range(iters):
        mid = (lo + hi) / 2
        dd, *_ = maxdd_mtm_of(pool, closes, make_sizing(mid), max_pos)
        if dd > target_dd:
            hi = mid
        else:
            lo = mid
    eqm, eqr, info = simulate(pool, closes, make_sizing(lo), max_pos=max_pos)
    return lo, eqm, eqr, info


def calibrate_robust(pool, closes, make_sizing, target_dd=0.20, max_pos=6,
                     n_boot=600, block=63, lo=0.02, hi=12.0, iters=18):
    """総エクスポージャ倍率 k を二分探索し、**ブートストラップ理論DD(p95)** == target_dd に揃える。

    経験的(単一パス)最大DDではなく、ブロックブートストラップ p95(20回に1回級の上振れDD)を
    20%に縛る厳しめの解釈。返り値: (k, eqm, eqr, info, p95)。
    """
    def p95_of(kk):
        eqm, _, _ = simulate(pool, closes, make_sizing(kk), max_pos=max_pos)
        return abs(bootstrap_maxdd(eqm, n_boot=n_boot, block=block)["p95"])
    if p95_of(hi) <= target_dd:
        eqm, eqr, info = simulate(pool, closes, make_sizing(hi), max_pos=max_pos)
        return hi, eqm, eqr, info, target_dd
    for _ in range(iters):
        mid = (lo + hi) / 2
        if p95_of(mid) > target_dd:
            hi = mid
        else:
            lo = mid
    eqm, eqr, info = simulate(pool, closes, make_sizing(lo), max_pos=max_pos)
    p95 = abs(bootstrap_maxdd(eqm, n_boot=n_boot, block=block)["p95"])
    return lo, eqm, eqr, info, p95


def evaluate_method(name, pool, closes, make_sizing, *, target_dd=0.20, max_pos=6,
                    tf=TF_DEFAULT, oos_start="2022-01-01", n_boot=1500) -> dict:
    """1手法を「経験的 MtM 最大DD=target に較正→CAGR等を測る」フル評価。

    返り値: フル期間の CAGR/DD/Sharpe/プラス年 + 理論DD(bootstrap p95/p99) +
    過剰最適化チェック(IS で較正→OOS で素の DD と CAGR)。
    """
    k, eqm, eqr, info = calibrate(pool, closes, make_sizing, target_dd, max_pos)
    s = stats(eqm, eqr, info, tf=tf)
    bs = bootstrap_maxdd(eqm, n_boot=n_boot)

    # IS で較正 → OOS で素検証(DD制御系の"過去当てはめ"を暴く)
    # グリッドも IS/OOS で切る(全期間グリッドのままだと未取引年がフラット=指標が歪む)
    is_pool = pool[pool["entry"] < oos_start].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= oos_start].reset_index(drop=True)
    is_closes = closes[closes.index < oos_start]
    oos_closes = closes[closes.index >= oos_start]
    oos = {}
    try:
        k_is, *_ = calibrate(is_pool, is_closes, make_sizing, target_dd, max_pos)
        eqm_o, eqr_o, info_o = simulate(oos_pool, oos_closes, make_sizing(k_is), max_pos=max_pos)
        so = stats(eqm_o, eqr_o, info_o, tf=tf)
        oos = {"k_is": k_is, "oos_cagr": so["cagr"], "oos_maxdd_mtm": so["maxdd_mtm"],
               "oos_pos_year": so["pos_year_rate"], "oos_sharpe": so["sharpe"]}
    except Exception as e:  # noqa: BLE001
        oos = {"error": str(e)}

    return {
        "method": name, "k": k,
        "cagr": s["cagr"], "total_return": s["total_return"],
        "maxdd_mtm": s["maxdd_mtm"], "maxdd_real": s["maxdd_real"],
        "sharpe": s["sharpe"], "sortino": s["sortino"],
        "pos_year_rate": s["pos_year_rate"], "worst_year": s["worst_year"],
        "boot_p95": bs["p95"], "boot_p99": bs["p99"], "boot_worst": bs["worst"],
        "max_conc": s["max_conc"], "n_taken": s["n_taken"], "skipped": s["skipped"],
        **oos,
    }


# --- 既製サイジング(ベースライン用) -----------------------------------
def fixed_fractional(deploy=1.0, max_pos=6):
    """満玉時に deploy 比率を運用する固定比率(account_sim 互換)。alloc = equity*deploy/max_pos。"""
    w = deploy / max_pos

    def _f(ctx):
        return ctx["equity_real"] * w
    return _f


def main() -> int:
    ap = argparse.ArgumentParser(description="資金管理ラボ: プール生成 + 固定比率 DD-レバレッジ曲線")
    ap.add_argument("--tf", default=TF_DEFAULT)
    ap.add_argument("--max-pos", type=int, default=6)
    ap.add_argument("--rebuild", action="store_true", help="プールを再生成")
    args = ap.parse_args()

    instruments = default_instruments()
    print(f"=== mm_lab: チャンピオンv2 資金管理基盤 ({args.tf}, 対象{len(instruments)}) ===")
    pool = build_pool(tf=args.tf, instruments=instruments, cache=not args.rebuild)
    closes = load_closes(tf=args.tf, instruments=instruments)
    print(f"トレード総数 {len(pool)}(年平均 {len(pool)/11:.0f}) / グリッド {len(closes)}本 / "
          f"建玉上限 {args.max_pos}\n")

    print("=== ベースライン: 固定比率(account_sim互換)の DD-レバレッジ曲線 ===")
    print("  deploy   総収益    CAGR    最大DD(MtM)  最大DD(実現)  Sharpe  ブート理論DD(p95)  プラス年")
    for deploy in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        eqm, eqr, info = simulate(pool, closes, fixed_fractional(deploy, args.max_pos),
                                  max_pos=args.max_pos)
        s = stats(eqm, eqr, info, tf=args.tf)
        bs = bootstrap_maxdd(eqm, n_boot=1000)
        print(f"  {deploy:>4.1f}x  {s['total_return']:>+8.1%}  {s['cagr']:>+6.1%}  "
              f"{s['maxdd_mtm']:>9.1%}  {s['maxdd_real']:>10.1%}  {s['sharpe']:>6.2f}  "
              f"{bs['p95']:>13.1%}      {s['pos_year_rate']:>4.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
