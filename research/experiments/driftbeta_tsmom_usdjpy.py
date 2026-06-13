"""検証1: USDJPY H1 lb24 tsmom が drift-beta か真の双方向モメンタムか。

(a) long/short/both GROSS Sharpe 比較
(b) 各暦年の GROSS PnL と USDJPY buy&hold 方向の関係
(c) USDJPY ドリフト(期間平均リターン)除去後に Sharpe が残るか
(d) reports/17 ドリフトベータ署名(年次PnLが原資産騰落と符号一致)該当性

GROSS = スプレッド0・手数料0。run() の vectorbt 評価と、
position-sign × asset-return の手計算ストリームの双方で確認する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import fxlab.config as C
from fxlab import metrics, run
from fxlab import universe as uni
from strategies.tsmom import generate_signals

PAIR = "USDJPY"
TF = "H1"
LB = 24
BAND = 0.0

# H1 の年率化係数(おおよそ。week=5d, 24h)
BARS_PER_YEAR = 24 * 252


def _position_series(data: pd.DataFrame, lookback: int, band: float) -> pd.Series:
    """tsmom のドテン後の保有方向(+1/-1/0)を各バーで返す。

    long_state/short_state はエントリー瞬間のみ True なので、ドテン挙動を
    再現するため state(mom>band / mom<-band)をそのまま方向とみなす。
    band=0 では常時 +1/-1 のフルインベスト(0 は mom==0 の境界のみ)。
    実約定は次バー以降に効くので方向を1バー shift して資産リターンに当てる。
    """
    close = data["close"]
    mom = close / close.shift(lookback) - 1.0
    pos = pd.Series(0, index=close.index, dtype=float)
    pos[mom > band] = 1.0
    pos[mom < -band] = -1.0
    return pos


def _gross_sharpe_via_run(side: str) -> dict:
    """run() を使い GROSS(コスト0)で side 別の Sharpe ほか。"""
    data = uni.instrument_data(PAIR, TF)
    pf = run(PAIR, TF, generate_signals, {"lookback": LB, "band": BAND},
             data=data, size_mode="value", side=side)
    m = metrics(pf)
    row = m.iloc[0] if isinstance(m, pd.DataFrame) else m
    return {k: float(row[k]) for k in
            ["total_return", "sharpe", "sortino", "max_drawdown",
             "win_rate", "profit_factor", "num_trades", "expectancy"]}


def _stream_sharpe(asset_ret: pd.Series, pos: pd.Series) -> tuple[float, pd.Series]:
    """position-sign(1バー遅延) × asset_ret のストリーム Sharpe と PnL系列。"""
    strat = (pos.shift(1) * asset_ret).dropna()
    mu, sd = strat.mean(), strat.std()
    sharpe = float(mu / sd * np.sqrt(BARS_PER_YEAR)) if sd > 0 else float("nan")
    return sharpe, strat


def main() -> None:
    # GROSS にするためスプレッドを0に上書き(プロセス内のみ)
    orig = dict(C.SPREADS_PIPS)
    C.SPREADS_PIPS[PAIR] = 0.0
    C.COMMISSION_FRACTION = 0.0

    data = uni.instrument_data(PAIR, TF).copy()
    close = data["close"]
    asset_ret = close.pct_change()           # 単純リターン
    asset_logret = np.log(close).diff()      # ログリターン(ドリフト除去用)

    pos = _position_series(data, LB, BAND)

    print("=" * 70)
    print(f"検証1: {PAIR} {TF} tsmom lb={LB} band={BAND} — drift-beta判定")
    print("=" * 70)

    # (a) long/short/both GROSS Sharpe -----------------------------------
    print("\n[a] side別 GROSS(コスト0)成績 — run()/vectorbt")
    side_rows = {}
    for side in ["both", "long", "short"]:
        r = _gross_sharpe_via_run(side)
        side_rows[side] = r
        print(f"  {side:5s}: Sharpe={r['sharpe']:+.3f}  totret={r['total_return']:+.3f}"
              f"  PF={r['profit_factor']:.2f}  trades={int(r['num_trades'])}"
              f"  expectancy={r['expectancy']:+.4f}")

    # ストリーム手計算でも side 別 Sharpe(位置×リターン)
    print("\n[a'] side別 GROSS ストリーム Sharpe(pos.shift(1)×asset_ret)")
    pos_long = pos.clip(lower=0)
    pos_short = pos.clip(upper=0)
    sh_both, st_both = _stream_sharpe(asset_ret, pos)
    sh_long, _ = _stream_sharpe(asset_ret, pos_long)
    sh_short, _ = _stream_sharpe(asset_ret, pos_short)
    print(f"  both : {sh_both:+.3f}")
    print(f"  long : {sh_long:+.3f}")
    print(f"  short: {sh_short:+.3f}")

    # (b) 各暦年 GROSS PnL と USDJPY buy&hold 方向 ------------------------
    print("\n[b] 暦年別: tsmom GROSS PnL vs USDJPY buy&hold")
    st_long = (pos_long.shift(1) * asset_ret).dropna()
    st_short = (pos_short.shift(1) * asset_ret).dropna()
    yrs = sorted(set(st_both.index.year))
    rows = []
    for y in yrs:
        strat_pnl = float(st_both[st_both.index.year == y].sum())   # その年のストリーム累積
        # buy&hold その年のリターン
        cy = close[close.index.year == y]
        if len(cy) < 2:
            continue
        bh = float(cy.iloc[-1] / cy.iloc[0] - 1.0)
        # ロング/ショートそれぞれの寄与
        long_pnl = float(st_long[st_long.index.year == y].sum())
        short_pnl = float(st_short[st_short.index.year == y].sum())
        rows.append((y, strat_pnl, bh, long_pnl, short_pnl))
    yr_df = pd.DataFrame(rows, columns=["year", "tsmom_pnl", "usdjpy_bh", "long_pnl", "short_pnl"])
    print(yr_df.to_string(index=False,
          formatters={c: (lambda v: f"{v:+.4f}") for c in
                      ["tsmom_pnl", "usdjpy_bh", "long_pnl", "short_pnl"]}))

    # 下落年(bh<0)でも tsmom がプラスを出すか
    down = yr_df[yr_df["usdjpy_bh"] < 0]
    up = yr_df[yr_df["usdjpy_bh"] >= 0]
    print(f"\n  USDJPY下落年(bh<0): {list(down['year'])}")
    print(f"    その年の tsmom PnL: {[f'{v:+.3f}' for v in down['tsmom_pnl']]}")
    print(f"    下落年で tsmom プラスの割合: "
          f"{(down['tsmom_pnl'] > 0).sum()}/{len(down)}")
    print(f"  USDJPY上昇年(bh>=0): {list(up['year'])}")
    print(f"    上昇年で tsmom プラスの割合: "
          f"{(up['tsmom_pnl'] > 0).sum()}/{len(up)}")

    # (d) ドリフトベータ署名: 年次 tsmom PnL の符号 が usdjpy_bh の符号 と一致するか
    sign_match = (np.sign(yr_df["tsmom_pnl"]) == np.sign(yr_df["usdjpy_bh"])).sum()
    n_yr = len(yr_df)
    corr = float(yr_df["tsmom_pnl"].corr(yr_df["usdjpy_bh"]))
    # long寄与だけの符号一致(ロングがドリフトに乗っているか)
    sign_match_long = (np.sign(yr_df["long_pnl"]) == np.sign(yr_df["usdjpy_bh"])).sum()
    print(f"\n[d] ドリフトベータ署名チェック")
    print(f"  年次 tsmom_pnl の符号が usdjpy_bh と一致: {sign_match}/{n_yr}")
    print(f"  年次 long_pnl の符号が usdjpy_bh と一致 : {sign_match_long}/{n_yr}")
    print(f"  corr(tsmom_pnl, usdjpy_bh) = {corr:+.3f}")

    # (c) ドリフト除去テスト ---------------------------------------------
    # 各バーの asset_logret から期間平均(ドリフト)を引いた系列で再評価。
    # ドリフトを引いても tsmom 方向決定に使う mom もドリフト除去版にすべき。
    print("\n[c] ドリフト除去テスト")
    drift = asset_logret.mean()
    demeaned_logret = asset_logret - drift
    # ドリフト除去後の価格(累積)でモメンタム状態を作り直す
    demeaned_close = np.exp(demeaned_logret.fillna(0).cumsum()) * close.iloc[0]
    demeaned_close = pd.Series(demeaned_close, index=close.index)
    mom_dm = demeaned_close / demeaned_close.shift(LB) - 1.0
    pos_dm = pd.Series(0.0, index=close.index)
    pos_dm[mom_dm > BAND] = 1.0
    pos_dm[mom_dm < -BAND] = -1.0

    # 評価リターンもドリフト除去版(超過リターン)
    strat_dm = (pos_dm.shift(1) * demeaned_logret).dropna()
    mu, sd = strat_dm.mean(), strat_dm.std()
    sh_dm = float(mu / sd * np.sqrt(BARS_PER_YEAR)) if sd > 0 else float("nan")

    # 比較用: ドリフト込みの同手計算 Sharpe(ログリターンで)
    strat_raw_log = (pos.shift(1) * asset_logret).dropna()
    mu0, sd0 = strat_raw_log.mean(), strat_raw_log.std()
    sh_raw_log = float(mu0 / sd0 * np.sqrt(BARS_PER_YEAR)) if sd0 > 0 else float("nan")

    # 補助: 元の方向(pos)のままドリフト除去リターンに当てる(方向はドリフト込みで決め、評価だけ超過)
    strat_demeval = (pos.shift(1) * demeaned_logret).dropna()
    mu1, sd1 = strat_demeval.mean(), strat_demeval.std()
    sh_demeval = float(mu1 / sd1 * np.sqrt(BARS_PER_YEAR)) if sd1 > 0 else float("nan")

    print(f"  年率ドリフト(asset logret mean × BARS/yr) = {drift * BARS_PER_YEAR:+.4f}")
    print(f"  ドリフト込み(参照)Sharpe(log)            = {sh_raw_log:+.3f}")
    print(f"  ドリフト除去(方向ドリフト込み・評価超過)   = {sh_demeval:+.3f}")
    print(f"  ドリフト除去(方向も評価も超過リターン)     = {sh_dm:+.3f}")
    # long/short 分解(完全除去版)
    pos_dm_long = pos_dm.clip(lower=0)
    pos_dm_short = pos_dm.clip(upper=0)
    sdl = (pos_dm_long.shift(1) * demeaned_logret).dropna()
    sds = (pos_dm_short.shift(1) * demeaned_logret).dropna()
    shdl = float(sdl.mean() / sdl.std() * np.sqrt(BARS_PER_YEAR)) if sdl.std() > 0 else float("nan")
    shds = float(sds.mean() / sds.std() * np.sqrt(BARS_PER_YEAR)) if sds.std() > 0 else float("nan")
    print(f"    除去後 long : {shdl:+.3f}")
    print(f"    除去後 short: {shds:+.3f}")

    # IS/OOS でも除去後 Sharpe
    for label, sl in [("IS 2016-2020", slice("2016", "2020")),
                      ("OOS 2021-2026", slice("2021", "2026"))]:
        seg = strat_dm.loc[sl]
        if len(seg) > 10:
            s = float(seg.mean() / seg.std() * np.sqrt(BARS_PER_YEAR)) if seg.std() > 0 else float("nan")
            print(f"    除去後 {label}: Sharpe={s:+.3f}")

    # 復元
    C.SPREADS_PIPS.clear()
    C.SPREADS_PIPS.update(orig)

    print("\n" + "=" * 70)
    print("要約数値(machine):")
    print(f"GROSS_run both/long/short Sharpe = "
          f"{side_rows['both']['sharpe']:+.3f}/{side_rows['long']['sharpe']:+.3f}/{side_rows['short']['sharpe']:+.3f}")
    print(f"stream both/long/short Sharpe = {sh_both:+.3f}/{sh_long:+.3f}/{sh_short:+.3f}")
    print(f"drift_annual = {drift * BARS_PER_YEAR:+.4f}")
    print(f"demean_full Sharpe = {sh_dm:+.3f} (long {shdl:+.3f}/short {shds:+.3f})")
    print(f"yr sign-match tsmom/bh = {sign_match}/{n_yr}, long/bh = {sign_match_long}/{n_yr}, corr = {corr:+.3f}")
    print(f"down-years tsmom positive = {(down['tsmom_pnl'] > 0).sum()}/{len(down)}")


if __name__ == "__main__":
    main()
