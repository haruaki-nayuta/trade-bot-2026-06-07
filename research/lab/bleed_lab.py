"""失血窓(bleed window)ベースの補完エッジ吟味 — チャンピオンv2の資金管理を底上げする土台。

発想(reports/09 後の戦略転換): 補完エッジを「平均相関の低さ」で選ぶのは浅い。DD≤20% に効く分散は
**チャンピオンが失血しているまさにその窓で稼ぐ**エッジだけ。平均無相関でも同じ窓で沈めばテール(=DD)に効かない。
→ 候補を測る物差しを「平均相関」から「**チャンピオンの失血窓における条件付きリターン**」へ変える。

本モジュール(全エージェント共通の検証済み基盤):
  champion_mtm()         : チャンピオン(z-size mp8, 20%較正)の MtM equity 曲線(失血のタイミング源)
  bleed_mask_monthly()   : 月次の「失血窓」マスク(MtM が水面下=ドローダウン中の月)
  regime_features()      : 月次レジーム特徴(バスケット trendiness=|ER|, vol)— 失血窓の正体の特徴づけ
  strategy_monthly_pnl() : 任意戦略の月次PnL(value $10k/銘柄を決済月で合算)= 候補のP&Lストリーム
  conditional_score()    : 候補月次PnL を失血窓 vs 平時で評価(窓内平均/合計/勝率/IS・OOS)= 吟味の核
  integrated_dd_test()   : champion + overlay を1口座(MtM)に統合し DD=20% 較正 → CAGR が改善するか(最終判定)

実行: uv run python bleed_lab.py     # チャンピオンの失血窓を特徴づけ(2022・高ER・塩漬けクラスタの確認)
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

import mm_lab as mm
from fxlab import config, universe as uni

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

TF = "H4"
DD_BLEED_THRESH = 0.05  # MtM が直近ピークから -5% 超で水面下 = 失血窓


# --- チャンピオンの MtM equity(失血タイミングの源)----------------------
def champion_mtm(max_pos=8):
    pool = mm.build_pool()
    closes = mm.load_closes()
    from mm_production import champion_sizing
    mk = champion_sizing(pool, max_pos=max_pos)
    k, eqm, eqr, info = mm.calibrate(pool, closes, mk, target_dd=0.20, max_pos=max_pos)
    return eqm, eqr, pool, closes


def _monthly_last(eq: pd.Series) -> pd.Series:
    return eq.groupby(eq.index.to_period("M")).last()


# --- 失血窓マスク(月次)------------------------------------------------
def bleed_mask_monthly(eqm: pd.Series, thresh=DD_BLEED_THRESH):
    """月末 MtM equity がドローダウン中(直近ピーク比 -thresh 超)の月= True。

    返り値: (mask: PeriodIndex(M)->bool, dd_monthly: 同 index の月末DD)。
    """
    me = _monthly_last(eqm)
    dd = me / me.cummax() - 1.0
    mask = dd < -thresh
    return mask, dd


# --- レジーム特徴(失血窓の正体)----------------------------------------
def _efficiency_ratio(close: pd.Series, w=40) -> pd.Series:
    direction = (close - close.shift(w)).abs()
    vol = close.diff().abs().rolling(w).sum()
    return (direction / vol).replace([np.inf, -np.inf], np.nan)


def regime_features(closes: pd.DataFrame) -> pd.DataFrame:
    """月次のバスケット・レジーム特徴: 平均|ER|(trendiness), 平均vol, USDトレンド強度。"""
    majors = [c for c in closes.columns if c in config.PAIRS]
    er = pd.DataFrame({c: _efficiency_ratio(closes[c], 40) for c in majors})
    vol = pd.DataFrame({c: closes[c].pct_change().rolling(30).std() for c in majors})
    # USDバスケット・トレンド: 各majorのlog価格を「USD建てで上がる=USD安」方向に揃えるのは煩雑なので
    # 代理として各majorの40本リターン絶対値の平均(=方向問わずどれだけ動いたか)を使う
    usdmove = pd.DataFrame({c: closes[c].pct_change(40).abs() for c in majors})
    feat = pd.DataFrame({
        "er": er.mean(axis=1),
        "vol": vol.mean(axis=1),
        "trend40": usdmove.mean(axis=1),
    })
    return feat.groupby(feat.index.to_period("M")).mean()


# --- 候補戦略の月次PnLストリーム ---------------------------------------
def strategy_monthly_pnl(strategy_name, params=None, instruments=None, tf=TF,
                         side="both", cross_spread=3.0) -> pd.Series:
    """任意戦略の月次PnL(value $10k/銘柄を決済月で合算)。候補エッジのP&Lストリーム。"""
    uni.register_cross_spreads(cross_spread)
    instruments = instruments or mm.default_instruments()
    mod = importlib.import_module(f"strategies.{strategy_name}")
    params = params if params is not None else dict(getattr(mod, "PARAMS", {}))
    from fxlab.backtest import run
    from fxlab.trades import trade_table
    rows = []
    for nm in instruments:
        data = uni.instrument_data(nm, tf)
        try:
            pf = run(nm, tf, mod.generate_signals, params, data=data, size_mode="value", side=side)
            tt = trade_table(pf, data)
        except Exception:  # noqa: BLE001
            continue
        for _, r in tt.iterrows():
            rows.append((r["exit"], r["pnl"]))
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["exit", "pnl"])
    df["m"] = pd.to_datetime(df["exit"]).dt.to_period("M")
    return df.groupby("m")["pnl"].sum()


# --- 条件付き貢献スコア(吟味の核)-------------------------------------
def conditional_score(cand_monthly: pd.Series, bleed_mask: pd.Series,
                      oos_start="2022-01") -> dict:
    """候補の月次PnL を「失血窓 vs 平時」で評価。良いヘッジ=失血窓で正(or 平時より高い)。

    返り値: 窓内/平時の平均月次PnL・合計・勝率、差(hedge_edge)、IS/OOS別の窓内平均。
    """
    s = cand_monthly.reindex(bleed_mask.index).fillna(0.0)
    inb = s[bleed_mask.values]
    out = s[~bleed_mask.values]
    oos_p = pd.Period(oos_start, "M")
    is_mask = bleed_mask & (bleed_mask.index < oos_p)
    oos_mask = bleed_mask & (bleed_mask.index >= oos_p)
    s_is = cand_monthly.reindex(is_mask.index).fillna(0.0)
    return {
        "n_bleed_months": int(bleed_mask.sum()),
        "mean_in_bleed": float(inb.mean()) if len(inb) else float("nan"),
        "mean_normal": float(out.mean()) if len(out) else float("nan"),
        "total_in_bleed": float(inb.sum()),
        "winrate_in_bleed": float((inb > 0).mean()) if len(inb) else float("nan"),
        "hedge_edge": float(inb.mean() - out.mean()) if len(inb) and len(out) else float("nan"),
        "mean_in_bleed_IS": float(s_is[is_mask.values].mean()) if is_mask.sum() else float("nan"),
        "mean_in_bleed_OOS": float(s_is[oos_mask.values].mean()) if oos_mask.sum() else float("nan"),
        "total_all": float(cand_monthly.sum()),
    }


# --- 統合DDテスト(最終判定)------------------------------------------
def integrated_dd_test(overlay_pool, overlay_weight=1.0, max_pos=8, target_dd=0.20):
    """champion(z-size) + overlay(固定比率) を1口座(MtM)に統合し、DD=20% 較正→CAGRを返す。

    overlay_weight = overlay の総建玉を champion 1玉あたりに対して何倍張るか(資本配分)。
    比較対象は champion 単独(reports/09 の z-size mp8)。overlay が失血窓で稼ぐなら、統合の
    DD が下がり→同20%でより高い総レバ k が許され→CAGR が champion 単独を上回るはず。
    """
    pool_c = mm.build_pool()
    closes = mm.load_closes()
    from mm_production import champion_sizing, _fz
    # 合成プール(champion + overlay)。tag で識別子を付ける
    pc = pool_c.copy(); pc["src"] = "champ"
    po = overlay_pool.copy(); po["src"] = "ovl"
    both = pd.concat([pc, po], ignore_index=True).sort_values("entry").reset_index(drop=True)

    fbar = float(np.mean([_fz(z) for z in pool_c["z_entry"].to_numpy()])) or 1.0
    src = both["src"].to_numpy()
    # champion は乖離連動z、overlay は固定比率(weight倍)。総量は k で線形。
    # ※ src を ctx に載せないので、(instr,ret,bars)一意キーで src を引く
    keysrc = {}
    instr = both["instr"].to_numpy(); ret = both["ret"].to_numpy(); bh = both["bars_held"].to_numpy()
    for i in range(len(both)):
        keysrc[(instr[i], round(float(ret[i]), 12), int(bh[i]))] = src[i]

    def make_sizing(k):
        base = k / max_pos
        def sizing(ctx):
            s = keysrc.get((ctx["instr"], round(float(ctx["ret"]), 12), int(ctx["bars_held"])), "champ")
            if s == "champ":
                return ctx["equity_real"] * base * (_fz(ctx["z"]) / fbar)
            return ctx["equity_real"] * base * overlay_weight
        return sizing

    k, eqm, eqr, info = mm.calibrate(both, closes, make_sizing, target_dd=target_dd, max_pos=max_pos)
    s = mm.stats(eqm, eqr, info)
    bs = mm.bootstrap_maxdd(eqm, n_boot=800)
    return {"k": k, "cagr": s["cagr"], "maxdd_mtm": s["maxdd_mtm"], "sharpe": s["sharpe"],
            "pos_year_rate": s["pos_year_rate"], "boot_p95": bs["p95"], "worst_year": s["worst_year"]}


def main():
    print("=== チャンピオンv2 失血窓プロファイル ===")
    eqm, eqr, pool, closes = champion_mtm()
    mask, dd = bleed_mask_monthly(eqm)
    feat = regime_features(closes)
    feat = feat.reindex(mask.index)

    n = len(mask); nb = int(mask.sum())
    print(f"月数 {n} / 失血窓(MtM水面下>-{DD_BLEED_THRESH:.0%}) {nb}ヶ月 ({nb/n:.0%})\n")

    print("=== 失血窓 vs 平時 のレジーム特徴(平均)===")
    comp = pd.DataFrame({
        "失血窓": feat[mask.values].mean(),
        "平時": feat[~mask.values].mean(),
    })
    comp["比"] = (comp["失血窓"] / comp["平時"]).round(2)
    print(comp.round(4).to_string())
    print("\n  er=バスケット効率比(trendiness, 高=一直線トレンド) / vol=平均ボラ / trend40=40本変化の大きさ")

    print("\n=== 最も深い失血窓 トップ10ヶ月(月末DD)===")
    worst = dd.sort_values().head(10)
    for m, d in worst.items():
        print(f"  {m}: DD={d:+.1%}  er={feat.loc[m,'er']:.3f}  vol={feat.loc[m,'vol']:.4f}")

    print("\n=== 年別の失血窓月数 ===")
    by_year = pd.Series(mask.values, index=[p.year for p in mask.index]).groupby(level=0).sum()
    print(by_year.to_string())


if __name__ == "__main__":
    main()
