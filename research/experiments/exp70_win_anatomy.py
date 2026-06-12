"""exp70: 勝ちトレードの解剖(d1 プール) — reports/21(損失の解剖)の勝ち側対称版。

テーマ「勝ちトレードに着目して利益を増やす」の事実基盤づくり。損失側は
reports/21 で「入口判別不能・一度も浮かない緩慢ブリード・介入は全て逆効果」と
確定済み。本実験は **現行 d1 プール全 19 銘柄**(build_pool_d1, キャッシュ済み)で
勝ち側を対称に解剖する。**分析のみ・採用判断なし**。

測定項目:
  A. 勝ちの基礎統計 — 勝率 / 勝ちの利益集中度(上位X%勝ちが総益の何%か、ローレンツ的)
     / 勝ち・負けの保有バー分布(中央値・p90)
  B. 早期シグネチャ — エントリー後 k バー時点(k=1,2,3,5,10)で含み益か。
     最終勝者のうち bar1 で浮いていた率 vs 最終敗者の同率(reports/21
     「負けの98%は一度も浮かない」の対称確認)。「一度でも含み益 X% 以上を見た
     トレード」の最終勝率(X=0.1/0.25/0.5%)
  C. MFE 捕捉率 — 勝者の実現リターン/MFE(保有中 H4 close ベース最大含み益)の
     分布。敗者の MFE(途中まで浮いてから死んだのか)
  D. 出口後の継続 — 決済後 +1/+2/+5/+10 H4 バーの「トレード方向への」さらなる
     値動き(bps)。勝者決済後にまだ収束方向へ動くなら取り残し。z_exit(決済バー
     の z)の分布(オーバーシュートしているか)
  E. 勝ち連鎖(hot-hand)— 同一銘柄で「直前トレードが勝ち」条件付き次トレード
     勝率/平均ret vs 無条件。プール全体の「直前5トレード(exit<entry の因果順)の
     勝ち数→次の勝率」。月次勝率の lag1 自己相関。銘柄内シャッフル1000回の
     ブートストラップ p 値
  F. 勝ちの地合い — 勝者の z_entry / vol_entry / 銘柄 / 年 の分布 vs 敗者
     (既知の「入口は方向情報ゼロ」reports/21 §4 と整合するか)

z 系列の規約(look-ahead を侵さない):
  ・エントリー時特徴量(z_entry / vol_entry)はプール列をそのまま使用
    (mm_production.build_pool_d1 規約 = window=PARAMS["window"], shift(1))。
  ・z_exit は **決済バー自身の(シフト無し)z**。出口ルールは「バー t の確定 close
    で z が exit 帯へ戻ったら同バー close で決済」なので、決済バーの z はその
    決済判断が使った因果値そのもの(先読みではない)。
  ・出口後継続(D)は決済より未来の close を使うが、これは診断統計であって
    取引ルールではない(取り残しの有無の測定)。

実行: PYTHONPATH=. uv run python research/experiments/exp70_win_anatomy.py
出力: research/outputs/exp70_result.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))

from mm_production import build_pool_d1  # noqa: E402
from fxlab import universe as uni  # noqa: E402
from strategies.confluence_meanrev_v2_d1 import PARAMS  # noqa: E402

OUT_DIR = ROOT / "research" / "outputs"
K_BARS = (1, 2, 3, 5, 10)          # 早期シグネチャの観測点
POST_BARS = (1, 2, 5, 10)          # 決済後の継続観測点
MFE_THRESH = (0.001, 0.0025, 0.005)  # 「一度でも含み益X%」の X


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def path_stats(pool: pd.DataFrame) -> pd.DataFrame:
    """各トレードの H4 終値パス統計(エントリー close 基準・方向調整済み)。

    exp62 と同じパス規約。mfe はエントリーバー(=path[0]=0)を除いた
    保有中最大含み益(close ベース・コスト前)。
    """
    n = len(pool)
    cols = ["mfe", "t_mfe_bars", "z_exit_dir"]
    cols += [f"pos_at_{k}" for k in K_BARS]
    cols += [f"post_{h}" for h in POST_BARS]
    out = {c: np.full(n, np.nan) for c in cols}
    win = PARAMS["window"]
    for instr, g in pool.groupby("instr"):
        d = uni.instrument_data(instr, "H4")
        close = d["close"]
        # 決済バー自身の z(シフト無し=出口判断が使った因果値。docstring 冒頭参照)
        z_raw = (close - close.rolling(win).mean()) / close.rolling(win).std()
        idx = close.index
        pos_of = pd.Series(np.arange(len(idx)), index=idx)
        e_pos = pos_of.reindex(g["entry"]).to_numpy()
        x_pos = pos_of.reindex(g["exit"]).to_numpy()
        carr = close.to_numpy()
        zarr = z_raw.to_numpy()
        for ti, e, x in zip(g.index.to_numpy(), e_pos, x_pos):
            if not (np.isfinite(e) and np.isfinite(x)):
                continue
            e, x = int(e), int(x)
            dirv = float(pool.at[ti, "dir"])
            ec = carr[e]
            path = dirv * (carr[e:x + 1] / ec - 1.0)
            if len(path) > 1:
                out["mfe"][ti] = path[1:].max()
                out["t_mfe_bars"][ti] = int(np.argmax(path))
            else:
                out["mfe"][ti] = 0.0
                out["t_mfe_bars"][ti] = 0
            for k in K_BARS:
                if x - e >= k:
                    out[f"pos_at_{k}"][ti] = float(path[k] > 0)
            # z_exit_dir = -dir*z: 0.5=ちょうど閾値 / 0=完全平均回帰 / 負=平均を突き抜け
            out["z_exit_dir"][ti] = -dirv * zarr[x]
            for h in POST_BARS:
                if x + h < len(carr):
                    out[f"post_{h}"][ti] = dirv * (carr[x + h] / carr[x] - 1.0)
    return pd.DataFrame(out, index=pool.index)


def hothand_instr(groups: dict[str, np.ndarray]):
    """銘柄内シーケンスで「直前勝ち/負け」条件付きの次トレード成績。"""
    nxt_w, nxt_l, ret_w, ret_l = [], [], [], []
    for arr in groups.values():
        if len(arr) < 2:
            continue
        prev_win = arr[:-1] > 0
        nxt = arr[1:]
        nxt_w.append(nxt[prev_win])
        nxt_l.append(nxt[~prev_win])
    w = np.concatenate(nxt_w)
    l = np.concatenate(nxt_l)
    return {
        "n_after_win": int(len(w)), "n_after_loss": int(len(l)),
        "winrate_after_win": float((w > 0).mean()),
        "winrate_after_loss": float((l > 0).mean()),
        "meanret_bps_after_win": float(w.mean() * 1e4),
        "meanret_bps_after_loss": float(l.mean() * 1e4),
    }


def monthly_wr_autocorr(months: np.ndarray, rets: np.ndarray) -> float:
    """エントリー月ごとの勝率系列の lag1 自己相関。"""
    s = pd.Series(rets > 0, index=months).groupby(level=0).mean()
    return float(s.autocorr(1))


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1().copy()
    print(f"=== exp70: 勝ちトレード解剖 (d1 pool n={len(pool)}) ===")
    ps = path_stats(pool)
    df = pd.concat([pool, ps], axis=1)
    df["year"] = df["entry"].dt.year
    win_m = df["ret"] > 0
    W, L = df[win_m], df[~win_m]
    res: dict = {"n_trades": int(len(df))}

    # --- A. 勝ちの基礎統計 ----------------------------------------------------
    sec("A. 勝ちの基礎統計")
    gross_win = float(W["ret"].sum())
    gross_loss = float(L["ret"].sum())
    res["A"] = {
        "win_rate": float(win_m.mean()),
        "n_wins": int(len(W)), "n_losses": int(len(L)),
        "gross_profit": gross_win, "gross_loss": gross_loss,
        "net": gross_win + gross_loss,
        "mean_win_bps": float(W["ret"].mean() * 1e4),
        "mean_loss_bps": float(L["ret"].mean() * 1e4),
    }
    print(f"勝率 {res['A']['win_rate']:.1%}  総益 {gross_win:+.3f}  総損 {gross_loss:+.3f}  "
          f"純益 {gross_win + gross_loss:+.3f}")
    print(f"平均勝ち {res['A']['mean_win_bps']:+.1f}bps / 平均負け {res['A']['mean_loss_bps']:+.1f}bps")
    wins_desc = W["ret"].sort_values(ascending=False)
    conc = {}
    for q in (0.01, 0.05, 0.10, 0.25, 0.50):
        k = max(1, int(len(wins_desc) * q))
        conc[f"top_{int(q*100)}pct_of_wins"] = float(wins_desc.iloc[:k].sum() / gross_win)
        print(f"  勝ち上位{q:.0%}({k}件) = 総益の {conc[f'top_{int(q*100)}pct_of_wins']:.1%}  "
              f"(最大 {wins_desc.iloc[0]:+.3%} 〜 {wins_desc.iloc[k-1]:+.3%})")
    res["A"]["profit_concentration"] = conc
    hold = {
        "win_median": float(W["bars_held"].median()),
        "win_p90": float(W["bars_held"].quantile(0.90)),
        "loss_median": float(L["bars_held"].median()),
        "loss_p90": float(L["bars_held"].quantile(0.90)),
    }
    res["A"]["bars_held"] = hold
    print(f"保有バー: 勝ち median {hold['win_median']:.0f} / p90 {hold['win_p90']:.0f}  "
          f"vs 負け median {hold['loss_median']:.0f} / p90 {hold['loss_p90']:.0f}")
    # 上位10%勝ちの構成(まぐれ単年でないか)
    top10 = W.nlargest(max(1, int(len(W) * 0.10)), "ret")
    print("勝ち上位10%の年次: " + " ".join(
        f"{int(y)}:{c}" for y, c in top10["year"].value_counts().sort_index().items()))
    print("  銘柄上位: " + "  ".join(
        f"{i}:{c}" for i, c in top10["instr"].value_counts().head(6).items()))
    res["A"]["top10pct_wins_by_year"] = {int(y): int(c) for y, c in
                                         top10["year"].value_counts().sort_index().items()}

    # --- B. 早期シグネチャ ------------------------------------------------------
    sec("B. 早期シグネチャ(エントリー後 k バー時点で含み益か)")
    early = {}
    for k in K_BARS:
        col = f"pos_at_{k}"
        sub = df.dropna(subset=[col])
        sw, sl = sub[sub["ret"] > 0], sub[sub["ret"] <= 0]
        early[f"bar{k}"] = {
            "n_eligible": int(len(sub)),
            "winners_pos_share": float(sw[col].mean()),
            "losers_pos_share": float(sl[col].mean()),
            "winrate_if_pos": float((sub[sub[col] == 1]["ret"] > 0).mean()),
            "winrate_if_neg": float((sub[sub[col] == 0]["ret"] > 0).mean()),
            "n_pos": int((sub[col] == 1).sum()),
        }
        e = early[f"bar{k}"]
        print(f"bar{k:>2}: 最終勝者の浮き率 {e['winners_pos_share']:.1%} / 最終敗者の浮き率 "
              f"{e['losers_pos_share']:.1%} | 浮いていれば勝率 {e['winrate_if_pos']:.1%} / "
              f"沈んでいれば {e['winrate_if_neg']:.1%} (n={e['n_eligible']})")
    res["B"] = {"pos_at_k": early}
    # 一度でも含み益X%以上を見たトレードの最終勝率(reports/21 の対称)
    mfe_tbl = {}
    for x in MFE_THRESH:
        seen = df[df["mfe"] >= x]
        never = df[df["mfe"] < x]
        mfe_tbl[f"ever_{x*100:.2f}pct"] = {
            "n": int(len(seen)), "win_rate": float((seen["ret"] > 0).mean()),
            "n_never": int(len(never)), "win_rate_never": float((never["ret"] > 0).mean()),
        }
        t = mfe_tbl[f"ever_{x*100:.2f}pct"]
        print(f"一度でも +{x:.2%} を見た: n={t['n']} 勝率 {t['win_rate']:.1%} / "
              f"見なかった: n={t['n_never']} 勝率 {t['win_rate_never']:.1%}")
    res["B"]["ever_mfe"] = mfe_tbl
    never_float_losers = float((L["mfe"] <= 0).mean())
    res["B"]["losers_never_in_profit_share"] = never_float_losers
    print(f"敗者のうち一度も含み益ゼロ超にならず: {never_float_losers:.1%}"
          f"(reports/21: ワースト10%の98%は+0.5%未達)")

    # --- C. MFE 捕捉率 ----------------------------------------------------------
    sec("C. MFE 捕捉率(実現リターン / 保有中最大含み益)")
    wpos = W[W["mfe"] > 0].copy()
    cap = wpos["ret"] / wpos["mfe"]
    res["C"] = {
        "winners": {
            "capture_median": float(cap.median()), "capture_mean": float(cap.mean()),
            "capture_p10": float(cap.quantile(0.10)), "capture_p90": float(cap.quantile(0.90)),
            "mfe_median": float(W["mfe"].median() * 1e4),
            "t_mfe_median_bars": float(W["t_mfe_bars"].median()),
            "exit_at_mfe_share": float((W["t_mfe_bars"] >= W["bars_held"]).mean()),
        },
        "losers": {
            "mfe_median_bps": float(L["mfe"].median() * 1e4),
            "mfe_mean_bps": float(L["mfe"].mean() * 1e4),
            "share_mfe_ge_10bps": float((L["mfe"] >= 0.001).mean()),
            "share_mfe_ge_25bps": float((L["mfe"] >= 0.0025).mean()),
            "share_mfe_ge_50bps": float((L["mfe"] >= 0.005).mean()),
        },
    }
    cw = res["C"]["winners"]
    print(f"勝者: capture median {cw['capture_median']:.2f} / mean {cw['capture_mean']:.2f} "
          f"(p10 {cw['capture_p10']:.2f} / p90 {cw['capture_p90']:.2f})")
    print(f"  MFE 中央値 {cw['mfe_median']:.0f}bps / MFE到達バー中央値 {cw['t_mfe_median_bars']:.0f}本 / "
          f"『MFE=決済バー(最高値で決済)』率 {cw['exit_at_mfe_share']:.1%}")
    cl = res["C"]["losers"]
    print(f"敗者: MFE median {cl['mfe_median_bps']:.1f}bps / mean {cl['mfe_mean_bps']:.1f}bps | "
          f"≥10bps {cl['share_mfe_ge_10bps']:.1%} / ≥25bps {cl['share_mfe_ge_25bps']:.1%} / "
          f"≥50bps {cl['share_mfe_ge_50bps']:.1%}")

    # --- D. 出口後の継続 --------------------------------------------------------
    sec("D. 出口後の継続(決済後 h バーのトレード方向への動き, bps)")
    post = {}
    for label, sub in (("winners", W), ("losers", L)):
        post[label] = {}
        for h in POST_BARS:
            v = sub[f"post_{h}"].dropna() * 1e4
            post[label][f"h{h}"] = {"mean_bps": float(v.mean()), "median_bps": float(v.median()),
                                    "n": int(len(v)),
                                    "t_stat": float(v.mean() / (v.std() / np.sqrt(len(v))))}
        row = "  ".join(f"+{h}: {post[label][f'h{h}']['mean_bps']:+.1f}bps"
                        f"(t={post[label][f'h{h}']['t_stat']:+.1f})" for h in POST_BARS)
        print(f"{label:8s} {row}")
    res["D"] = {"post_exit": post}
    zx = df["z_exit_dir"].dropna()
    zxw, zxl = W["z_exit_dir"].dropna(), L["z_exit_dir"].dropna()
    res["D"]["z_exit_dir"] = {
        "convention": "-dir*z: 0.5=exit閾値ちょうど / 0=平均完全回帰 / 負=平均を突き抜け",
        "median": float(zx.median()), "mean": float(zx.mean()),
        "p10": float(zx.quantile(0.10)), "p90": float(zx.quantile(0.90)),
        "share_overshoot_past_mean": float((zx < 0).mean()),
        "share_above_threshold": float((zx > PARAMS["exit_z"]).mean()),
        "winners_median": float(zxw.median()), "losers_median": float(zxl.median()),
    }
    dz = res["D"]["z_exit_dir"]
    print(f"z_exit_dir: median {dz['median']:+.2f} (勝者 {dz['winners_median']:+.2f} / "
          f"敗者 {dz['losers_median']:+.2f}) p10 {dz['p10']:+.2f} p90 {dz['p90']:+.2f}")
    print(f"  平均を突き抜けて決済(z<0): {dz['share_overshoot_past_mean']:.1%} / "
          f"閾値外のまま決済(z>{PARAMS['exit_z']}, 強制クローズ等): {dz['share_above_threshold']:.1%}")

    # --- E. 勝ち連鎖(hot-hand)---------------------------------------------------
    sec("E. 勝ち連鎖(hot-hand)")
    psort = pool.sort_values("entry").reset_index(drop=True)
    groups = {i: g["ret"].to_numpy() for i, g in psort.groupby("instr")}
    obs = hothand_instr(groups)
    base_wr = float((pool["ret"] > 0).mean())
    d_obs = obs["winrate_after_win"] - obs["winrate_after_loss"]
    dr_obs = obs["meanret_bps_after_win"] - obs["meanret_bps_after_loss"]
    print(f"無条件勝率 {base_wr:.1%}")
    print(f"直前勝ち→次の勝率 {obs['winrate_after_win']:.1%} (n={obs['n_after_win']}) / "
          f"直前負け→ {obs['winrate_after_loss']:.1%} (n={obs['n_after_loss']})  Δ={d_obs:+.1%}")
    print(f"直前勝ち→次の平均 {obs['meanret_bps_after_win']:+.1f}bps / "
          f"直前負け→ {obs['meanret_bps_after_loss']:+.1f}bps  Δ={dr_obs:+.1f}bps")
    # 銘柄内シャッフル 1000 回ブートストラップ(時系列構造を破壊して帰無分布を作る)
    months = psort["entry"].dt.tz_localize(None).dt.to_period("M").to_numpy()
    ac_obs = monthly_wr_autocorr(months, psort["ret"].to_numpy())
    rng = np.random.default_rng(0)
    n_boot = 1000
    d_null, dr_null, ac_null = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    instr_arr = psort["instr"].to_numpy()
    ret_arr = psort["ret"].to_numpy()
    instr_slots = {i: np.where(instr_arr == i)[0] for i in groups}
    for b in range(n_boot):
        shuf = ret_arr.copy()
        for i, slots in instr_slots.items():
            shuf[slots] = rng.permutation(shuf[slots])
        g2 = {i: shuf[slots] for i, slots in instr_slots.items()}
        o2 = hothand_instr(g2)
        d_null[b] = o2["winrate_after_win"] - o2["winrate_after_loss"]
        dr_null[b] = o2["meanret_bps_after_win"] - o2["meanret_bps_after_loss"]
        ac_null[b] = monthly_wr_autocorr(months, shuf)
    p_d = float((np.abs(d_null) >= abs(d_obs)).mean())
    p_dr = float((np.abs(dr_null) >= abs(dr_obs)).mean())
    p_ac = float((np.abs(ac_null) >= abs(ac_obs)).mean())
    print(f"ブート p 値(両側, 銘柄内シャッフル{n_boot}回): Δ勝率 p={p_d:.3f} / "
          f"Δ平均ret p={p_dr:.3f}")
    print(f"月次勝率 lag1 自己相関: {ac_obs:+.3f} (シャッフル帰無 p={p_ac:.3f})")
    # プール全体: 直前5トレード(exit < entry = 結果確定済み)の勝ち数→次の勝率
    ent = psort["entry"].dt.tz_localize(None).to_numpy()
    ext = psort["exit"].dt.tz_localize(None).to_numpy()
    order = np.argsort(ext, kind="stable")
    ext_sorted = ext[order]
    wins5 = np.full(len(psort), -1)
    for i in range(len(psort)):
        kk = int(np.searchsorted(ext_sorted, ent[i], side="left"))
        if kk >= 5:
            wins5[i] = int((ret_arr[order[kk - 5:kk]] > 0).sum())
    seq = {}
    for w in range(6):
        m = wins5 == w
        if m.sum() == 0:
            continue
        seq[str(w)] = {"n": int(m.sum()), "next_win_rate": float((ret_arr[m] > 0).mean()),
                       "next_mean_bps": float(ret_arr[m].mean() * 1e4)}
    print("直前5トレードの勝ち数 → 次の勝率:")
    for w, v in seq.items():
        print(f"  {w}/5 勝: n={v['n']:>4}  次勝率 {v['next_win_rate']:.1%}  "
              f"次平均 {v['next_mean_bps']:+.1f}bps")
    res["E"] = {
        "base_win_rate": base_wr, "instr_conditional": obs,
        "delta_winrate": d_obs, "delta_meanret_bps": dr_obs,
        "p_delta_winrate": p_d, "p_delta_meanret": p_dr,
        "monthly_wr_autocorr_lag1": ac_obs, "p_autocorr": p_ac,
        "last5_wins_to_next": seq,
    }

    # --- F. 勝ちの地合い ---------------------------------------------------------
    sec("F. 勝ちの地合い(z_entry / vol_entry / 銘柄 / 年)")
    try:
        from scipy import stats
        ks_z = stats.ks_2samp(W["z_entry"].dropna(), L["z_entry"].dropna())
        ks_v = stats.ks_2samp(W["vol_entry"].dropna(), L["vol_entry"].dropna())
        ks = {"z_entry_ks_p": float(ks_z.pvalue), "vol_entry_ks_p": float(ks_v.pvalue)}
    except Exception:
        ks = {}
    res["F"] = {
        "z_entry": {"win_mean": float(W["z_entry"].mean()), "loss_mean": float(L["z_entry"].mean()),
                    "win_median": float(W["z_entry"].median()),
                    "loss_median": float(L["z_entry"].median())},
        "vol_entry": {"win_mean": float(W["vol_entry"].mean()),
                      "loss_mean": float(L["vol_entry"].mean()),
                      "win_median": float(W["vol_entry"].median()),
                      "loss_median": float(L["vol_entry"].median())},
        **ks,
    }
    print(f"z_entry : 勝者 mean {res['F']['z_entry']['win_mean']:.3f} vs 敗者 "
          f"{res['F']['z_entry']['loss_mean']:.3f}" +
          (f"  (KS p={ks.get('z_entry_ks_p'):.3f})" if ks else ""))
    print(f"vol_entry: 勝者 mean {res['F']['vol_entry']['win_mean']:.5f} vs 敗者 "
          f"{res['F']['vol_entry']['loss_mean']:.5f}" +
          (f"  (KS p={ks.get('vol_entry_ks_p'):.3f})" if ks else ""))
    gi = df.groupby("instr")["ret"]
    instr_tbl = pd.DataFrame({"n": gi.size(), "win_rate": gi.apply(lambda s: (s > 0).mean()),
                              "mean_bps": gi.mean() * 1e4}).sort_values("win_rate")
    print("\n銘柄別(勝率昇順):")
    print(instr_tbl.to_string(float_format=lambda x: f"{x:.3f}"))
    gy = df.groupby("year")["ret"]
    year_tbl = pd.DataFrame({"n": gy.size(), "win_rate": gy.apply(lambda s: (s > 0).mean()),
                             "mean_bps": gy.mean() * 1e4})
    print("\n年次:")
    print(year_tbl.to_string(float_format=lambda x: f"{x:.3f}"))
    res["F"]["by_instr"] = instr_tbl.to_dict(orient="index")
    res["F"]["by_year"] = {int(y): v for y, v in year_tbl.to_dict(orient="index").items()}

    (OUT_DIR / "exp70_result.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print(f"\nsaved -> {OUT_DIR / 'exp70_result.json'}\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
