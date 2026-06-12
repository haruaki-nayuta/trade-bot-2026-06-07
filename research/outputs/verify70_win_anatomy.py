"""verify70: exp70(勝ちトレード解剖)の敵対検証 — 完全独立再計算。

exp70 のコードは import しない。プール構築(mm_production.build_pool_d1)と
fxlab のみ利用。監査 5 点:
  1. hot-hand: 「直前トレード」が entry 時点で exit 確定済みかの監査+
     厳密因果定義(直近 exit 済みトレードを条件)での再計算+ブート p。
  2. 決済後ドリフト: dir 符号・exit バー close 起点・コスト無し生値で再計算。
  3. MFE 捕捉率: MFE=コスト前 / ret=コスト後 の向き確認+再計算。
  4. 利益集中度: 分母・件数ベースの定義監査(exp62=全件ベース vs exp70=勝ちベース)。
  5. 早期シグネチャ: bars_held==k の同語反復除去(bars_held>k のみ)で再計算。

実行: PYTHONPATH=. uv run python research/outputs/verify70_win_anatomy.py
出力: research/outputs/verify70_result.json
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

from mm_production import build_pool_d1  # noqa: E402
from fxlab import universe as uni  # noqa: E402

K_BARS = (1, 2, 3, 5, 10)
POST_BARS = (1, 2, 5, 10)


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1().copy()
    n = len(pool)
    res: dict = {"n_trades": n, "sum_ret": float(pool["ret"].sum())}
    print(f"pool n={n}  sum(ret)={res['sum_ret']:+.4f}")
    assert n == 1207, "プール件数が想定と違う"

    win_m = pool["ret"] > 0
    W, L = pool[win_m], pool[~win_m]
    print(f"win_rate={win_m.mean():.4f}  n_wins={len(W)}  n_losses={len(L)}")
    res["win_rate"] = float(win_m.mean())
    res["bars_held"] = {
        "win_median": float(W["bars_held"].median()),
        "loss_median": float(L["bars_held"].median()),
    }

    # =========================================================================
    # 4. 利益集中度の定義監査
    # =========================================================================
    sec("4. 利益集中度: 定義監査(件数ベース: 勝ちのみ vs 全トレード)")
    wins_desc = W["ret"].sort_values(ascending=False)
    losses_asc = L["ret"].sort_values()  # 最悪から
    gross_win = float(wins_desc.sum())
    gross_loss = float(losses_asc.sum())
    conc = {}
    for q in (0.01, 0.05, 0.10):
        k_wins = max(1, int(len(wins_desc) * q))   # exp70: 勝ち865件ベース
        k_all = max(1, int(n * q))                 # exp62/reports21: 全1207件ベース
        k_loss = max(1, int(len(losses_asc) * q))  # 負け342件ベース
        conc[f"{int(q*100)}pct"] = {
            "win_top_share_winsbase_exp70def": float(wins_desc.iloc[:k_wins].sum() / gross_win),
            "win_top_share_allbase_exp62def": float(wins_desc.iloc[:k_all].sum() / gross_win),
            "loss_worst_share_allbase_exp62def": float(losses_asc.iloc[:k_all].sum() / gross_loss),
            "loss_worst_share_lossbase": float(losses_asc.iloc[:k_loss].sum() / gross_loss),
            "k_wins": k_wins, "k_all": k_all, "k_loss": k_loss,
        }
        c = conc[f"{int(q*100)}pct"]
        print(f"q={q:.0%}: 勝ち上位(勝ちベース{k_wins}件)={c['win_top_share_winsbase_exp70def']:.1%} | "
              f"勝ち上位(全件ベース{k_all}件)={c['win_top_share_allbase_exp62def']:.1%} | "
              f"負けワースト(全件ベース{k_all}件)={c['loss_worst_share_allbase_exp62def']:.1%} | "
              f"負けワースト(負けベース{k_loss}件)={c['loss_worst_share_lossbase']:.1%}")
    # 定義フリーの集中度: Gini(同一側内の |ret|)
    def gini(x):
        x = np.sort(np.abs(np.asarray(x, dtype=float)))
        nn = len(x)
        return float((2 * np.arange(1, nn + 1) - nn - 1).dot(x) / (nn * x.sum()))
    res["concentration"] = conc
    res["gini"] = {"wins": gini(W["ret"]), "losses": gini(L["ret"])}
    print(f"Gini(定義フリー): 勝ち {res['gini']['wins']:.3f} vs 負け {res['gini']['losses']:.3f}")

    # =========================================================================
    # パス統計(自前実装) — 2/3/5 で使用
    # =========================================================================
    sec("パス統計の独立再計算(close ベース・コスト前)")
    mfe = np.full(n, np.nan)
    t_mfe = np.full(n, -1)
    path_final = np.full(n, np.nan)      # close ベース・コスト前の最終リターン
    bars_chk = np.full(n, -1)
    pos_at = {k: np.full(n, np.nan) for k in K_BARS}
    post = {h: np.full(n, np.nan) for h in POST_BARS}
    n_overlap_strict = 0   # 同一銘柄で entry < 直前 exit(=未確定条件の証拠)
    n_overlap_eq = 0       # entry == 直前 exit(同一バー再エントリー)
    for instr, g in pool.groupby("instr"):
        d = uni.instrument_data(instr, "H4")
        close = d["close"]
        pos_of = pd.Series(np.arange(len(close)), index=close.index)
        gs = g.sort_values("entry")
        pe = gs["exit"].shift(1)
        n_overlap_strict += int((gs["entry"] < pe).sum())
        n_overlap_eq += int((gs["entry"] == pe).sum())
        e_pos = pos_of.reindex(g["entry"]).to_numpy()
        x_pos = pos_of.reindex(g["exit"]).to_numpy()
        carr = close.to_numpy()
        for ti, e, x, dirv in zip(g.index.to_numpy(), e_pos, x_pos, g["dir"].to_numpy()):
            if not (np.isfinite(e) and np.isfinite(x)):
                continue
            e, x = int(e), int(x)
            bars_chk[ti] = x - e
            path = float(dirv) * (carr[e:x + 1] / carr[e] - 1.0)
            path_final[ti] = path[-1]
            if len(path) > 1:
                mfe[ti] = path[1:].max()
                t_mfe[ti] = int(np.argmax(path))
            else:
                mfe[ti] = 0.0
                t_mfe[ti] = 0
            for k in K_BARS:
                if x - e >= k:
                    pos_at[k][ti] = float(path[k] > 0)
            for h in POST_BARS:
                if x + h < len(carr):
                    post[h][ti] = float(dirv) * (carr[x + h] / carr[x] - 1.0)
    bad_bars = int((bars_chk != pool["bars_held"].to_numpy()).sum())
    print(f"bars_held 整合: 不一致 {bad_bars} 件 / 銘柄内オーバーラップ: "
          f"entry<前exit {n_overlap_strict} 件, entry==前exit {n_overlap_eq} 件")
    res["audit"] = {"bars_held_mismatch": bad_bars,
                    "overlap_strict": n_overlap_strict, "overlap_eq": n_overlap_eq}

    # =========================================================================
    # 3. MFE 捕捉率 — コストの向き確認+再計算
    # =========================================================================
    sec("3. MFE 捕捉率")
    wm = win_m.to_numpy()
    # コストの向き: close ベース最終(コスト前) - ret(コスト後) は正のはず
    cost_gap = path_final - pool["ret"].to_numpy()
    print(f"コスト確認: median(path_final - ret) = {np.nanmedian(cost_gap)*1e4:+.2f}bps "
          f"(正= ret はコスト後 / MFE はコスト前 → 捕捉率は過小評価方向で安全)")
    share_neg_gap = float((cost_gap[np.isfinite(cost_gap)] < -1e-12).mean())
    print(f"  path_final < ret のトレード比率: {share_neg_gap:.1%}(0%に近いはず)")
    cap_mask = wm & (mfe > 0)
    cap = pool["ret"].to_numpy()[cap_mask] / mfe[cap_mask]
    cap_pre = path_final[cap_mask] / mfe[cap_mask]   # コスト前同士の捕捉率(上限)
    exit_at_mfe = float((t_mfe[wm] >= pool["bars_held"].to_numpy()[wm]).mean())
    res["C"] = {
        "capture_median_postcost": float(np.median(cap)),
        "capture_median_precost": float(np.median(cap_pre)),
        "mfe_median_bps_winners": float(np.median(mfe[wm]) * 1e4),
        "exit_at_mfe_share": exit_at_mfe,
        "cost_gap_median_bps": float(np.nanmedian(cost_gap) * 1e4),
    }
    print(f"勝者 capture median(コスト後/exp70規約)= {res['C']['capture_median_postcost']:.3f} "
          f"/ コスト前同士 = {res['C']['capture_median_precost']:.3f}")
    print(f"勝者 MFE median = {res['C']['mfe_median_bps_winners']:.1f}bps / "
          f"MFE=決済バー率 = {exit_at_mfe:.1%}")

    # =========================================================================
    # 2. 決済後ドリフト(トレード方向・close 起点・コスト無し)
    # =========================================================================
    sec("2. 決済後ドリフト")
    res["D"] = {}
    for label, mask in (("winners", wm), ("losers", ~wm)):
        res["D"][label] = {}
        row = []
        for h in POST_BARS:
            v = post[h][mask]
            v = v[np.isfinite(v)] * 1e4
            t = float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
            res["D"][label][f"h{h}"] = {"mean_bps": float(v.mean()), "t": t, "n": int(len(v))}
            row.append(f"+{h}: {v.mean():+.2f}bps(t={t:+.2f},n={len(v)})")
        print(f"{label:8s} " + "  ".join(row))
    # 方向符号の妥当性スポット監査: ロング/ショート別の h5 ドリフト
    dirs = pool["dir"].to_numpy()
    for dv, nmm in ((1, "long"), (-1, "short")):
        m2 = wm & (dirs == dv)
        v = post[5][m2]; v = v[np.isfinite(v)] * 1e4
        res["D"][f"winners_h5_{nmm}"] = {"mean_bps": float(v.mean()), "n": int(len(v))}
        print(f"  勝者 h5 {nmm}: {v.mean():+.2f}bps (n={len(v)})")

    # =========================================================================
    # 5. 早期シグネチャ — 同語反復(bars_held==k)除去
    # =========================================================================
    sec("5. 早期シグネチャ: bars_held==k の同語反復除去")
    rets = pool["ret"].to_numpy()
    early = {}
    for k in K_BARS:
        v = pos_at[k]
        elig = np.isfinite(v)                       # exp70 定義(bars_held >= k)
        strict = elig & (pool["bars_held"].to_numpy() > k)  # 同語反復除去
        n_tauto = int(elig.sum() - strict.sum())
        def wr(m, val):
            mm2 = m & (v == val)
            return float((rets[mm2] > 0).mean()) if mm2.sum() else float("nan")
        early[f"bar{k}"] = {
            "n_eligible_exp70": int(elig.sum()), "n_tautological": n_tauto,
            "exp70_winrate_if_pos": wr(elig, 1.0), "exp70_winrate_if_neg": wr(elig, 0.0),
            "strict_winrate_if_pos": wr(strict, 1.0), "strict_winrate_if_neg": wr(strict, 0.0),
            "strict_n": int(strict.sum()),
        }
        e = early[f"bar{k}"]
        print(f"bar{k:>2}: 同語反復 {n_tauto:>3}件除去 | exp70定義 浮き勝率 "
              f"{e['exp70_winrate_if_pos']:.1%}/沈み {e['exp70_winrate_if_neg']:.1%} → "
              f"厳密(bars_held>k, n={e['strict_n']}) 浮き {e['strict_winrate_if_pos']:.1%}"
              f"/沈み {e['strict_winrate_if_neg']:.1%}")
    res["B"] = early

    # =========================================================================
    # 1. hot-hand — 因果定義の監査と再計算
    # =========================================================================
    sec("1. hot-hand: 厳密因果定義(エントリー時点で exit 確定済みの直近トレード)")
    # 銘柄内: 直前(entry順)トレードが exit 済みかは上の overlap 監査で判明。
    # ここでは定義を変えて独立再計算: 条件トレード = 同一銘柄で exit <= entry_i の最新。
    psort = pool.sort_values("entry").reset_index(drop=True)
    ret_arr = psort["ret"].to_numpy()
    instr_arr = psort["instr"].to_numpy()
    ent_ns = psort["entry"].astype("int64").to_numpy()
    ext_ns = psort["exit"].astype("int64").to_numpy()

    def causal_prev_idx(strict: bool) -> np.ndarray:
        """各トレードの「同一銘柄・エントリー時点で確定済みの直近 exit」のインデックス(-1=なし)。"""
        prev = np.full(len(psort), -1)
        for instr in np.unique(instr_arr):
            ii = np.where(instr_arr == instr)[0]          # entry 昇順
            ex_i = ext_ns[ii]
            order = np.argsort(ex_i, kind="stable")        # exit 昇順
            ex_sorted = ex_i[order]
            for j, gi in enumerate(ii):
                side = "left" if strict else "right"
                kk = int(np.searchsorted(ex_sorted, ent_ns[gi], side=side))
                # 自分自身(exit>=entry)は exit_sorted 上で entry より後ろにしか
                # 来ないので混入しない(strict)。eq 側は同バー exit=entry の自分が
                # 入り得るため除外する。
                while kk > 0 and ii[order[kk - 1]] == gi:
                    kk -= 1
                if kk > 0:
                    prev[gi] = ii[order[kk - 1]]
        return prev

    def hothand_from_prev(prev: np.ndarray, r: np.ndarray):
        has = prev >= 0
        pw = r[prev[has]] > 0
        nxt = r[has]
        return {
            "n_after_win": int(pw.sum()), "n_after_loss": int((~pw).sum()),
            "wr_after_win": float((nxt[pw] > 0).mean()),
            "wr_after_loss": float((nxt[~pw] > 0).mean()),
            "mr_bps_after_win": float(nxt[pw].mean() * 1e4),
            "mr_bps_after_loss": float(nxt[~pw].mean() * 1e4),
        }

    hh = {}
    for strict in (True, False):
        prev = causal_prev_idx(strict)
        obs = hothand_from_prev(prev, ret_arr)
        d_obs = obs["wr_after_win"] - obs["wr_after_loss"]
        dr_obs = obs["mr_bps_after_win"] - obs["mr_bps_after_loss"]
        # ブート: 銘柄内シャッフルで帰無(prev マッピングは時刻固定)
        rng = np.random.default_rng(1)
        slots = {i: np.where(instr_arr == i)[0] for i in np.unique(instr_arr)}
        n_boot = 1000
        d_null = np.empty(n_boot)
        dr_null = np.empty(n_boot)
        for b in range(n_boot):
            shuf = ret_arr.copy()
            for s in slots.values():
                shuf[s] = rng.permutation(shuf[s])
            o2 = hothand_from_prev(prev, shuf)
            d_null[b] = o2["wr_after_win"] - o2["wr_after_loss"]
            dr_null[b] = o2["mr_bps_after_win"] - o2["mr_bps_after_loss"]
        p_d = float((np.abs(d_null) >= abs(d_obs)).mean())
        p_dr = float((np.abs(dr_null) >= abs(dr_obs)).mean())
        lab = "strict(exit<entry)" if strict else "nonstrict(exit<=entry)"
        hh[lab] = {**obs, "delta_wr": d_obs, "delta_mr_bps": dr_obs,
                   "p_delta_wr": p_d, "p_delta_mr": p_dr}
        print(f"{lab}: 直前勝ち→勝率 {obs['wr_after_win']:.1%}(n={obs['n_after_win']}) / "
              f"直前負け→ {obs['wr_after_loss']:.1%}(n={obs['n_after_loss']})  "
              f"Δ={d_obs:+.1%} (boot p={p_d:.3f})  Δret={dr_obs:+.1f}bps (p={p_dr:.3f})")
    res["E_hothand_causal"] = hh

    # プール全体の直前5(exit 確定済み)の独立再計算
    order = np.argsort(ext_ns, kind="stable")
    ext_sorted = ext_ns[order]
    seq = {}
    wins5 = np.full(len(psort), -1)
    for i in range(len(psort)):
        kk = int(np.searchsorted(ext_sorted, ent_ns[i], side="left"))
        if kk >= 5:
            wins5[i] = int((ret_arr[order[kk - 5:kk]] > 0).sum())
    for w in range(6):
        m = wins5 == w
        if m.sum():
            seq[str(w)] = {"n": int(m.sum()), "wr": float((ret_arr[m] > 0).mean()),
                           "mr_bps": float(ret_arr[m].mean() * 1e4)}
    res["E_last5"] = seq
    print("直前5(exit確定済)勝ち数→次勝率: " + "  ".join(
        f"{w}:{v['wr']:.1%}(n={v['n']})" for w, v in seq.items()))
    # 5/5 vs それ以外の単純2標本検定(レバー候補の最後の砦)
    m5 = wins5 == 5
    mrest = wins5 >= 0
    from scipy import stats as st
    a, b = ret_arr[m5], ret_arr[mrest & ~m5]
    tt = st.ttest_ind(a, b, equal_var=False)
    z55 = st.norm.sf(abs(
        ((a > 0).mean() - (b > 0).mean())
        / np.sqrt((np.concatenate([a, b]) > 0).mean()
                  * (1 - (np.concatenate([a, b]) > 0).mean())
                  * (1 / len(a) + 1 / len(b))))) * 2
    res["E_5of5_vs_rest"] = {"n_5of5": int(m5.sum()), "wr_5of5": float((a > 0).mean()),
                             "wr_rest": float((b > 0).mean()),
                             "t_meanret": float(tt.statistic), "p_meanret": float(tt.pvalue),
                             "p_wr_2prop": float(z55)}
    print(f"5/5 vs 残り: 勝率 {(a>0).mean():.1%} vs {(b>0).mean():.1%} "
          f"(2標本比率 p={z55:.3f}) / 平均ret t={tt.statistic:+.2f} p={tt.pvalue:.3f}")

    # 月次勝率 lag1 自己相関
    months = psort["entry"].dt.tz_localize(None).dt.to_period("M").to_numpy()
    s = pd.Series(ret_arr > 0, index=months).groupby(level=0).mean()
    res["E_monthly_ac1"] = float(s.autocorr(1))
    print(f"月次勝率 lag1 自己相関: {res['E_monthly_ac1']:+.3f}")

    # =========================================================================
    # F. 入口特徴 KS(独立)
    # =========================================================================
    sec("F. 入口特徴 KS 再計算")
    ks_z = st.ks_2samp(W["z_entry"].dropna(), L["z_entry"].dropna())
    ks_v = st.ks_2samp(W["vol_entry"].dropna(), L["vol_entry"].dropna())
    res["F"] = {"z_entry_ks_p": float(ks_z.pvalue), "vol_entry_ks_p": float(ks_v.pvalue)}
    print(f"z_entry KS p={ks_z.pvalue:.3f} / vol_entry KS p={ks_v.pvalue:.3f}")

    out = ROOT / "research" / "outputs" / "verify70_result.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print(f"\nsaved -> {out}\n経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
