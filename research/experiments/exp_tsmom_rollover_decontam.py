"""tsmom 短期モメンタムのロールオーバー(UTC20-22 BID)アーティファクト除染検証。

検証1: H1 lb24 / M30 lb48 / M15 lb96 の tsmom を7メジャーで、
  (A) 通常 GROSS
  (B) UTC20-23時の新規エントリーを禁止した GROSS
を比較。除染後もグロス正なら本物。さらに side=long/short の grossも見て
「ロング側だけ異常」=ロールオーバー署名 でないか確認する。
"""

from __future__ import annotations

import pandas as pd

import fxlab.config as C
from fxlab import load, metrics, run

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]

# (tf, lookback) の3設定
CONFIGS = [
    ("H1", 24),
    ("M30", 48),
    ("M15", 96),
]

BAND = 0.0
BAN_HOURS = {20, 21, 22, 23}  # UTC20-23の新規エントリー禁止


def make_signals(lookback: int, band: float = 0.0, ban_rollover: bool = False):
    """tsmom シグナル生成関数を返す。ban_rollover=True なら UTC20-23の新規エントリーをFalse化。"""

    def generate_signals(data: pd.DataFrame):
        close = data["close"]
        mom = close / close.shift(lookback) - 1.0
        long_state = mom > band
        short_state = mom < -band
        long_entries = long_state & ~long_state.shift(fill_value=False)
        short_entries = short_state & ~short_state.shift(fill_value=False)

        if ban_rollover:
            hours = data.index.hour
            banned = pd.Series(
                [h in BAN_HOURS for h in hours], index=data.index
            )
            long_entries = long_entries & ~banned
            short_entries = short_entries & ~banned

        long_exits = short_entries
        short_exits = long_entries
        return long_entries, long_exits, short_entries, short_exits

    return generate_signals


def set_gross():
    """コストゼロ(GROSS)に設定。"""
    C.SPREADS_PIPS = {k: 0.0 for k in C.SPREADS_PIPS}
    C.COMMISSION_FRACTION = 0.0


def sharpe_of(pf):
    m = metrics(pf)
    return float(m["sharpe"].iloc[0])


def main():
    set_gross()

    rows = []
    for tf, lb in CONFIGS:
        for pair in PAIRS:
            data = load(pair, tf)
            gen_raw = make_signals(lb, BAND, ban_rollover=False)
            gen_clean = make_signals(lb, BAND, ban_rollover=True)

            # both
            pf_raw = run(pair, tf, gen_raw, {}, data=data, size_mode="value", side="both")
            pf_clean = run(pair, tf, gen_clean, {}, data=data, size_mode="value", side="both")
            # long / short raw (ロールオーバー署名チェック)
            pf_long = run(pair, tf, gen_raw, {}, data=data, size_mode="value", side="long")
            pf_short = run(pair, tf, gen_raw, {}, data=data, size_mode="value", side="short")

            rows.append(
                {
                    "tf": tf,
                    "lb": lb,
                    "pair": pair,
                    "sharpe_raw": sharpe_of(pf_raw),
                    "sharpe_clean": sharpe_of(pf_clean),
                    "sharpe_long": sharpe_of(pf_long),
                    "sharpe_short": sharpe_of(pf_short),
                    "ret_raw": float(metrics(pf_raw)["total_return"].iloc[0]),
                    "ret_clean": float(metrics(pf_clean)["total_return"].iloc[0]),
                    "ntr_raw": int(metrics(pf_raw)["num_trades"].iloc[0]),
                    "ntr_clean": int(metrics(pf_clean)["num_trades"].iloc[0]),
                }
            )
            print(
                f"{tf} lb{lb} {pair}: raw={rows[-1]['sharpe_raw']:+.3f} "
                f"clean={rows[-1]['sharpe_clean']:+.3f} "
                f"long={rows[-1]['sharpe_long']:+.3f} short={rows[-1]['sharpe_short']:+.3f}"
            )

    df = pd.DataFrame(rows)

    print("\n===== per-config averages (7-pair mean) =====")
    for tf, lb in CONFIGS:
        sub = df[(df.tf == tf) & (df.lb == lb)]
        n_raw_pos = int((sub.sharpe_raw > 0).sum())
        n_clean_pos = int((sub.sharpe_clean > 0).sum())
        print(
            f"{tf} lb{lb}: raw_mean={sub.sharpe_raw.mean():+.4f} "
            f"clean_mean={sub.sharpe_clean.mean():+.4f} "
            f"(raw_pos {n_raw_pos}/7, clean_pos {n_clean_pos}/7) "
            f"long_mean={sub.sharpe_long.mean():+.4f} short_mean={sub.sharpe_short.mean():+.4f}"
        )

    # H1 lb24 詳細(タスクの主軸)
    print("\n===== H1 lb24 per-pair detail =====")
    h1 = df[(df.tf == "H1") & (df.lb == 24)]
    for _, r in h1.iterrows():
        print(
            f"  {r['pair']}: raw={r['sharpe_raw']:+.3f} clean={r['sharpe_clean']:+.3f} "
            f"long={r['sharpe_long']:+.3f} short={r['sharpe_short']:+.3f} "
            f"ntr {r['ntr_raw']}->{r['ntr_clean']}"
        )
    print(
        f"  H1 lb24 7-pair mean: raw={h1.sharpe_raw.mean():+.4f} "
        f"clean={h1.sharpe_clean.mean():+.4f} "
        f"long={h1.sharpe_long.mean():+.4f} short={h1.sharpe_short.mean():+.4f}"
    )

    # USD両方向同符号チェック(ロールオーバー署名)
    # long/short の符号がどうか
    print("\n===== rollover signature check (long vs short by config) =====")
    for tf, lb in CONFIGS:
        sub = df[(df.tf == tf) & (df.lb == lb)]
        print(
            f"{tf} lb{lb}: long>0 in {int((sub.sharpe_long>0).sum())}/7 pairs, "
            f"short>0 in {int((sub.sharpe_short>0).sum())}/7 pairs"
        )

    df.to_csv(
        C.ROOT / "research" / "outputs" / "tsmom_rollover_decontam.csv", index=False
    )
    print("\nsaved -> research/outputs/tsmom_rollover_decontam.csv")


if __name__ == "__main__":
    main()
