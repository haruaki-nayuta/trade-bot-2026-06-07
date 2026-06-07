# CLAUDE.md — FX 手法リサーチ&バックテスト環境

このリポジトリは **「FX の手法を考える(リサーチ)→ コード化 → バックテスト検証」までを一気通貫で AI に任せる**ための環境。新しいセッションの AI は、このファイルを読めば手法アイデアを受け取って自律的に検証できる。

---

## 0. このリポジトリで AI がやること(ワークフロー)

**ユーザーが雑にアイデアを投げてきたら(例:「H4 のボリンジャーバンド逆張りは効く?」「押し目買いって勝てる?」)、コード化して `evaluate.py` を回すだけで、10年スパンの検証結果と改善提案が出る。** これがこの環境の主目的。

1. **リサーチ(必要なら)** — 曖昧なアイデアは WebSearch / deep-research で具体化(指標・条件・パラメータ)。
2. **コード化** — `strategies/_template.py` をコピーして `strategies/<name>.py` を作り、`generate_signals` を書く(§4)。
3. **総合検証(これ1本でOK)** —
   ```bash
   uv run python evaluate.py <name>            # 既定 EURUSD H1。--pair/--tf で変更
   ```
   10年スパンで **パラメータ探索 / IS・OOS(過剰最適化チェック) / 7ペア横断 / マルチ時間足 / ロング・ショート分離 / 損切り・利確の自動テスト** を回し、**🔧 改善提案を自動生成**する(§5)。
4. **報告** — レポート冒頭の「**総合判定**」と「**🔧 改善提案**」をユーザーに伝える。数字は誇張しない(OOS で崩れていれば正直に「非推奨」と言う)。
5. **改善イテレーション** — 提案(例:「SL1%+TP2% で Sharpe 改善」「H4 の方が良い」「ロング専用化」)を戦略コードに反映し、再度 `evaluate.py` を回す。良くなるまで 2〜3 周。さらに深掘りするなら `extract_trades.py`(§5-4)で**ベスト/ワーストのトレードと直前の値動き**を見て、勝ちトレードに共通する地合いをフィルタ条件に落とし込む。

細かく1点だけ見たいときは `run_backtest.py`(§5末)。良い手法の目安・カーブフィット対策は §6。

---

## 1. 環境

- **パッケージ管理: `uv`**(Python 3.12 固定。システムの 3.14 は科学計算系が未対応のため使わない)。
- コマンドは必ず **`uv run python ...`** で実行する(`.venv` を自動で使う)。
- 主要ライブラリ: `vectorbt`(高速ベクトル化バックテスト), `dukascopy-python`(データ取得), `pandas` / `numpy` / `numba`, `ta`, `joblib`, `plotly`, `pyarrow`。
- 依存追加は `uv add <pkg>`。

```bash
uv run python -c "import fxlab; print('ok')"   # 動作確認
```

---

## 2. ディレクトリ構成

```
economy/
├── CLAUDE.md              ← これ
├── pyproject.toml         ← uv プロジェクト定義
├── fxlab/                 ← 中核パッケージ(基本いじらない)
│   ├── config.py          通貨ペア・パス・取引コスト・期間
│   ├── data.py            データ読込 / リサンプル / 状態確認
│   ├── backtest.py        run() 単発 / sweep() 並列探索 / metrics()
│   ├── evaluate.py        evaluate() 10年総合評価 / diagnose() 改善提案
│   └── trades.py          analyze() ベスト/ワーストのトレード+直前の値動き抽出
├── strategies/            ← ★手法はここに 1 ファイル 1 戦略で追加
│   ├── _template.py       テンプレ(コピー元)
│   ├── ma_cross.py        例: 移動平均クロス
│   ├── rsi_meanrev.py     例: RSI 逆張り
│   └── donchian_breakout.py 例: ブレイクアウト
├── scripts/
│   ├── download_data.py   10年分 M1 を取得
│   └── update_data.py     最新まで差分更新
├── evaluate.py            ← ★★ アイデアを投げる入口(10年総合評価+改善提案)
├── extract_trades.py      ← ベスト/ワーストのトレード+直前の値動きを抽出
├── run_backtest.py        ← 個別バックテスト CLI(単発/sweep/all-pairs)
├── data/raw/              M1 parquet(EURUSD_M1.parquet ...)※gitignore
└── results/              バックテスト結果 CSV/HTML/eval_*.md ※gitignore
```

---

## 3. データ

- **ソース: Dukascopy**(無料で入手できる中で最高精度・信頼性が高い)。
- **対象: 7 大メジャー** — EURUSD / USDJPY / GBPUSD / AUDUSD / USDCHF / USDCAD / NZDUSD。
- **基盤の足: M1(1分足)**。`data/raw/{PAIR}_M1.parquet` に UTC・OHLCV・BID で保存。
- **上位足は M1 からリサンプル**して使う(再取得不要)。対応: `M1/M5/M15/M30/H1/H4/D1/W1`。
- 期間: 過去 **10 年**(`fxlab/config.py: HISTORY_YEARS`)。

```python
from fxlab import load, summary
df = load("EURUSD", "H1")   # ← 一番よく使う。M1 を H1 に集約して返す
print(summary())            # 取得済みペアの行数・期間・容量を一覧
```

データ取得・更新(通常は構築済み。再取得が必要なときだけ):

```bash
uv run python scripts/download_data.py            # 全ペア10年(既存はスキップ)
uv run python scripts/download_data.py --force    # 既存も再取得
uv run python scripts/update_data.py              # 最新まで差分更新
```

---

## 4. 手法の書き方(これだけ覚える)

`strategies/_template.py` をコピーして 1 ファイル作る。中身は **`generate_signals` 1 関数だけ**。

```python
import pandas as pd
import vectorbt as vbt

PARAMS = {"period": 14, "low": 30, "high": 70}                       # 単発用デフォルト
PARAM_GRID = {"period": [7,14,21], "low": [20,30], "high": [70,80]}  # sweep 探索範囲

def generate_signals(data: pd.DataFrame, period=14, low=30, high=70):
    close = data["close"]
    rsi = vbt.RSI.run(close, period).rsi
    long_entries  = (rsi < low)  & (rsi.shift() >= low)
    long_exits    =  rsi > 50
    short_entries = (rsi > high) & (rsi.shift() <= high)
    short_exits   =  rsi < 50
    return long_entries, long_exits, short_entries, short_exits
```

**契約(必ず守る):**
- 引数 `data` は OHLCV の DataFrame(`open/high/low/close/volume`, UTC index)。
- 返り値は `data.index` に整列した **bool の pd.Series**:
  - ロングのみ → `(long_entries, long_exits)`
  - 両建て → `(long_entries, long_exits, short_entries, short_exits)`
- **先読み(look-ahead)禁止**:未確定バーの値で判断しない。指標は過去〜現在足のみ。rolling 系の極値を「自バーを含めず」使う場合は `.shift()` する(例: `donchian_breakout.py`)。
- `vbt.MA / vbt.RSI / vbt.BBANDS / vbt.ATR / vbt.MACD` などが使える。`ta` ライブラリも可。

---

## 5. 検証の実行

### 5-1. 総合評価 `evaluate.py`(メイン。これを使う)

```bash
uv run python evaluate.py ma_cross                      # EURUSD H1 で10年総合評価
uv run python evaluate.py rsi_meanrev --pair USDJPY --tf H4
uv run python evaluate.py ma_cross --save               # results/eval_*.md に保存
uv run python evaluate.py ma_cross --tfs M30 H1 H4      # マルチ時間足の対象を指定
```

1コマンドで以下を自動実行し、**総合判定 + 改善提案**を Markdown で出力:

| 検証 | 何を見るか |
|---|---|
| パラメータ探索 | 主ペア×主足の最適パラメータ(+グリッド全体で頑健性判定) |
| IS / OOS | 期間前半で最適化→後半で素の成績。**過剰最適化を一発で暴く** |
| 7ペア横断 | 同一パラメータが他通貨でも通用するか(まぐれ排除) |
| マルチ時間足 | M15/H1/H4/D1 のどれが効くか |
| ロング/ショート分離 | 片側だけ効いていないか |
| 損切り/利確 自動テスト | SL/TP/トレーリングで改善するかを**実測** |

改善提案はデータ駆動(`fxlab/evaluate.py: diagnose()`)。例:「IS→OOS で崩壊=過剰最適化」「最適値がグリッド端=範囲拡張」「SL1%+TP2% で Sharpe 改善(実測)」「EURUSD/GBPUSD は効くが JPY 系は負け=通貨依存」など、**根拠と次アクションをセットで**返す。

### 5-2. 個別バックテスト `run_backtest.py`(1点だけ細かく見たいとき)

```bash
uv run python run_backtest.py ma_cross --pair EURUSD --tf H1            # 単発
uv run python run_backtest.py ma_cross --pair EURUSD --tf H1 --params fast=10,slow=100
uv run python run_backtest.py ma_cross --pair EURUSD --tf H1 --sweep    # パラメータ総当り
uv run python run_backtest.py ma_cross --all-pairs --tf H1              # 7ペア横断
uv run python run_backtest.py ma_cross --pair EURUSD --tf H1 --save --plot  # CSV/資産曲線HTML
```

### 5-3. Python から直接(探索・デバッグ時)

```python
from fxlab import run, sweep, metrics
from fxlab.evaluate import evaluate
from strategies.ma_cross import generate_signals, PARAM_GRID

pf  = run("EURUSD", "H1", generate_signals, {"fast":20,"slow":50})  # 単発
res = sweep("EURUSD", "H1", generate_signals, PARAM_GRID)           # 並列探索→Sharpe降順
# run() は data=（期間スライス）, side="long"/"short", sl_stop/tp_stop/tsl_stop（割合）も取れる
```

**速度・並列の仕組み:** `sweep()` は全パラメータ組合せのシグナルを joblib で並列生成し、**1 回のベクトル化シミュレーションで全組合せを同時にシミュレート**(numba 並列)。数百〜数千通りでも高速。

### 5-4. トレードの事後分析 `extract_trades.py`(なぜ勝てた/負けたか)

特に良かった/悪かったトレードと、**その「トレード前の値動き」**を抜き出して、勝ち負けの条件を研究する。

```bash
uv run python extract_trades.py ma_cross --pair EURUSD --tf H1            # ベスト/ワースト各5件
uv run python extract_trades.py ma_cross --pair EURUSD --tf H1 --n 10 --lookback 80
uv run python extract_trades.py ma_cross --pair EURUSD --tf H1 --params fast=30,slow=100
uv run python extract_trades.py ma_cross --pair EURUSD --tf H1 --save --plot
```

出力:
- ベスト/ワースト n 件のトレード(損益・保有期間)と、**エントリー直前の特徴量**:
  直前リターン `pre_ret_%` / 傾き `pre_trend_%/bar` / ボラ `pre_vol_%` / 陽線率 `up_bar_ratio` /
  `rsi_at_entry` / `atr_at_entry_%` / 直近高値・安値からの距離 `dist_from_high_%` `dist_from_low_%`。
- **「ベスト vs ワースト の直前値動き平均」**の比較表 → 勝ちトレードに共通する地合いが見える(例:勝ちは RSI 高め・安値から離れて入る等)。フィルタ条件の発見に直結。
- `--save`: 全トレード+特徴量 `summary_all.csv` と、各トレードの**値動きOHLCV**(直前 `phase=pre` + 建玉中 `phase=trade`)CSV。
- `--plot`: 各トレードのローソク足チャート(エントリー=青線/エグジット=橙線、直前の値動き付き)。

→ ここで見つけた「勝ちトレードの地合い」を `generate_signals` のフィルタに足して `evaluate.py` で再検証、が改善の王道ループ。

Python から: `from fxlab import trades; r = trades.analyze(pf, data, n=5, lookback=50)`(`r["best"]`/`r["worst"]`/`r["contexts"]`)。

---

## 6. 評価の指針(過剰最適化を避ける)

`metrics()` / 各 runner が返す標準指標:

| 指標 | 意味 | 目安 |
|---|---|---|
| `total_return` | 累積リターン | プラスは前提 |
| `sharpe` | リスク調整後リターン | > 1 で良好、> 2 は要過剰最適化チェック |
| `sortino` | 下方リスク調整後 | sharpe と併読 |
| `max_drawdown` | 最大ドローダウン | 浅いほど良い(-0.2 以内が目安) |
| `win_rate` | 勝率 | 単体では判断しない(PF と併読) |
| `profit_factor` | 総利益/総損失 | > 1.3 で実用的 |
| `num_trades` | 取引数 | 少なすぎ(<30)は統計的に当てにならない |
| `expectancy` | 1取引あたり期待損益 | プラス必須 |

**カーブフィット対策(必ず意識する):**
- `--sweep` のトップ1だけを信じない。**パラメータ近傍が滑らかに良い**(高原状)かを見る。1点だけ突出は過剰最適化の疑い。
- `--all-pairs` で**複数通貨でも通用するか**を確認。1ペアだけ突出は偶然の可能性。
- 取引数が少ない結果は割り引いて見る。
- 取引コスト(スプレッド)は `fxlab/config.py: SPREADS_PIPS` で計上済み。**コスト無視の好成績に注意**(特に短期足は影響大)。
- 必要なら期間を前半/後半に分けて **in-sample / out-of-sample** で確認する。

---

## 7. 取引コストの扱い

- スプレッドを **バーごとの価格比スリッページ**として現実的に計上(エントリー+エグジットで往復1スプレッド)。
- ペア別の平均スプレッド(pips)は `fxlab/config.py: SPREADS_PIPS`。実口座に合わせて調整可。
- 手数料は `COMMISSION_FRACTION`(デフォルト 0、ECN 想定なら例: `$30/100万 ≒ 0.00003`)。
- JPY ペアは pip=0.01、その他は pip=0.0001(`config.pip_size`)。

---

## 8. 規約・注意

- 実行は常に `uv run`。`pip install` 単体は使わない(`uv add`)。
- 時刻は全て **UTC**。`data/`・`results/` は git 管理外。
- ライブ発注・送金は**しない**(検証専用環境)。
- `fxlab/` のコア API を変えたら §5 の例で動作確認する。
- 新しい手法を足すたびに `strategies/` に 1 ファイル追加し、`--sweep` と `--all-pairs` まで回して結論づける。
```
