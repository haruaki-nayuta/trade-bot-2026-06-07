# fx-lab — FX 手法リサーチ&バックテスト環境

FX の手法を **考える(リサーチ)→ コード化 → バックテスト検証** まで一気通貫で回す環境。

- バックテスト: **vectorbt**(numba ベクトル化・並列・高速)
- データ: **Dukascopy** の 7 大メジャー M1(1分足)・過去 10 年・UTC
- 環境: **uv** + Python 3.12

## クイックスタート

```bash
# データ取得(初回のみ。10年×7ペア)
uv run python scripts/download_data.py

# 取得状況の確認
uv run python -c "from fxlab import summary; print(summary())"

# バックテスト
uv run python run_backtest.py ma_cross --pair EURUSD --tf H1            # 単発
uv run python run_backtest.py ma_cross --pair EURUSD --tf H1 --sweep    # 並列パラメータ探索
uv run python run_backtest.py ma_cross --all-pairs --tf H1              # 7ペア横断

# 10年総合評価(パラメータ探索 / IS・OOS / 7ペア横断 / マルチ時間足 / 改善提案)
uv run python evaluate.py ma_cross

# テスト(合成データ・DL不要)
uv run pytest -q
```

新しい手法は `strategies/_template.py` をコピーして 1 ファイル作るだけ。
**詳しい使い方・設計・評価指針は [CLAUDE.md](CLAUDE.md) を参照。**

## ディレクトリ構成

| パス | 役割 |
|---|---|
| `fxlab/` | 中核パッケージ。`config`(ペア/コスト/期間)・`data`(読込・リサンプル)・`backtest`(run/sweep/metrics)・`evaluate`(10年総合評価+改善提案)・`trades`(事後分析) |
| `strategies/` | 手法。1 ファイル 1 戦略(`generate_signals` を書くだけ)。`_template.py` がひな型 |
| `scripts/` | データ取得・更新・品質チェック |
| `tests/` | pytest(先読み防止・コスト計上・リサンプル規則・サイジング等の安全網) |
| `reports/` | 検証の知見ログ。[`reports/00_INDEX.md`](reports/00_INDEX.md) が目次 |
| `research/` | 使い捨ての実験スクリプト群(下記) |
| `evaluate.py` / `run_backtest.py` / `leaderboard.py` / `extract_trades.py` | ルート直下の CLI エントリポイント |

### ルート直下の CLI

| コマンド | 用途 |
|---|---|
| `evaluate.py` | アイデアを投げる入口。10年総合評価 + 改善提案 |
| `run_backtest.py` | 個別バックテスト(単発 / `--sweep` / `--all-pairs`) |
| `leaderboard.py` | 全戦略を横並び比較(OOS Sharpe 降順) |
| `extract_trades.py` | ベスト/ワーストのトレード + 直前の値動きを抽出 |

### `research/` — 実験スクリプト

検証イテレーションの記録。コア(`fxlab/`)とは独立した「使い捨ての作業ログ」で、消えても本体は動く。

| サブフォルダ | 中身 |
|---|---|
| `research/experiments/` | `exp*.py` … 時系列の反復検証ログ(ブレイクアウト / ボラ / セッション系を含む) |
| `research/money_management/` | `mm_*.py` … 資金管理(ベットサイジング)ラボ |
| `research/lab/` | その他の探索ツール(`bleed_lab` / `portfolio` / `screen` / `signals` など) |
| `research/outputs/` | 実験が吐いた CSV |

実行はリポジトリ直下から `-m` で(直叩きではなく `-m`。`fxlab`/`strategies` への import を解決するため):

```bash
uv run python -m research.experiments.exp22_verify
uv run python -m research.money_management.mm_production
```
