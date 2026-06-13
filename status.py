#!/usr/bin/env python3
"""リポジトリの現状を一画面に要約する高速ダッシュボード。

  uv run python status.py        # 構成 / 戦略一覧 / 最新レポート / データ有無

新セッションの最初の「確認」用。標準ライブラリのみ・価格データ不要・即時。
知見の詳細は reports/00_INDEX.md 冒頭の「📌 現状サマリ」を読む。
research/ は時系列の使い捨てログなので探索対象から外すこと。
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 現状の正典は reports/00_INDEX.md。チャンピオン名だけは目印用にここに持つ
# (変わったら INDEX の現状サマリと一緒に更新する)。
CHAMPION = "confluence_meanrev_v2_d1"


def tracked(*globs: str) -> list[str]:
    r = subprocess.run(["git", "ls-files", *globs], cwd=ROOT,
                       capture_output=True, text=True)
    return [ln for ln in r.stdout.splitlines() if ln]


def count_lines(paths: list[str]) -> int:
    n = 0
    for p in paths:
        try:
            with (ROOT / p).open("rb") as f:
                n += sum(1 for _ in f)
        except OSError:
            pass
    return n


def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.0f}{unit}" if unit == "B" else f"{nbytes:.1f}{unit}"
        nbytes /= 1024
    return f"{nbytes:.1f}GB"


def main() -> int:
    py = tracked("*.py")
    core = [p for p in py if not p.startswith("research/")]
    research = [p for p in py if p.startswith("research/")]
    docs = tracked("*.md", "*.txt")
    csv = tracked("*.csv")
    json = tracked("*.json")

    print("══════════ FX リサーチ環境 ステータス ══════════")
    print(f"  💻 コード      {count_lines(py):>7,}行  (コア {count_lines(core):,} / "
          f"research {count_lines(research):,} ※使い捨てログ)")
    print(f"  📄 ドキュメント {count_lines(docs):>7,}行  (md/txt {len(docs)}本)")
    print(f"  📊 出力データ   csv {len(csv)} / json {len(json)} ファイル "
          f"(全て research/outputs)")

    # 価格データ(gitignore・ローカル)の有無
    raw = sorted((ROOT / "data" / "raw").glob("*.parquet")) if (ROOT / "data" / "raw").exists() else []
    if raw:
        total = sum(p.stat().st_size for p in raw)
        pairs = ", ".join(sorted({p.stem.split("_")[0] for p in raw}))
        print(f"  💾 価格データ   {len(raw)}ファイル / {human(total)}  ({pairs})")
    else:
        print("  💾 価格データ   なし(未取得)→ uv run python scripts/download_data.py")

    # 戦略一覧
    strat = sorted(p[len("strategies/"):-3] for p in tracked("strategies/*.py")
                   if not Path(p).name.startswith("_"))
    print(f"\n── 戦略 strategies/ ({len(strat)}本) ──")
    for s in strat:
        mark = "  ★現チャンピオン" if s == CHAMPION else ""
        print(f"  {s}{mark}")

    # 最新レポート(INDEX の表から番号最大の数行)
    idx = ROOT / "reports" / "00_INDEX.md"
    if idx.exists():
        rows = []
        for ln in idx.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\|\s*\[(\d+)\]\([^)]+\)\s*\|\s*([^|]*)\|\s*([^|]*)\|", ln)
            if m:
                rows.append((int(m.group(1)), m.group(2).strip(), m.group(3).strip()))
        rows.sort(reverse=True)
        print(f"\n── 最新レポート(全{len(rows)}本・reports/00_INDEX.md)──")
        for num, _date, topic in rows[:4]:
            print(f"  {num:>2}  {topic[:72]}")

    print("\n→ 現チャンピオン/天井/閉鎖済み軸の詳細は reports/00_INDEX.md 冒頭「📌 現状サマリ」")
    print("→ research/ は探索しない(使い捨てログ)。本体は fxlab/ + strategies/ + 直下CLI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
