# -*- coding: utf-8 -*-
"""原稿の文字種検査。

このフリートで繰り返し起きている 2 つの壊れ方を止める。どちらも
**構文エラーにならず、目で見ても気づけない**という共通の性質を持つ。

  1. キリル文字の混入(HC-037 系)。`surrogate` の一部がキリル文字になっても
     字形が同じなので読んでも分からない。検索も当たらなくなる
  2. 制御文字の混入。シェル経由で正規表現を書いたとき `\\b` が 0x08 に潰れる型で、
     検査器が**対象 0 件を無言で返す**ようになる

ギリシャ文字は数式記号として意図して使っている(γ λ β Σ ε)。禁止するのではなく
**許可した記号だけ**を通す。許可表に無いギリシャ文字は、たいてい混入の側である。

usage: python build/hygiene.py   違反が 1 件でもあれば exit 1
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 検査対象。**出荷データ(web/data)は対象外** —— あちらは青空文庫の本文そのもので、
# 何が入っているかは原稿の問題ではない
TARGETS = [
    "build/*.py",
    "build/*.mjs",
    "tests/*.py",
    "web/*.html",
    "web/js/*.js",
    "web/css/*.css",
    "*.md",
]

CYRILLIC = range(0x0400, 0x0500)
GREEK_OK = set("γλβΣεμσπΔα")  # 数式記号として意図して使うもの


def offenders(text: str) -> list[tuple[int, str, str]]:
    """(行番号, 理由, 該当行) の一覧を返す。空リストなら合格。"""
    out = []
    for i, line in enumerate(text.split("\n"), start=1):
        for ch in line:
            if ord(ch) in CYRILLIC:
                out.append((i, f"キリル文字 U+{ord(ch):04X} {ch!r}", line.strip()))
                break
            if "GREEK" in unicodedata.name(ch, "") and ch not in GREEK_OK:
                out.append((i, f"許可外のギリシャ文字 U+{ord(ch):04X} {ch!r}", line.strip()))
                break
            if unicodedata.category(ch) == "Cc" and ch not in "\t":
                out.append((i, f"制御文字 U+{ord(ch):04X}", line.strip()))
                break
    return out


def files() -> list[Path]:
    seen: list[Path] = []
    for pat in TARGETS:
        seen.extend(sorted(ROOT.glob(pat)))
    return seen


def main() -> int:
    paths = files()
    if not paths:
        print("検査対象が 0 件。検査器が壊れている可能性がある", file=sys.stderr)
        return 2
    bad = 0
    for p in paths:
        for line, why, src in offenders(p.read_text(encoding="utf-8")):
            print(f"{p.relative_to(ROOT)}:{line}: {why}\n    {src}")
            bad += 1
    print(f"文字種検査: {len(paths)} ファイル / 違反 {bad} 件")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
