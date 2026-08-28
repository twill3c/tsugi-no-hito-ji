# -*- coding: utf-8 -*-
"""文字種検査そのものの検査と、原稿への適用。

「違反 0 件」は「検査した」を意味しない(HC-041)。検査器が対象を 1 つも
見ていないときも、パターンが壊れて何も当たらないときも、同じ緑が出る。
そこで **(a) 必ず捕まえるべき悪い例 (b) 撃ってはならない正当な例** を並べ、
検査器自身をテストしてから、原稿に当てる。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build"))
import hygiene as H  # noqa: E402


# 捕まえなければならない例。
#
# **悪い字を原稿に直接書いてはならない。** 書くと、原稿を走査する T-072 が
# 自分の陽性対照を撃つ(chikuma-seiki loop_007 で踏んだ型 —— HC-002)。
# かといって「この行だけ見逃す」印を付けると、検査そのものに穴が開く。
# そこで**符号位置から実行時に組み立てる**。検査器が受け取るのは本物の悪い字で、
# 原稿に残るのは 16 進の数字だけになる。
def _ch(cp: int) -> str:
    return chr(cp)


BAD = [
    ("キリル小文字 a(U+0430)", "const " + _ch(0x0430) + " = 1;"),
    ("キリル小文字 o を含む英単語", "// surr" + _ch(0x043E) + "gate pair"),
    ("キリル小文字 e", "text.replac" + _ch(0x0435) + "(x)"),
    ("制御文字(単語境界が潰れた跡)", "re.compile('" + _ch(0x08) + "word" + _ch(0x08) + "')"),
    ("許可外のギリシャ文字 omega", "const " + _ch(0x03C9) + " = 2;"),
]

# 撃ってはならない例。実際に原稿で使っている書き方をそのまま並べる
GOOD = [
    ("日本語の説明文", "キリル文字が混入していないかを検査する"),
    ("数式のギリシャ文字", "γ_m = D · u_m / n_m とし、λ 型の混合は採らない"),
    ("総和記号", "下界 = Σ_i i · (q_i − q_{i+1}) · log2 i"),
    ("タブ", "\tconst x = 1;"),
    ("全角記号と絵文字混じり", "確率 → 0.87(⏎ は改行)"),
]


@pytest.mark.unit
def test_t070_checker_catches_every_known_bad_line():
    """T-070 / 陽性対照。既知の悪い例を、検査器が 1 つ残らず捕まえる。"""
    for name, line in BAD:
        assert H.offenders(line), f"捕まえられなかった: {name}: {line!r}"


@pytest.mark.unit
def test_t071_checker_ignores_every_legitimate_line():
    """T-071 / 陰性対照。正当な書き方を撃たない。

    ここが無いと、検査に通すために原稿から数式記号を追い出すことになる。
    緩めすぎを止めるのが T-070、締めすぎを止めるのがこちら。対で置く。
    """
    for name, line in GOOD:
        assert not H.offenders(line), f"誤って撃った: {name}: {H.offenders(line)}"


@pytest.mark.validation
def test_t072_manuscript_is_clean():
    """T-072 / 出所=SPEC N-03。原稿すべてに違反が無い。"""
    paths = H.files()
    assert len(paths) >= 8, f"検査対象が少なすぎる({len(paths)} 件)。走査の網が壊れている"
    found = []
    for p in paths:
        for line, why, src in H.offenders(p.read_text(encoding="utf-8")):
            found.append(f"{p.relative_to(ROOT)}:{line}: {why}\n    {src}")
    assert not found, "文字種違反:\n" + "\n".join(found)


@pytest.mark.validation
def test_t073_scan_actually_reaches_the_source_files():
    """T-073 / 陽性対照。走査の網が、実際に主要な原稿を含んでいる。

    対象が空でないことだけでは足りない。**名前で確かめる** ——
    パターンを 1 つ書き損ねても件数は減るだけで、緑のままになる。
    """
    names = {p.name for p in H.files()}
    for want in ("model.py", "model.js", "prepare.py", "js_dump.mjs", "index.html"):
        assert want in names, f"{want} が検査対象に入っていない: {sorted(names)}"
