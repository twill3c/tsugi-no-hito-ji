# -*- coding: utf-8 -*-
"""推測回数からのエントロピー上下界(Shannon 1951)の検算。

**公表値を暗記して期待値に書かない。** 1951 年の論文の英文 1.3 ビットという数字は
有名だが、本アプリが扱うのは日本語であり、そもそもこの式が手元の実装で
本当に挟むのかを先に確かめないと、人間の成績に意味のある数字を返せない。

そこで「エントロピーが厳密に分かっている分布」を無作為に大量に作り、
順位ごとの確率をそのまま q とみなして上下界が実際に挟むかを見る。
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))
import model as M  # noqa: E402


@pytest.mark.unit
def test_t050_bounds_bracket_the_true_entropy():
    """T-050 / 出所=定理を実測で確認。無作為 2,000 分布で下界 ≤ H ≤ 上界。

    分布の形を偏らせる(指数を 0.3 / 1 / 3 で振る)ことで、
    ほぼ一様な分布と、極端に尖った分布の両方を通す。
    片方だけだと、式を取り違えていても通ってしまう組み合わせがある。
    """
    rng = random.Random(20260829)
    for _ in range(2000):
        n = rng.randint(2, 40)
        w = [rng.random() ** rng.choice([0.3, 1.0, 3.0]) for _ in range(n)]
        s = sum(w)
        p = sorted((x / s for x in w), reverse=True)
        h = -sum(x * math.log2(x) for x in p if x > 0)
        lo, up = M.shannon_bounds(p)
        assert lo <= h + 1e-9, f"下界が上回った: {lo} > {h}"
        assert h <= up + 1e-9, f"上界が下回った: {h} > {up}"


@pytest.mark.unit
def test_t051_bounds_are_tight_for_uniform():
    """T-051 / 出所=閉形式。一様分布では上下界が一致して log2 n になる。

    上下が一致する場合を 1 つ持っておくと、片方だけ壊れたときに気づける
    (どちらも「それらしい数」を返すので、値を見比べるだけでは分からない)。
    """
    for n in (2, 27, 71, 128):
        q = [1 / n] * n
        lo, up = M.shannon_bounds(q)
        assert abs(lo - math.log2(n)) < 1e-9
        assert abs(up - math.log2(n)) < 1e-9


@pytest.mark.unit
def test_t052_perfect_guesser_has_zero_lower_bound():
    """T-052 / 出所=極限。毎回 1 回目で当たるなら、下界も上界も 0 ビット。

    「当てられる = 情報が無い」という向きを固定する。符号や log の底を
    取り違えると、ここが 0 にならずに負や 1 になる。
    """
    lo, up = M.shannon_bounds([1.0])
    assert lo == 0.0 and up == 0.0


@pytest.mark.unit
def test_t053_ideal_guesser_rank_matches_the_model_ranking(mdl):
    """T-053 / 出所=定義。理想予測器の推測回数は、確率順の順位そのもの。

    人間の成績と並べる相手がこれなので、順位の付け方(同率の扱い)が
    分布の並べ方と一致していなければならない。ずれると勝負が不公平になる。
    """
    p, _ = mdl.distribution("である。")
    order = [c for c, _ in sorted(p.items(), key=lambda kv: (-kv[1], kv[0]))]
    for want, ch in enumerate(order[:20], start=1):
        assert M.guess_ranks(p, ch) == want


@pytest.mark.integration
def test_t054_model_bounds_agree_with_its_own_entropy(mdl):
    """T-054 / 出所=定理。理想予測器の q を入れると、上界はエントロピーそのものになる。

    q が「分布を降順に並べたもの」であるとき、上界の式は
    − Σ p log p に一致する(並べ替えでエントロピーは変わらないため)。
    人間の q ではこの一致は起きない —— **その差が「人間はまだ最適に予測できていない」量**である。
    """
    p, _ = mdl.distribution("である。")
    q = sorted(p.values(), reverse=True)
    lo, up = M.shannon_bounds(q)
    assert abs(up - M.entropy(p)) < 1e-9
    assert lo <= M.entropy(p) + 1e-9


@pytest.mark.unit
def test_t055_lower_bound_is_only_valid_for_a_descending_guess_order():
    """T-055 / 陽性対照。前提を外れた q では下界が上界を上回りうる。

    **画面で実際にそうなった**(人間の下界 6.02 bit・上界 3.42 bit — HC-003)。
    下界の式は「確率の高い順に当てにいく」ことを前提にしており、
    人間の推測順はその前提を満たさない。ここでは
    (a) 前提を外れた q を検出できること、(b) 外れた q では実際に逆転が起きること、
    の両方を押さえる。**逆転が起きることを見ておかないと、
    番人だけ置いて安心する**ことになる。
    """
    ideal = [0.5, 0.3, 0.15, 0.05]
    assert M.is_non_increasing(ideal)
    lo, up = M.shannon_bounds(ideal)
    assert lo <= up + 1e-12

    # 下手な予測器。1 回目より 5 回目のほうがよく当たる、という形になりうる
    clumsy = [0.02, 0.03, 0.05, 0.10, 0.80]
    assert not M.is_non_increasing(clumsy)
    lo2, up2 = M.shannon_bounds(clumsy)
    assert lo2 > up2, "前提を外れているのに逆転が起きない。対照として役に立っていない"


@pytest.mark.unit
def test_t056_upper_bound_holds_for_any_guess_order():
    """T-056 / 出所=定理。上界は並べ替えても変わらない(どんな順で当てても言える)。

    上界は「推測回数の列から本文を復元できる」ことだけから出るので、
    予測器の巧拙によらない。だから人間にも出してよい。
    並べ替え不変であることが、その独立性の見える形である。
    """
    import random

    rng = random.Random(99)
    for _ in range(200):
        n = rng.randint(2, 20)
        w = [rng.random() for _ in range(n)]
        s = sum(w)
        q = [x / s for x in w]
        shuffled = q[:]
        rng.shuffle(shuffled)
        assert abs(M.shannon_bounds(q)[1] - M.shannon_bounds(shuffled)[1]) < 1e-12
