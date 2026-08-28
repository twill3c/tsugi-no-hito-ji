# -*- coding: utf-8 -*-
"""分布の加工(温度・top-k・top-p)の検算。

ここは LLM の「出口」そのもので、画面のつまみが直に触る部分である。
つまみの効きは目で見て「それらしい」だけになりやすいので、
**閉形式で答えが分かる場合を作って絶対値を固定する**。
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))
import model as M  # noqa: E402


def rand_dist(n: int, rng: random.Random) -> dict[str, float]:
    w = [rng.random() + 1e-3 for _ in range(n)]
    s = sum(w)
    return {chr(0x3042 + i): v / s for i, v in enumerate(w)}


# ---- 温度 -----------------------------------------------------------------------


@pytest.mark.unit
def test_t030_temperature_one_is_identity():
    """T-030 / 出所=定義。T=1 は p^(1/1) の正規化なので元の分布に戻る。"""
    rng = random.Random(20260829)
    for _ in range(20):
        p = rand_dist(12, rng)
        q = M.temperature(p, 1.0)
        for c in p:
            assert abs(p[c] - q[c]) < 1e-12


@pytest.mark.unit
def test_t031_temperature_two_point_closed_form():
    """T-031 / 出所=閉形式。2 点分布なら手で書ける式と一致するはず。

        q_1 = a^(1/T) / (a^(1/T) + (1-a)^(1/T))

    実装は対数側で最大値を引いてから戻す(桁溢れ対策)。
    その引き算が確率を歪めていないことを、素直な式で確かめる。
    """
    for a in (0.9, 0.5, 0.01, 0.999):
        for t in (0.1, 0.5, 1.0, 2.0, 10.0):
            p = {"あ": a, "い": 1 - a}
            got = M.temperature(p, t)
            x, y = a ** (1 / t), (1 - a) ** (1 / t)
            assert abs(got["あ"] - x / (x + y)) < 1e-9, (a, t)


@pytest.mark.unit
def test_t032_entropy_increases_with_temperature():
    """T-032 / 出所=定理。温度を上げるとエントロピーは下がらない。

    p^(1/T) 族は T について指数型分布族の逆温度をなぞるので、
    エントロピーは T の非減少関数になる。**画面のつまみの説明文が
    主張しているのはこの向きそのもの**なので、向きをテストに置く。
    """
    rng = random.Random(7)
    for _ in range(30):
        p = rand_dist(20, rng)
        hs = [M.entropy(M.temperature(p, t)) for t in (0.2, 0.5, 1.0, 2.0, 5.0, 20.0)]
        for a, b in zip(hs, hs[1:]):
            assert b >= a - 1e-9, f"温度を上げたのにエントロピーが下がった: {hs}"
        assert hs[-1] > hs[0], "一様に近づいていない(つまみが効いていない)"


@pytest.mark.unit
def test_t033_temperature_limits():
    """T-033 / 出所=極限。T→0 は最頻値へ集中し、T→大 は一様に近づく。"""
    p = {"あ": 0.5, "い": 0.3, "う": 0.2}
    zero = M.temperature(p, 0.0)
    assert zero == {"あ": 1.0, "い": 0.0, "う": 0.0}
    assert M.entropy(zero) == 0.0
    hot = M.temperature(p, 500.0)
    assert abs(M.entropy(hot) - math.log2(3)) < 1e-2

    # 同率のときは山分けする(片方に寄せると、種を変えるだけで結果が変わる)
    tie = M.temperature({"あ": 0.4, "い": 0.4, "う": 0.2}, 0.0)
    assert tie == {"あ": 0.5, "い": 0.5, "う": 0.0}


# ---- top-k / top-p ---------------------------------------------------------------


@pytest.mark.unit
def test_t034_top_k_keeps_exactly_k_and_renormalizes():
    """T-034 / 出所=定義。上位 k 個だけが残り、総和は 1 に戻る。"""
    rng = random.Random(3)
    p = rand_dist(30, rng)
    for k in (1, 3, 10, 29, 30, 99):
        q = M.top_k(p, k)
        assert len(q) == min(k, len(p))
        assert abs(sum(q.values()) - 1.0) < 1e-12
        # 残った字は、元の分布での上位 k 個と集合として一致する
        want = {c for c, _ in sorted(p.items(), key=lambda kv: (-kv[1], kv[0]))[: min(k, len(p))]}
        assert set(q) == want
    assert M.top_k(p, 0) == p, "k<=0 は制限なし"


@pytest.mark.unit
def test_t035_top_p_is_the_smallest_prefix_reaching_q():
    """T-035 / 出所=定義。累積が q を初めて超えるところまで、が正しい残し方。

    「超えるまで」と「超えたら止める」を取り違えると 1 個ずれる。
    ずれても総和は 1 に戻るので、正規化を見ているだけでは気づけない。
    そこで**残した集合から 1 個抜いたら q に届かない**ことを直接主張する。
    """
    rng = random.Random(11)
    for _ in range(20):
        p = rand_dist(25, rng)
        for q in (0.1, 0.5, 0.9, 0.99):
            kept = M.top_p(p, q)
            assert abs(sum(kept.values()) - 1.0) < 1e-12
            order = [c for c, _ in sorted(p.items(), key=lambda kv: (-kv[1], kv[0]))]
            k = len(kept)
            assert set(kept) == set(order[:k])
            assert sum(p[c] for c in order[:k]) >= q - 1e-12, "q に届いていない"
            if k > 1:
                assert sum(p[c] for c in order[: k - 1]) < q, "1 個多く残している"
    assert len(M.top_p({"あ": 0.99, "い": 0.01}, 0.5)) == 1
    assert M.top_p(p, 1.0) == p, "q>=1 は制限なし"


@pytest.mark.unit
def test_t036_top_p_is_monotone_in_q():
    """T-036 / 出所=定義。q を上げれば残る集合は縮まない(包含関係)。"""
    rng = random.Random(5)
    p = rand_dist(40, rng)
    prev: set[str] = set()
    for q in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0):
        cur = set(M.top_p(p, q))
        assert prev <= cur, f"q={q} で集合が縮んだ"
        prev = cur


# ---- エントロピーと当惑度 ----------------------------------------------------------


@pytest.mark.unit
def test_t037_entropy_closed_forms():
    """T-037 / 出所=閉形式。一様なら log2 n、一点なら 0。"""
    for n in (2, 3, 16, 71):
        uni = {chr(0x3042 + i): 1 / n for i in range(n)}
        assert abs(M.entropy(uni) - math.log2(n)) < 1e-12
    assert M.entropy({"あ": 1.0, "い": 0.0}) == 0.0


@pytest.mark.unit
def test_t038_perplexity_is_two_to_the_entropy(mdl, held):
    """T-038 / 出所=恒等式。PPL = 2^H。画面が両方出すので、両方が同じ計算から出ることを見る。"""
    r = mdl.bits_per_char(held[:400])
    assert abs(r["ppl"] - 2 ** r["bits"]) < 1e-9


@pytest.mark.unit
def test_t039_cross_entropy_matches_direct_average(mdl, held):
    """T-039 / 出所=定義の言い換え。1 字ずつ足した平均と、まとめて測った値が一致する。

    bits_per_char は内部で合計してから割る。ここでは外から 1 字ずつ
    -log2 p を集め直して同じ値になることを見る(集計の取り違えの検出)。
    """
    sample = held[:300]
    total = 0.0
    for i, ch in enumerate(sample):
        ctx = sample[max(0, i - mdl.max_order) : i]
        c = ch if ch in mdl.uni else M.UNK
        total += -math.log2(mdl.prob_of(ctx, c))
    assert abs(total / len(sample) - mdl.bits_per_char(sample)["bits"]) < 1e-9
