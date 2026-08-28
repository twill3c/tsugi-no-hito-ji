# -*- coding: utf-8 -*-
"""G-01 二実装照合 —— 参照実装(Python)と出荷実装(JS)が同じ数を出す。

画面に出る数字を作っているのは JS の方である。Python だけを丁寧に検算しても、
出荷物が正しいことは何も言えない。逆に JS だけを見ても、
「それらしい数」が出ている限り誤りに気づけない。

**両者は SPEC の式だけを共通の出どころとして、別々のデータ構造で書いてある。**
だから一致は偶然では起きない。片方の凡ミスはここで落ちる。

浮動小数の差は許容するが、許容幅は狭く取る(1e-12)。両実装とも IEEE754 倍精度で
同じ順序の演算をしているので、桁落ちの仕方まで揃うはずである。
広い許容幅は「揃っていないこと」を隠す。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))
import model as M  # noqa: E402

TOL = 1e-12


@pytest.mark.validation
def test_t060_corpus_shape_agrees(js, mdl):
    """T-060 / 出所=二実装照合。本文の長さと字種数が一致する。

    ここがずれていたら以降の一致は無意味なので、最初に見る。
    文字数の不一致は、UTF-8 の復号や BOM の扱いの違いで静かに起きる。
    """
    assert js["n"] == mdl.n
    assert js["v"] == mdl.v


@pytest.mark.validation
def test_t061_distributions_agree(js, mdl):
    """T-061 / 出所=二実装照合。各文脈の上位 12 字の確率と、寄与の帯が一致する。

    上位だけでなく **trace(各段の n・u・γ・share)も突き合わせる**。
    確率が合っていても帯の内訳がずれていれば、画面の説明が嘘になる。
    """
    assert js["dist"], "JS 側の探り文脈が空"
    for ctx, got in js["dist"].items():
        p, trace = mdl.distribution(ctx)
        assert abs(got["sum"] - 1.0) < 1e-9, f"{ctx!r}: JS 側の総和が 1 でない"
        assert abs(got["entropy"] - M.entropy(p)) < TOL, f"{ctx!r}: エントロピー"
        for ch, v in got["top"]:
            assert abs(v - p[ch]) < TOL, f"{ctx!r} → {ch!r}: 確率"
        assert len(got["trace"]) == len(trace), f"{ctx!r}: 段数"
        for a, b in zip(got["trace"], trace):
            assert a["order"] == b["order"] and a["ctx"] == b["ctx"], f"{ctx!r}: 段の並び"
            assert a["n"] == b["n"] and a["u"] == b["u"], f"{ctx!r}: n / u"
            assert abs(a["gamma"] - b["gamma"]) < TOL, f"{ctx!r}: γ"
            assert abs(a["share"] - b["share"]) < TOL, f"{ctx!r}: 寄与"
        for ch, v in got["probOf"].items():
            assert abs(v - mdl.prob_of(ctx, ch)) < TOL, f"{ctx!r} → {ch!r}: 1 字だけの経路"


@pytest.mark.validation
def test_t062_probe_set_covers_present_and_absent(js, mdl):
    """T-062 / 陽性対照。探り文脈が「よくある」「一度しかない」「無い」を実際に含む。

    照合が緑でも、探った文脈が全部同じ性格なら、
    混ぜ上げの下の段や打ち切りの枝は一度も通っていない。
    **検査対象が空でないことを別に確かめる**(HC-041)。
    """
    kinds = {"多い": 0, "少ない": 0, "無い": 0}
    for ctx in js["dist"]:
        if ctx == "":
            continue
        n = mdl.occ(ctx)
        kinds["無い" if n == 0 else "少ない" if n <= 3 else "多い"] += 1
    for k, v in kinds.items():
        assert v > 0, f"探り文脈に「{k}」が 1 つも無い: {kinds}"


@pytest.mark.validation
def test_t063_shaping_agrees(js, mdl):
    """T-063 / 出所=二実装照合。温度・top-k・top-p の結果が一致する。

    画面のつまみが動かすのはここ。JS 側は対数で引いてから戻す実装なので、
    素直に書いた Python 側と数値が揃うことを確かめる意味がある。
    """
    base, _ = mdl.distribution("である。")
    for t in (0, 0.25, 0.5, 1, 1.5, 2):
        want = M.temperature(base, float(t))
        for ch, v in js["shaped"][f"T={t}"]:
            assert abs(v - want[ch]) < TOL, f"T={t} → {ch!r}"
        assert abs(js["shaped"][f"H@T={t}"] - M.entropy(want)) < TOL, f"T={t}: エントロピー"
    for k in (1, 3, 10):
        want = M.top_k(base, k)
        got = dict(js["shaped"][f"k={k}"])
        assert set(got) == set(want), f"k={k}: 残った字の集合"
        for ch, v in got.items():
            assert abs(v - want[ch]) < TOL, f"k={k} → {ch!r}"
    for q in (0.5, 0.9, 0.99):
        want = M.top_p(base, q)
        got = dict(js["shaped"][f"p={q}"])
        assert set(got) == set(want), f"p={q}: 残った字の集合"
        for ch, v in got.items():
            assert abs(v - want[ch]) < TOL, f"p={q} → {ch!r}"


@pytest.mark.validation
def test_t064_random_streams_agree(js):
    """T-064 / 出所=二実装照合。同じ種の疑似乱数が同じ数列を出す。

    mulberry32 は 32 ビットの畳み込みで、Python の多倍長整数と
    JS の Math.imul とでは桁の落ち方が違いうる。ここがずれると
    「同じ種なら同じ文章」が言えなくなる。
    """
    rnd = M.mulberry32(20260829)
    for i, v in enumerate(js["rands"]):
        assert abs(rnd() - v) < 1e-15, f"{i} 番目の乱数がずれた"


@pytest.mark.validation
def test_t065_generated_text_agrees(js, mdl):
    """T-065 / 出所=二実装照合。同じ種・同じつまみなら、生成される文章が 1 字も違わない。

    分布・加工・乱数・逆関数法のどこか 1 つでもずれると、
    たいてい数十字めで分岐する。**文章そのものの一致**は、
    数値の一致より人間に読めて、しかも厳しい。
    """
    rnd = M.mulberry32(7)
    s = "吾輩は"
    for _ in range(120):
        p, _ = mdl.distribution(s[-M.MAX_ORDER :])
        s += M.sample(M.top_p(M.temperature(p, 0.8), 0.95), rnd)
    assert s == js["gen"], f"生成が分岐した:\n Python: {s}\n JS    : {js['gen']}"


@pytest.mark.validation
def test_t066_bits_per_char_agrees(js, mdl, held):
    """T-066 / 出所=二実装照合。検証本文のビット数が一致する。

    このアプリが最後に出す数字がこれ。ここまでの経路すべてを一度に通るので、
    どこかがずれていれば必ず落ちる。
    """
    want = mdl.bits_per_char(held[:2000])
    got = js["bits"]
    assert got["chars"] == want["chars"]
    assert got["oov"] == want["oov"]
    assert abs(got["bits"] - want["bits"]) < 1e-12
    assert abs(got["ppl"] - want["ppl"]) < 1e-9
