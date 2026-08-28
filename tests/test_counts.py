# -*- coding: utf-8 -*-
"""数え方の検算。

このアプリの確率はすべて「本文の中でその並びが何回出たか」から作られる。
数え方を間違えると、画面のすべての数字が静かにずれる(構文エラーにはならない)。
そこで**同じ量を、機構の違う 3 通りで数えて突き合わせる**:

  1. 本文を 1 回走査して作る表(長さ 1・2)
  2. 接尾辞配列の範囲を舐める(next_counts)
  3. 接尾辞配列の二分探索だけ(occ / distinct_followers — 範囲を舐めない)

3 つが一致することは、どれか 1 つが正しいことを意味しない。
だから (0) 総当たりで数える小さな本文を別に持ち、そこで絶対値を固定する。
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "build"))
import model as M  # noqa: E402

from conftest import _brute_sa  # noqa: E402


# ---- T-001 系: 接尾辞配列そのもの -------------------------------------------------


@pytest.mark.unit
def test_t001_suffix_array_matches_brute_force():
    """T-001 / 出所=総当たり。接尾辞配列は全接尾辞を素直に並べたものと一致する。

    prepare.py の接頭辞倍化は速いが読みにくい。読みにくい実装は、
    読める実装と突き合わせて初めて信用してよい。
    """
    import prepare as P

    for text in ["", "あ", "ああああ", "あいうえおあいうあいあ\n", "abracadabra", "ばばばばば。"]:
        if not text:
            continue
        codes = np.frombuffer(text.encode("utf-16-le"), dtype=np.uint16).astype(np.int64)
        got = P.suffix_array(codes)
        assert list(got) == _brute_sa(text), f"接尾辞配列が総当たりと違う: {text!r}"


@pytest.mark.validation
def test_t002_shipped_arrays_are_valid_for_every_corpus():
    """T-002 / 出所=不変量。出荷済みの全コーパスで、配列が置換かつ整列している。

    件数は書かない(コーパスを差し替えれば動く)。「本文と同じ長さの置換であること」
    「隣り合う接尾辞が昇順であること」という不変量だけを見る。
    """
    import prepare as P

    corpora = M.corpora()
    assert corpora, "コーパスが 1 つも出荷されていない"
    for c in corpora:
        text = (M.DATA / f"{c['id']}.txt").read_text(encoding="utf-8")
        sa = np.frombuffer((M.DATA / f"{c['id']}.sa.bin").read_bytes(), dtype=np.int32)
        assert len(text) == c["chars"], f"{c['id']}: corpora.json の字数と本文が食い違う"
        P.check_sa(text, sa)  # 置換 + 等間隔標本の順序


@pytest.mark.validation
def test_t003_shipped_text_is_bmp_only():
    """T-003 / 出所=SPEC N-02。BMP 外の字が残っていると Python と JS の添字がずれる。

    ずれても例外は出ない。**静かに違う場所を指すだけ**なので、ここで止める。
    """
    for c in M.corpora():
        text = (M.DATA / f"{c['id']}.txt").read_text(encoding="utf-8")
        bad = {ch for ch in text if ord(ch) > 0xFFFF}
        assert not bad, f"{c['id']}: BMP 外の字が残っている: {sorted(bad)[:8]}"
        # UTF-16 の符号単位数と文字数が一致することが、添字一致の本体
        assert len(text.encode("utf-16-le")) == 2 * len(text), f"{c['id']}: 符号単位と字数が不一致"


# ---- T-01x 系: 数え方の相互一致 ---------------------------------------------------


@pytest.mark.unit
def test_t010_tiny_counts_by_hand(tiny):
    """T-010 / 出所=手勘定。小さな本文で出現回数の絶対値を固定する。

    本文 "あいうえおあいうあいあ\\nあいうえお\\n" の添字は
    0あ 1い 2う 3え 4お 5あ 6い 7う 8あ 9い 10あ 11改行 12あ 13い 14う 15え 16お 17改行。
    「あい」は 0・5・8・12 の 4 か所に出て、直後は う・う・あ・う。
    """
    assert tiny.next_counts("あい") == Counter({"う": 3, "あ": 1})
    assert tiny.occ("あい") == 4
    assert tiny.occ("あいうえお") == 2
    assert tiny.occ("見えない") == 0
    # 本文末尾で終わる出現は「次の字」を持たない。
    # 「お改行」は 16-17 の 1 か所だけで、そこは本文の最後 —— 出現はするが数は 0 になる
    assert tiny.occ("お\n") == 1
    assert tiny.next_counts("お\n") == Counter()
    # 一方 4 の「お」は次が「あ」。同じ字でも位置で結果が変わることを対で押さえる
    assert tiny.next_counts("お") == Counter({"あ": 1, "\n": 1})


@pytest.mark.integration
def test_t011_table_and_suffix_array_agree(mdl):
    """T-011 / 出所=二経路一致。長さ 1・2 は表からも配列からも数えられる。

    表は本文の走査、配列は二分探索。**機構が違うので同義反復にならない。**
    速さのために表を使っている以上、表が本当に配列と同じものを返すかを見る。
    """
    probes = ["の", "は", "。", "\n", "であ", "です", "、そ", "吾輩"]
    for ctx in probes:
        from_table = mdl.next_counts(ctx)
        lo, hi = mdl.occ_range(ctx)
        from_sa: Counter = Counter()
        for i in range(lo, hi):
            j = int(mdl.sa[i]) + len(ctx)
            if j < mdl.n:
                from_sa[mdl.text[j]] += 1
        assert from_table == from_sa, f"表と配列が食い違う: {ctx!r}"

    # 陽性対照: 検査が働いていることを確かめる。わざと 1 件ずらせば落ちるはず
    broken = Counter(mdl.next_counts("の"))
    broken["の"] += 1
    assert broken != mdl.next_counts("の")


@pytest.mark.integration
def test_t012_distribution_and_probof_agree(mdl):
    """T-012 / 出所=二経路一致。分布を作る道と、1 字だけ求める道が一致する。

    片方は全字ぶんの表を作り、もう片方は二分探索だけで済ませる。
    式は同じでも、実装の経路がまるで違う。
    """
    for ctx in ["", "の", "である。", "吾輩は", "親譲りの無鉄砲で", "〓〓不在文脈"]:
        p, _ = mdl.distribution(ctx)
        targets = sorted(p, key=lambda c: -p[c])[:6] + [M.UNK, "の", "\n"]
        for ch in targets:
            assert abs(p.get(ch, 0.0) - mdl.prob_of(ctx, ch)) < 1e-12, f"{ctx!r} → {ch!r}"


@pytest.mark.integration
def test_t013_distinct_followers_matches_scan(mdl, tiny):
    """T-013 / 出所=二経路一致。種類数を「舐める」道と「跳ぶ」道が一致する。

    prob_of は速さのために跳び歩きで種類数を数える。跳び先の計算を 1 つ間違えても
    答えは「それらしい整数」になるので、走査側と突き合わせないと気づけない。
    """
    for m, ctx in [(1, "の"), (2, "であ"), (3, "である"), (4, "である。"), (8, "親譲りの無鉄砲で")]:
        assert len(mdl.next_counts(ctx)) == mdl.distinct_followers(ctx), f"種類数が食い違う: {ctx!r}"
    for ctx in ["あ", "あい", "あいう", "お\n"]:
        assert len(tiny.next_counts(ctx)) == tiny.distinct_followers(ctx), ctx


# ---- T-02x 系: 確率としての健全性 --------------------------------------------------


@pytest.mark.unit
def test_t020_distribution_sums_to_one(mdl, tiny):
    """T-020 / 出所=定義。混ぜ上げた分布は必ず総和 1 になる。

    絶対割引は「max(k-D,0) の総和」と「γ」が過不足なく 1 を分け合う形で書いてある。
    D >= 1 にすると k=1 の字が消えてここが崩れる — 式の前提そのものの検査。
    """
    for m in (mdl, tiny):
        for ctx in ["", "の", "あい", "である。", "親譲りの無鉄砲で", "〓〓不在文脈"]:
            p, trace = m.distribution(ctx)
            assert abs(sum(p.values()) - 1.0) < 1e-9, f"総和が 1 でない: {ctx!r}"
            assert all(v >= 0 for v in p.values()), f"負の確率がある: {ctx!r}"
            # 帯(各段の寄与)も 1 を分け合う。画面の帯はこの share だけから描く
            assert abs(sum(t["share"] for t in trace) - 1.0) < 1e-9, f"寄与の総和が 1 でない: {ctx!r}"


@pytest.mark.unit
def test_t021_unseen_characters_keep_positive_probability(mdl):
    """T-021 / 出所=SPEC F-02。見たことのない字にも 0 でない確率が残る。

    ここが 0 になると交差エントロピーが無限大に飛び、
    「次の一字を当てる相手」としてモデルが壊れる。打ち切りにしない理由そのもの。
    """
    p, _ = mdl.distribution("親譲りの無鉄砲で")
    assert p[M.UNK] > 0
    rare = min(mdl.uni, key=lambda c: mdl.uni[c])
    assert p[rare] > 0, f"本文に 1 回しか出ない字 {rare!r} の確率が 0 になった"


@pytest.mark.unit
def test_t022_longer_context_needs_the_prefix_to_exist(mdl):
    """T-022 / 出所=定義。長さ m で 0 回なら、それより長い文脈も必ず 0 回。

    distribution() はこの単調性に頼って途中で打ち切る。前提が崩れると
    「本当は使えた長い文脈」を黙って捨てることになる。
    """
    ctx = "〓〓不在文脈"
    assert mdl.occ(ctx[-1:]) >= mdl.occ(ctx[-2:]) >= mdl.occ(ctx)
    for m in range(1, len(ctx) + 1):
        if mdl.occ(ctx[-m:]) == 0:
            for longer in range(m, len(ctx) + 1):
                assert mdl.occ(ctx[-longer:]) == 0
            break
