# -*- coding: utf-8 -*-
"""目玉の主張を固定するゲート。

このアプリが画面で言い切っているのは 2 つだけである。

  「文脈は長いほど当たる、わけではない」
  「谷の位置は、読んだ量で決まる」

**1 つめは、もともと逆の主張を書こうとして実測に否定されたもの**(HC-001)。
だから主張の向きそのものをテストに置く。式や既定値を触ったら、ここが真っ先に落ちる。

数値そのものは書かない。コーパスを差し替えれば動くからである。
書くのは「下がる」「縮む」「符号が反転する」という**向き**だけ。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build"))
import model as M  # noqa: E402


@pytest.fixture(scope="module")
def scaling() -> dict:
    path = M.DATA / "scaling.json"
    assert path.exists(), "scaling.json が無い。python build/scaling.py を先に走らせる"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.validation
def test_t040_more_reading_lowers_the_whole_curve(scaling):
    """T-040 / 出所=不変量。学習量を増やすと、どの文脈長でもビット数が下がる。

    「読むほど当たる」はこのアプリの土台で、これが崩れていたら
    T-041 の谷の話は意味を持たない。**先に土台を見る。**

    比べるのは隣り合う 2 点ではなく、最小と最大の学習量。隣接だけで見ると、
    標本のぶれで 1 か所逆転しただけで落ちる(実際 12,500 → 25,000 では
    文脈長 0 が逆転している —— 走査の最小点は本文が短すぎて字種が揃わない)。
    """
    rows = scaling["rows"]
    assert len(rows) >= 4, f"走査の点が {len(rows)} 個しかない"
    small, large = rows[0]["curve"], rows[-1]["curve"]
    assert len(small) == len(large) == len(scaling["orders"])
    for L, (a, b) in enumerate(zip(small, large)):
        assert b < a, f"文脈長 {L}: 学習量を増やしたのにビット数が下がっていない({a} → {b})"


@pytest.mark.validation
def test_t041_the_valley_moves_right_as_the_corpus_grows(scaling):
    """T-041 / 出所=実測(HC-001 の主張の向きゲート)。

    画面が言っているのは「谷の位置は読んだ量で決まる」である。その根拠は
    **L=3 と L=2 の差が、学習量を増やすにつれて縮み、最後に符号が反転する**こと。

    ここでも数値は書かない。書くのは次の 3 点:
      (a) 学習量が最小のとき、谷は L=3 より手前にある
      (b) 学習量が最大のとき、谷は L=3 以降にある(= 右へ動いた)
      (c) 差の列が、途中から単調に縮んでいる
    """
    rows = scaling["rows"]
    gaps = scaling["gap_3_minus_2"]
    assert len(gaps) == len(rows)

    assert rows[0]["best_order"] < 3, f"最小の学習量で谷が {rows[0]['best_order']} 文字にある"
    assert rows[-1]["best_order"] >= 3, f"最大の学習量でも谷が {rows[-1]['best_order']} 文字のまま"
    assert gaps[-1] < 0, f"最大の学習量でも L=3 が L=2 を追い抜いていない(差 {gaps[-1]})"

    # 縮み方。最小の 1 点は本文が短すぎて挙動が安定しないので、2 点目以降で見る。
    # **どこから見るかを勝手に決めない** —— 2 点目以降で単調、と明示して固定する
    tail = gaps[1:]
    for a, b in zip(tail, tail[1:]):
        assert b < a, f"差が縮んでいない: {gaps}"


@pytest.mark.validation
def test_t042_longer_context_eventually_hurts_on_every_corpus():
    """T-042 / 出所=実測。どのコーパスでも、文脈を伸ばしきると最良より悪くなる。

    捨てた主張(「長いほど当たる」)が本当に成り立たないことを、
    出荷している全コーパスで押さえる。**捨てた主張は、捨てたままであることを
    見張っていないと、次に式を触った人が黙って復活させる。**
    """
    path = ROOT / "data" / "measured.json"
    assert path.exists(), "measured.json が無い。python build/measure.py を先に走らせる"
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert rows, "測定結果が空"
    for r in rows:
        curve = [x["bits"] for x in r["by_order"]]
        best = min(range(len(curve)), key=lambda i: curve[i])
        assert best <= 4, f"{r['id']}: 谷が {best} 文字。捨てた主張のほうが正しくなっている"
        assert curve[-1] > curve[best], f"{r['id']}: 伸ばしきっても悪化していない"
        # 文脈をまったく使わない状態よりは、必ず良くなっていること(土台)
        assert curve[best] < curve[0], f"{r['id']}: 文脈を使っても当たるようになっていない"


@pytest.mark.validation
def test_t043_calibration_and_reporting_use_different_texts():
    """T-043 / 出所=G-03。つまみを選んだ本文と、数字を報告する本文が別であること。

    循環の禁止は方針として書くだけでは守られない。**出荷物の上で確かめる。**
    dev と held が同じ作品になっていたら、報告値は「その値が良くなるように
    選んだ結果」になる。
    """
    sel = json.loads((ROOT / "data" / "selection.json").read_text(encoding="utf-8"))
    assert sel, "採録記録が空"
    for c in sel:
        assert c["dev"]["id"] != c["held"]["id"], f"{c['id']}: 調整用と報告用が同じ作品"
        # 学習側にも入っていないこと
        used = {u["id"] for u in c["used"]}
        assert c["dev"]["id"] not in used, f"{c['id']}: 調整用が学習側に入っている"
        assert c["held"]["id"] not in used, f"{c['id']}: 報告用が学習側に入っている"

    for cid in [c["id"] for c in sel]:
        dev, held = M.load_dev(cid), M.load_held(cid)
        assert dev and held and dev[:200] != held[:200], f"{cid}: 2 つの本文が同一"
