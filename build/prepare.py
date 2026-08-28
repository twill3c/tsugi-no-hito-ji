# -*- coding: utf-8 -*-
"""青空文庫の正規化済み本文から、出荷用のコーパスと接尾辞配列を作る。

出荷物は 1 コーパスにつき 2 つ。

    web/data/<id>.txt      本文(UTF-8)
    web/data/<id>.sa.bin   接尾辞配列(Int32 リトルエンディアン, 要素数 = 文字数 + 1)

**モデルファイルは出荷しない。** n-gram の確率表を焼き込む代わりに本文そのものを配り、
接尾辞配列の二分探索でその場で数える。こうすると次の 3 つが同時に手に入る:

  1. n の上限が無い(文脈が何文字でも、出現していれば数えられる)
  2. 出荷物と、テストが読む本文が同一である(モデル化の途中で嘘が入る隙が無い)
  3. 「モデルは読んだものでできている」を、コーパスを切り替えて体感できる

添字は **UTF-16 コード単位ではなく Python の文字単位**で振る。両者を一致させるため、
BMP 外の文字(JS では 2 個の符号単位になる文字)は採録時に落とし、残っていないことを
検算する。これをしないと Python 側の添字と JS 側の `charCodeAt` の添字がずれ、
二実装照合(G-01)が意味を失う。

usage: python build/prepare.py [--source <aozora-sakuin/data>] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "data"
DEFAULT_SOURCE = ROOT.parent / "aozora-sakuin" / "data"

# 出荷する 1 コーパスあたりの上限文字数。接尾辞配列は 1 文字あたり 4 バイトなので、
# 40 万字 = 本文 1.2MB(UTF-8)+ 配列 1.6MB。初回表示で読むのは既定コーパス 1 つだけ。
LIMIT = 400_000

# 検証用(学習に使わない)本文の上限。ビット数の推定はこの上で行う。
HELD_LIMIT = 20_000

# 採録は新字新仮名に限る。旧仮名を混ぜると、モデルが覚えた綴りと
# 読み手が期待する綴りがずれ、「人間 対 モデル」の勝負が仮名遣いの勝負になる。
KANA = "新字新仮名"

CORPORA = [
    {"id": "soseki", "label": "夏目漱石", "authors": ["夏目漱石"]},
    {"id": "dazai", "label": "太宰治", "authors": ["太宰治"]},
    {"id": "kenji", "label": "宮沢賢治", "authors": ["宮沢賢治"]},
    {"id": "mixed", "label": "近代小説つめあわせ", "authors": None},
]


def load_works(source: Path) -> list[dict]:
    meta = json.loads((source / "works.json").read_text(encoding="utf-8"))
    return meta["works"]


def clean(text: str) -> str:
    """出荷用の最小限の整形。

    正規化済み本文は既にルビ・注記・見出し・底本表記が落ちている。ここでは
    **表記を書き換えない**(NFKC もかけない)。落とすのは次の 2 つだけ:

      - BMP 外の文字(JS との添字一致のため)
      - 3 つ以上続く改行(段落の切れ目としては 1 つで足りる)

    改行そのものは残す。段落の終わりは文字 n-gram にとって実在の手がかりであり、
    落とすとモデルが「文が終わらない」ようになる。画面では ⏎ として見せる。
    """
    kept = []
    for ch in text:
        if ord(ch) > 0xFFFF:
            continue
        # 制御文字は改行だけ通す
        if ch != "\n" and unicodedata.category(ch) == "Cc":
            continue
        kept.append(ch)
    out = "".join(kept)
    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")
    return out.strip("\n") + "\n"


def pick(works: list[dict], authors: list[str] | None, exclude: set[str]) -> list[dict]:
    """採録候補を著者で選び、著者ごとに順番に取ることで偏りを抑える。

    素直に文字数順に取ると、長編 1 作でコーパスが埋まる。つめあわせ側は
    「近代小説一般の字並び」を代表してほしいので、著者を回しながら取る。
    """
    pool = [w for w in works if w["kana"] == KANA]
    if authors is not None:
        pool = [w for w in pool if w["author"] in authors]
    else:
        pool = [w for w in pool if w["author"] not in exclude]
    by_author: dict[str, list[dict]] = defaultdict(list)
    for w in sorted(pool, key=lambda w: w["id"]):
        by_author[w["author"]].append(w)
    order = sorted(by_author)
    picked, i = [], 0
    while True:
        added = False
        for a in order:
            if i < len(by_author[a]):
                picked.append(by_author[a][i])
                added = True
        if not added:
            break
        i += 1
    return picked


def suffix_array(codes: np.ndarray) -> np.ndarray:
    """接尾辞配列を接頭辞倍化で作る。

    終端に本文に現れない番兵(-1)を置き、全接尾辞が相異なるようにしてから
    順位が全て異なるまで倍化する。番兵があるので必ず停止する。
    """
    n = len(codes)
    _, rank = np.unique(codes, return_inverse=True)
    rank = rank.astype(np.int64)
    k = 1
    while True:
        second = np.full(n, -1, dtype=np.int64)
        if k < n:
            second[: n - k] = rank[k:]
        order = np.lexsort((second, rank))
        rs, ss = rank[order], second[order]
        new = np.zeros(n, dtype=np.int64)
        new[1:] = np.cumsum((rs[1:] != rs[:-1]) | (ss[1:] != ss[:-1]))
        rank_new = np.empty(n, dtype=np.int64)
        rank_new[order] = new
        rank = rank_new
        if int(rank.max()) == n - 1:
            break
        k *= 2
    return np.argsort(rank, kind="stable").astype(np.int32)


def check_sa(text: str, sa: np.ndarray, samples: int = 400) -> None:
    """出荷前の検算。ここが緑でないものは配らない。

    全順序の総当たり確認は O(n^2) になるので、(a) 置換であること(全域)と
    (b) 隣接接尾辞の順序(標本)に分ける。(b) は等間隔標本なので、
    先頭だけ・末尾だけを見て通ることがない。
    """
    n = len(text)
    assert len(sa) == n, f"接尾辞配列の長さが本文と違う: {len(sa)} != {n}"
    assert np.array_equal(np.sort(sa), np.arange(n, dtype=np.int32)), "置換になっていない"
    step = max(1, n // samples)
    for i in range(0, n - 1, step):
        a, b = int(sa[i]), int(sa[i + 1])
        assert text[a:] < text[b:], f"順序が壊れている: 行 {i}"


def build_one(spec: dict, works: list[dict], source: Path, exclude: set[str]) -> dict:
    candidates = pick(works, spec["authors"], exclude)

    # 採録の前に 2 作抜く。学習に使った本文でビット数を測ると必ず良く出る。
    #   dev  … つまみ(割引 D など)を選ぶための本文
    #   held … 最後に報告する数字を測るための本文
    # **同じ本文で選んで同じ本文で報告すると、較正が循環する**(G-03)。
    # 抜くのは候補の最後の 2 作。順序は id の昇順と著者の巡回で決まるので、
    # 走らせ直しても同じ作品が抜ける
    held, dev = candidates[-1], candidates[-2]
    picked = candidates[:-2]

    chunks, used, total = [], [], 0
    for w in picked:
        if total >= LIMIT:
            break
        body = clean((source / "normalized" / f"{w['id']}.txt").read_text(encoding="utf-8"))
        room = LIMIT - total
        if len(body) > room:
            # 途中で切るときは改行で切る。文の途中で切ると、末尾に
            # 本文には存在しない「文脈」が生まれる
            cut = body.rfind("\n", 0, room)
            body = body[: cut + 1] if cut > 0 else ""
            if not body:
                break
        chunks.append(body)
        used.append({"id": w["id"], "title": w["title"], "author": w["author"], "chars": len(body)})
        total += len(body)

    text = "".join(chunks)
    assert all(ord(c) <= 0xFFFF for c in text), "BMP 外の文字が残っている"

    codes = np.frombuffer(text.encode("utf-16-le"), dtype=np.uint16).astype(np.int64)
    assert len(codes) == len(text), "UTF-16 コード単位と文字数が一致しない(代用対が残っている)"
    sa = suffix_array(codes)
    check_sa(text, sa)

    def reserve(w: dict) -> str:
        body = clean((source / "normalized" / f"{w['id']}.txt").read_text(encoding="utf-8"))[:HELD_LIMIT]
        assert body and text.find(body[:200]) < 0, f"{w['id']} が学習側に混ざっている"
        return body

    held_text, dev_text = reserve(held), reserve(dev)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{spec['id']}.txt").write_text(text, encoding="utf-8", newline="")
    (OUT / f"{spec['id']}.sa.bin").write_bytes(sa.tobytes())
    (OUT / f"{spec['id']}.held.txt").write_text(held_text, encoding="utf-8", newline="")
    (OUT / f"{spec['id']}.dev.txt").write_text(dev_text, encoding="utf-8", newline="")

    freq = Counter(text)
    return {
        "id": spec["id"],
        "label": spec["label"],
        "chars": len(text),
        "types": len(freq),
        "works": len(used),
        "authors": sorted({u["author"] for u in used}),
        "bytes_text": (OUT / f"{spec['id']}.txt").stat().st_size,
        "bytes_sa": (OUT / f"{spec['id']}.sa.bin").stat().st_size,
        "held": {
            "id": held["id"],
            "title": held["title"],
            "author": held["author"],
            "chars": len(held_text),
        },
        "dev": {
            "id": dev["id"],
            "title": dev["title"],
            "author": dev["author"],
            "chars": len(dev_text),
        },
        "top": [[c, n] for c, n in freq.most_common(10)],
        "used": used,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.limit:
        global LIMIT
        LIMIT = args.limit

    if not (args.source / "works.json").exists():
        print(f"元データが見つからない: {args.source}", file=sys.stderr)
        return 1

    works = load_works(args.source)
    named = {a for c in CORPORA if c["authors"] for a in c["authors"]}
    report = []
    for spec in CORPORA:
        info = build_one(spec, works, args.source, named)
        report.append(info)
        print(
            f"{info['id']:8s} {info['label']:12s} {info['chars']:8,d}字 "
            f"{info['types']:5,d}種 {info['works']:3d}作 "
            f"本文{info['bytes_text']/1e6:.2f}MB 配列{info['bytes_sa']/1e6:.2f}MB"
        )

    manifest = {
        "limit": LIMIT,
        "kana": KANA,
        "corpora": [{k: v for k, v in c.items() if k != "used"} for c in report],
    }
    (OUT / "corpora.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "data" / "selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"→ {OUT/'corpora.json'} / {ROOT/'data'/'selection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
