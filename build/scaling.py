# -*- coding: utf-8 -*-
"""「読んだ量が増えると、使える文脈の長さはどこまで伸びるか」を測る。

このアプリの主題はここにある。文字 n-gram で文脈を伸ばしていくと、
ある長さから先はかえって当たらなくなる。**その折り返し点は、読んだ量で決まる。**
少ししか読んでいないモデルは短い文脈しか使えず、たくさん読むほど長い文脈が効き始める。

学習本文の先頭 N 字だけを使ったモデルを N を変えて作り、
文脈長ごとのビット数を測って曲線の束にする。評価は held 本文(学習にも
較正にも使っていない)で行う。

usage: python build/scaling.py
出力: data/scaling.json / web/data/scaling.json
"""
from __future__ import annotations

import json
from pathlib import Path

import model as _m
import numpy as np
import prepare as _p

ROOT = Path(__file__).resolve().parent.parent
SIZES = [12_500, 25_000, 50_000, 100_000, 200_000, 400_000]
ORDERS = list(range(0, 7))
CORPUS = "soseki"  # 40 万字あって、上限まで刻める唯一のコーパス
EVAL_CHARS = 8_000  # 曲線 42 本ぶんなので、held 全部は使わず先頭だけ


def main() -> int:
    full = (_m.DATA / f"{CORPUS}.txt").read_text(encoding="utf-8")
    held = _m.load_held(CORPUS)[:EVAL_CHARS]
    rows = []
    for n in SIZES:
        if n > len(full):
            continue
        text = full[:n]
        codes = np.frombuffer(text.encode("utf-16-le"), dtype=np.uint16).astype(np.int64)
        sa = _p.suffix_array(codes)
        curve = [
            _m.Model(text, sa, max_order=L).bits_per_char(held)["bits"] for L in ORDERS
        ]
        best = int(np.argmin(curve))
        rows.append({"chars": n, "curve": [round(x, 4) for x in curve], "best_order": best})
        print(f"{n:8,d}字  " + "  ".join(f"{v:.3f}" for v in curve) + f"   最良 L={best}")

    out = {
        "corpus": CORPUS,
        "eval_chars": len(held),
        "orders": ORDERS,
        "discount": _m.DISCOUNT,
        "rows": rows,
        # 「3 文字が 2 文字を追い抜くまでの差」。山の移動を 1 本の数列で見る
        "gap_3_minus_2": [round(r["curve"][3] - r["curve"][2], 4) for r in rows],
    }
    for target in (ROOT / "data" / "scaling.json", _m.DATA / "scaling.json"):
        target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"L=3 と L=2 の差: {out['gap_3_minus_2']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
