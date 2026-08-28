# -*- coding: utf-8 -*-
"""SPEC の実測表と、画面が読む基準値を作る。

**ここで測った値は SPEC.md に転記するためのものであって、テストの期待値ではない**
(HC-016)。テスト側は「単調である」「端点でない」といった不変量で書き、
ここの数字を定数で写さない。数字はコーパスを差し替えれば動く。

usage: python build/measure.py
出力: data/measured.json / web/data/baseline.json
"""
from __future__ import annotations

import json
from pathlib import Path

import model as _m  # 同ディレクトリ
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    rows = []
    for c in _m.corpora():
        cid = c["id"]
        held = _m.load_held(cid)
        text = (_m.DATA / f"{cid}.txt").read_text(encoding="utf-8")
        sa = np.frombuffer((_m.DATA / f"{cid}.sa.bin").read_bytes(), dtype=np.int32)
        by_order = []
        for L in range(0, 9):
            mdl = _m.Model(text, sa, max_order=L)
            r = mdl.bits_per_char(held)
            by_order.append({"order": L, "bits": round(r["bits"], 4), "ppl": round(r["ppl"], 2)})
            print(f"{cid:8s} L={L} {r['bits']:.4f} bits/字  PPL {r['ppl']:8.2f}")
        mdl8 = _m.Model(text, sa, max_order=8)
        r8 = mdl8.bits_per_char(held)
        rows.append(
            {
                "id": cid,
                "label": c["label"],
                "chars": c["chars"],
                "types": c["types"],
                "held": {"chars": len(held), "oov": r8["oov"]},
                "by_order": by_order,
                "best": min(by_order, key=lambda x: x["bits"]),
            }
        )
        print()

    (ROOT / "data" / "measured.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 画面が「モデルは何ビットで読むか」を出すために読む値。人間の成績と並べる相手。
    baseline = {
        r["id"]: {"bits": r["by_order"][-1]["bits"], "ppl": r["by_order"][-1]["ppl"],
                  "held_chars": r["held"]["chars"], "oov": r["held"]["oov"]}
        for r in rows
    }
    (_m.DATA / "baseline.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"→ {ROOT/'data'/'measured.json'} / {_m.DATA/'baseline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
