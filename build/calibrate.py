# -*- coding: utf-8 -*-
"""割引 D を選ぶ。**選ぶのは dev 本文の上だけで行う**(G-03 循環の禁止)。

報告用の held 本文で D を選ぶと、報告する数字が「その数字が良くなるように選んだ結果」
になる。較正と評価に同じ本文を使わない、はこのフリートで何度も踏んだ型なので、
選定は dev、報告は held、と場所で分ける。

D は 1 つだけ選び、全コーパスで共有する。コーパスごとに最良の D を選ぶと、
「コーパスを切り替えると数字が動く」のがモデルの性質なのか調整の結果なのか
分からなくなる。

usage: python build/calibrate.py
出力: data/calibration.json
"""
from __future__ import annotations

import json
from pathlib import Path

import model as _m
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GRID = [0.1, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
ORDERS = list(range(0, 9))


def main() -> int:
    corpora = _m.corpora()
    loaded = []
    for c in corpora:
        text = (_m.DATA / f"{c['id']}.txt").read_text(encoding="utf-8")
        sa = np.frombuffer((_m.DATA / f"{c['id']}.sa.bin").read_bytes(), dtype=np.int32)
        loaded.append((c["id"], text, sa, _m.load_dev(c["id"])))

    rows = []
    for d in GRID:
        per = {}
        for cid, text, sa, dev in loaded:
            mdl = _m.Model(text, sa, max_order=_m.MAX_ORDER, disc=d)
            per[cid] = mdl.bits_per_char(dev)["bits"]
        mean = sum(per.values()) / len(per)
        rows.append({"disc": d, "per_corpus": per, "mean_bits": mean})
        print(f"D={d:<5} 平均 {mean:.4f} bits  " + "  ".join(f"{k}:{v:.3f}" for k, v in per.items()))

    best = min(rows, key=lambda r: r["mean_bits"])
    print(f"\n選定: D = {best['disc']}(dev 平均 {best['mean_bits']:.4f} bits)")

    # 山の位置(最良の文脈長)も dev の上で見ておく。報告は held でやり直す
    peaks = {}
    for cid, text, sa, dev in loaded:
        curve = [
            _m.Model(text, sa, max_order=L, disc=best["disc"]).bits_per_char(dev)["bits"]
            for L in ORDERS
        ]
        peaks[cid] = {"curve": [round(x, 4) for x in curve], "best_order": int(np.argmin(curve))}
        print(f"  {cid:8s} 最良の文脈長 L={peaks[cid]['best_order']}")

    (ROOT / "data" / "calibration.json").write_text(
        json.dumps(
            {"grid": rows, "chosen": best["disc"], "dev_peaks": peaks, "orders": ORDERS},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"→ {ROOT/'data'/'calibration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
