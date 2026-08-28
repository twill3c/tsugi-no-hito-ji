# -*- coding: utf-8 -*-
"""共通フィクスチャ。

出荷実装(JS)の出力は `node build/js_dump.mjs` を 1 回だけ走らせて取る。
**pytest 側で JS の計算をやり直さない** — やり直した時点で照合相手が
出荷実装ではなくテストになる。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build"))

import model as M  # noqa: E402

CORPUS = "soseki"  # 照合の主コーパス。他コーパスは T-002 が形だけ見る


@pytest.fixture(scope="session")
def mdl() -> M.Model:
    return M.load(CORPUS)


@pytest.fixture(scope="session")
def held() -> str:
    return M.load_held(CORPUS)


@pytest.fixture(scope="session")
def js() -> dict:
    out = subprocess.run(
        ["node", str(ROOT / "build" / "js_dump.mjs"), CORPUS],
        capture_output=True,
        cwd=ROOT,
    )
    if out.returncode != 0:
        pytest.fail(f"js_dump.mjs が落ちた:\n{out.stderr.decode('utf-8', 'replace')}")
    return json.loads(out.stdout.decode("utf-8"))


@pytest.fixture(scope="session")
def tiny() -> M.Model:
    """総当たりと突き合わせるための小さなモデル。

    本文が短いので、接尾辞配列も出現回数も紙の上で数え直せる。
    実コーパスだけで検算すると、遅くて総当たりが書けないところが検査から漏れる。
    """
    import numpy as np

    text = "あいうえおあいうあいあ\nあいうえお\n"
    sa = _brute_sa(text)
    return M.Model(text, np.array(sa, dtype=np.int32), max_order=4)


def _brute_sa(text: str) -> list[int]:
    """総当たりの接尾辞配列(参照)。テスト側にしか置かない。"""
    return sorted(range(len(text)), key=lambda i: text[i:])
