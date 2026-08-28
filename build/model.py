# -*- coding: utf-8 -*-
"""文字 n-gram 言語モデルの参照実装(Python)。

出荷するのは `web/js/model.js` の方で、こちらはテストが照合に使う参照実装である
(G-01 の二実装照合)。**片方を見ながらもう片方を書くと同じ誤りが両方に入る**ので、
両者は SPEC §3 の式だけを共通の出どころとし、データ構造は敢えて別々に組む。

## 確率の作り方(SPEC §3)

長さ 0(単字)から長さ L(既定 8)までの文脈を、**絶対割引で下から混ぜ上げる**。

    p_0(c) = (N_c + 1) / (N + V + 1)              … 加算 1 の単字分布(未知字を 1 型として含む)
    p_m(c) = max(k_m(c) - D, 0) / n_m  +  γ_m · p_{m-1}(c)
    γ_m    = D · u_m / n_m

    n_m = 長さ m の文脈の出現回数 / u_m = その直後に来た字の種類数 / D = 0.95(既定)

**見た回数から D を引き、引いた分をまとめて短い文脈に回す。** 1 回しか出ていない文脈は
γ = 0.95、つまり 20 分の 19 を短い文脈に譲る —— たった 1 回の観測は、ほとんど信じない。
D は **dev 本文の上だけで選んだ**(0.1〜0.99 を掃いて平均が最小の 0.95)。
報告用の held 本文で選ぶと較正が循環する(G-03)。
D < 1 かつ k ≥ 1 なので max(·, 0) は常に正で、Σ_c p_m(c) = 1 が場合分けなしに成り立つ。

**λ = n/(n+β) 型の混合にしない理由は実測である。** β=8 で測ると、
検証本文のビット数が文脈長 3 以上で単調に悪化した(4 コーパスとも最良が L=2)。
3 回の観測に 27% の重みが乗るためで、「長い文脈ほど当たる」という
このアプリの主張と実装が逆を向いていた。詳細は HARNESS_CHANGELOG HC-001。

**打ち切り(hard backoff)ではなく混合にしてある理由**は、打ち切りだと選ばれた文脈で
一度も見ていない字の確率がちょうど 0 になり、交差エントロピーが無限大に飛ぶこと。
「その字は絶対に来ない」と言い切るモデルは、次の一字を当てる相手として不適格である。

## 数え方

長さ 1・2 の文脈は本文を 1 回走査して表に持つ。長さ 3 以上は接尾辞配列の二分探索で
その場で数える。**同じ量に 2 通りの数え方がある**ので、長さ 1・2 については
両者の一致をテストで突き合わせられる(T-011)。表と探索は独立な機構なので同義反復にならない。
"""
from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "web" / "data"

MAX_ORDER = 8       # 混ぜ上げる文脈の最大長 L
DISCOUNT = 0.95     # 絶対割引 D。dev 本文で選定(build/calibrate.py)。0 < D < 1 が式の前提
UNK = "�"      # 語彙に無い字をまとめる 1 記号(交差エントロピーを有限に保つため)


class Model:
    def __init__(self, text: str, sa: np.ndarray, max_order: int = MAX_ORDER, disc: float = DISCOUNT):
        self.text = text
        self.sa = sa
        self.max_order = max_order
        assert 0.0 < disc < 1.0, "D は 0 < D < 1 でなければならない"
        self.disc = disc

        self.uni = Counter(text)
        self.n = len(text)
        self.v = len(self.uni)

        # 長さ 1・2 の文脈は表で持つ(走査 1 回)
        # 数え直しを避けるための覚え書き。**数え方そのものは変えない**ので、
        # 二経路一致(T-012 / T-013)の意味は保たれる
        self._occ_cache: dict[str, int] = {}
        self._u_cache: dict[str, int] = {}

        self.tab: list[dict[str, Counter]] = [defaultdict(Counter), defaultdict(Counter)]
        for i in range(self.n - 1):
            self.tab[0][text[i]][text[i + 1]] += 1
        for i in range(self.n - 2):
            self.tab[1][text[i : i + 2]][text[i + 2]] += 1

    # ---- 接尾辞配列による出現範囲 -------------------------------------------------

    def occ_range(self, ctx: str) -> tuple[int, int]:
        """本文中で ctx が始まる位置の、接尾辞配列上の範囲 [lo, hi) を返す。"""
        text, sa, m = self.text, self.sa, len(ctx)
        key = lambda i: text[int(sa[i]) : int(sa[i]) + m]  # noqa: E731
        n = len(sa)
        lo = bisect_left(range(n), ctx, key=key)
        hi = bisect_right(range(n), ctx, key=key)
        return lo, hi

    def next_counts(self, ctx: str) -> Counter:
        """ctx の直後に来た字の出現回数。ctx が本文末尾で終わる出現は数に入らない。"""
        if len(ctx) == 0:
            return Counter(self.uni)
        if len(ctx) <= 2:
            return Counter(self.tab[len(ctx) - 1].get(ctx, Counter()))
        lo, hi = self.occ_range(ctx)
        out: Counter = Counter()
        m = len(ctx)
        for i in range(lo, hi):
            j = int(self.sa[i]) + m
            if j < self.n:
                out[self.text[j]] += 1
        return out

    # ---- 分布 -------------------------------------------------------------------

    def unigram(self) -> dict[str, float]:
        denom = self.n + self.v + 1
        p = {c: (k + 1) / denom for c, k in self.uni.items()}
        p[UNK] = 1 / denom
        return p

    def distribution(self, ctx: str) -> tuple[dict[str, float], list[dict]]:
        """文脈 ctx に続く字の分布と、各段の寄与の内訳を返す。

        内訳は画面の「どの長さの文脈が効いているか」の帯にそのまま使う。
        帯の数字を別途作らないので、図と文が食い違わない(HC-045)。
        """
        d = self.disc
        p = self.unigram()
        trace = [{"order": 0, "n": self.n, "u": self.v, "gamma": 1.0, "ctx": ""}]
        for m in range(1, self.max_order + 1):
            if m > len(ctx):
                break
            sub = ctx[-m:]
            counts = self.next_counts(sub)
            nm = sum(counts.values())
            if nm == 0:
                trace.append({"order": m, "n": 0, "u": 0, "gamma": 1.0, "ctx": sub})
                # 長さ m で 0 回なら、それより長い文脈も必ず 0 回
                break
            um = len(counts)
            gamma = d * um / nm
            p = {c: gamma * q for c, q in p.items()}
            for c, k in counts.items():
                p[c] = p.get(c, 0.0) + max(k - d, 0.0) / nm
            trace.append({"order": m, "n": nm, "u": um, "gamma": gamma, "ctx": sub})

        # 各段が最終確率にどれだけ寄与したか。γ の積で下へ流れる分が決まるので、
        # 上の段から順に (これまでの γ の積) × (その段が自分で持つ分) を配る。
        # 画面の帯はこの share だけから描く(帯の数字を別に作らない — HC-045)
        flow = 1.0
        for t in reversed(trace[1:]):
            t["share"] = flow * (1.0 - t["gamma"])
            flow *= t["gamma"]
        trace[0]["share"] = flow
        return p, trace

    def occ(self, s: str) -> int:
        hit = self._occ_cache.get(s)
        if hit is None:
            lo, hi = self.occ_range(s)
            hit = hi - lo
            self._occ_cache[s] = hit
        return hit

    def distinct_followers(self, ctx: str) -> int:
        """ctx の直後に来た字の種類数を、**範囲を舐めずに**数える。

        接尾辞配列上で ctx の範囲は「次の字」で整列しているので、
        先頭の字を見て ctx+その字 の上端へ跳ぶ、を繰り返せば種類数だけ数えられる。
        next_counts() の走査とは機構が違うので、両者の一致が検算になる(T-013)。
        """
        hit = self._u_cache.get(ctx)
        if hit is not None:
            return hit
        lo, hi = self.occ_range(ctx)
        m, u = len(ctx), 0
        while lo < hi:
            j = int(self.sa[lo]) + m
            if j >= self.n:  # 本文末尾で終わる出現。次の字が無いので数えない
                lo += 1
                continue
            _, nxt = self.occ_range(ctx + self.text[j])
            u += 1
            lo = nxt
        self._u_cache[ctx] = u
        return u

    def prob_of(self, ctx: str, ch: str) -> float:
        """1 字ぶんの確率だけを、分布を作らずに求める。

        distribution() が全字ぶんの表を作るのに対し、こちらは各段で
        n_m・u_m・k_m を二分探索だけで取る(範囲を舐めない)。
        **同じ量に対する独立な数え方**なので、
        distribution()[ch] との一致がそのまま検算になる(T-012)。
        """
        d = self.disc
        denom = self.n + self.v + 1
        p = (self.uni.get(ch, 0) + 1) / denom if ch != UNK else 1 / denom
        for m in range(1, self.max_order + 1):
            if m > len(ctx):
                break
            sub = ctx[-m:]
            nm = self.occ(sub) - (1 if self.text.endswith(sub) else 0)
            if nm <= 0:
                break
            # 長さ 1・2 は表から取る。跳び歩きは種類数に比例するので、
            # 「の」のように 2 千種類を連れてくる短い文脈では割に合わない。
            # 表と跳び歩きが一致することは T-013 で別に見る
            um = len(self.tab[m - 1][sub]) if m <= 2 else self.distinct_followers(sub)
            k = 0 if ch == UNK else self.occ(sub + ch)
            p = max(k - d, 0.0) / nm + (d * um / nm) * p
        return p

    # ---- 交差エントロピー ---------------------------------------------------------

    def bits_per_char(self, held: str) -> dict:
        """検証用本文 1 文字あたりのビット数。語彙に無い字は UNK 1 記号に寄せる。"""
        total, oov = 0.0, 0
        for i in range(len(held)):
            ctx = held[max(0, i - self.max_order) : i]
            ch = held[i]
            if ch not in self.uni:
                ch, oov = UNK, oov + 1
            total += -math.log2(max(self.prob_of(ctx, ch), 1e-300))
        return {
            "chars": len(held),
            "bits": total / len(held),
            "ppl": 2 ** (total / len(held)),
            "oov": oov,
        }


# ---- 分布の加工(温度・top-k・top-p)------------------------------------------------
# ここは LLM の「出口」そのもの。式は SPEC §4 に置き、両実装が同じ式を見る。


def temperature(p: dict[str, float], t: float) -> dict[str, float]:
    """p_i ∝ p_i^(1/T)。T→0 で最頻値に集中し、T→∞ で一様に近づく。

    T が小さいと p^(1/T) が桁溢れするので、対数で引いてから戻す。
    """
    if t <= 0:
        best = max(p.values())
        top = [c for c, q in p.items() if q == best]
        return {c: (1.0 / len(top) if c in top else 0.0) for c in p}
    logs = {c: math.log(max(q, 1e-300)) / t for c, q in p.items()}
    mx = max(logs.values())
    ex = {c: math.exp(v - mx) for c, v in logs.items()}
    s = sum(ex.values())
    return {c: v / s for c, v in ex.items()}


def top_k(p: dict[str, float], k: int) -> dict[str, float]:
    """確率上位 k 個だけを残して正規化する。k <= 0 は「制限なし」。"""
    if k <= 0 or k >= len(p):
        return dict(p)
    keep = [c for c, _ in sorted(p.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]
    s = sum(p[c] for c in keep)
    return {c: p[c] / s for c in keep}


def top_p(p: dict[str, float], q: float) -> dict[str, float]:
    """累積確率が q を初めて超えるところまでを残して正規化する。

    「超えるところまで」を含めるので、残る集合は必ず 1 個以上になる。
    q >= 1 は「制限なし」。
    """
    if q >= 1.0:
        return dict(p)
    order = sorted(p.items(), key=lambda kv: (-kv[1], kv[0]))
    keep, acc = [], 0.0
    for c, v in order:
        keep.append(c)
        acc += v
        if acc >= q:
            break
    s = sum(p[c] for c in keep)
    return {c: p[c] / s for c in keep}


def entropy(p: dict[str, float]) -> float:
    """シャノンエントロピー(ビット)。"""
    return -sum(v * math.log2(v) for v in p.values() if v > 0)


# ---- 疑似乱数(JS と同一の出力)--------------------------------------------------


def mulberry32(seed: int):
    """mulberry32。JS の Math.imul / >>> と同じ結果を出すよう 32 ビットで畳む。

    サンプリングの再現性は「同じ種なら同じ文章」の検査(T-031)に使う。
    言語をまたいで一致することが要件なので、演算幅を明示して書く。
    """
    state = seed & 0xFFFFFFFF

    def imul(a: int, b: int) -> int:
        r = (a & 0xFFFFFFFF) * (b & 0xFFFFFFFF) & 0xFFFFFFFF
        return r

    def nxt() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        t = imul(t ^ (t >> 15), 1 | t)
        t = (t + imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF ^ t
        t &= 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return nxt


def sample(p: dict[str, float], rnd) -> str:
    """字の昇順に並べた上での逆関数法。並べ方を決めておかないと両実装で結果がずれる。"""
    r = rnd()
    acc = 0.0
    items = sorted(p.items())
    for c, v in items:
        acc += v
        if r < acc:
            return c
    return items[-1][0]


# ---- 読み込み ---------------------------------------------------------------------


def load(cid: str, max_order: int = MAX_ORDER, disc: float = DISCOUNT) -> Model:
    text = (DATA / f"{cid}.txt").read_text(encoding="utf-8")
    sa = np.frombuffer((DATA / f"{cid}.sa.bin").read_bytes(), dtype=np.int32)
    return Model(text, sa, max_order, disc)


def load_held(cid: str) -> str:
    return (DATA / f"{cid}.held.txt").read_text(encoding="utf-8")


def corpora() -> list[dict]:
    return json.loads((DATA / "corpora.json").read_text(encoding="utf-8"))["corpora"]


# ---- 推測回数からのエントロピー上下界(Shannon 1951)---------------------------------


def shannon_bounds(q: list[float]) -> tuple[float, float]:
    """「何回目の推測で当たったか」の分布 q から、情報源のエントロピーの上下界を出す。

    q[i-1] = i 回目の推測で当たった割合(合計 1)。Shannon が 1951 年に
    英文で 1 文字 ≒ 1.3 ビットを出したときの道具立てで、
    **予測器の中身を知らなくても外側から測れる**のがこの式の値打ちである。
    人間の相手には確率を聞けないので、当てっこの成績からこの形で測る。

        下界 = Σ_i i · (q_i − q_{i+1}) · log2 i
        上界 = − Σ_i q_i log2 q_i

    式の妥当性は T-050 で確かめる: エントロピーが厳密に分かっている分布を
    無作為に 2,000 個作り、順位ごとの確率を q として上下界が実際に挟むかを見る。
    公表値の暗記ではなく、手元で成り立つことを見てから使う。
    """
    n = len(q)
    lo = 0.0
    for i in range(1, n + 1):
        nxt = q[i] if i < n else 0.0
        lo += i * (q[i - 1] - nxt) * math.log2(i)
    up = -sum(x * math.log2(x) for x in q if x > 0)
    return lo, up


def guess_ranks(p: dict[str, float], truth: str) -> int:
    """理想的な予測器(確率の高い順に言う)が truth を当てるまでの推測回数。"""
    order = [c for c, _ in sorted(p.items(), key=lambda kv: (-kv[1], kv[0]))]
    return order.index(truth) + 1


def load_dev(cid: str) -> str:
    return (DATA / f"{cid}.dev.txt").read_text(encoding="utf-8")


def is_non_increasing(q: list[float]) -> bool:
    """q が非増加か。**下界の式の前提**(確率の高い順に当てにいっていること)。

    人間の推測順はこれを満たさない。満たさない q に下界を当てると
    下界が上界を上回る(HC-003)。上界のほうは、推測回数の列から本文を
    復元できることだけから出るので、どんな予測器でも成り立つ。
    """
    return all(b <= a + 1e-12 for a, b in zip(q, q[1:]))
