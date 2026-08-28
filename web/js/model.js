// 文字 n-gram 言語モデル(出荷実装)。
//
// 参照実装は build/model.py にある。**あちらを見ながらこちらを書かない**こと。
// 両方が同じ誤りを持つと二実装照合(G-01)が何も捕まえなくなる。
// 共通の出どころは SPEC.md §3・§4 の式だけである。
//
// 確率は「長さ 0 の単字分布から、長さ L まで絶対割引で混ぜ上げる」形で作る:
//   p_0(c) = (N_c + 1) / (N + V + 1)
//   p_m(c) = max(k_m(c) - D, 0) / n_m + γ_m · p_{m-1}(c),  γ_m = D · u_m / n_m
//   n_m = 文脈の出現回数 / u_m = 直後に来た字の種類数 / D = 0.95(dev 本文で選定)
//
// 見た回数から D を引き、引いた分を短い文脈に回す。λ=n/(n+β) 型にしない理由は
// 実測(文脈長 3 以上でビット数が悪化した)。HARNESS_CHANGELOG HC-001 を見よ。
//
// 長さ 1・2 は本文の 1 回走査で作る表から、長さ 3 以上は接尾辞配列の二分探索から数える。
// 同じ量に 2 通りの数え方があるので、表と探索の一致がそのまま検算になる。

export const MAX_ORDER = 8;
export const DISCOUNT = 0.95;
export const UNK = "�"; // 語彙に無い字をまとめる 1 記号

export class Model {
  constructor(text, sa, { maxOrder = MAX_ORDER, disc = DISCOUNT } = {}) {
    this.text = text;
    this.sa = sa;
    this.maxOrder = maxOrder;
    if (!(disc > 0 && disc < 1)) throw new Error("D は 0 < D < 1 でなければならない");
    this.disc = disc;
    this.n = text.length;

    this.uni = new Map();
    for (let i = 0; i < this.n; i++) {
      const c = text[i];
      this.uni.set(c, (this.uni.get(c) || 0) + 1);
    }
    this.v = this.uni.size;

    // 数え直しを避けるための覚え書き。**数え方そのものは変えない**ので、
    // 二経路一致(T-012 / T-013)の意味は保たれる
    this._occ = new Map();
    this._u = new Map();

    // 長さ 1・2 の文脈表。走査 1 回ぶんの費用で、よく引かれる短い文脈を定数時間にする
    this.tab = [new Map(), new Map()];
    for (let i = 0; i + 1 < this.n; i++) bump(this.tab[0], text[i], text[i + 1]);
    for (let i = 0; i + 2 < this.n; i++) bump(this.tab[1], text.slice(i, i + 2), text[i + 2]);
  }

  // 本文中で ctx が始まる位置の、接尾辞配列上の範囲 [lo, hi)
  occRange(ctx) {
    const m = ctx.length;
    if (m === 0) return [0, this.n];
    const { text, sa, n } = this;
    // 下端: ctx 未満が終わる位置
    let lo = 0, hi = n;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (text.substr(sa[mid], m) < ctx) lo = mid + 1; else hi = mid;
    }
    const start = lo;
    hi = n;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (text.substr(sa[mid], m) <= ctx) lo = mid + 1; else hi = mid;
    }
    return [start, lo];
  }

  occ(s) {
    let hit = this._occ.get(s);
    if (hit === undefined) {
      const [lo, hi] = this.occRange(s);
      hit = hi - lo;
      this._occ.set(s, hit);
    }
    return hit;
  }

  // ctx の直後に来た字の出現回数。本文末尾で終わる出現は数に入らない
  nextCounts(ctx) {
    if (ctx.length === 0) return new Map(this.uni);
    if (ctx.length <= 2) {
      const hit = this.tab[ctx.length - 1].get(ctx);
      return hit ? new Map(hit) : new Map();
    }
    const [lo, hi] = this.occRange(ctx);
    const out = new Map();
    const m = ctx.length;
    for (let i = lo; i < hi; i++) {
      const j = this.sa[i] + m;
      if (j < this.n) {
        const c = this.text[j];
        out.set(c, (out.get(c) || 0) + 1);
      }
    }
    return out;
  }

  unigram() {
    const denom = this.n + this.v + 1;
    const p = new Map();
    for (const [c, k] of this.uni) p.set(c, (k + 1) / denom);
    p.set(UNK, 1 / denom);
    return p;
  }

  // 分布と、各段の寄与の内訳。画面の帯はこの trace だけから描く(数字を別に作らない)
  distribution(ctx) {
    const d = this.disc;
    let p = this.unigram();
    const trace = [{ order: 0, n: this.n, u: this.v, gamma: 1, ctx: "" }];
    for (let m = 1; m <= this.maxOrder && m <= ctx.length; m++) {
      const sub = ctx.slice(ctx.length - m);
      const counts = this.nextCounts(sub);
      let nm = 0;
      for (const k of counts.values()) nm += k;
      if (nm === 0) {
        trace.push({ order: m, n: 0, u: 0, gamma: 1, ctx: sub });
        break; // 長さ m で 0 回なら、それより長い文脈も必ず 0 回
      }
      const um = counts.size;
      const gamma = (d * um) / nm;
      const q = new Map();
      for (const [c, val] of p) q.set(c, gamma * val);
      for (const [c, k] of counts) q.set(c, (q.get(c) || 0) + Math.max(k - d, 0) / nm);
      p = q;
      trace.push({ order: m, n: nm, u: um, gamma, ctx: sub });
    }

    // 各段が最終確率にどれだけ寄与したか。γ の積で下へ流れる分が決まる
    let flow = 1;
    for (let i = trace.length - 1; i >= 1; i--) {
      trace[i].share = flow * (1 - trace[i].gamma);
      flow *= trace[i].gamma;
    }
    trace[0].share = flow;
    return { p, trace };
  }

  // ctx の直後に来た字の種類数を、範囲を舐めずに跳び歩きで数える
  distinctFollowers(ctx) {
    const hit = this._u.get(ctx);
    if (hit !== undefined) return hit;
    let [lo, hi] = this.occRange(ctx);
    const m = ctx.length;
    let u = 0;
    while (lo < hi) {
      const j = this.sa[lo] + m;
      if (j >= this.n) { lo++; continue; } // 本文末尾で終わる出現には次の字が無い
      lo = this.occRange(ctx + this.text[j])[1];
      u++;
    }
    this._u.set(ctx, u);
    return u;
  }

  // 分布を作らずに 1 字ぶんの確率だけを求める(独立な数え方)
  probOf(ctx, ch) {
    const d = this.disc;
    const denom = this.n + this.v + 1;
    let p = ch === UNK ? 1 / denom : ((this.uni.get(ch) || 0) + 1) / denom;
    for (let m = 1; m <= this.maxOrder && m <= ctx.length; m++) {
      const sub = ctx.slice(ctx.length - m);
      const nm = this.occ(sub) - (this.text.endsWith(sub) ? 1 : 0);
      if (nm <= 0) break;
      // 長さ 1・2 は表から。跳び歩きは種類数に比例するので短い文脈では割に合わない
      const um = m <= 2 ? (this.tab[m - 1].get(sub)?.size ?? 0) : this.distinctFollowers(sub);
      const k = ch === UNK ? 0 : this.occ(sub + ch);
      p = Math.max(k - d, 0) / nm + ((d * um) / nm) * p;
    }
    return p;
  }

  // 検証用本文 1 文字あたりのビット数
  bitsPerChar(held, onProgress) {
    let total = 0, oov = 0;
    for (let i = 0; i < held.length; i++) {
      const ctx = held.slice(Math.max(0, i - this.maxOrder), i);
      let ch = held[i];
      if (!this.uni.has(ch)) { ch = UNK; oov++; }
      total += -Math.log2(Math.max(this.probOf(ctx, ch), 1e-300));
      if (onProgress && (i & 1023) === 0) onProgress(i / held.length);
    }
    return { chars: held.length, bits: total / held.length, ppl: 2 ** (total / held.length), oov };
  }
}

function bump(tab, key, ch) {
  let m = tab.get(key);
  if (!m) { m = new Map(); tab.set(key, m); }
  m.set(ch, (m.get(ch) || 0) + 1);
}

// ---- 分布の加工(温度・top-k・top-p)-------------------------------------------

// p_i ∝ p_i^(1/T)。T が小さいと桁溢れするので対数側で引いてから戻す
export function temperature(p, t) {
  if (t <= 0) {
    let best = -Infinity;
    for (const v of p.values()) if (v > best) best = v;
    const top = [...p].filter(([, v]) => v === best).map(([c]) => c);
    const out = new Map();
    for (const c of p.keys()) out.set(c, top.includes(c) ? 1 / top.length : 0);
    return out;
  }
  const logs = new Map();
  let mx = -Infinity;
  for (const [c, v] of p) {
    const l = Math.log(Math.max(v, 1e-300)) / t;
    logs.set(c, l);
    if (l > mx) mx = l;
  }
  let s = 0;
  const ex = new Map();
  for (const [c, l] of logs) { const e = Math.exp(l - mx); ex.set(c, e); s += e; }
  const out = new Map();
  for (const [c, e] of ex) out.set(c, e / s);
  return out;
}

// 確率上位 k 個だけを残して正規化する。k <= 0 は「制限なし」
export function topK(p, k) {
  if (k <= 0 || k >= p.size) return new Map(p);
  const keep = sortedEntries(p).slice(0, k);
  let s = 0;
  for (const [, v] of keep) s += v;
  return new Map(keep.map(([c, v]) => [c, v / s]));
}

// 累積確率が q を初めて超えるところまでを残して正規化する(必ず 1 個以上残る)
export function topP(p, q) {
  if (q >= 1) return new Map(p);
  const keep = [];
  let acc = 0;
  for (const [c, v] of sortedEntries(p)) {
    keep.push([c, v]);
    acc += v;
    if (acc >= q) break;
  }
  let s = 0;
  for (const [, v] of keep) s += v;
  return new Map(keep.map(([c, v]) => [c, v / s]));
}

// 確率の降順、同率は字の昇順。並べ方を決めておかないと両実装で結果がずれる
export function sortedEntries(p) {
  return [...p].sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
}

export function entropy(p) {
  let h = 0;
  for (const v of p.values()) if (v > 0) h -= v * Math.log2(v);
  return h;
}

// ---- 疑似乱数(Python と同一の出力)---------------------------------------------

export function mulberry32(seed) {
  let a = seed | 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// 字の昇順に並べた上での逆関数法
export function sample(p, rnd) {
  const items = [...p].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  const r = rnd();
  let acc = 0;
  for (const [c, v] of items) {
    acc += v;
    if (r < acc) return c;
  }
  return items[items.length - 1][0];
}

// ---- 読み込み -------------------------------------------------------------------

export async function loadCorpus(id, base = "data") {
  const [text, sab] = await Promise.all([
    fetch(`${base}/${id}.txt`).then((r) => r.text()),
    fetch(`${base}/${id}.sa.bin`).then((r) => r.arrayBuffer()),
  ]);
  const sa = new Int32Array(sab);
  if (sa.length !== text.length) {
    throw new Error(`接尾辞配列と本文の長さが違う: ${sa.length} != ${text.length}`);
  }
  return { text, sa };
}
