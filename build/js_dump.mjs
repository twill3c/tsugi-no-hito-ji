// 出荷実装(web/js/model.js)の出力を JSON に落とす。pytest 側がこれと
// 参照実装(build/model.py)を突き合わせる(G-01 二実装照合)。
//
// **この台本は計算を一切しない。** ここで値を丸めたり整えたりすると、
// 照合しているのが「出荷実装」ではなく「この台本」になる。
//
// usage: node build/js_dump.mjs <corpus-id> > /tmp/js.json

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import * as M from "../web/js/model.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA = join(HERE, "..", "web", "data");

const cid = process.argv[2] || "soseki";
const text = readFileSync(join(DATA, `${cid}.txt`), "utf8");
const sa = new Int32Array(
  readFileSync(join(DATA, `${cid}.sa.bin`)).buffer.slice(0),
);
const held = readFileSync(join(DATA, `${cid}.held.txt`), "utf8");

const model = new M.Model(text, sa);

// 探る文脈は、長い一致・短い一致・未出現・空を混ぜる。
// 全部が長い一致だと、混ぜ上げの下の段が一度も効かないまま緑になる
const PROBES = ["", "の", "である。", "吾輩は", "親譲りの無鉄砲で", "こころ", "〓〓不在文脈"];

const dist = {};
for (const ctx of PROBES) {
  const { p, trace } = model.distribution(ctx);
  dist[ctx] = {
    trace,
    top: M.sortedEntries(p).slice(0, 12),
    // 三つ目の数え方との一致は Python 側でも見るが、JS 側の値も出しておく
    probOf: Object.fromEntries(
      M.sortedEntries(p).slice(0, 5).map(([c]) => [c, model.probOf(ctx, c)]),
    ),
    entropy: M.entropy(p),
    sum: [...p.values()].reduce((a, b) => a + b, 0),
  };
}

const base = model.distribution("である。").p;
const shaped = {};
for (const t of [0, 0.25, 0.5, 1, 1.5, 2]) {
  shaped[`T=${t}`] = M.sortedEntries(M.temperature(base, t)).slice(0, 8);
  shaped[`H@T=${t}`] = M.entropy(M.temperature(base, t));
}
for (const k of [1, 3, 10]) shaped[`k=${k}`] = M.sortedEntries(M.topK(base, k));
for (const q of [0.5, 0.9, 0.99]) shaped[`p=${q}`] = M.sortedEntries(M.topP(base, q));

const rnd = M.mulberry32(20260829);
const rands = Array.from({ length: 16 }, () => rnd());

const gen = (() => {
  const r = M.mulberry32(7);
  let s = "吾輩は";
  for (let i = 0; i < 120; i++) {
    const { p } = model.distribution(s.slice(-M.MAX_ORDER));
    s += M.sample(M.topP(M.temperature(p, 0.8), 0.95), r);
  }
  return s;
})();

process.stdout.write(
  JSON.stringify(
    {
      corpus: cid,
      n: model.n,
      v: model.v,
      dist,
      shaped,
      rands,
      gen,
      bits: model.bitsPerChar(held.slice(0, 2000)),
    },
    null,
    1,
  ),
);
