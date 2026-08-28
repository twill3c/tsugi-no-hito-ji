// 画面。計算そのものは model.js / shannon.js にあり、ここは描くことだけをする。
//
// **画面に出る文字は、図を作ったのと同じデータから導く**(HC-045)。
// 「文脈 7 文字が 25% 効いている」という文と、帯の幅と、表の行は、
// すべて distribution() が返す 1 つの trace から作る。別々に数えない。

import * as M from "./model.js";
import { shannonBounds, guessHistogram, guessRank } from "./shannon.js";

const $ = (id) => document.getElementById(id);
const fmt = (x, n = 2) => x.toFixed(n);
const pct = (x) => (x * 100).toFixed(1) + "%";

// 改行は候補として実在するが、そのままでは見えない。画面でだけ記号に置き換える
const show = (ch) => (ch === "\n" ? "⏎" : ch === " " ? "␣" : ch === M.UNK ? "未知" : ch);

const state = {
  corpora: [],
  id: null,
  model: null,
  held: "",
  base: "吾輩は",
  text: "吾輩は",
  temp: 0.8,
  topk: 0,
  topp: 1,
  order: M.MAX_ORDER,
  scaling: null,
};

// ---- 読み込み -------------------------------------------------------------------

async function boot() {
  const manifest = await fetch("data/corpora.json").then((r) => r.json());
  state.corpora = manifest.corpora;
  state.scaling = await fetch("data/scaling.json").then((r) => r.json()).catch(() => null);

  const sel = $("corpus");
  for (const c of state.corpora) {
    const o = document.createElement("option");
    o.value = c.id;
    o.textContent = `${c.label}(${c.chars.toLocaleString()}字)`;
    sel.appendChild(o);
  }
  sel.value = state.corpora[0].id;
  sel.addEventListener("change", () => switchCorpus(sel.value));

  $("disc-v").textContent = M.DISCOUNT;
  fillAbout();
  drawScaling();
  wire();
  await switchCorpus(state.corpora[0].id);
  $("boot").hidden = true;
  showTab("board");
}

async function switchCorpus(id) {
  $("boot").hidden = false;
  $("boot").textContent = "本文と索引を読み込んでいます…";
  const { text, sa } = await M.loadCorpus(id);
  state.id = id;
  state.model = new M.Model(text, sa, { maxOrder: state.order });
  state.held = await fetch(`data/${id}.held.txt`).then((r) => r.text());
  const c = state.corpora.find((x) => x.id === id);
  $("corpus-note").textContent =
    `${c.works} 作 / ${c.chars.toLocaleString()} 字 / 字種 ${c.types.toLocaleString()}。` +
    `報告用に『${c.held.title}』を抜いてある`;
  $("curve").innerHTML = "";
  $("curve-cap").textContent = "";
  $("boot").hidden = true;
  render();
}

// ---- 予測盤 ---------------------------------------------------------------------

function shaped(p) {
  let q = M.temperature(p, state.temp);
  q = M.topK(q, state.topk);
  q = M.topP(q, state.topp);
  return q;
}

function render() {
  const m = state.model;
  m.maxOrder = state.order;
  // slice(-0) は文字列全体を返してしまうので、0 は明示的に空文字にする
  const ctx = state.order > 0 ? state.text.slice(-state.order) : "";
  const { p, trace } = m.distribution(ctx);
  const cut = shaped(p);

  // 棒。並びは加工前の確率順にする。加工で順位が入れ替わることは無い(T-036 系)ので、
  // 「切られた候補」が下にまとまって見える
  const rows = M.sortedEntries(p).slice(0, 16);
  const max = rows.length ? rows[0][1] : 1;
  $("bars").innerHTML = rows
    .map(([ch, v]) => {
      const alive = cut.has(ch) && cut.get(ch) > 0;
      const w = ((v / max) * 100).toFixed(2);
      return `<div class="ch${ch === "\n" ? " nl" : ""}${alive ? "" : " cut"}">${esc(show(ch))}</div>
        <div class="track${alive ? "" : " cut"}"><div class="fill" style="width:${w}%"></div></div>
        <div class="pct${alive ? "" : " cut"}">${pct(v)}</div>`;
    })
    .join("");

  const shownCtx = ctx.length ? `「${ctx.replace(/\n/g, "⏎")}」` : "(文脈なし)";
  $("dist-note").textContent =
    `いま見ている文脈は ${shownCtx}。候補は ${p.size.toLocaleString()} 字、` +
    `つまみで残っているのは ${cut.size.toLocaleString()} 字。`;

  const hBefore = M.entropy(p);
  const hAfter = M.entropy(cut);
  $("dist-stat").innerHTML = `
    <div>加工前のエントロピー<b>${fmt(hBefore)} bit</b></div>
    <div>加工後<b>${fmt(hAfter)} bit</b></div>
    <div>当惑度(加工後)<b>${fmt(2 ** hAfter, 1)}</b></div>
    <div>いちばん確からしい字<b>${esc(show(M.sortedEntries(cut)[0][0]))}</b></div>`;

  drawStack(trace);
  $("temp-hint").textContent =
    state.temp === 0 ? "いつも最頻値。同じ文章しか書かない" :
    state.temp < 0.6 ? "尖っている。堅いが繰り返しやすい" :
    state.temp > 1.4 ? "溶けている。珍しい字も出る" : "";
  $("topp-hint").textContent =
    state.topp >= 1 ? "" : `上位 ${cut.size} 字で累積 ${pct(state.topp)} に届いた`;
}

function drawStack(trace) {
  const cols = trace.filter((t) => t.share > 0.0005);
  $("stack").innerHTML = cols
    .map(
      (t) =>
        `<div style="width:${(t.share * 100).toFixed(3)}%;background:var(--o${t.order})" ` +
        `title="長さ ${t.order}: ${pct(t.share)}"></div>`,
    )
    .join("");
  $("stack-legend").innerHTML = cols
    .map((t) => `<span><i style="background:var(--o${t.order})"></i>${t.order} 文字 ${pct(t.share)}</span>`)
    .join("");

  $("trace").querySelector("tbody").innerHTML = trace
    .map((t) => {
      const dead = t.n === 0;
      return `<tr class="${dead ? "dead" : ""}">
        <td>${t.order}</td>
        <td class="ctx">${t.order === 0 ? "—" : esc(t.ctx.replace(/\n/g, "⏎"))}</td>
        <td>${dead ? "0(出てこない)" : t.n.toLocaleString()}</td>
        <td>${dead ? "—" : t.u.toLocaleString()}</td>
        <td>${dead ? "—" : fmt(t.gamma, 3)}</td>
        <td>${pct(t.share)}</td></tr>`;
    })
    .join("");
}

function step(n) {
  const rnd = M.mulberry32(Number($("seed").value) | 0);
  // たねから毎回引き直すので、同じたね・同じつまみなら同じ続きになる。
  // 途中まで進んだ状態から 1 字だけ足すときも、先頭から引き直して同じ列を再現する
  const start = state.text.length;
  let s = state.base;
  const want = start - state.base.length + n;
  for (let i = 0; i < want; i++) {
    state.model.maxOrder = state.order;
    const { p } = state.model.distribution(state.order > 0 ? s.slice(-state.order) : "");
    s += M.sample(shaped(p), rnd);
  }
  state.text = s;
  $("prompt").value = s;
  renderGen();
  render();
}

function renderGen() {
  $("gen").innerHTML =
    `<span class="seed">${esc(state.base.replace(/\n/g, "⏎"))}</span>` +
    esc(state.text.slice(state.base.length).replace(/\n/g, "⏎"));
}

// ---- 文脈の山 -------------------------------------------------------------------

async function measureCurve() {
  const btn = $("btn-curve");
  btn.disabled = true;
  // 事前計算の走査(scaling.json)と同じ字数で測る。標本が違うと谷の位置が食い違い、
  // 同じ画面の 2 つの図が別のことを言い出す
  const sample = state.held.slice(0, state.scaling?.eval_chars ?? 8000);
  const curve = [];
  for (let L = 0; L <= M.MAX_ORDER; L++) {
    $("curve-busy").textContent = `文脈長 ${L} を測っています…`;
    await new Promise((r) => setTimeout(r, 0)); // 画面を描き替える隙を作る
    const m = new M.Model(state.model.text, state.model.sa, { maxOrder: L });
    curve.push(m.bitsPerChar(sample).bits);
  }
  $("curve-busy").textContent = "";
  btn.disabled = false;

  const best = curve.indexOf(Math.min(...curve));
  const c = state.corpora.find((x) => x.id === state.id);
  $("curve").innerHTML = lineChart([{ label: c.label, values: curve, color: "var(--accent)" }], {
    xs: curve.map((_, i) => i),
    xlabel: "文脈の長さ(文字)",
    ylabel: "1 字あたりのビット数",
    peaks: [best],
  });
  // 説明文は曲線と同じ配列から作る。数字を別に持たない
  $("curve-cap").textContent =
    `『${c.held.title}』の先頭 ${sample.length.toLocaleString()} 字で測った。` +
    `いちばん当たるのは文脈 ${best} 文字(${fmt(curve[best])} bit)。` +
    `文脈なし(${fmt(curve[0])} bit)から ${fmt(curve[0] - curve[best])} bit ぶん当たるようになり、` +
    `${M.MAX_ORDER} 文字まで伸ばすと ${fmt(curve[M.MAX_ORDER] - curve[best])} bit ぶん逆戻りする。`;
}

function drawScaling() {
  const s = state.scaling;
  if (!s) return;
  const series = s.rows.map((r, i) => ({
    label: `${(r.chars / 1000).toFixed(0)}千字`,
    values: r.curve,
    color: `var(--o${Math.min(8, i + 2)})`,
  }));
  $("scaling").innerHTML = lineChart(series, {
    xs: s.orders,
    xlabel: "文脈の長さ(文字)",
    ylabel: "1 字あたりのビット数",
    peaks: s.rows.map((r) => r.best_order),
  });
  $("scaling-cap").textContent =
    `${s.corpus} の本文を先頭から N 字だけ読ませたモデル ${s.rows.length} 本を、` +
    `同じ ${s.eval_chars.toLocaleString()} 字で測った。谷(赤)の位置に注目。`;
  $("scaling-tab").querySelector("tbody").innerHTML = s.rows
    .map((r, i) => {
      const gap = s.gap_3_minus_2[i];
      return `<tr><td>${r.chars.toLocaleString()}</td><td>${r.best_order} 文字</td>
        <td>${fmt(r.curve[2], 3)}</td><td>${fmt(r.curve[3], 3)}</td>
        <td style="color:${gap <= 0 ? "var(--ok)" : "inherit"}">${gap > 0 ? "+" : ""}${fmt(gap, 3)}</td></tr>`;
    })
    .join("");
}

// ---- かなあて勝負 ---------------------------------------------------------------

const quiz = { items: [], i: 0, tried: [], human: [], machine: [], kana: [] };
const QUESTIONS = 12;
const CONTEXT_SHOWN = 24;

function hiraganaOf(model) {
  // 語彙にあるひらがなだけを鍵盤に出す。**あると仮定せず、本文から拾う**
  const out = [];
  for (const ch of model.uni.keys()) {
    const c = ch.codePointAt(0);
    if (c >= 0x3041 && c <= 0x3096) out.push(ch);
  }
  return out.sort();
}

function startQuiz() {
  const m = state.model;
  m.maxOrder = state.order;
  quiz.kana = hiraganaOf(m);
  const set = new Set(quiz.kana);
  const held = state.held;
  // 出題位置: 次の一字がひらがなで、手前に十分な文脈がある場所
  const spots = [];
  for (let i = CONTEXT_SHOWN; i < held.length; i++) if (set.has(held[i])) spots.push(i);
  const rnd = M.mulberry32((Date.now() & 0x7fffffff) | 1);
  quiz.items = [];
  const used = new Set();
  while (quiz.items.length < QUESTIONS && used.size < spots.length) {
    const j = Math.floor(rnd() * spots.length);
    if (used.has(j)) continue;
    used.add(j);
    quiz.items.push(spots[j]);
  }
  quiz.i = 0;
  quiz.human = [];
  quiz.machine = [];
  $("quiz-area").hidden = false;
  $("tally").hidden = true;
  $("quiz-fig").hidden = true;
  drawKeyboard();
  askQuiz();
}

function askQuiz() {
  const at = quiz.items[quiz.i];
  const held = state.held;
  quiz.tried = [];
  $("quiz-progress").textContent = `${quiz.i + 1} / ${quiz.items.length} 問`;
  $("quiz-text").innerHTML =
    esc(held.slice(at - CONTEXT_SHOWN, at).replace(/\n/g, "⏎")) +
    `<span class="blank" id="blank">?</span>` +
    `<span class="after">${esc(held.slice(at + 1, at + 8).replace(/\n/g, "⏎"))}</span>`;
  $("quiz-hint").textContent = "次に来るひらがなを当ててください。当たるまで何度でも押せます。";
  for (const b of $("kana").children) {
    b.classList.remove("used");
    b.disabled = false;
  }
}

function drawKeyboard() {
  $("kana").innerHTML = quiz.kana.map((c) => `<button type="button" data-k="${esc(c)}">${esc(c)}</button>`).join("");
  $("kana").onclick = (ev) => {
    const b = ev.target.closest("button[data-k]");
    if (!b || b.disabled) return;
    guess(b.dataset.k, b);
  };
}

function guess(ch, btn) {
  const at = quiz.items[quiz.i];
  const truth = state.held[at];
  if (quiz.tried.includes(ch)) return;
  quiz.tried.push(ch);
  if (ch !== truth) {
    btn.classList.add("used");
    btn.disabled = true;
    $("quiz-hint").textContent = `${quiz.tried.length} 回目、まだ違う。`;
    return;
  }
  // 当たり。人間の回数と、同じ位置でのモデルの回数を記録する
  quiz.human.push(quiz.tried.length);
  quiz.machine.push(machineRank(at, truth));
  $("blank").textContent = ch;
  quiz.i++;
  if (quiz.i < quiz.items.length) setTimeout(askQuiz, 450);
  else finishQuiz();
}

function machineRank(at, truth) {
  // ひらがなだけに絞った上での順位。人間はひらがな以外を言えないので、
  // モデルにも同じ土俵で答えさせないと勝負が不公平になる
  const m = state.model;
  const { p } = m.distribution(state.held.slice(Math.max(0, at - state.order), at));
  const only = new Map();
  let s = 0;
  for (const c of quiz.kana) { const v = p.get(c) || 0; only.set(c, v); s += v; }
  for (const [c, v] of only) only.set(c, v / s);
  return guessRank(M.sortedEntries(only), truth);
}

function finishQuiz() {
  $("quiz-area").hidden = true;
  $("quiz-progress").textContent = "";
  const hh = guessHistogram(quiz.human);
  const mh = guessHistogram(quiz.machine);
  const hb = shannonBounds(hh);
  const mb = shannonBounds(mh);
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const first = (a) => a.filter((x) => x === 1).length;

  // **下界を出してよいのは、確率の降順で当てにいった側だけ**(HC-003)。
  // 理由は 2 通りあり、区別して書く:
  //   人間  … そもそも確率の順に当てにいっていない。何問やっても下界は言えない
  //   モデル … 構造上は確率の順だが、問題数が少ないと標本の並びが崩れることがある
  // 上界のほうは推測回数の列から本文を復元できることだけから出るので、どちらにも言える。
  const bits = (b) => (b.loValid ? `${fmt(b.lo)} – ${fmt(b.up)}` : `${fmt(b.up)} 以下`);
  const why = (b, ideal) =>
    b.loValid
      ? "確率の高い順に当てにいっているので、上下から挟める。"
      : ideal
        ? `問題数が ${quiz.items.length} 問と少なく、回数の並びが崩れた。この回は上界だけが言える。`
        : "当てにいった順が確率の順ではないので、<b>下からは挟めない</b>。上界だけが言える。";

  $("tally").hidden = false;
  $("tally").innerHTML = [
    ["人間", "h", quiz.human, hb, false],
    ["モデル", "m", quiz.machine, mb, true],
  ]
    .map(
      ([who, cls, arr, b, ideal]) => `<div class="card" style="margin:0">
        <div class="who ${cls}">${who}</div>
        <div class="stat">
          <div>一発で当てた<b>${first(arr)} / ${arr.length}</b></div>
          <div>平均の回数<b>${fmt(mean(arr), 1)}</b></div>
          <div>ビット数<b>${bits(b)}</b></div>
        </div>
        <p class="note" style="margin:.5rem 0 0">${why(b, ideal)}</p></div>`,
    )
    .join("");

  const nKana = quiz.kana.length;
  $("quiz-fig").hidden = false;
  $("quiz-chart").innerHTML = barPair(bucketize(quiz.human), bucketize(quiz.machine), BUCKETS);
  const hm = mean(quiz.human), mm = mean(quiz.machine);
  $("quiz-cap").textContent =
    `${quiz.items.length} 問。ひらがな ${nKana} 字から当てずっぽうに言うと平均 ${fmt((nKana + 1) / 2, 1)} 回、` +
    `情報量にして ${fmt(Math.log2(nKana))} bit。` +
    `平均の回数は人間 ${fmt(hm, 1)} 回、モデル ${fmt(mm, 1)} 回で ` +
    (hm < mm ? "人間のほうが少ない。" : mm < hm ? "モデルのほうが少ない。" : "同じ。") +
    `${quiz.items.length} 問しかないので、この差はぶれの内側にあるかもしれない。`;
}

// 推測回数の分布は裾が長い(70 回かかることもある)。**挟む計算は生の回数で行い、
// 図だけをまとめる。** まとめた分布で計算すると、まとめ方が答えを動かしてしまう
const BUCKETS = ["1", "2", "3", "4", "5", "6–10", "11–20", "21–40", "41+"];

function bucketize(counts) {
  const idx = (c) => (c <= 5 ? c - 1 : c <= 10 ? 5 : c <= 20 ? 6 : c <= 40 ? 7 : 8);
  const hist = new Array(BUCKETS.length).fill(0);
  for (const c of counts) hist[idx(c)]++;
  return hist.map((x) => x / counts.length);
}

// ---- 図(SVG は同じデータからラベルまで作る)-------------------------------------

function lineChart(series, opt) {
  const W = 720, H = 310, P = { l: 52, r: 14, t: 30, b: 40 };
  const all = series.flatMap((s) => s.values);
  const lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.12 || 0.5;
  const y0 = lo - pad, y1 = hi + pad;
  const X = (i) => P.l + (i / (opt.xs.length - 1)) * (W - P.l - P.r);
  const Y = (v) => P.t + (1 - (v - y0) / (y1 - y0)) * (H - P.t - P.b);

  let g = "";
  for (let k = 0; k <= 4; k++) {
    const v = y0 + ((y1 - y0) * k) / 4;
    g += `<line class="grid" x1="${P.l}" y1="${Y(v).toFixed(1)}" x2="${W - P.r}" y2="${Y(v).toFixed(1)}"/>
      <text x="${P.l - 8}" y="${(Y(v) + 4).toFixed(1)}" text-anchor="end">${v.toFixed(2)}</text>`;
  }
  for (const x of opt.xs) {
    g += `<text x="${X(x).toFixed(1)}" y="${H - P.b + 18}" text-anchor="middle">${x}</text>`;
  }
  let paths = "";
  series.forEach((s, si) => {
    const d = s.values.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join("");
    paths += `<path class="mark" d="${d}" stroke="${s.color}"/>`;
    const pk = opt.peaks?.[si];
    if (pk != null) {
      paths += `<circle class="peak" cx="${X(pk).toFixed(1)}" cy="${Y(s.values[pk]).toFixed(1)}" r="4"/>`;
      // ラベルは谷の座標から作る。位置を決め打ちしない(HC-045)
      paths += `<text x="${(X(pk) + 8).toFixed(1)}" y="${(Y(s.values[pk]) - 8).toFixed(1)}">${esc(s.label)}</text>`;
    }
  });
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opt.ylabel)}と${esc(opt.xlabel)}の関係">
    ${g}
    <line class="axis" x1="${P.l}" y1="${P.t}" x2="${P.l}" y2="${H - P.b}"/>
    <line class="axis" x1="${P.l}" y1="${H - P.b}" x2="${W - P.r}" y2="${H - P.b}"/>
    ${paths}
    <text x="${W / 2}" y="${H - 6}" text-anchor="middle">${esc(opt.xlabel)}</text>
    <text x="${P.l - 40}" y="14">${esc(opt.ylabel)}</text>
  </svg>`;
}

function barPair(a, b, labels) {
  const n = labels.length;
  const W = 720, H = 230, P = { l: 44, r: 14, t: 12, b: 44 };
  const hi = Math.max(...a, ...b, 0.1);
  const bw = (W - P.l - P.r) / n;
  let g = "";
  for (let i = 0; i < n; i++) {
    const av = a[i] || 0, bv = b[i] || 0;
    const x = P.l + i * bw;
    const ha = ((H - P.t - P.b) * av) / hi, hb = ((H - P.t - P.b) * bv) / hi;
    g += `<rect x="${(x + bw * 0.12).toFixed(1)}" y="${(H - P.b - ha).toFixed(1)}" width="${(bw * 0.36).toFixed(1)}" height="${ha.toFixed(1)}" fill="var(--human)"/>
      <rect x="${(x + bw * 0.52).toFixed(1)}" y="${(H - P.b - hb).toFixed(1)}" width="${(bw * 0.36).toFixed(1)}" height="${hb.toFixed(1)}" fill="var(--machine)"/>
      <text x="${(x + bw / 2).toFixed(1)}" y="${H - P.b + 16}" text-anchor="middle">${esc(labels[i])}</text>`;
  }
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="何回目で当たったかの分布">
    ${g}<line class="axis" x1="${P.l}" y1="${H - P.b}" x2="${W - P.r}" y2="${H - P.b}"/>
    <text x="${W / 2}" y="${H - 8}" text-anchor="middle">何回目で当たったか(青=人間 / 橙=モデル)</text></svg>`;
}

// ---- 雑務 ---------------------------------------------------------------------

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function fillAbout() {
  $("about-corpora").querySelector("tbody").innerHTML = state.corpora
    .map(
      (c) => `<tr><td>${esc(c.label)}</td><td>${c.chars.toLocaleString()}</td>
        <td>${c.types.toLocaleString()}</td><td>${c.works}</td>
        <td>${esc(c.dev.title)}</td><td>${esc(c.held.title)}</td></tr>`,
    )
    .join("");
}

function showTab(name) {
  for (const t of document.querySelectorAll('[role="tab"]')) {
    const on = t.id === `tab-${name}`;
    t.setAttribute("aria-selected", String(on));
    $(t.getAttribute("aria-controls")).hidden = !on;
  }
}

function wire() {
  for (const t of document.querySelectorAll('[role="tab"]')) {
    t.addEventListener("click", () => showTab(t.id.replace("tab-", "")));
  }
  $("prompt").addEventListener("input", () => {
    state.base = state.text = $("prompt").value;
    renderGen();
    render();
  });
  $("btn-step").addEventListener("click", () => step(1));
  $("btn-run").addEventListener("click", () => step(20));
  $("btn-reset").addEventListener("click", () => {
    state.text = state.base;
    $("prompt").value = state.base;
    renderGen();
    render();
  });
  $("seed").addEventListener("input", () => {
    state.text = state.base;
    $("prompt").value = state.base;
    renderGen();
    render();
  });
  $("temp").addEventListener("input", (e) => {
    state.temp = Number(e.target.value) / 100;
    $("temp-v").textContent = fmt(state.temp);
    render();
  });
  $("topk").addEventListener("input", (e) => {
    state.topk = Number(e.target.value);
    $("topk-v").textContent = state.topk === 0 ? "制限なし" : `上位 ${state.topk} 字`;
    render();
  });
  $("topp").addEventListener("input", (e) => {
    state.topp = Number(e.target.value) / 100;
    $("topp-v").textContent = state.topp >= 1 ? "制限なし" : fmt(state.topp);
    render();
  });
  $("order").addEventListener("input", (e) => {
    state.order = Number(e.target.value);
    $("order-v").textContent = state.order === 0 ? "使わない" : `${state.order} 文字`;
    render();
  });
  $("btn-curve").addEventListener("click", measureCurve);
  $("btn-quiz").addEventListener("click", startQuiz);
}

boot().catch((e) => {
  $("boot").className = "warn";
  $("boot").textContent = `読み込みに失敗しました: ${e.message}`;
});
