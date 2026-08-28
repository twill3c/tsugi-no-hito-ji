// 実ブラウザで開いて検品する。
//
// 目で見なければ分からない性質(図が読めるか・文と図が食い違っていないか)は
// テストで代替できない。だが「取得に失敗した画面を撮っても撮影しましたと出る」型の
// 事故があるので、**この道具は失敗を必ず終了コードで知らせる**(HC-041)。
//
// 検品するのは次の 5 点。どれも「画面に出ている数字」と「モデルの計算」の一致である。
//   1. 起動が終わり、棒・帯・表が実際に描かれている(件数 0 で緑にしない)
//   2. 帯の幅の合計が 100%、表の寄与の合計も 100%
//   3. 表の「出現 n」が、本文を素朴に数えた回数と一致する(画面の数字の出所の検算)
//   4. つまみを動かすと、切り落とされた候補の数が実際に変わる
//   5. フリート規約のフッタが 5 項目そろっている
//
// **本番にも向けられる。** ローカルで緑でも、出荷から漏れたファイルがあれば本番だけが壊れる
// (実際 .vercelignore の data/ が web/data/ にも当たり、本番でコーパスが 404 になった)。
// トップページは 200 を返し続けるので、URL の生存確認では気づけない。
//
// usage: node build/page_check.mjs [--shot <png>] [--url https://...]

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join, extname, normalize } from "node:path";
import { chromium } from "playwright";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = join(HERE, "..", "web");
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".bin": "application/octet-stream",
};

const problems = [];
const ok = (cond, msg) => { if (!cond) problems.push(msg); };

const urlAt = process.argv.indexOf("--url");
const remote = urlAt > 0 ? process.argv[urlAt + 1] : null;

const server = createServer(async (req, res) => {
  const url = decodeURIComponent(req.url.split("?")[0]);
  const path = normalize(join(WEB, url === "/" ? "index.html" : url));
  if (!path.startsWith(WEB)) { res.writeHead(403).end(); return; }
  try {
    const body = await readFile(path);
    res.writeHead(200, { "content-type": TYPES[extname(path)] || "application/octet-stream" });
    res.end(body);
  } catch {
    res.writeHead(404).end("not found");
  }
});
await new Promise((r) => server.listen(0, r));
const base = remote || `http://127.0.0.1:${server.address().port}/`;
if (remote) console.log(`検品先: ${remote}`);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1100, height: 1400 } });

const consoleErrors = [];
page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
page.on("pageerror", (e) => consoleErrors.push(String(e)));
// 取得できなかった資源は、画面が「読み込み中」のまま止まる形で現れる。
// 何が落ちたのかを名指しできるよう、応答コードで拾っておく
page.on("response", (r) => {
  if (r.status() >= 400) consoleErrors.push(`${r.status()} ${r.url()}`);
});

await page.goto(base, { waitUntil: "load" });
await page.waitForSelector("#p-board:not([hidden])", { timeout: 30000 });
await page.waitForFunction(() => document.querySelectorAll("#bars .ch").length > 0, { timeout: 30000 });

// 1. 描かれているか
const counts = await page.evaluate(() => ({
  bars: document.querySelectorAll("#bars .ch").length,
  stack: document.querySelectorAll("#stack > div").length,
  trace: document.querySelectorAll("#trace tbody tr").length,
  legend: document.querySelectorAll("#stack-legend span").length,
}));
ok(counts.bars >= 10, `棒が ${counts.bars} 本しか無い`);
ok(counts.stack >= 2, `帯が ${counts.stack} 区画しか無い`);
ok(counts.trace >= 2, `内訳の表が ${counts.trace} 行しか無い`);
ok(counts.legend === counts.stack, `凡例 ${counts.legend} と帯 ${counts.stack} の数が違う`);

// 2. 帯と表の合計
const sums = await page.evaluate(() => {
  const w = [...document.querySelectorAll("#stack > div")]
    .map((d) => parseFloat(d.style.width));
  const t = [...document.querySelectorAll("#trace tbody tr")]
    .map((tr) => parseFloat(tr.lastElementChild.textContent));
  return { stack: w.reduce((a, b) => a + b, 0), trace: t.reduce((a, b) => a + b, 0) };
});
ok(Math.abs(sums.stack - 100) < 0.6, `帯の幅の合計が ${sums.stack.toFixed(2)}%`);
ok(Math.abs(sums.trace - 100) < 0.6, `表の寄与の合計が ${sums.trace.toFixed(2)}%`);

// 3. 表の「出現 n」を、本文の素朴な数え直しと突き合わせる。
//    画面の数字はモデル経由で出ているので、こちらは indexOf で数える別経路にする
const nCheck = await page.evaluate(async () => {
  const text = await fetch("data/soseki.txt").then((r) => { if (!r.ok) throw new Error(`本文が取れない: ${r.status}`); return r.text(); });
  const rows = [...document.querySelectorAll("#trace tbody tr")].slice(1);
  const out = [];
  for (const tr of rows) {
    const ctx = tr.children[1].textContent.replace(/⏎/g, "\n");
    const shown = parseInt(tr.children[2].textContent.replace(/[^0-9]/g, ""), 10) || 0;
    let n = 0;
    for (let i = text.indexOf(ctx); i >= 0; i = text.indexOf(ctx, i + 1)) {
      if (i + ctx.length < text.length) n++;
    }
    out.push({ ctx, shown, n });
  }
  return out;
});
ok(nCheck.length > 0, "検算対象の行が無い");
for (const r of nCheck) {
  ok(r.shown === r.n, `「${r.ctx}」の出現回数: 画面 ${r.shown} / 数え直し ${r.n}`);
}

// 4. つまみが効いているか。top-p を絞ると残る候補が減るはず
const before = await page.textContent("#dist-note");
await page.fill("#prompt", "吾輩は");
await page.locator("#topp").fill("50");
await page.locator("#topp").dispatchEvent("input");
const after = await page.textContent("#dist-note");
const num = (s) => Number(s.match(/残っているのは ([\d,]+) 字/)[1].replace(/,/g, ""));
ok(num(after) < num(before), `top-p を絞っても候補が減らない: ${num(before)} → ${num(after)}`);
await page.locator("#topp").fill("100");
await page.locator("#topp").dispatchEvent("input");

// 生成が動くか(1 字書いて本文が伸びる)
const lenBefore = (await page.inputValue("#prompt")).length;
await page.click("#btn-step");
ok((await page.inputValue("#prompt")).length === lenBefore + 1, "「次の一字を書く」で本文が伸びない");

// 5. フッタ。要素名では探さず、中身(App Menu と MIT License)で選ぶ
const foot = await page.evaluate(() => {
  const bar = [...document.querySelectorAll("nav,footer,div")].find(
    (e) => /App Menu/.test(e.textContent) && /MIT License/.test(e.textContent) && e.querySelectorAll("a").length >= 4,
  );
  if (!bar) return null;
  return {
    links: [...bar.querySelectorAll("a")].map((a) => a.textContent.trim()),
    text: bar.textContent.replace(/\s+/g, " ").trim(),
    fixed: getComputedStyle(bar).position,
  };
});
ok(foot !== null, "規約フッタが見つからない");
if (foot) {
  ok(foot.links.length === 5, `フッタのリンクが ${foot.links.length} 本(規約は 5)`);
  ok(/© 2026 坂田哲朗/.test(foot.text), "著作権表示が無い");
  ok(foot.fixed === "fixed", `フッタが固定されていない: ${foot.fixed}`);
}

// 他のタブも開けること
for (const t of ["peak", "game", "about"]) {
  await page.click(`#tab-${t}`);
  ok(!(await page.locator(`#p-${t}`).isHidden()), `${t} タブが開かない`);
}
await page.click("#tab-board");

const shotAt = process.argv.indexOf("--shot");
if (shotAt > 0 && process.argv[shotAt + 1]) {
  await page.screenshot({ path: process.argv[shotAt + 1], fullPage: true });
  console.log(`撮影 → ${process.argv[shotAt + 1]}`);
}

ok(consoleErrors.length === 0, `コンソールエラー ${consoleErrors.length} 件:\n  ${consoleErrors.join("\n  ")}`);

await browser.close();
server.close();

if (problems.length) {
  console.error(`検品 NG(${problems.length} 件)`);
  for (const p of problems) console.error("  - " + p);
  process.exit(1);
}
console.log(`検品 OK — 棒 ${counts.bars} / 帯 ${counts.stack} / 内訳 ${counts.trace} 行、出現回数の検算 ${nCheck.length} 件`);
