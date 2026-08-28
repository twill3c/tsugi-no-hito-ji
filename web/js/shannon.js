// 推測回数からエントロピーの上下界を出す(Shannon 1951)。
//
// q[i-1] = 「i 回目の推測で当たった」割合。予測器の中身を知らなくても、
// 当てっこの成績だけから情報源のエントロピーを外側から挟める、というのがこの式の値打ち。
// 人間には確率を聞けないので、人間の側はこの形でしか測れない。
//
//   下界 = Σ_i i · (q_i − q_{i+1}) · log2 i
//   上界 = − Σ_i q_i log2 q_i
//
// 式が本当に挟むかは Python 側の T-050 で確かめてある(無作為 2,000 分布)。

export function shannonBounds(q) {
  const n = q.length;
  let lo = 0;
  for (let i = 1; i <= n; i++) {
    const nxt = i < n ? q[i] : 0;
    lo += i * (q[i - 1] - nxt) * Math.log2(i);
  }
  let up = 0;
  for (const x of q) if (x > 0) up -= x * Math.log2(x);
  return { lo, up, loValid: isNonIncreasing(q) };
}

// q が「i 回目で当たった割合」として非増加か。
//
// **下界の式はここを前提にしている。** 確率の高い順に言う予測器なら自然に成り立つが、
// 人間の推測順はそうならない。前提を外れた q に下界の式を当てると、
// 下界が上界を上回る(実際に画面でそうなった —— HARNESS_CHANGELOG HC-003)。
// 上界のほうは、推測回数の列から本文を復元できることだけから出るので、
// **どんな下手な予測器でも成り立つ**。だから人間には上界しか出さない。
export function isNonIncreasing(q) {
  for (let i = 1; i < q.length; i++) if (q[i] > q[i - 1] + 1e-12) return false;
  return true;
}

// 推測回数の生データ(1 始まり)から割合の配列を作る。
// 空のときに NaN を返さないよう、長さ 0 は長さ 0 のまま返す
export function guessHistogram(counts) {
  if (counts.length === 0) return [];
  const max = Math.max(...counts);
  const hist = new Array(max).fill(0);
  for (const c of counts) hist[c - 1]++;
  return hist.map((x) => x / counts.length);
}

// 理想的な予測器(確率の高い順に言う)が truth を当てるまでの回数。
// 並べ方は model.js の sortedEntries と同じでなければならない(同率の扱い)
export function guessRank(sorted, truth) {
  for (let i = 0; i < sorted.length; i++) if (sorted[i][0] === truth) return i + 1;
  return sorted.length + 1;
}
