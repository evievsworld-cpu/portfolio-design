/**
 * Browser-side retrieval over the prebuilt index (SPEC.md §3).
 *
 * This is a port of src/retrieve.py and must stay byte-for-byte equivalent in its
 * ranking: tests/test_parity.py runs both over every eval question and asserts the
 * top-5 match. If they diverge, the numbers in the README describe a retriever that
 * visitors are not using.
 *
 * Kept as its own ES module rather than inlined into ask.html precisely so it can be
 * imported by the test. Still no build step — the browser loads it directly.
 */

// Must match STOPWORDS in src/tokenize.py exactly.
const STOPWORDS = new Set(
  `a an and are as at be been but by for from had has have he her his i if in into is
it its of on or our ours she that the their them then there these they this to was
were what when where which who will with would you your`.split(/\s+/)
);

export function tokenize(text) {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length > 1 && !STOPWORDS.has(t));
}

/** Decode base64 to Int8Array. Works in the browser (atob) and in Node (Buffer). */
function b64ToInt8(b64) {
  if (typeof atob === "function") {
    const bin = atob(b64);
    const out = new Int8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = (bin.charCodeAt(i) << 24) >> 24;
    return out;
  }
  return new Int8Array(Buffer.from(b64, "base64"));
}

/** Inverse of index.quantize_int8. Mirrors dequantize() in src/retrieve.py. */
function dequantize(payload, rows, cols) {
  const raw = b64ToInt8(payload.b64);
  const scales = payload.scales;
  const out = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      out[r * cols + c] = (raw[r * cols + c] * scales[c]) / 127.0;
    }
  }
  return out;
}

function normalizeRows(mat, rows, cols) {
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    for (let c = 0; c < cols; c++) {
      const v = mat[r * cols + c];
      sum += v * v;
    }
    const norm = Math.sqrt(sum) || 1.0;
    for (let c = 0; c < cols; c++) mat[r * cols + c] /= norm;
  }
  return mat;
}

export class Retriever {
  constructor(index) {
    this.index = index;
    this.meta = index.meta;
    this.chunks = index.chunks;
    this.n = index.chunks.length;

    const bm = index.bm25;
    this.termIx = new Map(bm.vocab.map((t, i) => [t, i]));
    this.df = bm.df;
    this.docLen = bm.doc_len;
    this.avgdl = this.meta.avgdl;
    this.k1 = this.meta.bm25.k1;
    this.b = this.meta.bm25.b;
    this.tf = bm.postings.map((row) => new Map(row));

    const lsa = index.lsa;
    this.lsaIx = new Map(lsa.terms.map((t, i) => [t, i]));
    this.idf = lsa.idf;
    this.k = this.meta.components;
    this.nLsaTerms = lsa.terms.length;
    this.components = dequantize(lsa.components, this.nLsaTerms, this.k);
    this.docVectors = normalizeRows(
      dequantize(lsa.doc_vectors, this.n, this.k),
      this.n,
      this.k
    );
  }

  scoreBm25(tokens) {
    const scores = new Float64Array(this.n);
    for (const token of tokens) {
      const ti = this.termIx.get(token);
      if (ti === undefined) continue;
      const df = this.df[ti];
      const idf = Math.log(1.0 + (this.n - df + 0.5) / (df + 0.5));
      for (let doc = 0; doc < this.n; doc++) {
        const tf = this.tf[doc].get(ti);
        if (!tf) continue;
        const denom =
          tf + this.k1 * (1.0 - this.b + (this.b * this.docLen[doc]) / this.avgdl);
        scores[doc] += (idf * (tf * (this.k1 + 1.0))) / denom;
      }
    }
    return scores;
  }

  scoreLsa(tokens) {
    const counts = new Map();
    for (const token of tokens) {
      const col = this.lsaIx.get(token);
      if (col !== undefined) counts.set(col, (counts.get(col) || 0) + 1);
    }
    const scores = new Float64Array(this.n);
    if (counts.size === 0) return scores; // no query term in the LSA vocabulary

    const vec = new Float64Array(this.nLsaTerms);
    for (const [col, count] of counts) {
      vec[col] = (1.0 + Math.log(count)) * this.idf[col];
    }
    let norm = 0;
    for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm);
    if (norm) for (let i = 0; i < vec.length; i++) vec[i] /= norm;

    // Project into component space: q = vec @ components
    const q = new Float64Array(this.k);
    for (const [col] of counts) {
      const w = vec[col];
      if (!w) continue;
      const base = col * this.k;
      for (let c = 0; c < this.k; c++) q[c] += w * this.components[base + c];
    }
    let qn = 0;
    for (let c = 0; c < this.k; c++) qn += q[c] * q[c];
    qn = Math.sqrt(qn);
    if (qn === 0) return scores;
    for (let c = 0; c < this.k; c++) q[c] /= qn;

    for (let doc = 0; doc < this.n; doc++) {
      let dot = 0;
      const base = doc * this.k;
      for (let c = 0; c < this.k; c++) dot += this.docVectors[base + c] * q[c];
      scores[doc] = dot;
    }
    return scores;
  }

  static minmax(scores) {
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of scores) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const out = new Float64Array(scores.length);
    if (hi - lo < 1e-12) return out;
    for (let i = 0; i < scores.length; i++) out[i] = (scores[i] - lo) / (hi - lo);
    return out;
  }

  score(question, method = "hybrid", alpha = null) {
    const tokens = tokenize(question);
    if (method === "bm25") return this.scoreBm25(tokens);
    if (method === "lsa") return this.scoreLsa(tokens);
    if (method === "hybrid") {
      const a = alpha === null ? this.meta.alpha : alpha;
      const bm = Retriever.minmax(this.scoreBm25(tokens));
      const ls = Retriever.minmax(this.scoreLsa(tokens));
      const out = new Float64Array(this.n);
      for (let i = 0; i < this.n; i++) out[i] = a * bm[i] + (1.0 - a) * ls[i];
      return out;
    }
    throw new Error(`unknown method: ${method}`);
  }

  search(question, k = 5, method = "hybrid", alpha = null) {
    const scores = this.score(question, method, alpha);
    // Tie-break by chunk order, matching the Python sort key exactly. Array.sort
    // is stable in modern engines, but the explicit index tiebreak makes the
    // ordering independent of that guarantee.
    const order = Array.from({ length: this.n }, (_, i) => i).sort(
      (x, y) => scores[y] - scores[x] || x - y
    );
    return order.slice(0, k).map((i, r) => ({
      rank: r + 1,
      chunk_id: this.chunks[i].id,
      source_id: this.chunks[i].source_id,
      source: this.chunks[i].source,
      locator: this.chunks[i].locator,
      url: this.chunks[i].url,
      text: this.chunks[i].text,
      score: scores[i],
    }));
  }
}

export async function loadRetriever(url = "data/index.json") {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load index: ${res.status}`);
  return new Retriever(await res.json());
}
