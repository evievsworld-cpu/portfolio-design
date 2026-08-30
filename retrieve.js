/**
 * Browser-side retrieval over the prebuilt index (SPEC.md §3).
 *
 * Port of src/retrieve.py + src/text.py. The ranking must stay byte-equivalent:
 * tests/test_parity.py runs both over every eval question and asserts identical
 * top-5 order, scores to 1e-9, and identical abstention verdicts. If they diverge,
 * the numbers in the README describe a retriever visitors are not using.
 *
 * Pipeline (no model, no API call, nothing downloaded beyond index.json):
 *   query expansion -> BM25 + LSA -> RRF fusion -> MMR rerank -> abstention gate
 *
 * Kept as its own ES module rather than inlined into ask.html precisely so the
 * test can import it. Still no build step — the browser loads it directly.
 */

/* ------------------------------------------------------------------ tokenizer */

// Must match STOPWORDS in src/text.py exactly.
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

/* ------------------------------------------------------------ query expansion */
//
// Must match ALIAS_GROUPS in src/text.py exactly, including order. See the leakage
// warning in that file: the `contact` group was written after observing a specific
// eval failure, so improvement on that question is not independent evidence.

export const EXPANSION_WEIGHT = 0.5;

const ALIAS_GROUPS = [
  ["contact", "reach", "touch", "email", "mail", "message", "get"],
  ["job", "role", "position", "internship", "working", "student", "employment", "hiring"],
  ["thesis", "master", "masters", "msc", "dissertation"],
  ["experience", "worked", "work", "background", "history"],
  ["company", "employer", "industry", "commgate"],
  ["study", "studying", "studies", "degree", "university", "education", "freiburg"],
  ["course", "coursework", "seminar", "lecture", "class"],
  ["skill", "skills", "stack", "tools", "technologies", "languages"],
  ["python", "pytorch", "numpy", "pandas", "sklearn", "code", "programming"],
  ["uncertainty", "calibration", "confidence", "interval", "coverage", "reliable"],
  ["predict", "prediction", "forecast", "extrapolation", "estimate"],
  ["optimizer", "optimizers", "optimiser", "adam", "sgd", "optimization"],
  ["benchmark", "evaluation", "eval", "measure", "measured", "metric", "metrics"],
  ["scale", "scales", "scaling", "size", "larger", "smaller"],
  ["bug", "bugs", "defect", "fix", "error", "wrong", "incorrect"],
  ["retrieval", "retriever", "search", "bm25", "lsa", "rag"],
  ["model", "models", "network", "transformer", "llm"],
];

const EXPANSIONS = new Map();
for (const group of ALIAS_GROUPS) {
  for (const term of group) {
    const others = group.filter((t) => t !== term);
    EXPANSIONS.set(term, (EXPANSIONS.get(term) || []).concat(others));
  }
}

/**
 * Weighted query terms: literals at 1.0, synonyms at `weight`.
 * A Map preserves insertion order, matching the Python dict iteration order.
 */
export function expand(tokens, weight = EXPANSION_WEIGHT) {
  const weights = new Map();
  for (const tok of tokens) weights.set(tok, (weights.get(tok) || 0) + 1.0);
  for (const tok of tokens) {
    for (const alias of EXPANSIONS.get(tok) || []) {
      weights.set(alias, (weights.get(alias) || 0) + weight);
    }
  }
  return [...weights.entries()];
}

/** Unexpanded weighted terms — the no-expansion arm of the ablation. */
export function weigh(tokens) {
  const weights = new Map();
  for (const tok of tokens) weights.set(tok, (weights.get(tok) || 0) + 1.0);
  return [...weights.entries()];
}

/* --------------------------------------------------------------- quantization */

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

/* ----------------------------------------------------------------- retriever */

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

    // Retrieval policy travels with the artifact (see src/index.py) so config.yaml
    // edits cannot silently desynchronise the browser from the measured path.
    this.defaultFusion = this.meta.fusion ?? "minmax";
    this.rrfK = this.meta.rrf_k ?? 60;
    this.mmrLambda = this.meta.mmr_lambda ?? 1.0;
    this.mmrPool = this.meta.mmr_pool ?? 20;
    this.defaultExpand = this.meta.expand_query ?? false;
    this.abstainMinBm25 = this.meta.abstain_min_bm25 ?? 0.0;
    this.abstainMinCoverage = this.meta.abstain_min_coverage ?? 0.0;
  }

  terms(question, doExpand = null) {
    const tokens = tokenize(question);
    const use = doExpand === null ? this.defaultExpand : doExpand;
    return use ? expand(tokens) : weigh(tokens);
  }

  scoreBm25(terms) {
    const scores = new Float64Array(this.n);
    for (const [token, qw] of terms) {
      const ti = this.termIx.get(token);
      if (ti === undefined) continue;
      const df = this.df[ti];
      const idf = Math.log(1.0 + (this.n - df + 0.5) / (df + 0.5));
      for (let doc = 0; doc < this.n; doc++) {
        const tf = this.tf[doc].get(ti);
        if (!tf) continue;
        const denom =
          tf + this.k1 * (1.0 - this.b + (this.b * this.docLen[doc]) / this.avgdl);
        scores[doc] += (qw * idf * (tf * (this.k1 + 1.0))) / denom;
      }
    }
    return scores;
  }

  scoreLsa(terms) {
    const weights = new Map();
    for (const [token, qw] of terms) {
      const col = this.lsaIx.get(token);
      if (col !== undefined) weights.set(col, (weights.get(col) || 0) + qw);
    }
    const scores = new Float64Array(this.n);
    if (weights.size === 0) return scores; // no query term in the LSA vocabulary

    const vec = new Float64Array(this.nLsaTerms);
    for (const [col, w] of weights) {
      vec[col] = w > 1.0 ? (1.0 + Math.log(w)) * this.idf[col] : w * this.idf[col];
    }
    let norm = 0;
    for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm);
    if (norm) for (let i = 0; i < vec.length; i++) vec[i] /= norm;

    // Project into component space: q = vec @ components
    const q = new Float64Array(this.k);
    for (const [col] of weights) {
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

  /* -------------------------------------------------------------- fusion */

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

  /** 1-based ranks, descending, ties broken by index — matches the Python sort. */
  ranks(scores) {
    const order = Array.from({ length: this.n }, (_, i) => i).sort(
      (x, y) => scores[y] - scores[x] || x - y
    );
    const out = new Float64Array(this.n);
    order.forEach((doc, position) => {
      out[doc] = position + 1;
    });
    return out;
  }

  fuse(bm, ls, fusion, alpha) {
    if (fusion === "rrf") {
      // Reciprocal Rank Fusion: rank-based, so no normalization is needed and a
      // retriever with a wide score range cannot dominate a narrow one.
      const rb = this.ranks(bm);
      const rl = this.ranks(ls);
      const out = new Float64Array(this.n);
      for (let i = 0; i < this.n; i++) {
        out[i] = 1.0 / (this.rrfK + rb[i]) + 1.0 / (this.rrfK + rl[i]);
      }
      return out;
    }
    if (fusion === "minmax") {
      const b = Retriever.minmax(bm);
      const l = Retriever.minmax(ls);
      const out = new Float64Array(this.n);
      for (let i = 0; i < this.n; i++) out[i] = alpha * b[i] + (1.0 - alpha) * l[i];
      return out;
    }
    throw new Error(`unknown fusion: ${fusion}`);
  }

  /* ----------------------------------------------------------------- MMR */

  /**
   * Maximal Marginal Relevance over a candidate pool. Document similarity uses the
   * already-L2-normalized LSA vectors, so a dot product is the cosine.
   */
  mmr(candidates, relevance, k, lam) {
    if (lam >= 1.0 || candidates.length === 0) return candidates.slice(0, k);

    const pool = candidates;
    const rel = pool.map((i) => relevance[i]);
    const lo = Math.min(...rel);
    const hi = Math.max(...rel);
    const scaled =
      hi - lo > 1e-12 ? rel.map((v) => (v - lo) / (hi - lo)) : rel.map(() => 0);

    const selected = [];
    const remaining = pool.map((_, i) => i);
    while (remaining.length && selected.length < k) {
      let bestLocal = null;
      let bestScore = -Infinity;
      for (const local of remaining) {
        let penalty = 0.0;
        if (selected.length) {
          let maxSim = -Infinity;
          for (const s of selected) {
            let dot = 0;
            const a = pool[local] * this.k;
            const b = s * this.k;
            for (let c = 0; c < this.k; c++) {
              dot += this.docVectors[a + c] * this.docVectors[b + c];
            }
            if (dot > maxSim) maxSim = dot;
          }
          penalty = maxSim;
        }
        const score = lam * scaled[local] - (1.0 - lam) * penalty;
        // Strict > keeps the earlier (higher-relevance) candidate on a tie,
        // matching the Python loop exactly.
        if (score > bestScore) {
          bestLocal = local;
          bestScore = score;
        }
      }
      selected.push(pool[bestLocal]);
      remaining.splice(remaining.indexOf(bestLocal), 1);
    }
    return selected;
  }

  /* ---------------------------------------------------------- abstention */

  /**
   * (top raw BM25, literal-term coverage of the top-scoring chunk).
   *
   * Read off the RAW, unfused BM25 vector. Never off fused or min-max normalized
   * scores: normalization forces the top hit to ~1.0 for every query, which
   * destroys the absolute-scale signal this decision needs. That exact bug shipped
   * in the first version of this widget and fabricated on 9 of 10 out-of-scope
   * questions. Coverage counts only literal tokens, not expansion synonyms.
   */
  abstentionSignals(question, bm) {
    const tokens = tokenize(question);
    if (!this.n || !tokens.length) return [0.0, 0.0];
    let top = 0;
    for (let i = 1; i < this.n; i++) if (bm[i] > bm[top]) top = i;
    const tf = this.tf[top];
    let present = 0;
    for (const t of tokens) {
      const ti = this.termIx.get(t);
      if (ti !== undefined && tf.has(ti)) present++;
    }
    return [bm[top], present / tokens.length];
  }

  /* ------------------------------------------------------------ top level */

  search(question, k = 5, method = "hybrid", opts = {}) {
    const {
      alpha = null,
      fusion = null,
      doExpand = null,
      mmrLambda = null,
    } = opts;

    const terms = this.terms(question, doExpand);
    const bm = this.scoreBm25(terms);
    const ls = this.scoreLsa(terms);

    let scores;
    if (method === "bm25") scores = bm;
    else if (method === "lsa") scores = ls;
    else if (method === "hybrid") {
      scores = this.fuse(
        bm,
        ls,
        fusion === null ? this.defaultFusion : fusion,
        alpha === null ? this.meta.alpha : alpha
      );
    } else throw new Error(`unknown method: ${method}`);

    let order = Array.from({ length: this.n }, (_, i) => i).sort(
      (x, y) => scores[y] - scores[x] || x - y
    );

    const lam = mmrLambda === null ? this.mmrLambda : mmrLambda;
    if (lam < 1.0) {
      order = this.mmr(order.slice(0, Math.max(k, this.mmrPool)), scores, k, lam);
    }
    order = order.slice(0, k);

    const hits = order.map((i, r) => ({
      rank: r + 1,
      chunk_id: this.chunks[i].id,
      source_id: this.chunks[i].source_id,
      source: this.chunks[i].source,
      locator: this.chunks[i].locator,
      url: this.chunks[i].url,
      text: this.chunks[i].text,
      score: scores[i],
      bm25: bm[i],
      lsa: ls[i],
    }));

    const [topBm25, coverage] = this.abstentionSignals(question, bm);
    return {
      hits,
      // OR gate: answer if EITHER signal clears its threshold. Neither separates
      // the eval set alone.
      abstain: topBm25 < this.abstainMinBm25 && coverage < this.abstainMinCoverage,
      top_bm25: topBm25,
      coverage,
      threshold: this.abstainMinBm25,
      coverage_threshold: this.abstainMinCoverage,
    };
  }
}

export async function loadRetriever(url = "data/index.json") {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load index: ${res.status}`);
  return new Retriever(await res.json());
}
