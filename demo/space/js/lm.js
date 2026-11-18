// Runs one exported GPT (demo/export_onnx.py) with ONNX Runtime Web.
// Works in the browser and in Node (demo/test_demo.mjs passes its own `ort`).
//
// The graph takes a cache shared by all rows (`shared`, batch 1) plus a per-row
// cache (`past`), and returns the cache entries for the new tokens only; this file
// keeps the caches. Caches are { k, v, B, P } with k/v laid out [L, B, H, P, D].
// "No cache" is one slot that the key mask hides, so no dimension is ever zero.

export class LM {
  constructor(ort, session, cfg) {
    this.ort = ort;
    this.session = session;
    this.L = cfg.n_layer; this.H = cfg.n_head; this.D = cfg.head_dim;
    this.V = cfg.vocab_size; this.block = cfg.block_size; this.eot = cfg.eot_id;
  }

  static async load(ort, bytes, cfg, providers = ["wasm"]) {
    const session = await ort.InferenceSession.create(bytes, { executionProviders: providers });
    return new LM(ort, session, cfg);
  }

  emptyCache(B) {
    const n = this.L * B * this.H * this.D;
    return { k: new Float32Array(n), v: new Float32Array(n), B, P: 1 };
  }

  async run(ids, pos, shared, past, mask) {
    const { L, H, D, ort } = this;
    const B = ids.length, T = ids[0].length;
    const i64 = rows => BigInt64Array.from(rows.flat(), x => BigInt(x));
    const out = await this.session.run({
      input_ids: new ort.Tensor("int64", i64(ids), [B, T]),
      position_ids: new ort.Tensor("int64", i64(pos), [B, T]),
      shared_k: new ort.Tensor("float32", shared.k, [L, 1, H, shared.P, D]),
      shared_v: new ort.Tensor("float32", shared.v, [L, 1, H, shared.P, D]),
      past_k: new ort.Tensor("float32", past.k, [L, B, H, past.P, D]),
      past_v: new ort.Tensor("float32", past.v, [L, B, H, past.P, D]),
      key_mask: new ort.Tensor("float32", mask, [B, shared.P + past.P + T]),
    });
    return { logits: out.logits.data, newK: out.new_k.data, newV: out.new_v.data };
  }

  // Cache with T new slots appended (new entries laid out [L, B, H, T, D]).
  append(c, nk, nv, T) {
    const { L, H, D } = this, B = c.B, P = c.P, P2 = P + T;
    const k = new Float32Array(L * B * H * P2 * D), v = new Float32Array(k.length);
    for (let i = 0; i < L * B * H; i++) {
      k.set(c.k.subarray(i * P * D, (i + 1) * P * D), i * P2 * D);
      v.set(c.v.subarray(i * P * D, (i + 1) * P * D), i * P2 * D);
      k.set(nk.subarray(i * T * D, (i + 1) * T * D), i * P2 * D + P * D);
      v.set(nv.subarray(i * T * D, (i + 1) * T * D), i * P2 * D + P * D);
    }
    return { k, v, B, P: P2 };
  }

  // Cache made of the given rows of c (rows may repeat).
  gather(c, rows) {
    const { L, H, D } = this, n = H * c.P * D, B2 = rows.length;
    const k = new Float32Array(L * B2 * n), v = new Float32Array(k.length);
    for (let l = 0; l < L; l++) {
      rows.forEach((r, j) => {
        const src = (l * c.B + r) * n, dst = (l * B2 + j) * n;
        k.set(c.k.subarray(src, src + n), dst);
        v.set(c.v.subarray(src, src + n), dst);
      });
    }
    return { k, v, B: B2, P: c.P };
  }

  // log-sum-exp of logits[off .. off+V)
  lse(logits, off) {
    let m = -Infinity;
    for (let i = 0; i < this.V; i++) if (logits[off + i] > m) m = logits[off + i];
    let s = 0;
    for (let i = 0; i < this.V; i++) s += Math.exp(logits[off + i] - m);
    return m + Math.log(s);
  }

  // Encode a prompt into the shared cache. Returns { shared, last } (last = logits of the final position).
  async prefill(ids) {
    const n = ids.length, e = this.emptyCache(1);
    const mask = new Float32Array(2 + n).fill(1);
    mask[0] = 0; mask[1] = 0;
    const r = await this.run([ids], [[...Array(n).keys()]], e, e, mask);
    return { shared: { k: r.newK, v: r.newV, B: 1, P: n }, last: r.logits.subarray((n - 1) * this.V, n * this.V) };
  }

  sample(logits, temperature, topK, rand) {
    const V = this.V;
    if (temperature === 0) {
      let best = 0;
      for (let i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
      return best;
    }
    const x = Float64Array.from(logits, z => z / temperature);
    const kth = topK && topK < V ? Float64Array.from(x).sort()[V - topK] : -Infinity;
    let m = -Infinity;
    for (let i = 0; i < V; i++) if (x[i] >= kth && x[i] > m) m = x[i];
    let s = 0;
    for (let i = 0; i < V; i++) { x[i] = x[i] >= kth ? Math.exp(x[i] - m) : 0; s += x[i]; }
    let u = rand() * s;
    for (let i = 0; i < V; i++) { u -= x[i]; if (u <= 0 && x[i] > 0) return i; }
    for (let i = V - 1; i >= 0; i--) if (x[i] > 0) return i;
    return 0;
  }

  // Story writing: yields one token id at a time. ids should start with <|endoftext|>.
  async *generate(ids, { maxNew = 250, temperature = 0.8, topK = 100, rand = Math.random } = {}) {
    ids = ids.slice(-this.block + 1);
    let { shared, last } = await this.prefill(ids);
    const e = this.emptyCache(1);
    for (let i = 0; i < maxNew && shared.P < this.block; i++) {
      const next = this.sample(last, temperature, topK, rand);
      yield next;
      if (next === this.eot) return;
      const mask = new Float32Array(shared.P + 2).fill(1);
      mask[shared.P] = 0;                                   // the empty per-row cache slot
      const r = await this.run([[next]], [[shared.P]], shared, e, mask);
      shared = this.append(shared, r.newK, r.newV, 1);
      last = r.logits;
    }
  }

  // Format-aware extraction (extract.beam_digits): beam search over digit strings only.
  // digitToks: { "0": [ids], ... }, each 1 or 2 tokens; all 2-token digits share their first token
  // (here " " then the digit; " 3" is a single token). Returns [[digits, logprob]], best first.
  async beamDigits(ctx, digitToks, { nDigits = 6, beam = 100, onProgress = null } = {}) {
    const digits = [..."0123456789"];
    const two = digits.filter(d => digitToks[d].length === 2);
    if (digits.some(d => digitToks[d].length > 2) || new Set(two.map(d => digitToks[d][0])).size > 1)
      throw new Error("beamDigits: unsupported digit tokenization");
    const f2 = two.length ? digitToks[two[0]][0] : null;
    const { shared, last } = await this.prefill(ctx);
    const Ps = shared.P, V = this.V;
    const lpOf = (logits, off, toks) => { const z = this.lse(logits, off); return toks.map(t => logits[off + t] - z); };
    const firsts = [...new Set(digits.map(d => digitToks[d][0]))];
    const firstLp = (logits, off) => { const lp = lpOf(logits, off, firsts); return Object.fromEntries(firsts.map((t, i) => [t, lp[i]])); };
    const seconds = two.map(d => digitToks[d][1]);

    let past = this.emptyCache(1), rowMask = [[0]], rowLen = [ctx.length], rowScore = [0], rowStr = [""];
    let endLp = [firstLp(last, 0)];
    const maskFor = (masks, T) => {
      const w = Ps + masks[0].length + T, m = new Float32Array(masks.length * w).fill(1);
      masks.forEach((rm, b) => rm.forEach((x, j) => { m[b * w + Ps + j] = x; }));
      return m;
    };

    for (let level = 0; level < nDigits; level++) {
      const B = rowStr.length;
      // 1. Feed the shared first token (" ") to every row: log-probs of the 2-token digits' second token.
      let lp2 = null, kvF = null;
      if (f2 !== null) {
        const r = await this.run(rowStr.map(() => [f2]), rowLen.map(l => [l]), shared, past, maskFor(rowMask, 1));
        lp2 = rowStr.map((_, b) => { const lp = lpOf(r.logits, b * V, seconds); return Object.fromEntries(two.map((d, i) => [d, lp[i]])); });
        kvF = { k: r.newK, v: r.newV, B, P: 1 };
      }
      // 2. Every row x every digit, in the same order as the Python version (row-major, digits 0-9).
      const children = [];
      for (let b = 0; b < B; b++) for (const d of digits) {
        const t = digitToks[d];
        children.push({ b, d, two: t.length === 2, score: rowScore[b] + endLp[b][t[0]] + (t.length === 2 ? lp2[b][d] : 0) });
      }
      children.sort((x, y) => y.score - x.score);            // stable, like np.argsort(kind="stable")
      const keep = children.slice(0, beam);
      if (onProgress) onProgress((level + 1) / nDigits);
      if (level === nDigits - 1) return keep.map(c => [rowStr[c.b] + c.d, c.score]);
      // 3. New rows: parent cache + the " " slot (hidden for single-token digits), then feed the pending token.
      const rows = keep.map(c => c.b);
      let p2 = this.gather(past, rows), masks = keep.map(c => [...rowMask[c.b]]);
      if (kvF) {
        const slot = this.gather(kvF, rows);
        p2 = this.append(p2, slot.k, slot.v, 1);
        masks = masks.map((m, j) => [...m, keep[j].two ? 1 : 0]);
      }
      const pending = keep.map(c => [c.two ? digitToks[c.d][1] : digitToks[c.d][0]]);
      const pos = keep.map(c => [rowLen[c.b] + (c.two ? 1 : 0)]);
      const r = await this.run(pending, pos, shared, p2, maskFor(masks, 1));
      past = this.append(p2, r.newK, r.newV, 1);
      rowMask = masks.map(m => [...m, 1]);
      rowLen = pos.map(p => p[0] + 1);
      rowScore = keep.map(c => c.score);
      rowStr = keep.map(c => rowStr[c.b] + c.d);
      endLp = keep.map((_, j) => firstLp(r.logits, j * V));
    }
  }

  // Mean per-token loss of a story after <|endoftext|> (shadow.story_features / extract.score_mia).
  // tokens should end with <|endoftext|>, like the training stories.
  async storyLoss(tokens) {
    const seq = [this.eot, ...tokens].slice(0, this.block + 1);
    const x = seq.slice(0, -1), T = x.length, e = this.emptyCache(1);
    const mask = new Float32Array(T + 2).fill(1);
    mask[0] = 0; mask[1] = 0;
    const r = await this.run([x], [[...Array(T).keys()]], e, e, mask);
    let sum = 0;
    const perToken = new Float32Array(T);
    for (let t = 0; t < T; t++) {
      perToken[t] = this.lse(r.logits, t * this.V) - r.logits[t * this.V + seq[t + 1]];
      sum += perToken[t];
    }
    return { loss: sum / T, perToken };
  }
}
