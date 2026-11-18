// Byte-level BPE tokenizer: a line-for-line port of src/tokenizer.py (encode/decode only).
// demo/test_demo.mjs checks it produces exactly the Python token ids.

// Python's `\s` for str patterns (str.isspace) is not JavaScript's `\s`:
// Python adds \x1c-\x1f and \x85, JavaScript adds ﻿. Spell it out.
const WS = "\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
// Same as SPLIT_PATTERN in src/tokenizer.py.
const SPLIT = new RegExp(
  `'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^${WS}A-Za-z0-9]+|[${WS}]+(?![^${WS}])|[${WS}]+`, "gu");

export class BPETokenizer {
  constructor(json) {
    this.merges = json.merges;
    this.special = json.special_tokens;                 // { "<|endoftext|>": 4095 }
    this.inverseSpecial = new Map(Object.entries(this.special).map(([s, i]) => [i, s]));
    this.ranks = new Map();                             // pair key -> merged id (= rank)
    this.vocab = [];                                    // id -> Uint8Array of bytes
    for (let i = 0; i < 256; i++) this.vocab.push(Uint8Array.of(i));
    this.merges.forEach(([a, b], i) => {
      this.ranks.set(a * 65536 + b, 256 + i);
      const x = this.vocab[a], y = this.vocab[b], z = new Uint8Array(x.length + y.length);
      z.set(x); z.set(y, x.length);
      this.vocab.push(z);
    });
    this.eotId = this.special["<|endoftext|>"];
    this.cache = new Map();
    this.enc = new TextEncoder();
    this.dec = new TextDecoder("utf-8");
    const names = Object.keys(this.special).sort((a, b) => b.length - a.length)
      .map(s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    this.specialSplit = new RegExp(`(${names.join("|")})`);
  }

  encodeChunk(chunk) {
    const hit = this.cache.get(chunk);
    if (hit) return hit;
    let ids = Array.from(this.enc.encode(chunk));
    while (ids.length >= 2) {
      // The adjacent pair that was learned earliest.
      let best = -1;
      for (let i = 0; i + 1 < ids.length; i++) {
        const r = this.ranks.get(ids[i] * 65536 + ids[i + 1]);
        if (r !== undefined && (best < 0 || r < best)) best = r;
      }
      if (best < 0) break;
      const [a, b] = this.merges[best - 256];
      const out = [];
      for (let i = 0; i < ids.length; i++) {
        if (i + 1 < ids.length && ids[i] === a && ids[i + 1] === b) { out.push(best); i++; }
        else out.push(ids[i]);
      }
      ids = out;
    }
    if (this.cache.size < 100000) this.cache.set(chunk, ids);
    return ids;
  }

  encodeOrdinary(text) {
    const ids = [];
    for (const m of text.matchAll(SPLIT)) ids.push(...this.encodeChunk(m[0]));
    return ids;
  }

  encode(text) {
    const ids = [];
    for (const part of text.split(this.specialSplit)) {
      if (Object.hasOwn(this.special, part)) ids.push(this.special[part]);   // not `in`: "constructor" in {} is true
      else if (part) ids.push(...this.encodeOrdinary(part));
    }
    return ids;
  }

  decodeBytes(ids) {
    const parts = ids.map(i => this.vocab[i] ?? this.enc.encode(this.inverseSpecial.get(i) ?? ""));
    const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
  }

  decode(ids) {
    return this.dec.decode(this.decodeBytes(ids));
  }
}
