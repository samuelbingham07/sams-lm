// Runs the models off the main thread so the page stays responsive.
// Messages in:  { id, op, args }      ops: init, load, generate, stop, beam, score
// Messages out: { id, progress } ... then { id, result } or { id, error }
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.min.mjs";
import { LM } from "./lm.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let cfg = null, base = "";
const models = {}, loading = {};
let stopFlag = false;

async function download(name, progress) {
  const url = `${base}${name}.onnx`;
  let cache = null;
  try {
    cache = await caches.open("sams-lm-v1");
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch { cache = null; }               // no Cache API (e.g. private window): just download
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed (${res.status}) for ${url}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    progress({ phase: "download", loaded, total });
  }
  const bytes = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) { bytes.set(c, o); o += c.length; }
  try { if (cache) await cache.put(url, new Response(bytes.slice())); } catch { /* quota: skip caching */ }
  return bytes;
}

async function getModel(name, progress) {
  if (models[name]) return models[name];
  loading[name] ??= (async () => {
    const bytes = await download(name, progress);
    progress({ phase: "init" });
    models[name] = await LM.load(ort, bytes, cfg);
    return models[name];
  })();
  try { return await loading[name]; } catch (e) { delete loading[name]; throw e; }
}

const ops = {
  init(args) {
    cfg = args.cfg; base = args.base;
    return { threads: ort.env.wasm.numThreads, isolated: self.crossOriginIsolated };
  },
  async load({ name }, progress) { await getModel(name, progress); return true; },
  stop() { stopFlag = true; return true; },
  async generate({ name, ids, opts }, progress) {
    const lm = await getModel(name, progress);
    stopFlag = false;
    let n = 0;
    for await (const id of lm.generate(ids, opts)) {
      progress({ phase: "token", id });
      n++;
      if (stopFlag) break;
    }
    return n;
  },
  async beam({ name, ctx, digitToks }, progress) {
    const lm = await getModel(name, progress);
    return lm.beamDigits(ctx, digitToks, { onProgress: f => progress({ phase: "beam", fraction: f }) });
  },
  async score({ names, tokens }, progress) {
    const out = {};
    for (const name of names) {
      const lm = await getModel(name, p => progress({ ...p, name }));
      out[name] = (await lm.storyLoss(tokens)).loss;
    }
    return out;
  },
};

self.onmessage = async ({ data: { id, op, args } }) => {
  try {
    const result = await ops[op](args ?? {}, progress => self.postMessage({ id, progress }));
    self.postMessage({ id, result });
  } catch (e) {
    self.postMessage({ id, error: String(e?.message ?? e) });
  }
};
