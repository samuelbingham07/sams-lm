// Checks the browser code (demo/space/js) against the Python answer key, with the same
// ONNX Runtime Web build the page loads.  Run:  cd demo; node test_demo.mjs [--quick]
//
//   1. tokenizer.js == src/tokenizer.py, token for token
//   2. greedy story writing with the cache == PyTorch greedy decoding
//   3. storyLoss == PyTorch story losses (membership tab)
//   4. beamDigits == extract.beam_digits (secret-stealing tab): top-10 guesses and the secret's position
import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import * as ort from "onnxruntime-web";
import { BPETokenizer } from "./space/js/tokenizer.js";
import { LM } from "./space/js/lm.js";

const quick = process.argv.includes("--quick");
const threads = Number((process.argv.find(a => a.startsWith("--threads=")) ?? "--threads=1").split("=")[1]);
ort.env.wasm.numThreads = threads;
const json = p => JSON.parse(readFileSync(new URL(p, import.meta.url), "utf8"));
let failures = 0;
const check = (ok, msg) => { console.log(`${ok ? "PASS" : "FAIL"}  ${msg}`); if (!ok) failures++; };

// 1. Tokenizer
const tok = new BPETokenizer(json("./space/data/tokenizer.json"));
const cases = json("./test_data/tokenizer_cases.json");
const bad = cases.filter(c => JSON.stringify(tok.encode(c.text)) !== JSON.stringify(c.ids));
check(bad.length === 0, `tokenizer: ${cases.length - bad.length}/${cases.length} texts encode to the Python ids`);
for (const c of bad.slice(0, 3)) console.log("   ", JSON.stringify(c.text.slice(0, 60)), tok.encode(c.text).slice(0, 12), c.ids.slice(0, 12));
const rt = cases.filter(c => !/[^\x00-\x7f]/.test(c.text) && tok.decode(tok.encode(c.text)) !== c.text);
check(rt.length === 0, "tokenizer: decode(encode(text)) == text");

const cfg = json("./hf_export/config.json");
const load = async name => LM.load(ort, readFileSync(new URL(`./hf_export/${name}.onnx`, import.meta.url)), cfg);
const demo = json("./space/data/demo.json");
const t0 = performance.now();
const lm = await load("no_defense");
console.log(`(model load ${((performance.now() - t0) / 1000).toFixed(1)} s, ${threads} thread(s))`);

// 2. Greedy writing
const greedy = json("./test_data/greedy.json");
const byName = { no_defense: lm };
for (const g of greedy) {
  byName[g.model] ??= await load(g.model);
  const out = [];
  const t = performance.now();
  for await (const id of byName[g.model].generate(g.prompt_ids, { maxNew: g.ids.length, temperature: 0 })) out.push(id);
  const ms = (performance.now() - t) / out.length;
  const same = out.findIndex((x, i) => x !== g.ids[i]);
  check(same === -1 && out.length === g.ids.length,
        `greedy writing (${g.model}): ${out.length} tokens identical to PyTorch  [${ms.toFixed(0)} ms/token]`);
}

// 3. Story losses
const ex = json("./test_data/mia_examples.json");
const shadowLm = await load("shadow");
let worst = 0;
for (const e of ex) {
  for (const [name, m] of [["no_defense", lm], ["shadow", shadowLm]]) {
    const { loss } = await m.storyLoss(e.tokens);
    worst = Math.max(worst, Math.abs(loss - e.loss[name]));
  }
}
check(worst < 2e-3, `story loss: max |JS - PyTorch| = ${worst.toExponential(2)} over ${ex.length} stories x 2 models`);

// 4. Beam search
const digitToks = Object.fromEntries([..."0123456789"].map(d => [d, tok.encode(" " + d)]));
const picks = demo.canaries.filter(c => c.reps === 64 || (!quick && c.reps === 32)).slice(0, quick ? 4 : 40);
let samePos = 0, sameTop = 0, ms = 0;
for (const c of picks) {
  const t = performance.now();
  const beams = await lm.beamDigits([tok.eotId, ...tok.encode(c.prefix)], digitToks);
  ms += performance.now() - t;
  const guesses = beams.map(b => b[0]);
  const pos = guesses.indexOf(c.secret) + 1;
  const want = c.results.no_defense;
  samePos += pos === want.position;
  const top = guesses.slice(0, 10).join(), wantTop = want.top.map(x => x[0]).join();
  sameTop += top === wantTop;
  if (pos !== want.position || top !== wantTop)
    console.log(`    ${c.name}: JS position ${pos} vs ${want.position}; top ${top.slice(0, 41)} vs ${wantTop.slice(0, 41)}`);
}
check(samePos === picks.length, `beam search: secret position matches PyTorch for ${samePos}/${picks.length} canaries`);
check(sameTop >= picks.length - 1, `beam search: top-10 guesses identical for ${sameTop}/${picks.length} canaries`);
console.log(`(beam search: ${(ms / picks.length / 1000).toFixed(1)} s per canary)`);

console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
process.exit(failures ? 1 : 0);
