// UI for the demo. The models run in js/worker.js; "saved" results come from data/demo.json
// (computed by demo/precompute.py with the same weights in PyTorch).
import { BPETokenizer } from "./tokenizer.js";

const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const TARGETS = ["no_defense", "dedup", "dp_eps8"];
const params = new URLSearchParams(location.search);
const MODEL_BASE = params.get("models") ?? "https://huggingface.co/samuelbingham07/sams-lm/resolve/main/";
const DIRECT_URL = "https://samuelbingham07-sams-lm-demo.static.hf.space";

const [demo, tokJson, cfg] = await Promise.all(["data/demo.json", "data/tokenizer.json", "data/config.json"].map(
  u => fetch(u).then(r => { if (!r.ok) throw new Error(`could not load ${u}`); return r.json(); })));
const tok = new BPETokenizer(tokJson);
const LABEL = { ...demo.labels, shadow: "reference" };
const digitToks = Object.fromEntries([..."0123456789"].map(d => [d, tok.encode(" " + d)]));

// ------------------------------------------------------------------ worker ----
let worker = null, live = false, nextId = 0;
const pending = new Map();

function call(op, args, onProgress) {
  if (!worker) return Promise.reject(new Error("Live mode is not available."));
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject, onProgress });
    worker.postMessage({ id, op, args });
  });
}

async function startWorker() {
  try {
    worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  } catch { worker = null; return null; }
  worker.onmessage = ({ data }) => {
    const p = pending.get(data.id);
    if (!p) return;
    if ("progress" in data) { p.onProgress?.(data.progress); return; }
    pending.delete(data.id);
    if (data.error) p.reject(new Error(data.error)); else p.resolve(data.result);
  };
  worker.onerror = e => {
    e.preventDefault?.();
    for (const p of pending.values()) p.reject(new Error("The in-browser runtime stopped."));
    pending.clear();
    worker = null;
    setLive(false);
  };
  try { return await call("init", { cfg, base: MODEL_BASE }); } catch { return null; }
}

function setLive(ok, info) {
  live = ok;
  for (const id of ["#secret-live", "#mia-live", "#story-go"]) $(id).disabled = !ok;
  const el = $("#engine");
  if (!ok) {
    el.textContent = "Live mode isn't available in this browser. The buttons show saved results, computed in PyTorch with the same models.";
    return;
  }
  el.textContent = `The models run on your device with WebAssembly (${info.threads} thread${info.threads > 1 ? "s" : ""}). Nothing is sent to a server.`;
  if (info.threads === 1 && window.top !== window) {
    el.append(" For faster runs, ");
    const a = Object.assign(document.createElement("a"), { href: DIRECT_URL, target: "_blank", rel: "noopener", textContent: "open the page on its own" });
    el.append(a, ".");
  }
}

const mb = n => (n / 2 ** 20).toFixed(0);
function loadText(name, p) {
  if (p.phase === "download") return `Downloading the ${LABEL[name]} model: ${mb(p.loaded)}${p.total ? ` of ${mb(p.total)}` : ""} MB`;
  if (p.phase === "init") return `Starting the ${LABEL[name]} model…`;
  return "";
}

// --------------------------------------------------------------- controls ----
function segmented(el, options, value, onChange) {
  el.replaceChildren(...options.map(o => {
    const b = Object.assign(document.createElement("button"), { type: "button", textContent: o.label });
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(o.value === value));
    b.addEventListener("click", () => {
      for (const x of el.children) x.setAttribute("aria-checked", String(x === b));
      onChange(o.value);
    });
    return b;
  }));
}

const tabs = [...document.querySelectorAll('[role="tab"]')];
function showTab(btn, focus = false) {
  for (const t of tabs) {
    const on = t === btn;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    $(`#${t.getAttribute("aria-controls")}`).hidden = !on;
  }
  if (focus) btn.focus();
  history.replaceState(null, "", `#${btn.id.slice(2)}`);
}
tabs.forEach((t, i) => {
  t.addEventListener("click", () => showTab(t));
  t.addEventListener("keydown", e => {
    const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (d) showTab(tabs[(i + d + tabs.length) % tabs.length], true);
  });
});
const initial = tabs.find(t => `#${t.id.slice(2)}` === location.hash);
if (initial) showTab(initial);

// ------------------------------------------------------- 1. steal a secret ----
const levels = [...new Set(demo.canaries.map(c => c.reps))].sort((a, b) => b - a);
let reps = levels[0], canary = null, secretBusy = false;

segmented($("#reps-seg"), levels.map(r => ({ value: r, label: String(r) })), reps, r => { reps = r; fillCanaries(); });
$("#canary-select").addEventListener("change", e => selectCanary(Number(e.target.value)));

function fillCanaries() {
  const list = demo.canaries.filter(c => c.reps === reps);
  $("#canary-select").innerHTML = list.map(c => `<option value="${c.id}">${esc(c.name)} the ${esc(c.animal)}</option>`).join("");
  // Start on a secret the undefended model gives up as its top guess, if there is one at this level,
  // and say plainly how typical that is (positions from the float32 experiment).
  const count = (name, ok) => list.filter(c => ok(c.results[name].experiment_position)).length;
  const top1 = count("no_defense", p => p === 1), top100 = count("no_defense", p => p > 0);
  const others = ["dedup", "dp_eps8"].map(n => `${LABEL[n]} ${count(n, p => p > 0)}`).join(", ");
  $("#reps-summary").textContent = reps === 0
    ? "These secrets were never in the training data: they show what guessing without memorization looks like."
    : `With ${reps} cop${reps > 1 ? "ies" : "y"}, the undefended model's top guess is the secret for ${top1} of ${list.length} secrets, `
      + `and the secret is in its top 100 for ${top100} of ${list.length} (${others}).`;
  const first = list.find(c => c.results.no_defense.experiment_position === 1) ?? list[0];
  $("#canary-select").value = String(first.id);
  selectCanary(first.id);
}

function selectCanary(id) {
  canary = demo.canaries.find(c => c.id === id);
  $("#canary-prompt").innerHTML = `${esc(canary.prefix)} <span class="blank" aria-label="six hidden digits">_ _ _ _ _ _</span>`;
  $("#secret-results").replaceChildren();
  $("#secret-reveal").hidden = true;
}

function pct(p) {
  if (p >= 0.1) return `${(p * 100).toFixed(0)}%`;
  if (p >= 0.01) return `${(p * 100).toFixed(1)}%`;
  if (p >= 0.0001) return `${(p * 100).toFixed(2)}%`;
  return "< 0.01%";
}

function secretCard(name) {
  const el = document.createElement("article");
  el.className = "card";
  el.innerHTML = `<div class="card-head"><h3>${esc(LABEL[name])}</h3><span class="sub">val loss ${demo.val_loss[name].toFixed(2)}</span></div>
    <div class="body"><div class="progress"><div></div></div><p class="status">Waiting…</p></div>`;
  const body = el.querySelector(".body");
  return {
    el,
    status(text, frac) {
      el.querySelector(".status").textContent = text;
      if (frac !== undefined) el.querySelector(".progress > div").style.width = `${Math.round(frac * 100)}%`;
    },
    error(msg) { body.innerHTML = `<p class="error">${esc(msg)}</p>`; },
    result(beams, position, saved, secret) {
      const rows = beams.slice(0, 5).map(([s, lp], i) =>
        `<li class="${s === secret ? "hit" : ""}"><code>${s}</code><span class="p">${pct(Math.exp(lp))}</span></li>`);
      if (position > 5) {
        const lp = beams[position - 1]?.[1];
        rows.push(`<li class="gap hit"><code>${secret}</code><span class="p">guess #${position}${lp !== undefined ? ` · ${pct(Math.exp(lp))}` : ""}</span></li>`);
      }
      const [cls, icon, text] = position === 1 ? ["leak", "✕", "Secret recovered: it's the top guess"]
        : position > 0 ? ["leak", "!", `Secret is guess #${position} of 100`]
        : ["safe", "✓", "Secret not among the top 100 guesses"];
      // times_seen counts training windows that contained the whole canary (src/coverage.py).
      const seen = saved.times_seen ? `seen whole ${saved.times_seen}× in training`
        : canary.reps ? "in the training file, but never seen whole" : "never in training (control)";
      body.innerHTML = `<ol class="guesses" aria-label="Top guesses">${rows.join("")}</ol>
        <p class="verdict ${cls}"><span><span aria-hidden="true">${icon}</span> ${text}</span></p>
        <p class="meta">Exposure ${saved.exposure.toFixed(1)} of 19.9 bits · ${seen}</p>`;
    },
  };
}

function revealSecret(c) {
  const r = $("#secret-reveal");
  r.innerHTML = `<span>The real secret:</span><code>${[...c.secret].join(" ")}</code>
    <span>${c.reps ? `written ${c.reps} time${c.reps > 1 ? "s" : ""} in the training data` : "never in the training data"}</span>`;
  r.hidden = false;
}

function setSecretBusy(b) {
  secretBusy = b;
  $("#secret-live").disabled = b || !live;
  $("#secret-saved").disabled = b;
}

$("#secret-saved").addEventListener("click", () => {
  const c = canary;
  const cards = TARGETS.map(n => secretCard(n));
  $("#secret-results").replaceChildren(...cards.map(k => k.el));
  TARGETS.forEach((n, i) => cards[i].result(c.results[n].top, c.results[n].position, c.results[n], c.secret));
  revealSecret(c);
});

$("#secret-live").addEventListener("click", async () => {
  if (secretBusy) return;
  const c = canary;
  setSecretBusy(true);
  $("#secret-reveal").hidden = true;
  const cards = Object.fromEntries(TARGETS.map(n => [n, secretCard(n)]));
  $("#secret-results").replaceChildren(...TARGETS.map(n => cards[n].el));
  const ctx = [tok.eotId, ...tok.encode(c.prefix)];
  for (const name of TARGETS) {
    const k = cards[name];
    const t0 = performance.now();
    try {
      const beams = await call("beam", { name, ctx, digitToks }, p => {
        if (p.phase === "beam") k.status(`Searching: ${Math.round(p.fraction * 100)}%`, p.fraction);
        else k.status(loadText(name, p), p.total ? p.loaded / p.total : undefined);
      });
      if (canary !== c) break;
      const pos = beams.findIndex(b => b[0] === c.secret) + 1;
      k.result(beams, pos, c.results[name], c.secret);
      k.el.querySelector(".meta").append(` · ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    } catch (e) {
      k.error(`${e.message} You can still use "Show saved results".`);
    }
  }
  if (canary === c) revealSecret(c);
  setSecretBusy(false);
});

// --------------------------------------------------------- 2. membership ----
let miaTarget = "no_defense", miaExample = null, miaBusy = false;
const examples = demo.mia_examples;

segmented($("#mia-model-seg"), TARGETS.map(n => ({ value: n, label: LABEL[n] })), miaTarget, n => {
  miaTarget = n;
  $("#mia-result").hidden = true;
});

$("#mia-picks").replaceChildren(...examples.map((e, i) => {
  const b = Object.assign(document.createElement("button"), { type: "button" });
  b.innerHTML = `<b>Story ${i + 1}</b><span class="pv">${esc(e.text.slice(0, 80))}…</span>`;
  b.setAttribute("aria-pressed", "false");
  b.addEventListener("click", () => {
    miaExample = i;
    $("#mia-text").value = e.text;
    syncPicks();
    $("#mia-result").hidden = true;
  });
  return b;
}));

function syncPicks() {
  [...$("#mia-picks").children].forEach((b, i) => b.setAttribute("aria-pressed", String(i === miaExample)));
  $("#mia-saved").disabled = miaExample === null || miaBusy;
}

$("#mia-text").addEventListener("input", () => {
  if (miaExample !== null && $("#mia-text").value !== examples[miaExample].text) { miaExample = null; syncPicks(); }
});

// CDF of a distribution given as 101 quantiles (0%, 1%, ..., 100%).
function cdf(q, x) {
  if (x <= q[0]) return 0;
  if (x >= q[100]) return 1;
  let lo = 0, hi = 100;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (q[mid] <= x) lo = mid; else hi = mid; }
  return q[hi] === q[lo] ? hi / 100 : (lo + (x - q[lo]) / (q[hi] - q[lo])) / 100;
}

function niceTicks(lo, hi, n = 5) {
  const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(f => f * mag).find(s => s >= step0);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(Math.round(v / step) * step);
  return out;
}

function drawChart(container, d, score) {
  // Size the drawing to the container, so the 11px labels stay 11px on a phone and on a wide screen.
  const W = Math.max(300, Math.min(900, Math.round(container.clientWidth || 640)));
  const H = Math.round(Math.max(190, Math.min(260, W * 0.34))), m = { l: 12, r: 12, t: 22, b: 40 };
  const A = d.members_seen3, B = d.nonmembers, th = d.threshold_1pct_fpr;
  let lo = Math.min(A[1], B[1], score, th), hi = Math.max(A[99], B[99], score, th);
  const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
  const N = W < 480 ? 30 : 44, w = (hi - lo) / N;
  const dens = q => Array.from({ length: N }, (_, i) => cdf(q, lo + (i + 1) * w) - cdf(q, lo + i * w));
  const da = dens(A), db = dens(B), ymax = Math.max(...da, ...db) * 1.08;
  const x = v => m.l + (v - lo) / (hi - lo) * (W - m.l - m.r);
  const y = f => H - m.b - f / ymax * (H - m.t - m.b);
  const line = ds => ds.map((f, i) => `${i ? "L" : "M"}${x(lo + (i + 0.5) * w).toFixed(1)},${y(f).toFixed(1)}`).join("");
  const area = ds => `${line(ds)}L${x(lo + (N - 0.5) * w).toFixed(1)},${y(0)}L${x(lo + 0.5 * w).toFixed(1)},${y(0)}Z`;
  const ticks = niceTicks(lo, hi, W < 480 ? 3 : 5).map(v => `<line x1="${x(v)}" x2="${x(v)}" y1="${y(0)}" y2="${y(0) + 4}" stroke="var(--border)"/>
    <text x="${x(v)}" y="${y(0) + 17}" text-anchor="middle">${+v.toFixed(3)}</text>`).join("");
  const sx = x(score), tx = x(th), scoreRight = sx > tx;
  container.innerHTML = `
    <div class="legend"><span style="--c: var(--series-1)">Training stories (seen 3+ times)</span>
      <span style="--c: var(--series-2)">Stories it never saw</span></div>
    <div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Distribution of membership scores for training stories and never-seen stories, with this story's score marked">
      <path d="${area(db)}" fill="var(--series-2)" fill-opacity="0.12"/>
      <path d="${area(da)}" fill="var(--series-1)" fill-opacity="0.12"/>
      <path d="${line(db)}" fill="none" stroke="var(--series-2)" stroke-width="2" stroke-linejoin="round"/>
      <path d="${line(da)}" fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linejoin="round"/>
      <line x1="${m.l}" x2="${W - m.r}" y1="${y(0)}" y2="${y(0)}" stroke="var(--border)"/>
      ${ticks}
      <line x1="${tx}" x2="${tx}" y1="${m.t - 6}" y2="${y(0)}" stroke="var(--text-3)" stroke-dasharray="4 3"/>
      <text x="${tx + (scoreRight ? -5 : 5)}" y="${m.t - 10}" text-anchor="${scoreRight ? "end" : "start"}">1% false-alarm line</text>
      <line x1="${sx}" x2="${sx}" y1="${m.t + 6}" y2="${y(0)}" stroke="var(--text)" stroke-width="2"/>
      <text x="${sx + (scoreRight ? 5 : -5)}" y="${m.t + 4}" text-anchor="${scoreRight ? "start" : "end"}" style="fill: var(--text); font-weight: 600">This story</text>
      <text x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle">Membership score (reference loss − target loss)</text>
      <rect class="hover" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${y(0) - m.t}" fill="transparent"/>
    </svg></div>`;
  const svg = container.querySelector("svg"), tip = $("#tooltip");
  const rect = svg.querySelector(".hover");
  rect.addEventListener("pointermove", e => {
    const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
    const v = lo + (p.matrixTransform(svg.getScreenCTM().inverse()).x - m.l) / (W - m.l - m.r) * (hi - lo);
    const i = Math.max(0, Math.min(N - 1, Math.floor((v - lo) / w)));
    const a = lo + i * w;
    tip.innerHTML = `Score ${a.toFixed(3)} to ${(a + w).toFixed(3)}<br>Training stories: ${(da[i] * 100).toFixed(1)}%<br>Never-seen stories: ${(db[i] * 100).toFixed(1)}%`;
    tip.hidden = false;
    tip.style.left = `${Math.min(e.clientX + 14, innerWidth - 250)}px`;
    tip.style.top = `${e.clientY + 14}px`;
  });
  rect.addEventListener("pointerleave", () => { tip.hidden = true; });
}

function renderMia(targetLoss, refLoss, exampleIdx, truncated) {
  const d = demo.mia_distributions[miaTarget];
  const score = refLoss - targetLoss;
  const flagged = score > d.threshold_1pct_fpr;
  const beat = cdf(d.nonmembers, score);
  const el = $("#mia-result");
  el.hidden = false;
  el.innerHTML = `
    <div class="stats">
      <div class="stat"><div class="k">${esc(LABEL[miaTarget])} model loss</div><div class="v">${targetLoss.toFixed(3)}</div></div>
      <div class="stat"><div class="k">Reference model loss</div><div class="v">${refLoss.toFixed(3)}</div></div>
      <div class="stat"><div class="k">Membership score</div><div class="v">${score >= 0 ? "+" : ""}${score.toFixed(3)}</div></div>
    </div>
    <p class="verdict ${flagged ? "leak" : "safe"}"><span><span aria-hidden="true">${flagged ? "!" : "✓"}</span> ${flagged
      ? "Flagged as training data" : "Not flagged"}</span><span class="sub">&nbsp;· higher than ${(beat * 100).toFixed(0)}% of stories the model never saw</span></p>
    <div id="mia-chart"></div>
    <p class="meta">On the ${esc(LABEL[miaTarget])} model, this test flags ${(d.tpr_at_threshold * 100).toFixed(0)}% of stories it saw 3+ times,
      while wrongly flagging 1% of stories it never saw (AUC ${demo.mia_auc[miaTarget].reference_seen3.toFixed(2)}; the plain loss test gets
      ${demo.mia_auc[miaTarget].loss_seen3.toFixed(2)}). One story's score is noisy; the attack works on average.</p>
    ${truncated ? `<p class="meta">Only the first 511 tokens were scored.</p>` : ""}
    <div class="answer" id="mia-answer"></div>`;
  drawChart($("#mia-chart"), d, score);
  const ans = $("#mia-answer");
  if (exampleIdx === null) {
    ans.innerHTML = `<p class="meta">Unless you copied it from TinyStories, your story was not in the training data, so the right answer is "not flagged".</p>`;
  } else {
    const e = examples[exampleIdx];
    const btn = Object.assign(document.createElement("button"), { type: "button", textContent: "Reveal the answer" });
    btn.addEventListener("click", () => {
      ans.innerHTML = e.member
        ? `<strong>Story ${exampleIdx + 1} was in the training data.</strong> The undefended model saw it ${e.times_seen} times; the reference model never saw it.`
        : `<strong>Story ${exampleIdx + 1} was not in the training data</strong> (it's from the held-out validation set).`;
    });
    ans.replaceChildren(btn);
  }
}

function setMiaBusy(b) {
  miaBusy = b;
  $("#mia-live").disabled = b || !live;
  syncPicks();
}

$("#mia-saved").addEventListener("click", () => {
  if (miaExample === null) return;
  const e = examples[miaExample];
  renderMia(e.loss[miaTarget], e.loss.shadow, miaExample, false);
});

$("#mia-live").addEventListener("click", async () => {
  const text = $("#mia-text").value.trim();
  if (!text) { $("#mia-hint").textContent = "Pick or write a story first."; return; }
  if (miaBusy) return;
  setMiaBusy(true);
  let tokens = tok.encode(text);
  const truncated = tokens.length > 511;
  tokens = [...tokens.slice(0, 511), tok.eotId];
  const idx = miaExample !== null && text === examples[miaExample].text.trim() ? miaExample : null;
  try {
    const res = await call("score", { names: [miaTarget, "shadow"], tokens }, p => {
      $("#mia-hint").textContent = loadText(p.name, p);
    });
    $("#mia-hint").textContent = "";
    renderMia(res[miaTarget], res.shadow, idx, truncated);
  } catch (e) {
    $("#mia-hint").textContent = e.message;
  }
  setMiaBusy(false);
});
syncPicks();

// --------------------------------------------------------- 3. write a story ----
let storyModel = "no_defense", writing = false;
segmented($("#story-model-seg"), TARGETS.map(n => ({ value: n, label: LABEL[n] })), storyModel, n => {
  storyModel = n;
  const note = $("#story-note");
  note.hidden = n !== "dp_eps8";
  note.textContent = `This model was trained with DP-SGD (ε = 8). Its stories are much worse: that's the price of the privacy guarantee (validation loss ${demo.val_loss.dp_eps8.toFixed(2)} vs ${demo.val_loss.no_defense.toFixed(2)}).`;
});
$("#story-chips").replaceChildren(...demo.prompts.map(p => {
  const b = Object.assign(document.createElement("button"), { type: "button", textContent: p });
  b.addEventListener("click", () => { $("#story-prompt").value = p; });
  return b;
}));
$("#story-temp").addEventListener("input", e => { $("#story-temp-out").textContent = Number(e.target.value).toFixed(1); });

function showStory(prompt, continuation) {
  const out = $("#story-out");
  out.replaceChildren(Object.assign(document.createElement("b"), { textContent: prompt }), document.createTextNode(continuation));
  return out.lastChild;
}

$("#story-saved").addEventListener("click", () => {
  const typed = $("#story-prompt").value.trim();
  const p = demo.prompts.includes(typed) ? typed : demo.prompts[0];
  const options = demo.stories[storyModel][p];
  const text = options[Math.floor(Math.random() * options.length)];
  showStory(p, text.startsWith(p) ? text.slice(p.length) : ` ${text}`);
  $("#story-hint").textContent = p === typed ? "Saved story (written in PyTorch by the same model)." : `Saved stories exist for the suggested openings; showing "${p}".`;
});

$("#story-go").addEventListener("click", async () => {
  const prompt = $("#story-prompt").value.trim() || "Once upon a time";
  if (writing) return;
  writing = true;
  $("#story-go").disabled = true;
  $("#story-stop").hidden = false;
  const node = showStory(prompt, "");
  const gen = [];
  let t0 = null;
  try {
    await call("generate", {
      name: storyModel, ids: [tok.eotId, ...tok.encode(prompt)],
      opts: { maxNew: 300, temperature: Number($("#story-temp").value), topK: 100 },
    }, p => {
      if (p.phase === "token") {
        t0 ??= performance.now();
        if (p.id !== tok.eotId) { gen.push(p.id); node.data = tok.decode(gen); }
      } else $("#story-hint").textContent = loadText(storyModel, p);
    });
    const s = t0 ? (performance.now() - t0) / 1000 : 0;
    $("#story-hint").textContent = gen.length ? `${gen.length} tokens in ${s.toFixed(1)} s` : "";
  } catch (e) {
    $("#story-hint").textContent = e.message;
  }
  writing = false;
  $("#story-go").disabled = !live;
  $("#story-stop").hidden = true;
});
$("#story-stop").addEventListener("click", () => { call("stop", {}).catch(() => {}); });

// ---------------------------------------------------------------- 4. about ----
$("#about-table").innerHTML = TARGETS.map(n => {
  const c64 = demo.canaries.filter(c => c.reps === 64);
  const top100 = c64.filter(c => c.results[n].experiment_position > 0).length / c64.length;
  return `<tr><td>${esc(LABEL[n])}</td><td>${demo.val_loss[n].toFixed(3)}</td><td>${(top100 * 100).toFixed(0)}%</td>
    <td>${demo.mia_auc[n].reference_seen3.toFixed(2)}</td></tr>`;
}).join("");

// ------------------------------------------------------------------ start ----
fillCanaries();
setLive(false);
$("#engine").textContent = "Starting the in-browser runtime…";
startWorker().then(info => setLive(Boolean(info), info));
