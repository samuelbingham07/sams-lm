// Records the README GIF: drives the page (python demo/serve.py must be running) and saves frames,
// then `python demo/make_gif.py` assembles demo/demo.gif.   cd demo; node record_gif.mjs
import { chromium } from "playwright-core";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";

const dir = "e2e_out/frames";
rmSync(dir, { recursive: true, force: true });
mkdirSync(dir, { recursive: true });
const browser = await chromium.launch({ channel: "msedge", headless: true });
const page = await browser.newPage({ viewport: { width: 1000, height: 820 }, deviceScaleFactor: 1 });
await page.goto(process.env.E2E_URL ?? "http://localhost:8000/?models=/models/");
await page.waitForFunction(() => /run on your device/.test(document.querySelector("#engine").textContent), null, { timeout: 60000 });

// Recording aids: a visible cursor, room to scroll, and the tabs pinned to the top of the frame.
await page.addStyleTag({ content: `
  main { min-height: 1600px; }
  #cursor { position: fixed; z-index: 99; width: 22px; height: 22px; margin: -4px 0 0 -4px; pointer-events: none;
    transition: left .45s ease, top .45s ease; left: 700px; top: 500px; }
  #cursor.down svg { transform: scale(.85); }` });
await page.evaluate(() => {
  const c = document.createElement("div");
  c.id = "cursor";
  c.innerHTML = `<svg viewBox="0 0 22 22" width="22" height="22"><path d="M3 2l15 9-7 1.5L7.5 19z" fill="#111" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>`;
  document.body.append(c);
  window.scrollTo(0, document.querySelector(".tabs").getBoundingClientRect().top + scrollY - 12);
});

const frames = [];
let n = 0;
async function snap(ms) {
  const path = `${dir}/f${String(n++).padStart(4, "0")}.png`;
  await page.screenshot({ path });
  frames.push({ path, ms });
}
async function hold(ms) { await snap(ms); }
async function moveTo(sel) {
  const box = await page.locator(sel).first().boundingBox();
  await page.evaluate(([x, y]) => Object.assign(document.querySelector("#cursor").style, { left: `${x}px`, top: `${y}px` }),
    [box.x + box.width / 2, box.y + box.height / 2]);
  for (let i = 0; i < 4; i++) { await page.waitForTimeout(110); await snap(110); }
}
async function click(sel) {
  await moveTo(sel);
  await page.evaluate(() => document.querySelector("#cursor").classList.add("down"));
  await snap(120);
  await page.click(sel);
  await page.evaluate(() => document.querySelector("#cursor").classList.remove("down"));
}
async function recordUntil(cond, every = 250) {
  while (!(await page.evaluate(cond))) { await snap(every); await page.waitForTimeout(every - 100); }
}
const scrollTabsTop = () => page.evaluate(() =>
  window.scrollTo(0, document.querySelector(".tabs").getBoundingClientRect().top + scrollY - 12));

// Scene 1: steal a secret (default: a 64-copy secret)
await hold(1600);
await click("#secret-live");
await recordUntil(() => document.querySelectorAll("#secret-results .verdict").length === 3, 300);
await page.waitForTimeout(150);
await hold(4200);

// Scene 2: membership inference on Story 3 (a training story the attack flags)
await click("#t-member");
await scrollTabsTop();
await hold(900);
await click("#mia-picks button:nth-child(3)");
await hold(500);
await click("#mia-live");
await recordUntil(() => !document.querySelector("#mia-result").hidden && document.querySelector("#mia-result svg"), 300);
await page.evaluate(() => window.scrollTo({ top: document.querySelector("#mia-result").getBoundingClientRect().top + scrollY - 70 }));
await hold(1800);
await click("#mia-answer button");
await hold(3200);

// Scene 3: write a story
await click("#t-story");
await scrollTabsTop();
await hold(600);
await click("#story-go");
// Writing takes about a second, so capture often until it finishes.
await recordUntil(() => document.querySelector("#story-stop").hidden && /tokens in/.test(document.querySelector("#story-hint").textContent), 130);
await hold(3000);

writeFileSync(`${dir}/frames.json`, JSON.stringify(frames));
console.log(`${frames.length} frames`);
await browser.close();
