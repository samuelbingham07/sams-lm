// End-to-end check of the page in a real browser (installed Microsoft Edge, headless).
// Start `python demo/serve.py` first, then:  cd demo; node e2e.mjs [outDir]
import { chromium } from "playwright-core";
import { mkdirSync } from "node:fs";

const out = process.argv[2] ?? "e2e_out";
mkdirSync(out, { recursive: true });
const url = "http://localhost:8000/?models=/models/";
const browser = await chromium.launch({ channel: "msedge", headless: true });
const page = await browser.newPage({ viewport: { width: 1100, height: 900 } });
const errors = [];
page.on("console", m => { if (m.type() === "error") errors.push(m.text()); });
page.on("pageerror", e => errors.push(String(e)));
const t = () => (performance.now() / 1000).toFixed(0) + "s";

await page.goto(url);
await page.waitForFunction(() => /run on your device|isn't available/.test(document.querySelector("#engine").textContent), null, { timeout: 60000 });
console.log(t(), "engine:", await page.textContent("#engine"));
console.log(t(), "isolated:", await page.evaluate(() => crossOriginIsolated));

// 1. Steal a secret (64 copies, first canary), live
await page.click("#secret-live");
await page.waitForFunction(() => document.querySelectorAll("#secret-results .verdict, #secret-results .error").length === 3, null, { timeout: 600000 });
const verdicts = await page.$$eval("#secret-results .card", cs => cs.map(c => c.querySelector("h3").textContent + ": " +
  (c.querySelector(".verdict")?.textContent ?? c.textContent) + " | " + (c.querySelector(".meta")?.textContent ?? "")));
console.log(t(), "secret:", await page.textContent("#secret-reveal"));
verdicts.forEach(v => console.log("   ", v.replace(/\s+/g, " ")));
await page.screenshot({ path: `${out}/secret.png`, fullPage: true });
// saved results for the same canary should agree with the live ones
await page.click("#secret-saved");
const saved = await page.$$eval("#secret-results .verdict", vs => vs.map(v => v.textContent));
console.log(t(), "saved verdicts:", saved.join(" | "));

// 2. Membership, live, first example
await page.click("#t-member");
await page.click("#mia-picks button:nth-child(1)");
await page.click("#mia-live");
await page.waitForSelector("#mia-result:not([hidden]) svg", { timeout: 300000 });
console.log(t(), "membership:", (await page.textContent("#mia-result .verdict")).replace(/\s+/g, " "));
const stats = await page.$$eval("#mia-result .stat .v", v => v.map(x => x.textContent));
await page.click("#mia-saved");
const statsSaved = await page.$$eval("#mia-result .stat .v", v => v.map(x => x.textContent));
console.log("    live", stats.join(" / "), " saved", statsSaved.join(" / "));
await page.click("#mia-live");
await page.waitForSelector("#mia-result:not([hidden]) svg", { timeout: 300000 });
await page.hover("#mia-chart svg", { position: { x: 300, y: 100 } });
await page.screenshot({ path: `${out}/membership.png`, fullPage: true });

// 3. Story, live
await page.click("#t-story");
await page.click("#story-go");
await page.waitForFunction(() => document.querySelector("#story-stop").hidden && /tokens in/.test(document.querySelector("#story-hint").textContent), null, { timeout: 300000 });
console.log(t(), "story:", await page.textContent("#story-hint"));
console.log("   ", (await page.textContent("#story-out")).slice(0, 300).replace(/\s+/g, " "));
await page.screenshot({ path: `${out}/story.png`, fullPage: true });

// Dark mode + phone width, on the secret tab with saved results
await page.emulateMedia({ colorScheme: "dark" });
await page.click("#t-secret");
await page.click("#secret-saved");
await page.screenshot({ path: `${out}/secret_dark.png`, fullPage: true });
await page.setViewportSize({ width: 390, height: 844 });
await page.emulateMedia({ colorScheme: "light" });
await page.screenshot({ path: `${out}/secret_phone.png`, fullPage: true });
const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
console.log(t(), "horizontal overflow at 390px:", overflow);
await page.click("#t-member");
await page.click("#mia-saved");                       // redraw the chart at phone width
await page.screenshot({ path: `${out}/member_phone.png`, fullPage: true });
console.log("summary:", await page.textContent("#reps-summary"));

console.log(errors.length ? `console errors:\n  ${errors.join("\n  ")}` : "no console errors");
await browser.close();
