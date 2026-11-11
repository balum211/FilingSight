// Takes the README / documentation screenshots against a running app (http://localhost:8501).
// Used by .github/workflows/screenshots.yml, where the free model is reachable.
const { chromium } = require('playwright');
const D = 'docs/screenshots/';
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1440, height: 1000 } });
  const ask = async (re) => { await p.getByRole('button', { name: re }).click();
    await p.waitForSelector('[data-testid="stChatMessage"] .meta', { timeout: 180000 }); await p.waitForTimeout(1500); };
  await p.goto('http://localhost:8501', { waitUntil: 'networkidle' });
  await p.waitForSelector('.hero', { timeout: 180000 }); await p.waitForTimeout(3000);
  await p.screenshot({ path: D + '01_home.png' });
  await ask(/Compare Uber/);
  await p.locator('[data-testid="stChatMessage"] [data-testid="stExpander"] summary').last().click(); await p.waitForTimeout(800);
  await p.locator('[data-testid="stChatMessage"]').last().scrollIntoViewIfNeeded();
  await p.screenshot({ path: D + '02_comparison.png' });
  await p.locator('.src').first().scrollIntoViewIfNeeded(); await p.waitForTimeout(500);
  await p.screenshot({ path: D + '02b_sources.png' });
  await p.getByRole('button', { name: 'Clear chat' }).click(); await p.waitForTimeout(2000);
  await ask(/revenue in 2023/);
  await p.screenshot({ path: D + '03_refusal.png' });
  await p.getByRole('tab', { name: 'Evaluation' }).click(); await p.waitForTimeout(1000);
  await p.locator('.tbl').first().scrollIntoViewIfNeeded();
  await p.screenshot({ path: D + '04_evaluation.png' });
  await p.getByRole('tab', { name: 'How it works' }).click(); await p.waitForTimeout(1000);
  await p.locator('.steps').scrollIntoViewIfNeeded();
  await p.screenshot({ path: D + '05_how_it_works.png' });
  await p.getByRole('tab', { name: 'Ask the filings' }).click();
  await p.getByRole('button', { name: 'Clear chat' }).click(); await p.waitForTimeout(47000);  // free tier: 1 reply / 45 s
  await ask(/Uber's total revenue in 2021/);
  await p.locator('[data-testid="stChatMessage"]').last().scrollIntoViewIfNeeded();
  await p.screenshot({ path: D + '06_answer.png' });
  await b.close();
})().catch(e => { console.error(e); process.exit(1); });
