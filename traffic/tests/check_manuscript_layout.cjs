const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { execFileSync } = require("node:child_process");
const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.launch({ headless: true, channel: "chrome" });
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  const file = path.resolve("traffic/manuscript.html");
  const output = path.resolve("traffic/artifacts/manuscript_layout");
  fs.mkdirSync(output, { recursive: true });
  const results = [];
  try {
    await page.goto(pathToFileURL(file).href, { waitUntil: "load" });
    for (const width of [1440, 1024, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.locator("#figure-2").scrollIntoViewIfNeeded();
      const result = await page.evaluate(() => {
        const figure = document.querySelector("#figure-2");
        const image = figure.querySelector("img");
        const caption = figure.querySelector("#figure-2-caption");
        const box = element => {
          const b = element.getBoundingClientRect();
          return { x: b.x, y: b.y, width: b.width, height: b.height, right: b.right, bottom: b.bottom };
        };
        return {
          width: innerWidth, scrollWidth: document.documentElement.scrollWidth,
          figure: box(figure), image: box(image), caption: box(caption),
          imagesLoaded: [...document.images].every(i => i.complete && i.naturalWidth > 0),
          captionText: caption.innerText, equations: document.querySelectorAll(".equation math").length,
          mathErrors: document.querySelectorAll("merror").length,
        };
      });
      assert(result.imagesLoaded);
      assert.equal(result.equations, 4);
      assert.equal(result.mathErrors, 0);
      assert(result.captionText.includes("Dotted line: 5% reference"));
      assert(result.scrollWidth <= width, "Document overflows horizontally");
      assert(Math.abs(result.image.width / result.image.height - 6.6 / 4.5) < 0.01);
      if (width > 760) {
        assert(result.caption.x >= result.image.right + 20, "Caption must be a separate right column");
        assert(result.image.width < result.figure.width * 0.65);
      } else {
        assert(result.caption.y >= result.image.bottom + 12, "Mobile caption must follow plot");
      }
      await page.locator("#figure-2").screenshot({ path: path.join(output, `figure-2-${width}.png`) });
      results.push(result);
    }
    for (const colorScheme of ["light", "dark"]) {
      await page.emulateMedia({ colorScheme });
      await page.setViewportSize({ width: 1440, height: 1000 });
      await page.goto(pathToFileURL(file).href, { waitUntil: "load" });
      await page.locator("#figure-2").scrollIntoViewIfNeeded();
      const colors = await page.evaluate(() => ({
        body: getComputedStyle(document.body).backgroundColor,
        figure: getComputedStyle(document.querySelector("#figure-2")).backgroundColor,
        caption: getComputedStyle(document.querySelector("#figure-2-caption")).backgroundColor,
        bodyText: getComputedStyle(document.body).color,
        captionText: getComputedStyle(document.querySelector("#figure-2-caption")).color,
      }));
      assert.equal(colors.figure, "rgba(0, 0, 0, 0)");
      assert.equal(colors.caption, "rgba(0, 0, 0, 0)");
      assert.equal(colors.captionText, colors.bodyText);
      await page.locator("#figure-2").screenshot({ path: path.join(output, `figure-2-${colorScheme}.png`) });
      results.push({ colorScheme, colors });
    }
    const native = execFileSync("uv", ["run", "--extra", "research", "python", "-c",
      "from pathlib import Path; from markdown_it import MarkdownIt; print(MarkdownIt('commonmark', {'html': False}).enable('table').render(Path('traffic/manuscript.md').read_text()))"
    ], { encoding: "utf8" });
    const styles = await page.locator("style").innerText();
    await page.setContent(`<html><head><style>${styles}</style></head><body><main>${native}</main></body></html>`);
    const nativeTable = page.locator("table").nth(1);
    const nativeImage = nativeTable.locator('img[src="figures/observation_limits_preview.png"]');
    assert.equal(await page.locator("table img").count(), 1);
    assert(!(await page.locator("main").innerText()).includes("<table"));
    for (const width of [900, 1100, 1600]) {
      await page.setViewportSize({ width, height: 900 });
      await nativeImage.scrollIntoViewIfNeeded();
      const plot = await nativeImage.boundingBox();
      const caption = await nativeTable.locator("tbody td").nth(1).evaluate(cell => {
        const box = cell.getBoundingClientRect();
        return { text: cell.innerText, x: box.x, y: box.y, width: box.width };
      });
      assert(plot.width <= 528 && plot.height <= 360.1);
      assert(Math.abs(plot.width / plot.height - 22 / 15) < 0.01, "Plot must retain original proportions");
      assert(caption.text.startsWith("Figure 2."));
      assert(caption.x >= plot.x + plot.width);
      assert(caption.width >= 220);
      await page.screenshot({ path: path.join(output, `figure-2-portable-markdown-${width}.png`) });
      results.push({ htmlDisabledMarkdown: true, width, image: plot, caption });
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output, "validation.json"), JSON.stringify(results, null, 2) + "\n");
    console.log(JSON.stringify(results, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
