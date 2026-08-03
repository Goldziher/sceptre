// Rasterize the brand SVGs into the PNGs the site and social cards need.
// Run with `npm run assets` (from website/) or `node scripts/render-assets.mjs`.
// Outputs are committed; CI never runs this — it only serves the PNGs.
import sharp from "sharp";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const publicDir = join(here, "..", "public");
const docsAssets = join(here, "..", "..", "docs", "assets");

const DARK = { r: 8, g: 7, b: 10, alpha: 1 };
const TRANSPARENT = { r: 0, g: 0, b: 0, alpha: 0 };

async function raster(svgPath, outPath, width, height, background) {
  const svg = readFileSync(svgPath);
  await sharp(svg, { density: 384 })
    .resize(width, height, { fit: "contain", background })
    .png({ compressionLevel: 9 })
    .toFile(outPath);
  console.log(`  ${outPath}  ${width}x${height}`);
}

const og = join(publicDir, "og.svg");
const favicon = join(publicDir, "favicon.svg");

console.log("Rendering brand PNGs:");
// 1280x640 social / OG card — doubles as the GitHub social preview.
await raster(og, join(publicDir, "og.png"), 1280, 640, DARK);
await raster(og, join(docsAssets, "social-preview.png"), 1280, 640, DARK);
// Favicons (the source favicon.svg is a rounded dark tile with the amber crown).
await raster(favicon, join(publicDir, "favicon-32.png"), 32, 32, TRANSPARENT);
await raster(favicon, join(publicDir, "apple-touch-icon.png"), 180, 180, TRANSPARENT);
console.log("Done.");
