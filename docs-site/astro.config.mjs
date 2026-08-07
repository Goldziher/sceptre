// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import starlightLlmsTxt from "starlight-llms-txt";
import { visit } from "unist-util-visit";

const BASE = "/sceptre";

// The site is hosted at a project sub-path (goldziher.github.io/sceptre), so root-absolute
// internal links authored in Markdown must be prefixed with the base. Starlight rewrites its own
// nav but not content links, so we do it here for every in-page <a>/<img> that targets a root path.
function rehypeBaseLinks() {
  const rewrite = (url) =>
    typeof url === "string" &&
    url.startsWith("/") &&
    !url.startsWith("//") &&
    !url.startsWith(`${BASE}/`) &&
    url !== BASE
      ? BASE + url
      : url;
  return (tree) => {
    visit(tree, "element", (node) => {
      if (node.tagName === "a" && node.properties?.href) node.properties.href = rewrite(node.properties.href);
      if ((node.tagName === "img" || node.tagName === "source") && node.properties?.src)
        node.properties.src = rewrite(node.properties.src);
    });
  };
}

export default defineConfig({
  site: "https://goldziher.github.io",
  base: BASE,
  markdown: {
    rehypePlugins: [rehypeBaseLinks],
  },
  integrations: [
    starlight({
      title: "sceptre",
      description:
        "EasyOCR's accuracy, Rust's speed and footprint. A from-scratch Rust reimplementation of the " +
        "EasyOCR pipeline — CRAFT detection then gen2 CRNN recognition over ONNX — as a library, CLI, and MCP server.",
      logo: {
        src: "./src/assets/logo.svg",
        alt: "sceptre",
        replacesTitle: false,
      },
      favicon: "/favicon.svg",
      customCss: ["./src/styles/custom.css"],
      social: [{ icon: "github", label: "GitHub", href: "https://github.com/Goldziher/sceptre" }],
      editLink: {
        baseUrl: "https://github.com/Goldziher/sceptre/edit/main/website/",
      },
      head: [
        {
          tag: "link",
          attrs: { rel: "apple-touch-icon", href: "/sceptre/apple-touch-icon.png" },
        },
        {
          tag: "link",
          attrs: { rel: "icon", type: "image/png", sizes: "32x32", href: "/sceptre/favicon-32.png" },
        },
        {
          tag: "meta",
          attrs: { property: "og:image", content: "https://goldziher.github.io/sceptre/og.png" },
        },
        {
          tag: "meta",
          attrs: { name: "twitter:card", content: "summary_large_image" },
        },
        {
          tag: "meta",
          attrs: { name: "twitter:image", content: "https://goldziher.github.io/sceptre/og.png" },
        },
      ],
      plugins: [
        starlightLlmsTxt({
          promote: ["index*", "start/**", "concepts/**"],
          minify: { collapseCodeBlocks: true },
          details:
            "sceptre is a Rust reimplementation of EasyOCR (CRAFT detection + gen2 CRNN recognition over ONNX), " +
            "shipped as a library, a CLI, and an MCP server. Repository: https://github.com/Goldziher/sceptre",
        }),
      ],
      sidebar: [
        {
          label: "Start here",
          items: [
            { label: "Introduction", slug: "start/introduction" },
            { label: "Installation", slug: "start/installation" },
            { label: "Quickstart", slug: "start/quickstart" },
          ],
        },
        {
          label: "Concepts",
          items: [
            { label: "How it works", slug: "concepts/how-it-works" },
            { label: "Models & parity", slug: "concepts/models-and-parity" },
            { label: "Backends", slug: "concepts/backends" },
          ],
        },
        {
          label: "Guides",
          items: [
            { label: "CLI", slug: "guides/cli" },
            { label: "Library", slug: "guides/library" },
            { label: "MCP server", slug: "guides/mcp-server" },
            { label: "Configuration", slug: "guides/configuration" },
            { label: "Offline and CI", slug: "guides/offline-and-ci" },
          ],
        },
        {
          label: "Reference",
          items: [
            { label: "Benchmarks", slug: "reference/benchmarks" },
            { label: "Feature flags", slug: "reference/feature-flags" },
            { label: "Image formats", slug: "reference/image-formats" },
            { label: "Decision records", slug: "reference/decision-records" },
          ],
        },
      ],
    }),
  ],
});
