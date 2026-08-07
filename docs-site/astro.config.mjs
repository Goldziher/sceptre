// @ts-check
import starlight from "@astrojs/starlight";
import { xbergStarlightConfig } from "@xberg-io/docs-theme";
import { defineConfig } from "astro/config";
import starlightLlmsTxt from "starlight-llms-txt";

export default defineConfig({
  site: "https://docs.sceptre.xberg.io",
  integrations: [
    starlight(
      xbergStarlightConfig({
        title: "sceptre",
        description:
          "EasyOCR's accuracy, Rust's speed and footprint. A from-scratch Rust reimplementation of the " +
          "EasyOCR pipeline — CRAFT detection then gen2 CRNN recognition over ONNX — as a library, CLI, and MCP server.",
        githubUrl: "https://github.com/xberg-io/sceptre",
        editBaseUrl: "https://github.com/xberg-io/sceptre/edit/main/docs-site/",
        plugins: [
          starlightLlmsTxt({
            promote: ["index*", "start/**", "concepts/**"],
            minify: { collapseCodeBlocks: true },
            details:
              "sceptre is a Rust reimplementation of EasyOCR (CRAFT detection + gen2 CRNN recognition over ONNX), " +
              "shipped as a library, a CLI, and an MCP server. Repository: https://github.com/xberg-io/sceptre",
          }),
        ],
        sidebar: [
          { label: "Home", link: "/" },
          {
            label: "Get Started",
            items: [
              { label: "Introduction", slug: "start/introduction" },
              { label: "Installation", slug: "start/installation" },
              { label: "Quickstart", slug: "start/quickstart" },
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
            label: "Concepts",
            items: [
              { label: "How it works", slug: "concepts/how-it-works" },
              { label: "Models & parity", slug: "concepts/models-and-parity" },
              { label: "Backends", slug: "concepts/backends" },
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
          {
            label: "More",
            items: [
              { label: "Contributing", slug: "more/contributing" },
              { label: "Changelog", slug: "more/changelog" },
              { label: "Ecosystem", slug: "more/ecosystem" },
            ],
          },
        ],
      }),
    ),
  ],
});
