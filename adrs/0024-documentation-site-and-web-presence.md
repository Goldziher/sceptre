---
status: accepted
date: 2026-08-03
deciders: Na'aman Hirschfeld
---

# Documentation site, social assets, and CI docs deployment

## Context and Problem Statement

sceptre shipped `v0.1.0` to crates.io and the repository is public, but its only prose surface was the
README plus the ADRs. It needs a browsable documentation site (guides, concepts, a full CLI/library/
MCP reference), a shareable social/OG image for link previews, and a repeatable way to publish the
site. The sibling `basemind` repository already runs a proven documentation stack; reusing it avoids
re-litigating a well-solved problem.

## Decision Drivers

- One documentation stack shared with `basemind` (Astro + Starlight), so conventions and CI transfer.
- Zero-DNS hosting for the first iteration — no custom domain to provision.
- Docs content must stay accurate to the shipped API surface, not an invented one.
- The site build and social assets must be reproducible in CI and from a single command locally.
- Keep the amber/pixel brand identity already established in `docs/assets/`.

## Considered Options

- **Astro + Starlight** (as `basemind` uses), hosted on GitHub Pages.
- mdBook (Rust-native) or a `docs.rs`-only presence.
- A custom domain vs. the GitHub Pages project path.

## Decision Outcome

Chosen: an **Astro + Starlight** site under `website/`, mirroring `basemind`'s stack
(`@astrojs/starlight`, `starlight-llms-txt`, `sharp`, npm, Node 22), deployed to **GitHub Pages at the
project path `https://goldziher.github.io/sceptre`**.

- Because the site is served from a sub-path, `astro.config.mjs` sets `base: "/sceptre"`, and a small
  in-repo `rehype` plugin prefixes that base onto root-absolute internal links authored in Markdown
  (Starlight base-prefixes its own nav but not content links). Hero-action and JSX-component links,
  which bypass the Markdown pipeline, carry the base explicitly (`import.meta.env.BASE_URL`).
- `.github/workflows/docs.yaml` builds `website/` and deploys via the native GitHub Pages actions on
  every push to `main` that touches `website/**`. No `CNAME` — the project path needs no custom domain.
- Social/brand PNGs (`og.png` at 1280×640, favicons, apple-touch-icon) are rasterized from committed
  brand SVGs by `website/scripts/render-assets.mjs` using `sharp`; `og.png` doubles as the GitHub
  social preview (mirrored to `docs/assets/social-preview.png`). The README banner was reduced to the
  `sceptre` wordmark only.
- Documentation content is derived strictly from the shipped source (CLI, `lib.rs`, config, MCP,
  ADRs) and every non-trivial API claim was verified against that source before publishing.
- CI adopts the `xberg-io/actions` reusable validate workflow (owned by the same author, as
  `basemind` does), and `poly.local.toml` is added so the `ai-rulez` hook resolves and commits no
  longer need `--no-verify`. The release workflow is renamed `release.yaml` → `publish.yaml` to match
  the crates.io trusted-publisher configuration (see ADR 0023).

### Consequences

- Good: docs, brand assets, and deploy are reproducible; the stack and CI match `basemind`.
- Good: enabling a custom domain later is a one-file change (`public/CNAME` + `site`), not a rewrite.
- Neutral: GitHub Pages must be set to "GitHub Actions" as the source once, in repository settings.
- Bad: the `base`-prefixing of internal links is a small maintenance tax that a root-domain host would
  not carry; it is centralized in the `rehype` plugin to keep content base-agnostic.
