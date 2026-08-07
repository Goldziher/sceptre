---
status: accepted
date: 2026-08-07
deciders: Na'aman Hirschfeld
supersedes: 0024
---

# Docs on `docs.sceptre.xberg.io` with the shared `@xberg-io/docs-theme`

## Context and Problem Statement

[ADR 0024](0024-documentation-site-and-web-presence.md) stood the documentation site up under
`website/`, hosted on GitHub Pages at the **project path** `https://goldziher.github.io/sceptre`,
with a per-repo amber/pixel brand (`src/styles/custom.css`, rasterized favicons and OG cards) and
npm as the package manager. That was the right call for a single repository with no DNS.

sceptre has since moved to the `xberg-io` organization, which already runs several documentation
sites off one convention: `docs.<repo>.xberg.io`, an `Astro Starlight` project under `docs-site/`,
and the shared `@xberg-io/docs-theme` npm package carrying the brand, the CDN assets, the Open Graph
defaults, and the canonical analytics IDs. sceptre was the only repo not on it, which meant three
divergences to maintain: a second brand system, a second package manager, and a sub-path host.

The sub-path is the expensive one. Because the site was served from `/sceptre`, `astro.config.mjs`
carried a `base` plus an in-repo `rehypeBaseLinks` plugin that rewrote every root-absolute link in
Markdown, and every JSX link had to interpolate `import.meta.env.BASE_URL` by hand. ADR 0024 named
that a maintenance tax and predicted that a custom domain would remove it.

## Decision Drivers

- One documentation system across the organization; a brand refresh must be a package bump, not a
  per-repo edit.
- The docs URL should be stable and product-shaped, not tied to a personal GitHub account.
- Content should be host-agnostic: an internal link written in Markdown must be correct with no
  build-time rewriting.
- Analytics, OG cards, and favicons belong in exactly one place.
- The deploy pipeline should be the shared reusable workflow, so its fixes land everywhere at once.

## Considered Options

- **Keep the project path and only rename the org.** Cheapest, and it keeps `base`, the rehype
  plugin, and the per-repo brand — the three things that make sceptre's docs different from its
  siblings'.
- **Custom domain, but keep the local brand CSS and assets.** Removes the sub-path tax and leaves
  sceptre visually divergent, with a second analytics property and its own OG card to re-render.
- **`docs.sceptre.xberg.io` on the shared theme**, matching every other xberg.io repo.

## Decision Outcome

Chosen option: **`docs.sceptre.xberg.io` on the shared `@xberg-io/docs-theme`.**

- `website/` becomes **`docs-site/`**, the directory name the shared tooling and the reusable deploy
  workflow expect. The move preserves git history.
- `astro.config.mjs` sets `site: "https://docs.sceptre.xberg.io"` and **no `base`**. The `BASE`
  constant, the `rehypeBaseLinks` plugin, the `markdown.rehypePlugins` wiring, and the
  `unist-util-visit` dependency are all deleted; root-absolute links in content now mean what they
  say. `public/CNAME` carries the domain and `robots.txt` points at the site's own sitemap.
- Starlight config is built by **`xbergStarlightConfig()`** wrapped in `starlight()`. The theme
  injects the brand stylesheet, the CDN logo (`SiteTitle` override), favicons, Open Graph and
  Twitter tags, and the shared GA4 / Google Ads tags. Accordingly the site no longer declares
  `logo`, `favicon`, `customCss`, `social`, or a `head` array, and the vendored brand files
  (`src/styles/custom.css`, `public/{og,favicon}*`, `scripts/render-assets.mjs`) are removed. The
  only assets that stay in-repo are the light/dark hero pair used by the splash page.
- The sidebar adopts the mandated organization-wide order — **Home → Get Started → Guides →
  Concepts → Reference → More** — which adds a `More` group carrying the changelog, a contributing
  pointer, and the ecosystem list, and moves `Start here` to `Get Started`.
- **npm → pnpm.** `package-lock.json` is replaced by `pnpm-lock.yaml`, and `pnpm-workspace.yaml`
  declares the build allowlist (`esbuild`, `sharp`) and exempts the pinned theme version from the
  minimum-release-age gate. The reusable deploy workflow runs `pnpm install --frozen-lockfile`, so
  the lockfile is a committed build input.
- `.github/workflows/docs.yaml` — a hand-rolled build-and-deploy pair — is replaced by
  `.github/workflows/ci-docs.yaml`, which calls
  `xberg-io/actions/.github/workflows/reusable-docs-deploy.yml@v1` and deploys only on a push to
  `main`. Pull requests build without deploying, which the old workflow did not do at all.

### What this supersedes

This supersedes ADR 0024 in full: the host, the directory name, the brand mechanism, the asset
pipeline, the package manager, and the deploy workflow all change. What survives from 0024 is its
one durable choice — Astro + Starlight as the documentation stack — and its rule that content is
derived from the shipped source rather than invented.

### Consequences

- Good: the sub-path tax is gone. Links in Markdown are plain root-absolute paths with no rewriting,
  and JSX links no longer interpolate a base.
- Good: a brand change is a `@xberg-io/docs-theme` version bump; sceptre owns no presentation code.
- Good: analytics, OG card, and favicons are shared, so the six docs sites report into one property.
- Neutral: the domain needs a DNS `CNAME` record and GitHub Pages must verify it once. The
  `public/CNAME` file is what keeps the custom domain across deploys.
- Bad: the old `goldziher.github.io/sceptre` URLs break. GitHub redirects the repository itself
  after the org move, but Pages content at the old project path does not survive, so external links
  to specific pages are lost.
- Bad: pnpm is a second Node package manager for contributors who already had npm. It is the
  organization-wide choice and what the reusable workflow installs, so matching it is cheaper than
  diverging.

## Related

- [ADR 0024](0024-documentation-site-and-web-presence.md) — the superseded decision: `website/`,
  the GitHub Pages project path, the per-repo brand, and the `base`-prefixing rehype plugin.
