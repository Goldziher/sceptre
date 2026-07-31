---
status: accepted
date: 2026-07-31
deciders: Na'aman Hirschfeld
---

# Tooling scaffolding adapted from basemind

## Context and Problem Statement

We need a lint/format/supply-chain/CI toolchain from day one. A sibling project,
basemind, already has a mature setup worth reusing rather than reinventing.

## Decision Outcome

Chosen option: **adapt basemind's tooling scaffolding**. We copy and trim
`poly.toml` (polylint driving fmt/clippy/typos/markdown/deny plus the `uncomment`
hook and the pinned `ai-rulez` hook source), `rustfmt.toml` (`max_width = 120`),
`deny.toml` (license allowlist), the `.gitignore` structure (including the
managed ai-rulez block), a `Taskfile.yaml`, and a `.github/workflows/ci.yaml`
(fmt + clippy + test matrix + cargo-deny). basemind-specific stanzas (its index
schema, harden harness, release-sync) are dropped.

### Consequences

- Good: a proven, consistent toolchain immediately; `line_length = 120` aligned
  across rustfmt, poly, and markdown.
- Good: the `uncomment` hook enforces the `~keep` comment convention.
- Bad: `poly` and `ai-rulez` become required local tools (installed via the
  Taskfile `setup` target).
