---
status: accepted
date: 2026-07-31
deciders: Na'aman Hirschfeld
---

# Cargo workspace: rlib core + thin CLI

## Context and Problem Statement

The project must ship as a reusable library, a CLI, and an MCP server, and later
target WASM/Android. We need a structure that keeps the OCR logic reusable and
the entry points thin.

## Decision Drivers

- Library-first reuse; binaries should only orchestrate.
- Centralized dependency, lint, and profile management.
- A layout that extends to binding crates without churn.

## Considered Options

- Single crate with `lib.rs` + `main.rs`.
- Cargo workspace: `crates/easyocr` (rlib) + `crates/easyocr-cli` (bin).
- Multi-app monorepo with several binaries up front.

## Decision Outcome

Chosen option: **workspace with an rlib core and a thin CLI**, mirroring the
xberg design. `[workspace.package]`, `[workspace.dependencies]`,
`[workspace.lints]`, and tuned `[profile.*]` live at the root; members inherit
with `.workspace = true`. The CLI depends only on the core plus argument
parsing and never contains OCR logic.

### Consequences

- Good: the core is independently testable and embeddable; binding crates can be
  added later as peripheral members.
- Good: one place governs versions, lints, and build profiles.
- Bad: slightly more manifest boilerplate than a single crate.
