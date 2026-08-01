---
status: accepted
date: 2026-08-01
deciders: Na'aman Hirschfeld
---

# Release strategy: tag-triggered multi-platform build with crates.io dry-run

## Context and Problem Statement

sceptre ships as a library (`sceptre`) and a CLI (`sceptre-cli`) and will
eventually be published to crates.io with prebuilt CLI binaries. We want release
mechanics — versioning, multi-platform builds, and crates.io packaging — wired up
and validated in CI now, but we are **not** ready to make the crates public: the
decision to publish is deferred to an explicit future point. The release path must
therefore be reproducible and continuously exercised without ever pushing to
crates.io.

## Decision Drivers

- Release must be prepared and validated in CI without any real publish.
- Multi-platform CLI binaries (Linux, macOS, Windows) built from a tag.
- Publish order is load-bearing: `sceptre-cli` depends on `sceptre`, so the
  library must package first.
- Workflows call `task` targets, not raw cargo, per the pipeline-standards
  task-runner rule.
- Turning on a real publish later must be a minimal, low-risk change.

## Considered Options

- Publish for real on tag now (crates.io + binary artifacts).
- Tag-triggered multi-platform build plus a crates.io **dry-run** only, deferring
  the real publish behind an explicit future decision.
- Manual, off-CI releases.

## Decision Outcome

Chosen option: **tag-triggered (`v*`) multi-platform build plus a crates.io
dry-run**, no real publish until an explicit decision.

- `.github/workflows/release.yaml` triggers on `v*` tags (and `workflow_dispatch`).
  A `build` matrix over `ubuntu-latest`, `macos-latest`, and `windows-latest` runs
  `task release:build` and uploads the release CLI binary as a per-OS artifact.
- A `publish-dry-run` job runs `task release:dry-run`, which invokes
  `cargo publish --dry-run` for `sceptre` — the library. `sceptre-cli` cannot be
  dry-run-published: cargo resolves its `sceptre` dependency from crates.io, where
  `0.1.0` does not exist yet, so the cli dry-run fails with "no matching package
  named sceptre". The cli's build is covered by `release:build` and its publish
  packaging is validated at real-publish time, once the library is on crates.io.
  The dry-run uses the `ort-dynamic` feature so packaging does not download the
  ONNX Runtime; `release:build` uses `ort-bundled` for a runnable binary.
- Versioning is workspace-wide, currently `0.1.0`, with Conventional-Commit-driven
  bumps and a Keep-a-Changelog `CHANGELOG.md`.
- Model artifacts stay pinned by the existing sha256 verification in the registry;
  no model bytes are vendored into the crates.
- A real `cargo publish` is intentionally omitted. It is a one-line change (drop
  `--dry-run`, add crates.io token auth) gated on the future release decision.

### Consequences

- Good: the full release pipeline is reproducible and validated on every tag —
  packaging breakage surfaces in CI, not at publish time.
- Good: enabling a real publish later is a minimal, well-scoped change.
- Good: no accidental public release; nothing reaches crates.io until decided.
- Neutral: prebuilt binaries are uploaded as CI artifacts, not attached to a
  GitHub Release; wiring release assets is deferred with the publish decision.
- Bad: the dry-run cannot fully exercise cross-crate publish ordering (the library
  is not actually on crates.io during the cli dry-run), so the first real publish
  still needs a live verification.
