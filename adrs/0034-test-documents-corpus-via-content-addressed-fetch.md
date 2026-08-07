---
status: accepted
date: 2026-08-07
deciders: Na'aman Hirschfeld
---

# Test corpus: the `test_documents` submodule via content-addressed fetch, not vendored fixtures

> **Supersedes the "Corpus" decision of [ADR 0016](0016-parity-harness-and-test-corpus.md)**;
> that ADR's model-resolution and dual-golden decisions are unaffected and remain accepted.

## Context and Problem Statement

[ADR 0016](0016-parity-harness-and-test-corpus.md) adopted the `xberg` `test_documents`
Git-LFS submodule as sceptre's OCR corpus. In practice sceptre read only 38 images and 27
transcripts out of a 578 MiB Git-LFS repository, so commit `36c64e3` reversed that decision
without recording a new ADR: it vendored the 38 images and 27 transcripts directly into
`crates/sceptre/tests/data/` (14 MiB), dropped the submodule, and excluded the vendored
directories from the published crate (two follow-up fixes, `ce7b55b` and `b44e161`, were
needed to get the `exclude` globs right for `cargo package`).

That objection is now gone. `xberg-io/test_documents` no longer uses Git-LFS: it tracks only
text in git and keeps binaries in the public `gs://xberg-test-documents` bucket,
content-addressed by sha256 and pinned by a committed `corpus.lock.json`, fetched per-glob by
`scripts/fetch_corpus.py` (no credentials, no LFS client). This is also now the idiom every
other repo in the `xberg-io` polyrepo uses (`xberg`, `xberg-enterprise`): a plain git
submodule plus a `TEST_DOCUMENTS_DIR` env var with a repo-relative fallback for path
resolution.

## Decision Drivers

- Do not vendor binaries in this repository; a corpus update should land once upstream
  instead of being hand-copied into sceptre (and every other consumer repo).
- Match the org-wide `test_documents` submodule + `TEST_DOCUMENTS_DIR` idiom instead of a
  bespoke, sceptre-only layout.
- Keep default `cargo test` fast, offline, and green — unchanged from ADR 0016 — and never
  reintroduce the silent stand-in fallback that `36c64e3` explicitly killed: an absent or
  unfetched corpus must skip explicitly or fail loudly, never substitute a different image.
- Ground-truth transcripts are git-tracked plain text upstream, so they arrive with the
  submodule at clone time; only image binaries need an explicit fetch step.

## Considered Options

- Keep vendoring the fixture subset in-tree (status quo since `36c64e3`).
- Re-adopt the old Git-LFS `test_documents` submodule from ADR 0016.
- Adopt the new, LFS-free `test_documents` submodule (content-addressed via
  `corpus.lock.json` + `fetch_corpus.py`).

## Decision Outcome

Adopt the new `test_documents` submodule. `.gitmodules` points `test_documents` at
`https://github.com/xberg-io/test_documents.git`, matching `xberg`/`xberg-enterprise`
exactly. The vendored `crates/sceptre/tests/data/{images,ground_truth}/` directories and the
`exclude` entries they required in `crates/sceptre/Cargo.toml` are removed;
`tests/data/golden/` and `tests/data/metrics_vectors.json` stay, since those are sceptre's own
generated artifacts, not corpus.

Every path-resolution site (the library's `bench` seam, the tier-1/tier-2/backend-agreement/
MCP integration tests, the Python `sceptre_rs_tools` corpus/golden tooling, and the
`sceptre-tools` snapshot generator) resolves the corpus root the same way: a `TEST_DOCUMENTS_DIR`
environment variable override, falling back to the `test_documents` submodule at the
repository root. Ground truth is resolved across upstream's partitioned
`ground_truth/{images,jpg,png,jpeg}/` layout by trying each base in order, since manifest
stems are unique across the corpus. `task setup` fetches the image corpus with
`python3 test_documents/scripts/fetch_corpus.py --include 'images/**'`.

Absent-corpus behavior carries the ADR 0016 / `36c64e3` invariant forward: every resolver
returns `None` (Python) or an explicit early `return` before assertions (Rust tests) instead
of substituting a different file. Tests gated behind `SCEPTRE_REQUIRE_MODELS` keep failing
loudly when that flag is set and the corpus/models are unavailable.

The submodule is pinned at upstream `main`. Nine images this repo names in its manifests and
golden generators (`english.png`, `example.png`, `french.jpg`, `chinese.jpg`, `japanese.jpg`,
`korean.png`, `cyrillic.png`, `telugu.png`, `kannada.png`) exist only on an unmerged upstream
branch (`chore/publish-sceptre-parity-images`, commit `fc85e69`) as of this decision; until
that branch merges and the pin is bumped, resolution for those specific images skips rather
than substituting anything.

### Consequences

- Good: no binaries vendored in this repository; a `test_documents` update lands once
  upstream instead of being hand-copied per consumer repo.
- Good: matches the org-wide submodule + `TEST_DOCUMENTS_DIR` + `fetch_corpus.py` idiom used
  by `xberg` and `xberg-enterprise`.
- Good: `crates/sceptre/Cargo.toml` no longer needs `exclude` entries or upkeep for the
  corpus, and the published crate carries no fixture bytes either way.
- Bad: `cargo test` and `task setup` now have an implicit dependency on the submodule being
  initialized and, for images, fetched; unresolved locally or in a fresh CI checkout,
  corpus-dependent tests skip instead of exercising the real path.
- Bad: pinned at upstream `main`, several sceptre-specific images remain unavailable until
  `chore/publish-sceptre-parity-images` merges and the submodule pin is re-bumped — tracked
  as follow-up work, not resolved by this ADR.
