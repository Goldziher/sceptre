---
status: accepted
date: 2026-08-01
deciders: Na'aman Hirschfeld
---

# Criterion microbenchmarks behind a `bench` feature seam

## Context and Problem Statement

The detection and recognition hot paths — CRAFT preprocess/postprocess/grouping and the
recognizer crop/preprocess/CTC-decode stages — are the functions future perf work (fused
softmax, SIMD, allocation trimming) will target. To measure them we need a stable microbenchmark
harness. The obstacle is visibility: these stages are `pub(crate)` (the whole `detect` and
`recognize` modules are crate-internal), so a Criterion `benches/` crate, which compiles against
`sceptre` as an external dependency, cannot reach them without widening the public API — which we
do not want to do purely to enable benchmarking.

## Decision Drivers

- Measure the real internal hot paths, not a proxy, so perf work is guided by true numbers.
- Do not widen the crate's public API to expose internals for benchmarking.
- Benches must build offline and in CI (no model downloads, no mandatory corpus submodule).
- Fit the existing tooling (`task`, the CI job style) and the 1000-line module cap.

## Considered Options

- **Public-API-only benches** — benchmark through `Reader`. Rejected: exercising a real end-to-end
  run requires downloaded ONNX models (network + cache), conflates the stages under test with
  model inference, and cannot isolate a single hot path.
- **`#[bench]` / libtest harness** — still unstable, nightly-only, and far weaker than Criterion
  (no statistics, no regression tracking). Rejected.
- **A `bench` cargo feature exposing a `#[doc(hidden)] pub` seam module** — a thin wrapper layer
  that reaches the internal stages through feature-gated `pub(crate)` shim functions in the
  `detect`/`recognize` module roots and re-exposes them through `#[doc(hidden)] pub fn
  *_for_benchmark` wrappers, driven by Criterion.

## Decision Outcome

Chosen option: **the `bench` feature seam**. A new `bench` cargo feature compiles
`crates/sceptre/src/bench.rs` as `#[cfg(feature = "bench")] #[doc(hidden)] pub mod bench`. The
`detect` and `recognize` module roots add `#[cfg(feature = "bench")] pub(crate)` shim functions
(`bench_preprocess`, `bench_postprocess`, `bench_group`, `bench_crop_preprocess`,
`bench_ctc_decode`) that call the still-`pub(super)` stage entry points (`prepare`,
`get_det_boxes`, `adjust_coordinates`, `group_boxes`, `prepare_batch`, `decode_greedy`,
`crop_region`) from their parent module. A Rust re-export cannot widen visibility (E0364), so a
shim — not a `pub(crate) use` — is what bridges the boundary; the shims are feature-gated, so they
add nothing when the feature is off. The wrappers take and return only public types (`Image`,
`Array2`, `DetectionConfig`, `GrayImage`, `()`), bind each result, and hand it to
`std::hint::black_box`, so no crate-private type crosses the public seam and the optimizer cannot
elide the work.

Criterion is pinned at `0.8` with `html_reports` at the workspace level and consumed as a
`sceptre` dev-dependency. Two `harness = false`, `required-features = ["bench"]` bench targets
(`detect`, `recognize`) live under `crates/sceptre/benches/`. Image-input stages run on real
corpus images through a `load_corpus_image` loader that falls back to a committed test image when
the `test_documents/` corpus is absent, so benches run offline; model-output stages (postprocess
heat-maps, grouping boxes, CTC logits) take synthetic inputs, because those stages consume model
output rather than images.

A `task bench` (`cargo bench -p sceptre --features bench`) runs the suite locally. CI adds a
dedicated `bench` job that compile-checks (`cargo check -p sceptre --benches --features bench`)
and clippy-lints the benches; it does not run timing passes, since perf numbers are not a CI
gate.

### Consequences

- Good: the true internal hot paths are measurable without any public-API change; the seam is
  hidden from docs and gated off by default.
- Good: benches build offline and are compile-checked in CI, so they cannot silently rot.
- Good: Criterion gives statistics and regression tracking, unblocking future perf work.
- Bad/limited: benchmarks are run manually (perf timings are not a CI gate), so regressions are
  caught only when someone runs `task bench`.
- Neutral: a small feature-gated wrapper layer and a set of feature-gated re-exports to maintain
  alongside the stage signatures they mirror.
