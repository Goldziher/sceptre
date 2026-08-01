---
status: accepted
date: 2026-08-01
deciders: Na'aman Hirschfeld
---

# Parity harness: dual golden fixtures, HF-cache model resolution, and the test corpus

## Context and Problem Statement

sceptre must be validated against EasyOCR, but the pieces needed to do so are heavy:
multi-GB ONNX models, a Python torch+easyocr reference environment, and a real-world
image corpus. We need a parity harness that stands the infrastructure up now and keeps
the default `cargo test` fast and green, while making the heavy download/generation an
opt-in, well-documented step. Three questions had real alternatives with lasting
consequences: what image corpus to test against, how the test harness locates models,
and what a golden fixture is compared against.

## Decision Drivers

- Default `cargo test` must stay fast and green with no network, no models, no submodule.
- CI must be able to *force* the real path and fail loudly on a misconfigured cache.
- Reuse existing, proven patterns (the xberg corpus + skip-if-missing convention).
- No new runtime dependencies for path resolution; downloading stays deferred.

## Considered Options

- **Corpus:** vendor a handful of images in-tree only / adopt the xberg `test_documents`
  Git-LFS submodule / build a new corpus.
- **Model resolution in tests:** a bespoke env-var model provider only / resolve from the
  Hugging Face hub cache / add `hf-hub` and download in the harness.
- **Golden comparison:** exact text only / fuzzy only / a dual golden (EasyOCR reference
  compared fuzzily + a sceptre self-snapshot compared exactly).

## Decision Outcome

- **Corpus:** adopt the xberg `test_documents` Git-LFS submodule as the OCR corpus. It is
  added without pulling LFS blobs by default; the harness tolerates it being absent or
  unpopulated (`test_documents_dir` + `skip_if_missing`).
- **Model resolution:** the test harness resolves CRAFT + gen2 models from the Hugging
  Face hub cache (`HF_HUB_CACHE` / `HF_HOME` / `~/.cache/huggingface/hub`) via a
  `HfCacheModelProvider` built on `std::fs` only — no `hf-hub`, no download. Adopting
  HF-cache resolution in the *library* is deferred as a follow-up; for now it lives in
  `tests/helpers`.
- **Golden comparison:** dual goldens. The `easyocr` side (authoritative Python EasyOCR
  reference) is compared with bag-of-words F1 plus per-line box-IoU (>= 0.5); the
  `sceptre` side (self-snapshot) is compared for exact text equality.
- **Gating:** real-model tests read `SCEPTRE_REQUIRE_MODELS`. Unset/falsy and models
  absent → skip (test passes); truthy and models absent → panic, so CI surfaces the
  misconfiguration. Pipeline-running bodies are `#[cfg(feature = "ort")]`-gated so the
  backend-less default build still compiles.

### Consequences

- Good: default `cargo test` stays green offline; the pure helpers are unit-tested and
  always run; CI can force the real path via one env var and a `workflow_dispatch` job.
- Good: no new dependency for model resolution; the download remains deferred and opt-in.
- Good: the dual golden catches both reference-parity drift and self-regressions.
- Bad: goldens ship as placeholders until the heavy generation is run and committed.
- Bad: two generators (Python `sceptre_rs_tools.golden` for the reference, `sceptre-tools
  snapshot` for the snapshot) must be kept in sync with the fixture schema.
- Bad: HF-cache resolution is duplicated in the harness until the library adopts it.

## Status update (2026-08-01)

The library adopted Hugging Face hub-cache resolution (ADR 0017), so the bespoke
`HfCacheModelProvider` and its `std::fs` resolver were removed from
`tests/helpers`. Real-model tests now gate availability via the library's
`model_manifest` and build the reader with the default provider; the dual-golden
comparison and `SCEPTRE_REQUIRE_MODELS` gating are unchanged.
