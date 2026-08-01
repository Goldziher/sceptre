---
status: accepted
date: 2026-07-31
deciders: Na'aman Hirschfeld
---

# Inference backend seam: ort / tract / candle

## Context and Problem Statement

We run ONNX models on desktop/server today and want WASM/Android later. ONNX
Runtime (`ort`) is fast natively but is not a good fit for pure-Rust targets.
`tract` is pure-Rust but its recurrent (LSTM/Scan) support is "not bulletproof",
and the gen2 recognizers contain a BiLSTM. We need to avoid coupling the pipeline
to any one runtime.

## Decision Drivers

- CPU/RSS-optimized native execution.
- A pure-Rust path for WASM/Android.
- A hedge if `tract` cannot run the recognizer BiLSTM.

## Considered Options

- `ort` only.
- `ort` native + `tract` pure-Rust, both behind a runtime-neutral trait.
- Reimplement the networks in `candle` (pure-Rust tensors), no ONNX runtime.

## Decision Outcome

Chosen option: **a runtime-neutral `ModelBackend` seam** with `ort` as the
default native impl and `tract` as the pure-Rust impl, both feature-gated;
`candle` is reserved as a third impl (reimplemented nets) to fall back on if
`tract` fails the recognizer BiLSTM. The pipeline and all config/result types
stay backend-agnostic. A single `ConcurrencyConfig` budget feeds Rayon and the
backend's intra-op threads.

### Consequences

- Good: backend choice is not load-bearing; we can ship `ort` first and add
  pure-Rust backends without touching the pipeline.
- Good: the pure-Rust target has a defined fallback.
- Bad: an abstraction layer plus a per-backend BiLSTM verification step.

## Status update (2026-08-01)

Both native backends have landed behind the `ModelBackend` seam: `ort` (default,
`inference/ort_backend.rs`) and `tract` (pure-Rust, `inference/tract_backend.rs`),
each feature-gated and unit-tested. `candle` remains reserved and unimplemented
(the cargo feature is an empty stub); see
[ADR 0009](0009-candle-evaluation-ort-primary.md) for why it is off the critical path.
