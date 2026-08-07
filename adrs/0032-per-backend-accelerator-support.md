---
status: accepted
date: 2026-08-07
deciders: Na'aman Hirschfeld
---

# Per-backend accelerator support (amends ADR 0029)

## Context and Problem Statement

[ADR 0029](0029-cli-provisioning-default-and-runtime-scoped-parity.md) introduced
`ModelConfig::accelerator` as vocabulary named for the backend seam rather than for ONNX
Runtime, so that "`tract` and `candle` answer the same field and reject a hardware
accelerator at config-validation time." The implementation encoded that as a single test:

```rust
if self.accelerator.is_cpu_only() || self.backend == Backend::Ort { return Ok(()) }
```

That was true exactly as long as `ort` was the only backend that could reach hardware.
[ADR 0031](0031-hand-written-candle-networks.md) makes `candle` run on Metal and CUDA, so
the special case is now wrong in both directions: it would reject a valid
`candle` + `metal` configuration, and the vocabulary has no word for Metal at all.

A second problem is specific to Apple. `ort` reaches the Apple GPU through **CoreML**;
`candle` reaches the same GPU through **Metal**. Neither name is a synonym for the other,
and a user who knows one backend will reasonably type the wrong one on the other.

## Decision Drivers

- Backend/accelerator validity is a matrix, not a single privileged backend.
- The wire vocabulary flows verbatim into published benchmark provenance (ADR 0030,
  `publish.py`), so it must name what actually ran.
- ADR 0029's two-layer invariant must survive: config validates *vocabulary*, load validates
  *availability*, and an explicit request never silently degrades to CPU.
- A rejection should name the remedy, not merely the fault.

## Considered Options

- **Fold Metal into `CoreMl`** as "the Apple accelerator", resolved per backend. One less
  word, and wrong: they are different frameworks with different numerics, and a benchmark
  report saying `coreml` for a Metal run misattributes the result.
- **Namespace every value by backend** (`ort:coreml`, `candle:metal`). Unambiguous, and it
  duplicates the backend field into the accelerator field, breaking the ADR 0029 principle
  that the vocabulary is seam-level rather than backend-level.
- **A per-backend support table** over a flat vocabulary.

## Decision Outcome

Chosen option: **a per-backend support table**, `Backend::hardware_accelerators()`, with
`validate_accelerator` consulting it instead of testing for `ort`.

| Backend | Hardware accelerators |
| --- | --- |
| `ort` | `coreml`, `directml`, `cuda` |
| `tract` | — (CPU only) |
| `candle` | `metal`, `cuda` |

- **`Cuda` is deliberately shared.** It names hardware, not an ONNX Runtime execution
  provider; `ort` reaches it through the CUDA EP and `candle` through its own kernels, and
  which one ran is already recorded by the `backend` field beside it.
- **`Metal` is a new variant, not an alias for `CoreMl`.** They drive one GPU through two
  frameworks; keeping them distinct is what lets provenance stay truthful.
- **A rejection names the remedy.** For the Apple pair the message points at the equivalent
  on the configured backend (`--backend candle --accelerator coreml` says the same hardware
  is reached with `metal`); otherwise it names a backend that does support the request.
- **The two-layer invariant is preserved.** `candle/device.rs` mirrors `ort_ep.rs`: `Auto`
  walks candidates and may settle for the CPU, while an explicit request that this build did
  not compile in is an error naming the cargo feature (`candle-metal`, `candle-cuda`).
- **`runtime_info_for` stops hardcoding the CPU for `candle`.** It previously reported
  `Some(Cpu)` unconditionally, which would have put `"cpu"` into published benchmark
  provenance for a run that happened on the GPU. It now resolves the device the backend
  would actually open, and reports `None` — undetermined — when the backend is not compiled
  in, rather than promising a run that cannot happen.
- **`onnxruntime` stays a nullable key** in the CLI's environment report, because
  `benchmark.py` and `publish.py` parse it; only its rendering omits the runtime clause when
  null.

### What this does and does not amend

This amends **only** the "candle rejects hardware accelerators" clause of ADR 0029, and only
for `candle`. Everything else in 0029 stands unchanged: the provisioning default, the
`Cpu`-not-`Auto` default and its reasoning, `error_on_failure` registration, and the scoping
of every published parity figure to `ort` + CPU.

### Consequences

- Good: adding a backend's hardware support is a table entry, not a new branch in validation.
- Good: provenance distinguishes CoreML from Metal, so benchmark reports stay attributable.
- Good: the most likely user error — the wrong Apple framework — answers with the right one.
- Bad: `Accelerator` gains a variant, which is a breaking change for exhaustive matches
  downstream. Taken at 0.5.0, while pre-1.0. The enum is deliberately left exhaustive rather
  than `#[non_exhaustive]`: it is a small closed configuration vocabulary that consumers have
  a legitimate reason to match on.
- Neutral: **GPU numerics are validated on real hardware only.** No hosted CI runner has a
  GPU, so the Metal path is compiled in CI and exercised locally; CUDA is compiled and never
  executed. The docs and tests say so rather than implying coverage.

## Related

- [ADR 0029](0029-cli-provisioning-default-and-runtime-scoped-parity.md) — establishes the
  accelerator vocabulary and the fail-loud invariant this ADR generalizes.
- [ADR 0031](0031-hand-written-candle-networks.md) — the candle backend whose GPU support
  makes the single-backend special case wrong.
- [ADR 0030](0030-published-benchmark-artifact-and-drift-gate.md) — the published artifact
  whose provenance strings this vocabulary feeds.
