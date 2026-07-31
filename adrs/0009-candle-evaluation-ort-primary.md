---
status: accepted
date: 2026-07-31
deciders: Na'aman Hirschfeld
---

# Backend evaluation: `ort` primary, `candle` deferred off the critical path

## Context and Problem Statement

[ADR 0004](0004-inference-backend-seam.md) established a runtime-neutral `ModelBackend`
seam with `ort` as the default native backend, `tract` as the pure-Rust path, and
`candle` reserved as a reimplemented-network fallback. Before committing to that ordering
we wanted concrete evidence on whether `candle` (pure-Rust tensors, no C++ runtime) could
instead be a *primary* CPU backend — it is attractive because it removes the ONNX Runtime
native dependency and keeps a single pure-Rust story. This ADR records the outcome of a
hands-on `candle` feasibility spike so the "evaluate candle first" question is closed with
data rather than left open.

## Decision Drivers

- CPU-optimized native latency and peak RSS for the gen2 CRNN recognizer and CRAFT detector.
- A pure-Rust story for WASM/Android without hand-maintaining network reimplementations.
- Minimizing model-parity risk (exact CTC output vs EasyOCR) and ongoing maintenance burden.

## Considered Options

- **`candle`-primary** — load or reimplement the nets in `candle`, drop the ONNX runtime.
- **`ort`-primary, `candle` co-primary** — keep both native backends behind the seam.
- **`ort`-primary, `candle` deferred** — ship `ort` now, keep `tract` for pure-Rust, treat
  `candle` as a last resort only if `tract` cannot run the models.

## Decision Outcome

Chosen option: **`ort`-primary, `candle` deferred off the critical path.** The spike found
`candle` is neither loadable nor competitive for these models today:

- **`candle-onnx` cannot load either model** (measured on `candle` 0.9.2, gaps confirmed
  unchanged on upstream `main`): `english_g2` fails with *"LSTM currently only supports
  direction == forward"* (the recognizer uses two bidirectional LSTMs); `craft` fails with
  *"MaxPool with pads != 0"* (its VGG pooling is padded). Both are hard `bail!`s.
- The only `candle` route is a **from-scratch architecture reimplementation** in
  `candle_nn`, including a hand-rolled BiLSTM (reverse-run-reverse-concat, ONNX iofc→candle
  ifco gate reorder), with real CTC-parity risk concentrated there (~1 week, high
  correctness sensitivity).
- Even done perfectly, **`candle`'s CPU BiLSTM alone (~24 ms) is ~8× slower than `ort`'s
  entire recognizer (~2.9 ms)** — `candle` decomposes the LSTM into hundreds of tiny
  matmuls from a Rust loop with no fused CPU kernel. `ort` measured 110 ms for CRAFT and
  2.9 ms for `english_g2` at batch 1.

The pure-Rust requirement does **not** depend on `candle`: `tract` consumes ONNX directly
(no reimplementation) and remains the designated WASM/Android backend per ADR 0004, subject
to its own BiLSTM/Resize validation spike. `torch` is not required to obtain weights — the
Hugging Face ONNX files carry them as initializers, extractable to `safetensors` without it.

This refines ADR 0004's ordering with evidence; it does not reverse it. `candle` stays a
feature-gated option behind the `ModelBackend` seam but is not on the delivery path.

### Consequences

- Good: the primary path is `ort` — loads the real models unchanged, fastest measured, no
  reimplementation or parity risk.
- Good: the "candle-first" question is settled with concrete measurements, not speculation.
- Good: dropping `candle` from the near-term plan removes ~1 week of high-risk BiLSTM work.
- Bad: the native primary keeps a C++ runtime (`ort`); the fully-pure-Rust path rests on
  `tract`, which still needs its own validation on these models.
- Revisit if: `candle-onnx` upstream adds bidirectional LSTM, padded MaxPool, and a fast CPU
  LSTM kernel; or a hard no-C++-runtime requirement lands and `tract` cannot run the models —
  in which case the hand-written `candle` path becomes the fallback despite its cost.
