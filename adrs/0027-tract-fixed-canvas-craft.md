---
status: accepted
date: 2026-08-03
deciders: Na'aman Hirschfeld
---

# Fixed-canvas CRAFT detection on the tract backend

## Context and Problem Statement

[ADR 0004](0004-inference-backend-seam.md) makes `tract` the pure-Rust backend for WASM/Android.
For that path to be a real alternative to `ort`, the whole pipeline — detection and recognition —
must run under tract. Two obstacles surfaced when validating this against real models:

1. The upstream itextresearch recognizer exports do not load under tract at all: their
   `AdaptiveAvgPool2d` lowers to a pooling op tract cannot shape-infer under a dynamic width.
2. The CRAFT detector does not load under tract with a dynamic input: its U-net upsamples with
   `Resize` and concatenates the result with a skip connection, and tract cannot prove the
   `Resize`-upsampled extent equals the skip extent when H/W are symbolic
   (`Undetermined symbol … (height)/8`). itextresearch's CRAFT export fails identically.

## Decision Drivers

- A working end-to-end pure-Rust pipeline on tract, not just recognition.
- Keep the `ort` path unchanged (dynamic shapes, best throughput).
- Keep the model-execution seam model-agnostic (ADR 0004): no per-model `#[cfg]` on config/types.

## Considered Options

- **Upgrade tract** (0.22 → 0.23) hoping newer shape inference resolves the dynamic CRAFT — tested;
  it does not (same `Resize` symbolic failure).
- **Fixed-size CRAFT export for tract** — a second, statically-shaped ONNX; adds a backend-specific
  model artifact and a second thing to host and pin.
- **Fixed square detection canvas on tract**, pinning the existing dynamic CRAFT ONNX to that shape
  at load time.

## Decision Outcome

Chosen option: **on the tract backend, run CRAFT on a fixed square canvas.**

- The first-party export pipeline (ADR 0025) replaces the recognizer's `AdaptiveAvgPool2d` with a
  numerically identical mean over the height axis (a `ReduceMean`), which **makes the recognizers
  load and run under tract with a dynamic width** — something the upstream exports never did. No
  fixed canvas is needed for recognition.
- For detection, the engine pads every image to a fixed `canvas × canvas` square (the configured
  `detection.canvas_size`, rounded up to CRAFT's 32-pixel alignment) on the tract backend, and pins
  the CRAFT model's input to the matching `[1, 3, canvas, canvas]` shape before `into_optimized()`.
  With concrete dimensions tract resolves the U-net shapes and optimizes cleanly. `ort` keeps the
  dynamic, aspect-ratio-padded path unchanged.
- The seam gains a runtime `fixed_input: Option<&[usize]>` on `load_backend` (a value, not a
  type-level `#[cfg]`), so ADR 0004's "config/types stay backend-agnostic" rule holds; `ort` ignores
  it.

### Consequences

- Good: a full pure-Rust OCR pipeline runs on tract; `backend_agreement` proves tract recognizes the
  same words as ort for Latin and Cyrillic.
- Bad: tract detection pads to a square, so it computes over more pixels than the aspect-fitted `ort`
  canvas, and tract's CRAFT optimization is a heavier one-time per-load cost.
- Neutral: the fixed canvas can shift CRAFT's heat-map enough to split or merge a line differently
  from ort (a trailing word landing on its own line); the recognized words are identical, so the
  cross-backend test compares the word multiset rather than raw line grouping.
- Neutral: pinning couples detection padding to the backend at runtime; recorded here rather than
  leaking into the config/result types.
