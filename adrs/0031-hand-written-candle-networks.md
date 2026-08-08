---
status: accepted
date: 2026-08-07
deciders: Na'aman Hirschfeld
---

# Hand-written `candle` networks as the GPU and no-C++-runtime backend

## Context and Problem Statement

[ADR 0009](0009-candle-evaluation-ort-primary.md) evaluated `candle` as a *primary CPU*
backend and deferred it: `candle-onnx` could not load either model, and `candle`'s CPU
BiLSTM alone measured ~8× slower than `ort`'s entire recognizer. That answer was correct
and, as re-measured below, still is.

The question reopened here is a different one. `ort` reaches a GPU only through an ONNX
Runtime execution provider, which means the GPU story is tied to provisioning a C++ runtime
built with the right provider — the matrix [ADR 0029](0029-cli-provisioning-default-and-runtime-scoped-parity.md)
had to work around. Two things were wanted that `ort` and `tract` together do not offer:

1. **GPU execution** (Apple Metal, NVIDIA CUDA) without a C++ runtime and without
   provider-specific ONNX Runtime builds.
2. **A native path with no ONNX Runtime dependency at all**, for environments where
   shipping or loading `libonnxruntime` is the obstacle rather than speed.

ADR 0009's "revisit if" clause names exactly this: a hard no-C++-runtime requirement, with
the hand-written `candle` path as the fallback despite its cost.

## Decision Drivers

- GPU execution that does not depend on the ONNX Runtime provisioning matrix.
- A native backend with no C++ runtime to ship, load, or match to a provider.
- No new model artifacts: the registry, sha256 pins, and export pipeline must not move.
- No build-time toolchain requirements that would break `cargo install` (ADR 0029).
- Bit-level accountability: a hand-written network can diverge from the graph invisibly.

## Considered Options

- **Stay deferred.** Costs nothing, delivers neither driver.
- **Execute the ONNX graph with `candle-onnx`.** Re-verified against upstream `main`, and
  it is disqualified twice over. It still `bail!`s on all three constructs these models
  need — bidirectional `LSTM`, `MaxPool` with non-zero pads, and `Resize` with
  `mode != nearest` / `coordinate_transformation_mode != asymmetric` (the `Resize` gap is
  new since ADR 0009, which never recorded it). Independently, its `build.rs` runs
  `prost_build::compile_protos`, and prost-build has not bundled `protoc` since 0.11, so
  depending on it would break `cargo install --features candle` on any machine without
  protoc — reintroducing the zero-config-install defect ADR 0029 fixed.
- **Upstream the three ops to `candle-onnx`.** `deny.toml` sets `allow-git = []`, so a fork
  cannot ship; this would block on an upstream crates.io release.
- **Hand-write the two networks against `candle_nn`**, reading weights from the existing
  ONNX initializers.

## Decision Outcome

Chosen option: **hand-write CRAFT and the gen2 CRNN against `candle_nn`**, with weights
read from the same ONNX files the other backends load.

- **No new artifacts.** `prost` alone has no build script, so ~6 ONNX messages are declared
  by hand and decoded with prost's wire reader. No `protoc`, no codegen, no second weight
  format. ADR 0025's model source is untouched.
- **Weights are named positionally**, not by initializer name: roughly half of those names
  are exporter-generated counters (`onnx::Conv_299`) that shift on every re-export, and
  ADR 0025 makes re-export routine. The `n`-th `Conv` becomes `conv.{n}`, which depends
  only on graph topology.
- **The seam gains `NetworkKind`** on `BackendOptions`. A backend that runs a hand-written
  forward pass cannot recover the architecture from bytes alone, and both engine call sites
  know the answer already. This mirrors ADR 0027's `fixed_input`: a runtime *value*, not a
  type-level `#[cfg]`, so config and result types stay backend-agnostic. `ort` and `tract`
  ignore it. Because weights are resolved positionally, the graph is still *validated*
  against the requested kind — a host-supplied stranger graph (ADR 0028) must error rather
  than load whatever tensors sit at those positions and produce confident nonsense.
- **candle keeps the dynamic CRAFT canvas.** A hand-written CRAFT reads each skip tensor's
  extent at runtime, which is precisely what tract could not do (ADR 0027), so candle sees
  byte-identical preprocessed input to `ort`.
- **GPU support is a per-backend accelerator vocabulary**, recorded in
  [ADR 0032](0032-per-backend-accelerator-support.md).

### Pinning `candle` at 0.11, and the honesty cost

ADR 0009 measured against 0.9.2, and this work started there. `candle-core` 0.9.x is
**unusable for CRAFT**: its tiled im2col fast path detects a channels-last tensor by testing
`stride == [i_h*i_w*c_in, i_w*c_in, c_in, 1]`, which a contiguous NCHW tensor also satisfies
whenever `C == W == H`. CRAFT hits that coincidence at 128 channels on a 128×128 feature map
and one convolution is silently computed wrong. This was reproduced with a synthetic
convolution independent of sceptre (128×128 fails, 16×16 passes, exactly as the coincidence
predicts). Upstream deleted the shortcut in 0.10.0.

0.10.0 is also the release that made `tokenizers` a non-optional dependency of `candle-core`
with the `onig` feature on non-wasm targets, which compiles oniguruma from C. So:

> **`candle` is not a pure-Rust backend.** It pulls `onig_sys`, the one C-building crate in
> the tree. What it removes is the *ONNX Runtime* C++ runtime and its provisioning matrix —
> a real and different benefit. Documentation must claim that and not more.

The 0.11 pin is therefore deliberate on both ends and must not be swept up by
`cargo upgrade --incompatible`; the workspace manifest carries the reason inline.

### Verification

A hand-written forward pass can diverge from the graph anywhere — a mistransposed weight, a
misordered LSTM gate, a skip connection taken one convolution too late — and every one of
those produces plausible output. Two independent bars, both against `ort`:

- **Tensor level:** whole-model output on fixed pseudo-random input, over the same shapes
  `sceptre_rs_tools.export` already validates torch against onnxruntime, within `1e-4`
  absolute. Not `0.0`: candle accumulates bilinear interpolation weights in `f64` where
  ONNX Runtime uses `f32`, so bit-exactness is unachievable and is not claimed.
- **End to end:** exact sorted word-multiset equality with `ort` on `english.png` and
  `cyrillic.png` — no numeric tolerance to hide behind.

Two op-level semantics are pinned against ONNX Runtime ground truth in unit tests, because
both are easy to get plausibly wrong: ONNX `MaxPool` pads with `-infinity` rather than zero
(CRAFT's padded pool consumes a raw convolution output with no rectifier between, so a
fabricated `0.0` would win the maximum), and ONNX `Resize` `half_pixel` is candle's
`upsample_bilinear2d(.., align_corners = false)`.

One trap deserves naming: `candle_nn`'s `Direction::Backward` only selects the `_reverse`
weight-name suffix — `RNN::seq` always iterates forwards — so the bidirectional layer
reverses the sequence itself.

### Performance, stated plainly

Measured on this repository's images (release build, Apple Silicon, best of runs):

| Image | `ort` / cpu | `candle` / cpu | `candle` / metal |
| --- | --- | --- | --- |
| `english.png` | 0.29 s | 2.88 s | 0.89 s |
| `balance_sheet_1.png` | 1.59 s | 12.89 s | 5.12 s |
| `doclaynet_page_01.jpg` | 2.04 s | 20.82 s | 8.48 s |

**`candle` on a GPU is still slower than `ort` on the CPU**, by roughly 3–4× here, and its
CPU path is ~10× slower — consistent with ADR 0009's measurement, which stands. Metal buys
~2.5× over candle's own CPU path and nothing relative to `ort`.

> **Status update (2026-08-08).** These ratios are host-dependent and the numbers above are
> specific to the stated setup. The ADR 0035 backend matrix measures `candle`/cpu against
> `ort`/cpu at **~3.6×** on `runner-large-arm64` and **~1.8–2.0×** on `macos-latest`
> (`canvas_size` 1024), against the ~10× recorded here — three aarch64 hosts, three different
> ratios. The absolute times move with them: `ort`/cpu takes 1154 ms on `runner-large-arm64` and
> 2269–2632 ms on `macos-latest` for the same work, so the gap compresses on constrained hosts
> where `ort`'s multithreading has fewer cores to exploit, rather than because either backend
> changed. `tract` spans even wider — 4.2× to 26.5× — on the same two runners.
>
> The **ordering is invariant across every host measured**: `ort` fastest on CPU, then `candle`,
> then `tract`. That ordering, not any particular multiple, is what this ADR's decision rests on,
> and it stands as written. The lesson for anyone quoting these numbers is that the multiple is
> meaningless without its host. `candle` is a compatibility
and deployment option, never a performance one, and it is excluded from
[ADR 0030](0030-published-benchmark-artifact-and-drift-gate.md)'s benchmark drift gate.

### Consequences

- Good: a native backend with no ONNX Runtime dependency, and a GPU path that needs no
  provider-specific runtime build.
- Good: no new model artifacts, no `protoc`, no build-time codegen.
- Good: correctness is held to `ort` at two independent levels rather than asserted.
- Bad: two networks are now maintained by hand. A re-export that changes either
  architecture — not merely its weights — breaks candle and only candle. The positional
  weight naming and the node-count validation are what turn that into an error instead of
  silent corruption.
- Bad: `onig_sys` compiles C, so the "pure Rust" claim belongs to `tract` alone.
- Neutral: `candle` is opt-in and not in any default feature set.
- Revisit if: `candle-onnx` grows the three ops *and* drops its `protoc` build dependency,
  which would let the graph be executed rather than reimplemented.

## Supersedes

[ADR 0009](0009-candle-evaluation-ort-primary.md). Its CPU conclusion is not overturned —
it is confirmed by fresh measurement. What changes is the question: 0009 asked whether
`candle` should be the primary CPU backend (no, still no), and this ADR answers whether it
should exist as a GPU and no-C++-runtime option (yes).
