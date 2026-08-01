---
status: accepted
date: 2026-08-01
deciders: Na'aman Hirschfeld
---

# Parity-safe performance optimization and SIMD

## Context and Problem Statement

The pipeline is pinned to EasyOCR output by committed golden fixtures
(`crates/sceptre/tests/data/golden/*.json`) that record exact recognized text and
confidences. We want to speed up the numeric hot paths — recognition and detection
normalization, and CTC softmax + argmax — but any change that shifts a single ULP
would change a confidence and fail the golden check. We also enforce
`unsafe_code = "deny"` workspace-wide and ship no nightly toolchain, so hand-written
`unsafe` SIMD and fast-math intrinsics are off the table.

## Decision Drivers

- Output must stay **bit-identical**: the goldens require a zero diff, so every
  optimization must preserve IEEE-754 arithmetic and its evaluation order.
- No `unsafe`, no nightly, no `target-cpu` pinning — portability and the
  workspace `unsafe` ban are hard constraints.
- Each optimization must be provable, not asserted: a differential test and an A/B
  benchmark, both retaining the original implementation as the reference.

## Considered Options

- Explicit SIMD via `std::arch` intrinsics or the `wide` crate.
- Fast-math / reassociated reductions (e.g. telescoping the softmax renormalizer).
- Compiler autovectorization of refactored contiguous-slice loops, holding the
  arithmetic and its order fixed.

## Decision Outcome

Chosen option: **compiler autovectorization of contiguous-slice loops with the
arithmetic and evaluation order held identical to the reference.** The bit-identical
rule governs all future perf work in the numeric paths.

Applied to three hot paths:

- **Recognition normalize** (`recognize/preprocess.rs::fill_plane`): iterate the
  grayscale image's contiguous `GrayImage::as_raw()` backing slice with a
  `zip`, applying the same `(v/255 - 0.5)/0.5` per pixel, instead of the
  bounds-checked per-pixel `get_pixel`.
- **Detection normalize** (`detect/preprocess.rs::normalize_into_tensor`): iterate
  the resized image's contiguous interleaved-RGB `as_raw()` slice via
  `chunks_exact(3)`, writing each channel plane with the same `(raw - mean)/std`,
  instead of the channel-strided `get_pixel`. Padding pixels keep the normalized
  zero `(0 - mean)/std`.
- **Fused softmax + argmax** (`recognize/ctc.rs::decode_greedy`): a single pass per
  timestep that drops the full `Array2<f32>` probability buffer and the redundant
  full-width renorm-division loop, producing the same `(class, probability)`.

### Fused-softmax division-order constraints

Bit-identity of the fused decoder is preserved only by keeping numpy's operation
order exactly; these constraints are load-bearing and must not be "simplified":

- The exp-`sum` is over **all** classes, including `ignore` classes, in class order
  — the reference sums the softmax weights before zeroing the ignored ones.
- `renorm` sums the per-cell `weight / sum` quotients over the non-ignored classes,
  in class order. It must **not** be algebraically telescoped to
  `weight_a / Σ_{non-ignore} weight`: that reorders the rounding and is not
  bit-identical.
- The argmax is folded into the exp pass over the non-ignored classes with numpy's
  first-index tie rule; the reported probability is the fused
  `(weight_argmax / sum) / renorm`, matching the reference's value at that cell.
- `custom_mean`, `collect_max_probs` ordering, blank handling, and repeat-collapse
  are unchanged.

### Proof obligations

Every optimization retains its original implementation as a `*_reference` function
gated `#[cfg(any(test, feature = "bench"))]`, and is proved by:

- a **differential test** asserting the optimized path equals the reference exactly
  (tensor buffers compared on `f32` bit patterns; CTC text and confidence compared
  on text equality and `f32` bit patterns) over varied inputs plus the fixtures;
- **A/B Criterion benches** timing an `.../optimized` and an `.../reference` arm in
  one run (via the `bench` seam), plus a saved `pre-opt` baseline; and
- the **unchanged golden fixtures**, regenerated with a zero diff.

### Consequences

- Good: measurable speedups with no parity risk; the reference impls make any future
  regression a failing differential test rather than a silent golden drift.
- Good: no `unsafe`, no nightly, no non-portable codegen flags.
- Bad: the reference impls and A/B bench arms are carried as `bench`/`test`-only
  code; the fused decoder's division order is subtle and must be respected.
- Neutral: gains depend on the host compiler's autovectorizer. Where it proves
  insufficient, the fix is a fresh ADR — not `unsafe` SIMD bolted on here.

### SIMD explicitly deferred

Vectorizing `exp` (via a polynomial approximation or a SIMD math crate) is **not**
bit-identical: it changes the rounding of the softmax weights and would require its
own tolerance budget and re-baselined goldens. It is deferred and, if pursued, needs
its own ADR that supersedes the bit-identical rule for that path.
