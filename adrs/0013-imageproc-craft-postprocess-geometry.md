---
status: accepted
date: 2026-07-31
deciders: Na'aman Hirschfeld
---

# `imageproc` for CRAFT postprocess geometry

## Context and Problem Statement

EasyOCR's CRAFT postprocess (`craft_utils.py:getDetBoxes_core`) leans on OpenCV for four
image-geometry primitives: threshold the region/link heat-maps, label connected components
with per-component stats, dilate each component's segmentation map with a rectangular kernel,
and fit a minimum-area (rotated) rectangle to the component pixels. We reimplement CRAFT in
Rust with no OpenCV/native dependency, so we need those primitives in pure Rust.

## Decision Drivers

- Faithful reproduction of the CRAFT box-extraction algorithm (connected components + rotated
  min-area rect are the load-bearing steps).
- Avoid re-deriving numerically fiddly geometry (convex hull + rotating calipers) by hand.
- Pure-Rust, WASM-compatible, no native/OpenCV dependency.
- Consistency with the sibling xberg project, which already ships an ONNX text detector
  (DBNet) postprocess in Rust.

## Considered Options

- **Hand-roll** connected components, morphological dilation, and min-area-rect. Full control,
  but rotating-calipers min-area-rect is error-prone and pure duplication of solved work.
- **`imageproc`** — mature pure-Rust image-processing crate providing
  `region_labelling::connected_components`, `morphology::dilate`, `contrast::threshold`, and
  `geometry::min_area_rect` (convex hull + rotating calipers). xberg already uses it for its
  DBNet postprocess (`find_contours` + `min_area_rect`).
- **OpenCV bindings** (`opencv` crate) — exact parity with EasyOCR, but a heavy native
  dependency that breaks the pure-Rust WASM/Android story ([ADR 0009](0009-candle-evaluation-ort-primary.md)).

## Decision Outcome

Chosen option: **`imageproc`** (pinned `0.27`, `default-features = false`), a core (always
compiled) dependency of the detection stage. It supplies every geometry primitive CRAFT's
postprocess needs in pure Rust: connected-component labelling (4-connectivity, matching
`cv2.connectedComponentsWithStats(connectivity=4)`), morphological dilation, thresholding, and
`min_area_rect` — which returns corners already in `[top-left, top-right, bottom-right,
bottom-left]` order, the clockwise convention EasyOCR's `boxPoints` path also produces.

Per-component *stats* (area, bounding box) are not provided by `connected_components` (it
returns only a labelled image); we compute them from the label image in a single pass, which is
trivial and keeps us independent of a stats API. The rectangular structuring element EasyOCR
builds with `getStructuringElement(MORPH_RECT, (1+niter, 1+niter))` is approximated by
`imageproc`'s `dilate(Norm::LInf, k)`; exact kernel-shape parity is not required for detection
quality and is validated later against golden box output (box IoU threshold), not asserted
pixel-exactly.

`imageproc` stays confined to the `detect` module's postprocess stage; it is not part of the
backend seam and does not touch `config`/`types`.

### Consequences

- Good: the numerically delicate min-area-rect (convex hull + rotating calipers) is a vetted
  library routine, not hand-rolled.
- Good: pure Rust — no OpenCV, preserving the WASM/Android pure-Rust path.
- Good: aligns with xberg's detector-postprocess stack, so patterns transfer.
- Bad/limited: `imageproc`'s dilation kernel shape differs slightly from OpenCV's rectangular
  element; parity is asserted at the box level via golden IoU, not per pixel.
- Neutral: `imageproc` pulls `nalgebra`/`num` into the tree (pure Rust, covered by `cargo-deny`).

## Status update (2026-08-01)

The dilation step is **amended by ADR 0018**: it now uses a custom cv2-exact `MORPH_RECT(1 + niter)`
kernel instead of `imageproc`'s `LInf` dilation, closing the "differs slightly from OpenCV"
limitation above. Connected-component labelling and min-area rectangle fitting stay on `imageproc`.

## Status update (2026-08-08)

The min-area-rect step is **superseded by ADR 0039**: `imageproc::geometry::min_area_rect`
snapped each rotated-rectangle corner outward with a per-corner `.floor()`/`.ceil()`, which
OpenCV's `minAreaRect`/`boxPoints` never does; that snap was measurably fragmenting rotated text
lines (`french.jpg`). Rotated-quad fitting now uses an in-crate rotating-calipers implementation
that reuses `imageproc`'s (unaffected) `convex_hull` but keeps corners in floating point with no
outward rounding. Connected-component labelling and the ADR 0018 dilation kernel are unaffected
and remain as described above.
