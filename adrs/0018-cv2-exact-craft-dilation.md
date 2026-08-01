---
status: accepted
date: 2026-08-01
deciders: Na'aman Hirschfeld
---

# cv2-exact CRAFT dilation (amends ADR 0013)

## Context and Problem Statement

ADR 0013 adopted `imageproc` for CRAFT post-processing geometry (connected components,
morphological dilation, min-area rectangle). For the dilation step it used a deliberate
approximation: EasyOCR dilates each component's segmentation map with
`cv2.dilate` over a `cv2.getStructuringElement(MORPH_RECT, (1 + niter, 1 + niter))` kernel,
whereas `imageproc`'s `dilate(Norm::LInf, radius)` grows a symmetric `(2·radius + 1)` square,
so the code used `radius = niter / 2`.

That approximation matches EasyOCR only for **even** `niter`. For **odd** `niter` the imageproc
kernel is one pixel smaller (`niter` vs `1 + niter`) and symmetric, so component boxes come out
marginally narrower and inter-word gaps marginally wider. On borderline gaps (near
`width_ths · height`) this flips EasyOCR's horizontal word-merge into a split, fragmenting text
lines and dropping per-line box-IoU parity below threshold (observed on `french.jpg`). The line
grouping itself (`detect::group`) is a faithful port of `group_text_box`; the divergence was
purely this box geometry.

## Decision Outcome

Replace the imageproc dilation with a small custom dilation that reproduces OpenCV exactly:
`dst(x, y) = max` over kernel offsets `(i, j) ∈ [0, 1 + niter)` of `src(x + i − anchor, y + j −
anchor)`, with OpenCV's default center anchor `anchor = (1 + niter) / 2` (integer division) and
out-of-buffer neighbours treated as background (`BORDER_CONSTANT`). The kernel is symmetric for
even `niter` and grows one pixel more on the anchor side for odd `niter`, matching `cv2.MORPH_RECT`.
The per-component window margin is set to `niter` (matching EasyOCR's dilated slice bounds), which
comfortably contains the grown halo, so the windowed per-component build stays identical to a
whole-map build (guarded by the existing windowing-equivalence test).

This **amends ADR 0013** for the dilation step only; connected-component labelling and min-area
rectangle fitting remain on `imageproc`.

### Consequences

- Good: box geometry now matches EasyOCR bit-for-bit through the dilation, so horizontal
  word-merges resolve as they do upstream; parity improved (the `Mairie du` split is gone).
- Good: no new dependency — a ~20-line safe scalar kernel, no `unsafe`.
- Neutral: the dilation is O(area · (1 + niter)²) instead of imageproc's separable pass; `niter`
  is small (single digits) and the scan is already windowed to the component bbox.
- Limited: strongly-rotated multi-word lines can still diverge via min-area-rect corner rounding
  (a separate geometry detail, tracked by the ignored `parity_french_jpg` case); this ADR does not
  address that.
