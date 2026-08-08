---
status: accepted
date: 2026-08-08
deciders: Na'aman Hirschfeld
---

# OpenCV-faithful min-area-rect fitting (supersedes ADR 0013's min-area-rect half)

## Context and Problem Statement

ADR 0013 adopted `imageproc` for CRAFT post-processing geometry, including
`imageproc::geometry::min_area_rect` for fitting a rotated bounding rectangle to each
component's pixels. That function computes the rectangle correctly in `f64` via convex hull +
rotating calipers, but before returning it **unconditionally snaps each of the four corners
outward** with a per-corner `.floor()`/`.ceil()` (top-left floors both axes, top-right ceils `x`
and floors `y`, and so on). OpenCV's `minAreaRect`/`boxPoints` — what EasyOCR's
`craft_utils.py:getDetBoxes_core` actually calls — never does this: EasyOCR keeps the box in
floating point all the way through `group_text_box`, and `cv2.boxPoints` returns `float32`
corners with no rounding at all.

For axis-aligned boxes this snap is a no-op (the true corners are already integers). For a truly
rotated box it inflates the rectangle by up to ~1px per corner in heat-map space, which becomes
up to ~2px after the CRAFT postprocess's `x2` scale-up (`adjust_coordinates`, `ratio_net = 2`).
That was enough to flip borderline `group_text_box` merge decisions on real images:

- On `french.jpg`, sceptre's `LOUVRE` box corners drifted from EasyOCR's by 1-2px per corner,
  pushing the line's slope from cv2's ~0.096 to ~0.129 — just over `group.rs`'s `slope_ths`
  (0.1) — so it was routed to the free/rotated list instead of merging with `[Palais du`,
  splitting one reference line into two.
- The same box's height came out 46px in sceptre vs. 44px in EasyOCR, independently tripping the
  `height_ths` merge boundary for a second line.

Both are box-geometry artifacts of the outward snap, not detector or recognizer error — the line
grouping in `detect::group` is already a faithful port of `group_text_box`.

Also fixed in passing: `imageproc::geometry::rotating_calipers` only tests hull edges via
`windows(2)`, which never tests the edge from the last hull vertex back to the first. Rotating
calipers requires testing every hull edge (the true minimum-area rectangle is always flush with
one of them); omitting the closing edge can only ever pick an equal-or-larger rectangle. This is
a real, if usually small, extra source of over-large boxes independent of the rounding bug.

## Decision Drivers

- Match OpenCV's `minAreaRect` + `boxPoints` contract exactly enough that rotated-quad geometry
  agrees with EasyOCR's, not just axis-aligned geometry (which already matched bit-for-bit).
- Stay pure Rust (no OpenCV/native dependency) per ADR 0013's original driver.
- Minimal surface change: keep `imageproc` for connected components (unaffected) and its (correct,
  exact-arithmetic) `convex_hull`; only the rotating-calipers + corner-rounding step needed
  replacing.

## Considered Options

- **Patch around `imageproc`'s output** (e.g. shrink each corner back by ~1px). Rejected: a
  blind compensating offset does not reproduce OpenCV's actual geometry and would be wrong for
  some orientations.
- **Vendor/fork `imageproc`'s `rotating_calipers`.** Its rotation/inversion helpers
  (`Point<f64>::rotate`, `invert_rotation`, `Rotation`) are `pub(crate)` to `imageproc`, so a fork
  would need to duplicate that arithmetic anyway; forking the whole function for a ~15-line change
  adds more surface than writing it directly.
- **Hand-roll rotating calipers in-crate, reusing `imageproc::geometry::convex_hull`.** The convex
  hull step itself has no rounding bug (exact integer cross-product orientation test), so only the
  calipers + corner-selection step needs reimplementing. Chosen.

## Decision Outcome

Chosen option: a small in-crate rotating-calipers implementation
(`crates/sceptre/src/detect/min_area_rect.rs`) that reuses `imageproc::geometry::convex_hull`
(unaffected by this bug) and then, for the hull's minimum-area rectangle:

- tests **every** hull edge, including the closing one `imageproc` omits;
- keeps the winning rectangle's corners as `f64` for the whole computation;
- returns those `f64` corners **un-rounded** — no per-corner floor/ceil.

The caller (`detect::postprocess::fit_box`) does the single final cast to `f32` that
`BoxPoints` requires, mirroring `cv2.boxPoints`'s own `float32` output dtype rather than four
independent outward roundings. Corner ordering (top-left, top-right, bottom-right, bottom-left)
is preserved exactly as ADR 0013 described it.

This **supersedes ADR 0013's min-area-rect half only**; connected-component labelling stays on
`imageproc` (ADR 0013), and dilation stays on the custom cv2-exact kernel (ADR 0018).

### Consequences

- Good: rotated-quad boxes now agree with EasyOCR/OpenCV geometry, not just axis-aligned ones.
  Measured full `SCEPTRE_REQUIRE_MODELS=1` tier-2 corpus run, before/after, every number held or
  improved with no regressions:

  | image | word_f1 (before → after) | line_recall | line_precision | line_f1 |
  |---|---|---|---|---|
  | english.png | 1.000 → 1.000 | 1.000 → 1.000 | 0.923 → **1.000** | 0.960 → **1.000** |
  | french.jpg | 0.933 → 0.933 | 0.833 → **1.000** | 0.625 → **1.000** | 0.714 → **1.000** |
  | chinese.jpg | 0.900 → 0.900 | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 |
  | japanese.jpg | 0.895 → 0.895 | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 |
  | korean.png | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 |
  | cyrillic.png | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 |
  | telugu.png | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 |
  | kannada.png | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 |

  `french.jpg` — the motivating case — moves from the worst line scores in the corpus to a
  perfect line match; `english.png`'s only non-1.000 number (line precision, from an unrelated
  minor over-split) also reaches 1.000. No other image moved at all, and none regressed. This was
  a clean win, not a trade-off.
- Good: also picks up the "missing closing edge" correctness fix for free, which can only shrink
  (never grow) a fitted box.
- Good: `parity_french_jpg`'s long-standing documented split (see its doc comment in
  `tests/tier2_golden.rs`) is resolved; the sceptre-snapshot golden fixtures were regenerated
  (`cargo run -p sceptre-tools --features ort -- snapshot`) to reflect the corrected box/line
  output.
- Neutral: `imageproc` stays a dependency (connected components, dilation input types, and
  `crop.rs`'s `geometric_transformations` still use it); only its `min_area_rect` is no longer
  called.
- Neutral: the new module is ~200 lines of straightforward rotation arithmetic, unit-tested
  directly against known cases (including one built by differentially comparing against
  `imageproc::geometry::min_area_rect` on the same input, and one against a windows(2)-only
  variant of the calipers scan) rather than against the full pipeline.
