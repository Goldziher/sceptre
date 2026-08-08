---
status: accepted
date: 2026-08-08
deciders: Na'aman Hirschfeld
---

# Opt-in megapixel budget for detection, `canvas_size` default unchanged

## Context and Problem Statement

Peak process memory is dominated by the CRAFT forward pass, not by anything sceptre
allocates itself: `detect` alone measures 6236 MB RSS versus 6281 MB for a full
`readtext` run on the same large page, recognition adds only ~45 MB, and recognizer
language is irrelevant (6586 MB Japanese vs. 6580 MB English on the same image). Every
ORT session knob was measured and none help — disabling the arena is 700 MB *worse*.
Sceptre's own allocations are ~1.5% of peak; ORT's arena holds the rest.

`DetectionConfig::canvas_size` already exists as a way to shrink the detection input,
but it bounds the *longest side*, not memory. Fitting RSS against 91 (image, canvas)
points over the actual resized/padded detection input gives:

- `RSS_MB ≈ 1063 * megapixels + 326` (R² = 0.984)
- `RSS_MB ≈ 2.26 * longest_side - 889` (R² = 0.847)

Area is decisively the better predictor. A caller who wants a memory ceiling and reaches
for `canvas_size` is aiming at the wrong variable: two images with the same longest side
can differ 2× in area depending on aspect ratio, and `canvas_size` cannot express "keep
memory under N" without the caller first computing an aspect-ratio-dependent longest-side
cap by hand.

## Decision Drivers

- Bound peak detection memory without forcing every caller to hand-derive a
  longest-side cap from their own image's aspect ratio.
- The EasyOCR parity gate (ADR 0021) is the correctness bar: default detection output
  must stay byte-identical unless an ADR says otherwise, and none does here.
- The parity corpus cannot referee a `canvas_size` default change: all eight tier-2
  images are natively ≤ 1024 px on their longest side, so raising or lowering the
  default anywhere above that is invisible to the gate. Line counts on genuinely large
  pages swing 8/18/64/116/75/35/30 across `canvas_size` 768–3200 in exploratory testing,
  and none of those pages have ground truth — changing the default would be tuned
  against nothing.

## Considered Options

- **Lower the `canvas_size` default.** Rejected: the parity corpus cannot see the
  effect (all images already fit under any plausible new default), so there is no
  measurement to justify a specific number, and the large pages where it *would* matter
  have no ground truth to check the accuracy trade against. This would be moving a
  load-bearing default on vibes.
- **Document a formula for callers to derive a `canvas_size` from a memory target.**
  Pushes aspect-ratio and padding-to-32 arithmetic onto every caller and silently
  breaks the moment `canvas_size`'s relationship to memory changes (e.g. a future
  backend with a different arena growth curve).
- **A separate megapixel budget on `DetectionConfig`, opt-in, composing with
  `canvas_size` as a minimum.** Bounds the actual measured driver of peak memory
  directly, defaults to `None` so it cannot move today's output, and composes rather
  than replaces the existing knob.

## Decision Outcome

Chosen: **`DetectionConfig::max_megapixels: Option<f32>`, default `None`.**

When set, it further constrains the resize target that `canvas_size` and `mag_ratio`
already compute, so the *padded* (multiple-of-32) input area — the quantity the RSS fit
above actually measured, not the unpadded resize target — stays within the budget. The
two constraints compose as their minimum: whichever of `canvas_size` or
`max_megapixels` is more restrictive for a given image wins. Implementation lives in
`detect::preprocess::resize_dimensions_within_budget`, which binary-searches the
largest effective canvas cap in `[1, canvas_size]` whose padded area fits the budget —
sound because `resize_dimensions`'s target dimensions (and therefore the padded area)
are monotonically non-decreasing in that cap, so the search converges on the
least-restrictive size that still satisfies the budget without ever exceeding what
`canvas_size` alone allows.

`None` takes an early return before any of that runs, reproducing today's
`canvas_size`/`mag_ratio` sizing bit-for-bit — verified by a test that compares the
`f32` ratio by bit pattern, not by epsilon, across several image sizes. The tier-2
parity gate is therefore unaffected by construction, not just by intent.

On the `tract` backend's fixed-square canvas path (ADR 0027), the padded output area is
pinned by the fixed canvas regardless of the budget — only the real (unpadded) content
pasted into that square shrinks. The budget's memory benefit is therefore specific to
the dynamic-shape path (`ort`, the default backend), which is also where the RSS
measurement below was taken.

### Measured: RSS tracks the fit

Built the release CLI and ran `/usr/bin/time -l` over
`ndl_meiji_vertical_01.jpg --lang japanese` (3630×2777, large enough that the default
`canvas_size` binds and the padded detection input sits at 5.08 MP):

| `max_megapixels` | achieved padded area | measured peak RSS | fit (`1063 * MP + 326`) |
|---|---|---|---|
| unset | 5.079 MP (unchanged) | 6592.2 MB | 5725 MB |
| `2.0` | 1.997 MP | 2229.7 MB | 2449.6 MB |
| `0.5` | 0.486 MP | 841.0 MB | 843.1 MB |

The unset row reproduces its own previously-measured baseline (~6590 MB) exactly, as it
must — this ADR does not touch that path. Against the independently-derived area fit,
the two budgeted points land within 9% (`2.0` MP) and 0.3% (`0.5` MP); the unset point
sits about 15% above it, which is ordinary scatter for a single point against an
R² = 0.984 fit built from the other 91 and does not change the conclusion: RSS falls as
the budget's achieved area falls, tracking the fit closely enough at the two points this
feature actually controls (the budgeted ones) to trust the knob does what it claims.

### Consequences

- Good: bounds the metric that actually drives peak memory, not a proxy for it.
- Good: default output is bit-for-bit unchanged; the parity gate needs no update and no
  new ADR to justify a moved baseline.
- Good: composes with `canvas_size` rather than replacing it, so existing configs and
  the CLI's `--canvas-size` flag keep working unchanged.
- Neutral: on `tract`'s fixed-square canvas, the budget shrinks real content but not
  the padded memory footprint — a limitation inherent to ADR 0027's fixed-shape
  requirement, not something this feature can fix without revisiting that ADR.
- Bad: a budget small enough to bind hard trades detection accuracy for memory on large
  pages, the same trade `canvas_size` already makes — this ADR does not change that
  trade-off, only how precisely a caller can target it.

## Related

- ADR 0021 — the EasyOCR parity gate this decision leaves untouched by construction.
- ADR 0027 — the `tract` fixed-square canvas this budget cannot shrink the memory
  footprint of, only the real content within it.
