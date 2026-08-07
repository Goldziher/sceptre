---
status: accepted
date: 2026-08-07
deciders: Na'aman Hirschfeld
---

# Opt-in whole-page orientation pre-pass, disabled by default

## Context and Problem Statement

Sceptre had no whole-page orientation handling: a page scanned or photographed at 90°,
180°, or 270° went into CRAFT as-is. The cost is not marginal. On the 27-image labeled
corpus, eleven images are rotated or vertical, and they score CER 0.77–0.99 with
token-F1 near 0.00 — effectively total failure. The other sixteen average CER 0.360 /
token-F1 0.596. The corpus mean is therefore a average over two disjoint populations and
carries almost no information; excluding the eleven moves it from CER 0.584 to 0.360.

Two facts frame the decision:

- **EasyOCR fails identically.** `easyocr.Reader(['en']).readtext(ocr_test_rotated_90.png)`
  returns `['WI','2','Hl','8','5','9','8','7','g','UL','3']`. This is a shared
  architectural gap, not a sceptre regression, so fixing it is not a parity repair — it
  is a deliberate divergence that puts sceptre ahead.
- **Recognition was never the problem.** Pre-rotating that same image by 90° externally
  makes sceptre emit the ground truth verbatim. The detector and recognizer are both
  fine; nothing upstream tells them which way is up.

The question is how to decide which way is up, and whether to do it by default.

## Decision Drivers

- EasyOCR parity is the baseline contract. The tier-2 golden corpus must stay
  byte-identical unless an ADR says otherwise.
- Any orientation signal must come from something already in the pipeline. Adding a
  separate classifier model would mean a second download, a second `NetworkKind` for the
  `candle` backend to hand-write, and a new entry in the model registry.
- The pre-pass costs four extra CRAFT forward passes. That is a real latency tax on
  every page, including the overwhelming majority that are already upright.
- A wrong answer is worse than no answer. Rotating an already-upright page corrupts a
  page that previously worked.

## Considered Options

- **Do nothing.** Matches EasyOCR exactly and costs nothing. Leaves eleven of
  twenty-seven corpus images at near-total failure and forgoes the single largest
  quality win available.
- **A dedicated orientation-classifier model.** Most accurate in principle, and the
  standard approach in document-OCR stacks. Costs a new model download, a new hand-written
  `candle` network, registry and ADR 0035 backend-matrix churn, and a new parity surface —
  a large amount of new machinery for a pre-pass.
- **Reuse CRAFT: probe all four rotations, score the heat-maps, detect on the best.**
  No new model, no new backend work, no registry change. Costs four reduced-canvas
  forward passes and needs a scoring function that actually discriminates.
- **Probe, but only on demand (opt-in).** The above, with the default left unchanged.

## Decision Outcome

Chosen: **reuse CRAFT via a four-rotation probe, exposed as opt-in
`DetectionConfig::detect_orientation`, defaulting to `false`.**

The pre-pass lives in `crates/sceptre/src/detect/orientation.rs` and runs inside
`CraftDetector::detect`, so the engine and the `TextDetector` seam are untouched. Each
rotation is probed at `orientation_probe_canvas_size` (default 1280, below the 2560
detection default) to keep the probe cheap, and the winning rotation is then detected at
the real `canvas_size`.

### Scoring: region mass alone cannot see 180°

The score is the mean of region-head values above `low_text` plus link-head values above
`link_threshold`, normalized by pixel count so probes of different (rotation-dependent)
shapes stay comparable.

The link term is load-bearing, and it is the whole reason the scorer works. CRAFT's
region head responds to stroke density far more than to glyph orientation, so an
upside-down line of text still looks like text to it — region mass alone cannot tell 0°
from 180°. The affinity head is trained on horizontally-flowing character pairs, and its
response drops for all three wrong rotations, 180° included. Scoring variants were
compared per rotation before settling here; region mass alone was rejected on that
evidence rather than on argument.

A rotation must beat the 0° score by a relative `orientation_margin` (default 0.05)
before it wins, and a `BASELINE_FLOOR` keeps a near-blank page from flipping on noise.
Ties favor leaving the page alone.

### Mapping regions back rotates the corner ordering, not just the coordinates

This is the part that silently breaks if done naively. Rotating the page cycles which
physical corner is "top-left", so the inverse transform must re-index the corners as well
as move them. Coordinates alone still produce a geometrically valid quadrilateral — and
the recognizer then perspective-crops it sideways or upside-down, so the failure moves
from the detector to the recognizer instead of disappearing.

`unrotate_corners` re-derives "clockwise from top-left" fresh in the original frame
rather than applying a fixed per-rotation index offset, so it holds for any quad shape.
`Deg0` returns early as a true no-op: running the re-derivation unconditionally would
reorder a free (non-axis-aligned) quad whenever its existing order did not already start
at the minimum-`y` corner, corrupting the crop on pages that were never rotated.

### Why the default is `false`

Measured with the real CRAFT model:

- All six known-rotated pages get the correcting rotation
  (`ocr_test_rotated_{90,180,270}.png`, `complex_document_rotated_{90,180,270}.png`).
- Seven of the eight tier-2 parity images stay at `Deg0`.
- **`kannada.png` false-positives to `Deg270`**, clearing the 5% margin by roughly 12%.
  Its short lines and dense, loopy glyph strokes give it high baseline activation in
  every orientation, so the margin gate that protects the other seven does not protect
  it.

Enabling this by default would trade a parity image scoring word-F1 1.000 for the
rotated ones. That trade is not the pre-pass's to make silently, and it would break the
tier-2 byte-identical contract without the ADR that contract requires.

The false positive is pinned by
`should_document_the_known_kannada_false_positive_when_enabled`, deliberately written as
a characterization test: if a future scorer stops flipping kannada, that test fails, and
the correct response is to promote it to a `should_leave_*` case and revisit this
default — not to treat it as a break.

### Measured: the first implementation delivered no benefit, and why

Placing the pre-pass inside `CraftDetector::detect` was wrong, and end-to-end measurement
over the 27-image labeled corpus proved it. Enabling the flag moved **no** rotated or
vertical image toward correctness (n=11, mean CER 0.909 → 0.935) and regressed four
otherwise-fine upright images (`financial_table_1`, `layout_parser_paper_with_table`,
`cord_receipt_01`, `invoice_image`; other-16 mean CER 0.360 → 0.448). Rotation
*selection* was never the problem — the real-model selection test passes, and rotating a
page externally still yields ground truth verbatim.

The cause is structural. The detector rotated the image, ran CRAFT, and mapped regions
back to the original frame — but `SceptreEngine::recognize` then grayscales and crops the
**original** image. A 90° rotation maps an axis-aligned box to an axis-aligned box, so
every line kept `axis_aligned: true` and hit `crop_axis_aligned`, which slices the
bounding rectangle and discards corner order outright. Every crop therefore still held
sideways glyphs, and `clockwise_from_top_left` made it strictly worse by normalizing
corner order back into the original frame — erasing the only channel that could have
carried orientation.

The lesson generalizes past this feature: **an orientation decision cannot live in the
detector, because recognition is what needs the rotated pixels.** It has to be hoisted so
detection and recognition share one frame, with only the final output quads mapped back.

### The corrected design, and what it measures

`SceptreEngine` now resolves the rotation once and runs detection, grayscale conversion,
and cropping against that single frame, mapping only the final output quads back to the
caller's frame. `Cow` borrows at `Deg0`, so the disabled path allocates nothing and is
byte-identical to before.

Re-measured over the same 27 images:

| group | n | CER off → on | token-F1 off → on |
|---|---|---|---|
| truly rotated | 6 | 0.858 → **0.134** | 0.003 → **0.911** |
| vertical Japanese | 5 | 0.970 → 0.983 | ~0 → ~0 |
| upright | 16 | 0.360 → 0.487 | 0.597 → 0.429 |
| all | 27 | 0.583 → **0.500** | 0.354 → **0.457** |

Every rotated page lands on *exactly* its upright baseline — `complex_document` scores CER
0.2672 upright and all three of its rotated variants now score 0.2672; `ocr_test_original`
scores 0.0000 and all three of its variants now score 0.0000. The rotation is fully undone,
not merely improved.

Two corrections to earlier framing this measurement forces:

- **Vertical Japanese was never in scope.** The five `ndl_meiji_vertical_*` pages do not
  move, and should not: vertical writing direction is not page rotation. Grouping them
  with the rotated pages into one "eleven broken images" bucket overstated what any
  orientation pre-pass could ever fix.
- **The false-positive rate, not the win, is the blocker.** Four of sixteen upright images
  (`financial_table_1`, `invoice_image`, `layout_parser_paper_with_table`,
  `cord_receipt_01`) flip wrongly, and a wrong rotation is now a *total* loss rather than a
  no-op, because it corrupts recognition too. `kannada.png` makes five known cases. The
  corpus aggregate is net positive even so, but an average over a bimodal population is not
  a reason to enable something that destroys a quarter of upright pages.

The `false` default therefore stands on the false-positive rate, not on the pre-pass being
unproven. Making it default-on requires separating true from false rotations — whether the
relative-margin gate can do that is the open question this ADR leaves.

### Consequences

- Good: no new model, no registry entry, no new `candle` network, and no change to the
  backend matrix in ADR 0035 — the pre-pass reuses CRAFT through the existing seam.
- Good: default output is bit-for-bit unchanged, so ADR 0021's parity gate and the
  tier-2 golden fixtures are unaffected.
- Good: a capability EasyOCR does not have, which is a genuine differentiator rather
  than parity work.
- Bad: opt-in features are under-discovered. Most users with rotated pages will never
  find the flag.
- Bad: four extra forward passes when enabled. Mitigated by the reduced probe canvas,
  but it is not free, and the probe is serial.
- Bad: the scorer is a heuristic over heat-map mass, not a trained classifier, and its
  false-positive rate is the feature's binding constraint — five known cases, four of
  which are dense tables and receipts rather than the non-Latin scripts the kannada case
  first suggested. A wrong rotation is a total loss, not a degradation.
- Neutral: the four probes go through the same `ModelBackend` seam as detection, so
  every backend gets the feature at once with no per-backend work.

## Related

- ADR 0021 — the EasyOCR parity gate this decision deliberately leaves untouched.
- ADR 0035 — the backend/accelerator matrix, unchanged because the pre-pass adds no
  network.
- ADR 0036 — the other opt-in-by-measurement quality lever; same discipline of
  defaulting to the measured-safe behavior rather than the theoretically better one.
