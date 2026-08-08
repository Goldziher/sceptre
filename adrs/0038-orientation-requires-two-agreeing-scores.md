---
status: accepted
date: 2026-08-08
deciders: Na'aman Hirschfeld
---

# A rotation is applied only when two independent scores agree (amends ADR 0037)

## Context and Problem Statement

[ADR 0037](0037-opt-in-whole-page-orientation-pre-pass.md) shipped the whole-page
orientation pre-pass and left it disabled by default for one reason: the scorer flipped
upright pages. Five of the twenty-three orientation-labeled corpus images were rotated
wrongly — `financial_table_1`, `invoice_image`, `layout_parser_paper_with_table`,
`cord_receipt_01` and `kannada.png` — mostly dense tables and receipts. Since ADR 0037
also made the engine run detection *and* recognition in the rotated frame, a wrong
rotation is a total loss, not a degradation. That false-positive rate, and nothing else,
was blocking the default.

Calibration could not fix it. The winning margins over the `Deg0` baseline were 5.84 /
7.98 / 8.82 / 9.20 / 15.84 / 16.12 % for the six true positives and 5.80 / 9.02 / 12.14 /
12.38 / 13.67 % for the five false positives. Since `max(false) = 13.67% > min(true) =
5.84%`, no threshold on `orientation_margin` separates the two populations. This was a
discrimination problem, not a calibration problem.

## Decision Drivers

- The signal must stay inside CRAFT's existing two heat-map heads. Adding an orientation
  classifier would mean a new model download, a new `NetworkKind` for `candle` to
  hand-write, and a registry entry — all rejected in ADR 0037 and still rejected here.
- No true positive may be lost. Losing a correcting rotation costs far more than leaving
  a false positive in place, because rotated pages score near-zero without the pre-pass.
- The rule has to be explainable from what the two heads actually measure, not fitted to
  five images.

## Decision

**A non-zero rotation is applied only when the combined region+link score and the
link-only score independently select the same rotation, each clearing
`orientation_margin`. Otherwise the page is left as given.**

The two heads fail in opposite directions, which is what makes the conjunction work:

- CRAFT's **region head** responds to stroke density, not glyph orientation. On dense
  tables and receipts it drifts to whichever rotation happens to pack more activation,
  and because it dominates the combined sum it carries the decision with it. All five
  false positives were region-driven.
- CRAFT's **link (affinity) head** is trained on horizontally-flowing character pairs, so
  it is the only orientation-discriminating signal available — but on photographed scenes
  it is noisy enough to flip an upright page on its own (`textocr_scene_03` selects
  `Deg180` by 14% on link mass alone).

A rotation that the orientation-blind half proposes and the orientation-sensitive half
refuses is precisely the signature of a false positive. Requiring consensus rejects both
failure modes without inventing a new statistic.

Measured over the corpus, the two halves disagree on the *rotation* for three of the five
false positives (link picks `Deg0` for `financial_table_1` and
`layout_parser_paper_with_table`, and `Deg180` against the combined score's `Deg270` for
`cord_receipt_01` and `kannada.png`), and for `invoice_image` they agree on `Deg270` but
the link score's lead is 4.16%, under the margin. Both rejection paths are load-bearing;
neither alone is sufficient.

## Consequences

Over the 23 orientation-labeled images (the five vertical-Japanese pages are excluded —
they are a writing-direction problem no rotation fixes, per ADR 0037):

| | correct | wrong |
|---|---|---|
| ADR 0037 scorer | 18 | 5 |
| this ADR | **23** | **0** |

All six correcting rotations are preserved; all five wrong rotations are dropped; no new
false positive is introduced. The tightest surviving true positive is
`ocr_test_rotated_180` at 5.84% on the combined score and 9.59% on the link score, both
above the 5% default margin.

- Good: the blocker ADR 0037 recorded is removed on the measured corpus. The recognition
  win it reported is unchanged — this ADR only changes *when* a rotation is applied, never
  which rotation or what happens afterwards.
- Good: no new model, no new config field, no extra CRAFT pass. Link mass was already
  computed per rotation and summed into the combined score; it is now also kept separately.
- Neutral: `detect_orientation` **stays `false` by default.** Twenty-three images is a
  thin basis for flipping a default that costs four CRAFT passes on every page, and the
  cost side of ADR 0037's decision is unchanged. Flipping it is a separate decision that
  should be taken against a broader corpus and a measured latency budget.
- Bad: the rule is strictly more conservative, so a page where only one half of the signal
  survives — very sparse text, a heavily degraded scan — is now left unrotated where the
  old scorer might have corrected it. No corpus image exhibits this, so the cost is
  unmeasured rather than zero.
- Bad: the conjunction is validated on one corpus. A future image that fools both heads in
  the same direction is not protected against, and nothing here detects that case.

## Related

- Amends [ADR 0037](0037-opt-in-whole-page-orientation-pre-pass.md); does not supersede it.
  ADR 0037's scoring function, cost analysis, and the decision to run both detection and
  recognition in one rotated frame all stand.
