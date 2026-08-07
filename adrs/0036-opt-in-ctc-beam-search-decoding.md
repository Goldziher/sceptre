---
status: accepted
date: 2026-08-07
deciders: Na'aman Hirschfeld
---

# Opt-in CTC beam-search decoding, greedy stays default

## Context and Problem Statement

EasyOCR's `Reader.recognize` (`easyocr.py:352`) exposes `decoder='greedy'|'beamsearch'
|'wordbeamsearch'` with `beamWidth=5`. Sceptre's `RecognitionConfig` already carried a
`Decoder` enum with all three variants (added ahead of this ADR as scaffolding), but
`CrnnRecognizer::recognize` rejected anything but `Greedy` with a config error — the
CTC decode path (`recognize/ctc.rs`) implemented greedy only.

Beam search is the standard accuracy lever CTC systems reach for beyond greedy: instead
of committing to the single most probable class at each timestep, it sums the
probability of every path that collapses to the same label, so a label whose
individually-most-likely path never wins a per-timestep argmax can still win overall.
The question this ADR answers is whether sceptre should implement it, how faithfully to
port EasyOCR's specific algorithm, and what the default should be.

## Decision Drivers

- EasyOCR parity is the baseline contract, not a ceiling (see `adr-discipline` /
  project rules): a new decoder must not change default output.
- `ctc.rs` is close to but under the 1000-line module cap; a beam-search port is
  substantial enough to warrant its own sibling module rather than growing `ctc.rs`.
- Confidence must stay comparable across decoders, since EasyOCR itself computes
  confidence identically regardless of `decoder` (`recognition.py::recognizer_predict`
  computes `preds_max_prob` from the greedy argmax once, before branching on the
  decoder choice).
- Any accuracy claim needs a real measurement on the tier-2 golden corpus, not an
  assumption that beam search is strictly better than greedy.

## Considered Options

- **Leave the scaffolding as a config error.** Keeps the surface honest about what's
  implemented, but wastes the groundwork already laid and leaves `readtext`'s most
  commonly reached-for accuracy lever unavailable.
- **Implement a "corrected" beam search** that fixes `ctcBeamSearch`'s blank-handling
  quirk (its per-timestep extension candidates are not blank-excluded, so an explicit
  blank extension and the implicit "beam unchanged" continuation both add mass to the
  same blank-ending labeling — not a normalized probability update). Tempting, but this
  stops being an EasyOCR decoder and starts being a different algorithm sceptre would
  own the correctness of, with no upstream reference to validate against — and, per the
  "Follow-up" section below, it was measured and is worse net, not better.
- **Port `ctcBeamSearch` faithfully, quirk included, opt-in via `Decoder::BeamSearch`**,
  leaving `Decoder::WordBeamSearch` a config error.

## Decision Outcome

Chosen option: **a faithful port of `ctcBeamSearch`, opt-in, greedy remains the
default.** New module `recognize/beam.rs`: `fast_simplify_label` (the same-vs-different
repeated-class blank bookkeeping), a `BeamEntry`/beam-state prefix search keyed by raw
label sequence, and `decode_beam_search`, which reuses `ctc::probability_matrix` (the
ignore-masked, renormalized probability row, generalized from the greedy path to keep
every class, not just the argmax) and `ctc::decode_row`/`custom_mean` for confidence, so
confidence is bit-identical to what greedy would report for the same logits — matching
EasyOCR's own decoder-independent confidence computation. `collapse` was split into a
shared `collapse_classes` so both decoders reuse the same CTC-collapse rule.

`Decoder::WordBeamSearch` stays a config error. EasyOCR's word-beam-search needs
per-language dictionaries and, when `separator_list` is non-empty, script-boundary word
segmentation (`utils.py::word_segmentation`) — infrastructure sceptre has no equivalent
of today. Implementing it without that infrastructure would be a different, weaker
algorithm wearing the same name.

### A determinism bug found and fixed along the way

The first measurement pass gave a different `english.png` word-F1 on different process
runs of the same code and config (0.919, then 0.901). Root cause: beam pruning and the
final "pick the highest-mass labeling" step both broke ties on plain `f32` comparison
over a `HashMap<Vec<usize>, BeamEntry>`; `HashMap`'s hasher is reseeded randomly per
instance (not just per process), so iteration order — and so which entry a tied
`sort_by`/`max_by` lands on — varies from call to call with identical input. For a
decoder, that means the same crop could recognize to different text on different runs
of the same binary. Fixed by `rank_beams`, a total order over `(total mass, labeling)`
that breaks every float tie on the labeling itself (unique per `HashMap` key), used by
both the per-timestep pruning sort and the final selection; regression-tested by
`should_decode_deterministically_across_repeated_calls` (a beam-width-1 fixture chosen
to force frequent ties). This is orthogonal to the accuracy question below but changes
the exact figures from a first draft of this ADR: all numbers here are post-fix and
reproduced identically across repeated runs.

### Measured accuracy: mixed, and net negative at the EasyOCR default

Word-F1 against the EasyOCR reference, tier-2 corpus, `beam_width=5` (EasyOCR's
default) vs. the existing `Greedy` default, everything else unchanged:

| image | greedy | beamsearch | delta |
| --- | ---: | ---: | ---: |
| english.png | 1.000 | 0.927 | −0.073 |
| chinese.jpg | 0.900 | 0.900 | +0.000 |
| japanese.jpg | 0.895 | 0.973 | **+0.078** |
| korean.png | 1.000 | 0.857 | **−0.143** |
| cyrillic.png | 1.000 | 1.000 | +0.000 |
| telugu.png | 1.000 | 1.000 | +0.000 |
| kannada.png | 1.000 | 1.000 | +0.000 |
| french.jpg | 0.933 | 0.933 | +0.000 |

Net: one clear win (Japanese), two regressions (English, Korean) that outweigh it, four
unaffected (net Σdelta = −0.138 over the 8 images). Inspecting the regressions
(`english.png`: "coronavirus infection" → "coronaviru ifect"; "coughing" → "coughg";
`korean.png`: "205Km" → "20Km") shows a consistent pattern: beam search drops a
character mid-word rather than substituting a wrong one. Widening `beam_width` (5 → 15
→ 50 on `english.png`) does not recover it (F1 0.927 → 0.901 → 0.893, monotonically
*worse*) — this is not a pruning artifact fixable by a wider search, it is the actual
highest-total-probability-mass label under a model with no language-model prior on word
length or content. Decode time cost is 0.83×–1.3× greedy per image — not the
bottleneck either way.

Measured with `crates/sceptre/tests/decoder_beam_search_parity.rs`, a throwaway
comparison harness added alongside this ADR (not a gate — `tier2_golden.rs`'s
thresholds describe the greedy default and must not move for an opt-in decoder).

### Follow-up: is the blank-handling quirk the actual cause?

A natural hypothesis is that the character-dropping pattern above traces to the exact
quirk this ADR chose to port faithfully rather than fix (see Considered Options): when
a labeling already ends in a blank, `char_highscore` still offers the blank as an
extension candidate, `fast_simplify_label` returns the labeling unchanged for it (its
"consecutive blanks" branch), and that unchanged key is the same one the continuation
step just wrote to — so blank-continuation mass gets counted twice, the second time
booked into the *non-blank-ending* bucket. The systematic effect would be to inflate
the mass of labelings that end in blank, i.e. shorter labelings — plausibly exactly
what a dropped character looks like.

Tested directly: `extend_beam`'s extension loop was changed to skip `class ==
BLANK_CLASS` (the standard Graves formulation, where blank-ending mass comes solely
from the continuation branch's `prBlank`), re-measured across all 8 images at
`beam_width=5`, confirmed reproducible across repeated runs (the determinism fix
above applies to this variant too):

| image | greedy | beamsearch (faithful) | beamsearch (blank skipped) |
| --- | ---: | ---: | ---: |
| english.png | 1.000 | 0.927 | 0.929 |
| chinese.jpg | 0.900 | 0.900 | 0.900 |
| japanese.jpg | 0.895 | **0.973** | 0.895 |
| korean.png | 1.000 | 0.857 | **1.000** |
| cyrillic.png | 1.000 | 1.000 | **0.750** |
| telugu.png | 1.000 | 1.000 | 1.000 |
| kannada.png | 1.000 | 1.000 | 1.000 |
| french.jpg | 0.933 | 0.933 | 0.933 |

Net Σdelta over the 8 images: faithful −0.138, blank-skipped **−0.321** — worse, not
better. The hypothesis is half right: it correctly predicts and fixes the Korean
regression (`2O5Km` decodes correctly once blank-continuation mass stops leaking into
the non-blank bucket) and nudges English from −0.073 to −0.071. But it does not
generalize — it costs the entire Japanese win (`NO LTTB` → `NOLTTB`, closing a spurious
greedy-inserted space, stops happening once shorter-labeling mass is no longer
inflated in that direction either) and opens a new, larger regression on Cyrillic
("Россия" → "Росия", a repeated-letter word losing its repeat exactly like
`coronavirus` → `coronaviru`). The quirk is a real, identifiable defect in upstream's
implementation, and it is *a* contributor to the character-dropping pattern, but the
pattern's root cause is the broader one already documented above: a length bias with no
language model to counteract it, which the quirk amplifies in some places and
(via the Japanese case) accidentally cancels in others. Removing the quirk does not
remove the bias, it just redistributes which words it lands on.

**Decision on this follow-up: keep the faithful port.** Diverging from EasyOCR's
algorithm here is not license to invent a strictly-better one without re-measuring —
and the strictly-better hypothesis measured worse. `extend_beam` is unchanged from the
faithful port.

### Consequences

- Good: `RecognitionConfig::decoder` now does what its scaffolding promised for two of
  its three variants; a user who has a reason to try beam search (e.g., Japanese-heavy
  input, per this corpus) can opt in per `Reader` without a code change.
- Good: confidence stays comparable across decoders, so `filter_ths` and the
  contrast-adjustment second pass (both keyed on confidence) behave the same regardless
  of which decoder produced the text.
- Good: the default output is unchanged — every `tier2_golden.rs` figure in this ADR's
  table is the pre-existing greedy baseline, unmoved.
- Bad: beam search is not a general accuracy win on this corpus; recommending it
  requires knowing it helps for the specific script/content in question, not "richer
  decoding is strictly better." This is called out in the config-level doc comment on
  `Decoder::BeamSearch` and here, so a future reader doesn't reach for it as a default
  accuracy fix.
- Neutral: `Decoder::WordBeamSearch` remains unimplemented; building it would need a
  dictionary/word-segmentation subsystem this ADR does not scope.
- Good (incidental): the investigation surfaced and fixed a real determinism bug
  (`rank_beams`) that would otherwise have shipped — the same crop recognizing to
  different text on different runs is a correctness defect independent of whether beam
  search helps accuracy at all.

## Related

- [ADR 0016](0016-parity-harness-and-test-corpus.md) — the tier-2 golden corpus and
  harness this decision is measured against.
- [ADR 0019](0019-parity-safe-perf-and-simd.md) — establishes the fused-softmax greedy
  decode path that `ctc::probability_matrix` generalizes for beam search's use.
