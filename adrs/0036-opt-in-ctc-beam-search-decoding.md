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
  own the correctness of, with no upstream reference to validate against.
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

### Measured accuracy: mixed, and net negative at the EasyOCR default

Word-F1 against the EasyOCR reference, tier-2 corpus, `beam_width=5` (EasyOCR's
default) vs. the existing `Greedy` default, everything else unchanged:

| image | greedy | beamsearch | delta |
| --- | ---: | ---: | ---: |
| english.png | 1.000 | 0.919 | −0.081 |
| chinese.jpg | 0.900 | 0.900 | +0.000 |
| japanese.jpg | 0.895 | 0.973 | **+0.078** |
| korean.png | 1.000 | 0.857 | **−0.143** |
| cyrillic.png | 1.000 | 1.000 | +0.000 |
| telugu.png | 1.000 | 1.000 | +0.000 |
| kannada.png | 1.000 | 1.000 | +0.000 |
| french.jpg | 0.933 | 0.933 | +0.000 |

Net: one clear win (Japanese), two regressions (English, Korean) that outweigh it, four
unaffected. Inspecting the regressions (`english.png`: "coronavirus infection" →
"coronaviru ifect"; "coughing" → "coughg"; `korean.png`: "205Km" → "20Km") shows a
consistent pattern: beam search drops a character mid-word rather than substituting a
wrong one. Widening `beam_width` (5 → 15 → 50 on `english.png`) does not recover it
(F1 0.937 → 0.919 → 0.919, plateauing) — this is not a pruning artifact fixable by a
wider search, it is the actual highest-total-probability-mass label under a model with
no language-model prior on word length or content. This matches a known limitation of
un-language-modeled CTC beam search: a shorter labeling can integrate more total path
mass across timesteps than the correct, one-character-longer path, and only a language
model (which is exactly what `wordbeamsearch` adds via its dictionary) reliably fixes
it. Decode time cost is 0.96×–1.4× greedy per image — not the bottleneck either way.

Measured with `crates/sceptre/tests/decoder_beam_search_parity.rs`, a throwaway
comparison harness added alongside this ADR (not a gate — `tier2_golden.rs`'s
thresholds describe the greedy default and must not move for an opt-in decoder).

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

## Related

- [ADR 0016](0016-parity-harness-and-test-corpus.md) — the tier-2 golden corpus and
  harness this decision is measured against.
- [ADR 0019](0019-parity-safe-perf-and-simd.md) — establishes the fused-softmax greedy
  decode path that `ctc::probability_matrix` generalizes for beam search's use.
