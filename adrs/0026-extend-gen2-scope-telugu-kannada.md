---
status: accepted
date: 2026-08-03
deciders: Na'aman Hirschfeld
---

# Extend gen2 scope to Telugu and Kannada

## Context and Problem Statement

[ADR 0002](0002-scope-gen2-recognizers-and-craft.md) scoped sceptre to CRAFT plus exactly six gen2
recognizers (`english_g2`, `latin_g2`, `zh_sim_g2`, `japanese_g2`, `korean_g2`, `cyrillic_g2`),
noting gen1-only languages are out of scope "until a future ADR revisits scope". EasyOCR's gen2
family actually has **eight** recognizers: the six above plus `telugu_g2` and `kannada_g2`. Since we
are already standing up a first-party export pipeline (ADR 0025), completing the gen2 set is low
marginal cost and closes the gap between "gen2 support" and "the six we happened to start with".

## Decision Drivers

- Cover EasyOCR's *entire* gen2 recognizer family, not an arbitrary subset.
- Stay within the gen2-only architecture (VGG-BiLSTM-CTC, image-only ONNX input) — no gen1 tail.
- Reuse the existing add-a-language mechanism (enum + registry + charset + CLI arg).

## Decision Outcome

Chosen option: **add `telugu_g2` and `kannada_g2`**, extending ADR 0002's enumerated scope to all
eight gen2 recognizers. Both are genuine EasyOCR gen2 models (EasyOCR has no gen1 Telugu/Kannada),
so this is a scope extension within the existing architecture, not a gen1 exception. They are
exported by the ADR 0025 pipeline and hosted at `sceptre-ocr/telugu_g2` and `sceptre-ocr/kannada_g2`.

Languages that exist only as EasyOCR gen1 (Arabic, Thai, Devanagari, Bengali, Tamil, Traditional
Chinese) remain out of scope, unchanged from ADR 0002.

### Consequences

- Good: sceptre covers the complete gen2 recognizer family — no gen2 gaps.
- Good: Telugu and Kannada scripts become supported with the existing greedy-CTC pipeline.
- Neutral: two new golden parity fixtures (image + dual golden) are added under ADR 0016; none
  existed for these scripts before.
- Bad: two more models to export, host, and keep parity-tested.

## Supersedes

Supersedes [ADR 0002](0002-scope-gen2-recognizers-and-craft.md) (widens its enumerated recognizer
list from six to eight). The gen1-out-of-scope decision is carried forward unchanged.
