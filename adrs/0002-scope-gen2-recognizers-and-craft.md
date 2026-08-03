---
status: superseded
date: 2026-07-31
deciders: Na'aman Hirschfeld
---

# Scope to gen2 recognizers + CRAFT

> **Superseded by [ADR 0026](0026-extend-gen2-scope-telugu-kannada.md)**, which extends the
> enumerated gen2 recognizer list from six to eight (adds Telugu and Kannada). The
> gen1-out-of-scope decision below is carried forward unchanged.

## Context and Problem Statement

EasyOCR spans 80+ languages, two detector families (CRAFT, DBNet), and two
recognizer generations (gen1 ResNet-BiLSTM-CTC, gen2 VGG-BiLSTM-CTC). Full
parity is a very large surface. We need a scope that matches EasyOCR's current
recommendation without the legacy tail.

## Decision Drivers

- "Latest version" parity, not historical completeness.
- Smaller, consistent model architecture and registry.
- A clear path to add languages by configuration.

## Considered Options

- Full parity (all languages, CRAFT + DBNet, gen1 + gen2).
- gen2 recognizers (`*_g2`) + CRAFT only.
- English-only first, expand later.

## Decision Outcome

Chosen option: **gen2 recognizers + CRAFT only**, covering `english_g2`,
`latin_g2`, `zh_sim_g2`, `japanese_g2`, `korean_g2`, `cyrillic_g2`. gen2 shares a
single architecture (VGG-BiLSTM-CTC, image-only ONNX input) and is EasyOCR's
current default. Languages that exist only as gen1 (e.g. Thai, Arabic,
Devanagari) are out of scope.

### Consequences

- Good: one recognizer architecture; a compact model registry.
- Good: greedy CTC decoding is sufficient for the initial target.
- Bad: gen1-only languages are unsupported until a future ADR revisits scope.
- Neutral: DBNet and beam/word-beam decoding are deferred, not precluded.
