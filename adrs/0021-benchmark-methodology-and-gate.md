---
status: accepted
date: 2026-08-03
deciders: Na'aman Hirschfeld
---

# Head-to-head benchmark methodology and regression gate

## Context and Problem Statement

sceptre's reason to exist is delivering EasyOCR-equivalent (or better) OCR at substantially
better speed and memory. That claim is only credible if it is measured fairly and defended
against regression. An earlier harness (`python/sceptre_rs_tools/benchmark.py`) compared the
sceptre release CLI against upstream EasyOCR, but three choices with lasting consequences were
unresolved: how to measure peak RSS comparably, what corpus and quality signal to score
against, and whether the benchmark is a static report or a repeatable development instrument
with an enforced floor.

The prior harness measured EasyOCR peak RSS in-process (`getrusage(RUSAGE_SELF)`, including the
whole torch runtime) while measuring sceptre as a `/usr/bin/time` subprocess — an
apples-to-oranges comparison the report itself had to disclaim. It also ran each timing once
and mixed undecodable-format failures into the results as opaque skips.

## Decision Drivers

- RSS and speed numbers must be like-for-like, so the "substantially better" claim is honest.
- The harness is a development tool: fast subsets, stable run-over-run numbers, per-image
  drill-down, and diffable machine-readable output.
- Regressions in speed, memory, or quality must fail loudly, mirroring the existing
  `SCEPTRE_REQUIRE_MODELS` opt-in discipline (ADR 0016) so the default lanes stay light.
- No new runtime dependencies in the library; the harness stays a dev-only Python tool.

## Considered Options

- **RSS measurement:** keep the mixed in-process/subprocess methodology with a caveat / run
  both engines each as a fresh subprocess under `/usr/bin/time` for identical whole-process
  peak RSS / embed sceptre in Python to match EasyOCR's process.
- **Threading:** pin both engines single-threaded for stability / run both at their native
  multi-threaded default / benchmark only sceptre's configured concurrency.
- **Quality signal:** cross-engine agreement only / add absolute CER/WER/token-F1 against
  ground truth / require box-level parity for every image.
- **Gate:** report-only / an opt-in `--assert` gate with fixed floors run on demand and on a
  schedule / block every PR on the full benchmark.

## Decision Outcome

- **RSS + warm timing:** both engines run as a fresh subprocess per language group under
  `/usr/bin/time`, loading their model/Reader once and processing every image (the warm/batch
  axis). Peak RSS is captured identically for both; EasyOCR's figure legitimately includes the
  Python + torch runtime, which is its real cost. A secondary per-image "cold" sceptre figure
  (fresh process per image) reports the per-invocation CLI cost. EasyOCR is driven by a
  dedicated `_easyocr_runner` subprocess so it is symmetric with the sceptre batch.
- **Threading:** both engines run at their native multi-threaded default (representative of
  real deployment on the same hardware), with `--threads N` available to pin both for a
  controlled per-core comparison. Stability comes from `--repeats` (median wall time, max peak
  RSS), not from crippling parallelism.
- **Quality signal:** score absolute CER/WER/token-F1 against `test_documents` ground truth on
  the labeled corpus (including scene text, receipts, dense tables, and vertical Japanese),
  plus cross-engine agreement (char/word-F1, box-IoU) on breadth images. Formats sceptre cannot
  decode (BMP/HEIF/AVIF/JP2) are a distinct `capability` group: reported as gaps, excluded from
  speed/quality aggregates so a fast-failing decode never inflates sceptre's numbers.
- **Gate:** an opt-in `--assert` mode enforces three floors — warm speedup ≥ 2.0×, peak-RSS
  ratio ≥ 3×, and labeled token-F1 within 0.05 of EasyOCR. The floors sit below the measured
  like-for-like margins so normal variation does not trip the gate. It runs via
  `workflow_dispatch` and a
  weekly `schedule` in CI (heavy: torch + easyocr + models + a release build), never on every
  PR. The pure harness logic is covered by fast `pytest` unit tests that do run on every PR.

### Consequences

- The headline RSS advantage is stated honestly as a like-for-like whole-process peak ratio
  (smaller and more defensible than the earlier in-process-vs-subprocess number), and the speed
  advantage reflects real multi-threaded deployment.
- Capability gaps (input-format coverage) become tracked, reported findings that drive
  follow-up work rather than silent skips. See the `image` crate feature set in the root
  `Cargo.toml` for the current decode coverage.
- The gate can fail on three independent axes; thresholds live in one constant block in
  `benchmark.py` and are revised here if the corpus or reference EasyOCR version changes.

## Related

- ADR 0016 (parity harness, dual golden fixtures, opt-in heavy path) — this benchmark reuses
  the same corpus, HF-cache model resolution, and `SCEPTRE_REQUIRE_MODELS` opt-in philosophy.
- ADR 0015 (criterion bench seam) — orthogonal: those microbenchmark internal hot paths, this
  measures end-to-end OCR against EasyOCR.

## Status update (2026-08-08)

The **peak-RSS floor is amended by ADR 0042**. The "peak-RSS ratio ≥ 3×" floor above divides
EasyOCR's peak RSS by sceptre's, where each side is a maximum taken independently over language
groups: the two are not a paired measurement, the quotient compounds two extreme-value
statistics, and peak RSS is dominated by the CRAFT canvas *both* engines allocate, so the ratio
tends toward 1 as pages grow. Measured on `runner-medium` it is 1.08×, against the 3.11×
recorded on macOS/arm64 — so the claim above that "the floors sit below the measured
like-for-like margins" no longer holds, and the gate could be tripped by adding one large image
to the corpus or by an EasyOCR release, neither of which is a sceptre regression. The gate is
now an absolute ceiling on sceptre's *own* peak RSS relative to the committed published
baseline for the same host; the cross-engine ratio remains a reported figure, computed as the
median of the per-image ratios. Everything else here — the like-for-like measurement
methodology, the threading choice, the quality signal, the capability-gap handling, and the
warm-speedup floor — is unchanged.

The labeled-quality floor above ("labeled token-F1 within 0.05 of EasyOCR") is likewise no
longer a corpus mean in `check_thresholds`: the harness enforces per-image guardrail contracts
(`derive_guardrails`, checked with `--guardrails`), since a corpus mean cannot see a regression
isolated to one image or one language.

The **warm-speedup floor is superseded by ADR 0043**, which applies ADR 0042's reasoning to it.
"Warm speedup ≥ 2.0×" is a cross-engine ratio, so a faster EasyOCR release lowers it without
sceptre changing; and it is host-dependent, because sceptre's advantage on this corpus is almost
entirely parallel scaling (a 1.064× core-seconds work ratio against 5.92 versus 3.23 effective
cores), which an 8-core runner caps. It read 2.28× on the macOS/arm64 0.4.0 baseline and 1.95× on
the `runner-medium` 0.6.0 one — two points that differ in version as well as host, so ADR 0043
rests on the within-run core-seconds decomposition rather than on that pair.
`check_thresholds` now gates an absolute, host-scoped floor on sceptre's own warm/batch
throughput against the committed published baseline; `warm_speedup` remains a reported figure.
With that, none of the three floors named above survives in its original form.
