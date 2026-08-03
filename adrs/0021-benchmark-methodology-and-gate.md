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
