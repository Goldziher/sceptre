---
status: accepted
date: 2026-08-08
deciders: Na'aman Hirschfeld
---

# Host-scoped absolute benchmark floors (amends ADR 0021)

## Context and Problem Statement

ADR 0021 gave the `--assert` gate a peak-RSS floor: EasyOCR's peak RSS divided by sceptre's
must be at least 3×. It also stated that the floors "sit below the measured like-for-like
margins so normal variation does not trip the gate". Both the floor's shape and that claim are
now false, and the measurements say so.

The ratio is built from `_peak_rss`, which takes the maximum over a run's language-group
batches — independently for each engine. Three defects follow.

1. **It is not a paired measurement.** Nothing constrains the numerator's batch to be the
   denominator's batch: the two maxima are selected separately, so the published figure can be
   a quotient of two measurements that never co-occurred, over different images in different
   languages. It is therefore not an estimate of anything a user experiences.
2. **It is a ratio of two extreme-value statistics.** Each side is a max over `--repeats`, then
   a max over groups, and the gate divides two of those. Every layer discards all but the most
   extreme sample, so the quantity has strictly more variance than the measurements underneath
   it, and the ratio compounds both sides' variance.
3. **It largely measures the corpus, not sceptre.** Peak RSS is dominated by the CRAFT
   detector — a model *both* engines run — at roughly 1063 MB per canvas megapixel. As the
   largest page grows, that shared term dominates both sides and the ratio tends toward 1. On
   `runner-medium` the max-of-maxes is **1.08×**, while the per-batch ratios span 1.08×
   (japanese) to 3.25× (chinese), with a per-batch median of 2.00× and a geometric mean of
   1.91×. The committed baseline, measured on macOS/arm64, reads **3.11×** and clears the 3.0
   floor by 4%.

So a passing gate can be tripped by adding one large image to the corpus, with no code change;
or by an EasyOCR or torch release that trims the Python-side footprint, with no sceptre change.
Neither is a sceptre regression, and both would be reported as one. Conversely a real sceptre
memory regression on large pages is partly hidden, because the shared CRAFT term sits in both
the numerator and the denominator. The gate also runs on a host (`runner-medium`) that is not
the host the committed baseline was measured on (macOS/arm64), where absolute figures — and a
floor calibrated against them — are not comparable at all (ADR 0035).

## Decision Drivers

- A gate must fail for the reason it names. "sceptre regressed" must never be signalled by a
  corpus edit or by a dependency's release.
- The cross-engine memory advantage is real and worth publishing; it needs an honest statistic
  and an honest scope, not a floor to defend.
- Absolute figures are comparable only within one host class; a runner label is part of a
  host's identity, not a footnote to it (ADR 0035).
- Unmeasurable must read as failure, never as success — the discipline `load_guardrails` and
  `check_guardrails` already apply to the per-image quality contract.

## Considered Options

- **Gate shape:** keep the cross-engine ratio and lower the floor / keep it but compute it
  paired per image / replace it with an absolute, self-referential ceiling on sceptre's own
  peak RSS / drop the memory gate entirely and rely on review.
- **Reported statistic:** max-of-maxes (status quo) / arithmetic mean of per-image ratios /
  median of per-image ratios / geometric mean of per-image ratios.
- **Scope of an absolute floor:** global / per `(os, arch)` / per `(os, arch, runner label)`.
- **Absent or foreign baseline:** skip the check / warn and pass / fail the gate.

## Decision Outcome

1. **The memory gate is absolute and self-referential.** `--assert` requires this run's sceptre
   warm/batch peak RSS to be at most `SCEPTRE_RSS_CEILING_FACTOR` (1.10) times the
   `headline.sceptre_warm.peak_rss_mb` carried by the committed
   `benchmarks/published/latest.json`. EasyOCR appears nowhere in it, so no change on the
   reference side can move it in either direction. The 10% tolerance is the mirror image of
   `GUARDRAIL_THRESHOLD_FACTOR`'s 10% slack and is justified the same way: peak RSS is driven
   by allocator and canvas rounding rather than by timing noise, and repeats to within a few
   percent on a fixed host and corpus, so 10% clears the noise without hiding the growth an
   extra resident model or a leaked buffer would produce.
2. **Floors over absolute figures are scoped to a host**, identified by
   `(os, arch, runner_label)`. `Linux/x86_64` names machine classes whose absolute figures are
   not comparable, so the runner label is part of the identity (ADR 0035). A baseline from a
   different host is neither silently used nor silently skipped.
3. **Unmeasurable is a breach.** No committed baseline, a baseline carrying no peak-RSS field,
   a baseline from another host, and a run that measured no RSS at all each fail the gate with
   a message naming the host and the remedy.
4. **The cross-engine ratio survives as a reported figure, computed paired.**
   `headline.rss_ratio` is the median, over scored images, of `easyocr_rss_mb / sceptre_rss_mb`
   as recorded per `ImageRecord` — each side being the peak RSS of the batch that image was
   actually measured in. The median is the central tendency because these are ratios: it is
   robust to one outlying batch and invariant under inversion, so the sceptre-over-EasyOCR
   median is exactly its reciprocal. It is taken over images rather than over batches so each
   language group counts for as much of the corpus as it actually covers. Pairing must *not* be
   done on batch keys: sceptre keys its batches `english` / `english+korean` while EasyOCR keys
   the very same images `en` / `en+ko`, and nothing in the report maps one key space onto the
   other.
5. **The published table names the statistic it reports** — `(~3.8× lower, per-image median)` —
   because that aside is not the quotient of the two peak-RSS cells beside it, which remain each
   engine's corpus-wide maximum. `publish.SCHEMA_VERSION` moves to 2: a version-1 artifact
   carries every field the renderer reads, so nothing else would stop it from being rendered
   with its old, unpaired number under the new statistic's label.
6. **`WARM_SPEEDUP_FLOOR` (2.0) is untouched.** Whether a cross-engine speedup floor suffers a
   related problem is a separate question and is deliberately left open here.

### Consequences

- The gate now fails only for sceptre-side memory growth, on the host it was baselined on. A
  corpus addition, an EasyOCR upgrade, or a torch upgrade cannot trip it.
- Gating requires a published baseline measured on the gating host. The committed artifact is
  macOS/arm64 and the scheduled head-to-head job runs on `runner-medium`, so the first
  scheduled run after this ADR fails with "no baseline for this host" until a `runner-medium`
  run is published. That is a pre-existing incomparability made visible, not a new obstacle.
- The ceiling is scoped to a host *and* a corpus: peak RSS moves with the largest canvas, so a
  corpus change requires re-publishing the baseline. Unlike the ratio, this failure is legible —
  it names the measured figure, the ceiling, the baseline, and the host — and its remedy
  (re-publish) is the correct response to a changed corpus.
- The published memory claim changes both number and meaning, from 3.11× (max-of-maxes) to
  3.77× (per-image median), recomputed from the *same* macOS measurement run. No new
  measurement was taken and no provenance changed.
- The floors that remain in `check_thresholds` are warm speedup and the sceptre peak-RSS
  ceiling; labeled quality is enforced separately by the per-image guardrail contract
  (`derive_guardrails` / `--guardrails`), which had already replaced ADR 0021's corpus-mean
  token-F1 floor in the harness.

## Related

- ADR 0021 (head-to-head benchmark methodology and regression gate) — **amended, not
  superseded**. The methodology stands unchanged: fresh subprocess per language group under
  `/usr/bin/time`, warm/batch axis, native threading, capability gaps excluded from aggregates.
  Only the memory floor's shape changes.
- ADR 0030 (published benchmark artifact and drift gate) — the committed artifact this gate now
  reads its baseline from, which is why the baseline is reviewable rather than machine-local.
- ADR 0035 (backend × accelerator benchmark matrix) — established that a runner class is
  provenance a figure cannot be attributed without; this ADR makes it part of a floor's scope.
