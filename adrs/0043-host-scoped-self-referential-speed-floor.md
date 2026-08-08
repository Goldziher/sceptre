---
status: accepted
date: 2026-08-08
deciders: Na'aman Hirschfeld
---

# Host-scoped, self-referential warm-speed floor (amends ADR 0021)

## Context and Problem Statement

ADR 0021 gave the `--assert` gate a warm-speedup floor: EasyOCR's warm/batch total wall time
divided by sceptre's, per image, must be at least 2.0×. ADR 0042 replaced the sibling peak-RSS
floor for two reasons and explicitly left this one open ("whether a cross-engine speedup floor
suffers a related problem is a separate question"). It does, and it suffers both of them.

**1. The floor is host-dependent.** The v0.6.0 run says sceptre and EasyOCR do nearly the same
amount of CPU work on this corpus: the core-seconds ratio is **1.064×**. Essentially all of
sceptre's wall-clock advantage is parallel scaling — **5.92 effective cores against EasyOCR's
3.23**, a 1.833× factor, and `1.064 × 1.833 = 1.95`, which is the measured speedup to three
digits. Effective cores are bounded by the machine: on the **8-core** `runner-medium` sceptre is
already using 74% of the host, so a smaller host compresses the ratio further while sceptre's
per-core efficiency is unchanged.

| Figure | Darwin/arm64, sceptre 0.4.0 | `runner-medium` (8 cores), sceptre 0.6.0 |
| --- | --- | --- |
| Warm speedup vs EasyOCR | 2.28× | **1.95×** |
| Core-seconds ratio (EasyOCR / sceptre) | not recorded | 1.064× |
| sceptre effective cores | not recorded | 5.92 |
| EasyOCR effective cores | not recorded | 3.23 |
| sceptre warm throughput | not recorded | 0.2329 img/s |
| EasyOCR warm throughput | not recorded | 0.1196 img/s |
| `WARM_SPEEDUP_FLOOR` | 2.00 | 2.00 |

The two columns differ in **both** host and sceptre version — 0.4.0 on Darwin/arm64 against
0.6.0 on `runner-medium`, because core-seconds did not exist as a measurement before 0.6.0 — so
the pair on its own does not isolate the host as the cause, and it is not offered as proof. The
decomposition that does is entirely inside the single `runner-medium` run: work is equal to
within 6%, the parallelism gap accounts for the rest exactly, and the parallelism gap is bounded
by a core count. That reasoning needs no second host. The cross-host pair only shows that the
floor's margin is thin enough for the effect to matter in practice, and that the gate fails on
`runner-medium` today on code that regressed in nothing.

This is the same
shape as ADR 0042's finding — a floor calibrated on macOS/arm64 asserted against a Linux runner
whose absolute figures are not comparable (ADR 0035) — arriving through timing rather than
memory.

**2. The floor is coupled to EasyOCR.** It is a cross-engine ratio, so its denominator is a
dependency's release cadence. A torch or EasyOCR release that makes the reference engine 20%
faster drops the measured ratio by 20% and trips a gate whose failure message says "sceptre
regressed". Nothing in the quantity distinguishes that from an actual sceptre slowdown. ADR 0042
removed exactly this coupling from the memory floor; leaving it in the speed floor keeps a gate
that can be broken by someone else's changelog.

## Decision Drivers

- A gate must fail for the reason it names. Neither a smaller runner nor a faster EasyOCR is a
  sceptre regression, and both are currently reported as one.
- The cross-engine speed advantage is real and worth publishing; it needs an honest scope, not a
  floor to defend.
- Absolute figures — and floors calibrated against them — are comparable only within one host
  class (ADR 0035, ADR 0042).
- Unmeasurable must read as failure, never as success, as it already does for `check_rss_ceiling`
  and for `load_guardrails` / `check_guardrails`.
- Two floors that answer the same question ("did sceptre get worse on this host?") should have
  the same shape, so a reader learns the rule once.

## Considered Options

- **Keep the cross-engine ratio and lower the floor to 1.8.** Rejected. It tunes the gate to the
  measurement rather than fixing what the measurement is: 1.8 is chosen precisely because 1.95
  was observed, so the gate encodes one host's core count as a constant. It is still
  host-dependent — a 4-core runner would push the ratio under 1.8 with no code change — and still
  EasyOCR-coupled, so a faster reference release still reports itself as a sceptre regression.
  It buys quiet until the next runner class or the next torch release.
- **Host-scope the ratio.** Compare this run's cross-engine ratio against the ratio the committed
  baseline published for the same host. This fixes the host dependence — the 1.95 would be
  gated against 1.95, not against a macOS-derived 2.0 — but it keeps EasyOCR in the denominator,
  so defect 2 survives untouched. A gate that is right about the machine and wrong about who can
  break it is not worth the asymmetry with `check_rss_ceiling`.
- **Drop the speed gate entirely** and rely on the published artifact's drift review (ADR 0030).
  Rejected: memory has a gate, and speed is the axis this project exists to move; an unnoticed
  30% slowdown is exactly what an `--assert` mode is for.
- **Absolute, host-scoped, self-referential floor on sceptre's own warm throughput.** Chosen.

## Decision Outcome

1. **The speed gate is absolute and self-referential.** `--assert` requires this run's sceptre
   warm/batch throughput to be at least `SCEPTRE_THROUGHPUT_FLOOR_FACTOR` (0.90) times the
   `headline.sceptre_warm.throughput_img_s` carried by the committed
   `benchmarks/published/latest.json`. EasyOCR appears nowhere in it, so no release on the
   reference side can move it in either direction, and a host's core count is baked into both
   sides rather than into the threshold.
2. **The 10% tolerance is symmetric with `SCEPTRE_RSS_CEILING_FACTOR`'s 1.10.** Both express the
   same judgement from opposite directions: a tenth of run-to-run movement is noise on a shared
   runner, and more than that is a change worth reading. Throughput is the noisier of the two —
   it is wall time on a machine with neighbours, not an allocator's rounding — which is an
   argument for not tightening it, not for widening it.
3. **The floor is scoped to a host**, identified by `(os, arch, runner_label)`, reusing
   `_host_identity` / `published_host` unchanged. `Linux/x86_64` names machine classes whose
   absolute throughput is not comparable (ADR 0035).
4. **Unmeasurable is a breach.** No committed baseline, a baseline carrying no
   `throughput_img_s`, a baseline from another host, and a run that measured no batch timings
   each fail the gate with a message naming the host and the remedy. `check_warm_throughput_floor`
   is structurally identical to `check_rss_ceiling` for this reason: the two gates should be
   readable as one rule applied twice.
5. **`warm_speedup` survives as a reported figure.** It stays in `headline.warm_speedup`, in
   `publish.py`, and in the rendered README and docs tables, exactly as ADR 0042 kept
   `rss_ratio`. It is the number a reader wants; it is simply not a number a CI gate can defend,
   because the reader can see the host and the EasyOCR version beside it and the gate cannot.
6. **`WARM_SPEEDUP_FLOOR` is deleted**, not lowered. A constant left in place would be re-adopted
   by the next gate that needs a speed threshold.

### Consequences

- The gate now fails only for sceptre-side slowdown, on the host it was baselined on. A smaller
  runner, a faster EasyOCR, and a torch upgrade cannot trip it.
- Gating requires a published baseline measured on the gating host, as the RSS ceiling already
  did. The committed artifact is `runner-medium` and the scheduled job runs there, so this floor
  is evaluable today; a run on any other host fails with "no baseline for this host" until one is
  published, which is the incomparability made visible rather than a new obstacle.
- The floor is scoped to a host *and* a corpus: throughput moves with the images measured, so a
  corpus change requires re-publishing the baseline. The failure names the measured figure, the
  floor, the baseline, and the host, so the remedy is legible.
- The published speed claim does not change: `warm_speedup` is still computed and rendered the
  same way, from the same measurements. Unlike ADR 0042, this ADR changes only what is gated.
- `check_thresholds` now consists of two structurally identical self-referential checks against
  the published baseline. Nothing cross-engine gates anything.

## Related

- ADR 0021 (head-to-head benchmark methodology and regression gate) — **amended, not
  superseded**. This is the second of its three original floors to be replaced: ADR 0042 took
  the peak-RSS floor, this one takes the warm-speedup floor, and the remaining labeled-quality
  floor had already moved to per-image guardrail contracts (`derive_guardrails` / `--guardrails`)
  in the harness. The measurement methodology — fresh subprocess per language group under
  `/usr/bin/time`, warm/batch axis, native threading, capability gaps excluded from aggregates —
  is untouched.
- ADR 0042 (host-scoped absolute benchmark floors) — **extended**. This ADR applies its two
  findings, host-scoping and self-reference, to the floor it deliberately left open, and reuses
  its host identity, its baseline source, and its unmeasurable-is-a-breach discipline verbatim.
- ADR 0030 (published benchmark artifact and drift gate) — the committed artifact this floor
  reads its baseline from, which is why the baseline is reviewable rather than machine-local.
- ADR 0035 (backend × accelerator benchmark matrix) — established that a runner class is
  provenance a figure cannot be attributed without.
