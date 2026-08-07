---
status: accepted
date: 2026-08-07
deciders: Na'aman Hirschfeld
---

# Published benchmark artifact and drift gate

## Context and Problem Statement

[ADR 0021](0021-benchmark-methodology-and-gate.md) settled how sceptre *measures* itself against
EasyOCR and how a regression gate gets enforced. It did not settle how a measured number becomes a
*published* number, and that gap produced a class of defect we then had to go and fix by hand.

The harness writes `benchmark-results/comparison.{json,md}`. That directory is gitignored and
machine-local. Every figure quoted in `README.md` and on the docs site was copied out of such a run
by a human, at some point, from some machine. The consequences were all realized, not hypothetical:

- **No number could be attributed to a runtime.** The harness learned to record a
  `metadata.environment` block — sceptre version, backend, accelerator, ONNX Runtime version, model
  pins, and the reference EasyOCR/torch/Python versions. But the artifact the published tables were
  copied from predated that block, so the published figures named no runtime at all. ADR 0029 had
  just finished arguing that an accuracy or performance number is meaningless without the runtime
  that produced it; the published tables were the counterexample.
- **The same numbers were transcribed into several places in different units** — gigabytes in the
  README, megabytes on the docs site — plus ratio claims restated in prose in a third and fourth
  place. Nothing kept them consistent, and an audit found the copies had already diverged from each
  other in wording and rounding.
- **Nothing could detect drift.** A stale table is indistinguishable from a fresh one by inspection.

## Decision Drivers

- A published figure must carry the runtime that produced it, or not be publishable.
- Numbers must have exactly one authoritative location; anything else is a copy that can rot.
- The drift check must be cheap enough to run on every pull request — which means it must not
  require torch, easyocr, models, or a release binary.
- Regenerating the numbers is expensive and needs a quiet machine, so measuring and publishing must
  be separable steps.

## Considered Options

1. **Keep hand-transcription, add a review checklist.** Zero machinery. Rejected: this is precisely
   the process that produced the defects, and a checklist does not survive contact with a release.
2. **Generate the docs tables directly from `benchmark-results/comparison.json` at build time.**
   Rejected: that file is gitignored and machine-local, so the docs would not build reproducibly,
   and CI — which cannot run the benchmark on every PR — would have nothing to read.
3. **Commit the full `comparison.json` and generate from it.** Rejected: it is a ~36 KB report
   dominated by per-image records, most of which is irrelevant to a published table, and its shape
   is free to change as the harness evolves. Committing it couples the docs to the harness's
   internal serialization.
4. **Commit a small, stable-schema artifact distilled from the report, generate the tables from
   that, and gate on drift.** Chosen.

## Decision Outcome

Chosen: **option 4**. `sceptre_rs_tools.publish` distils a comparison report into
`benchmarks/published/latest.json` — a committed, versioned artifact carrying the headline figures
and the `environment` provenance block — then renders the headline table from that artifact into
every marked region in the README and the docs site.

The split is deliberate: `task python:benchmark` *measures* (expensive, needs torch and a release
binary, wants a quiet machine), `task python:publish` *publishes* (pure JSON and Markdown). Only the
second runs in CI on a pull request, as `task python:publish:check`.

Three rules make it hold:

- **No provenance, no publish.** A report without a `metadata.environment` block is rejected rather
  than published. A table that cannot be attributed to a runtime is the defect this ADR exists to
  prevent, so it is unrepresentable rather than discouraged.
- **One authoritative location.** Numeric claims live in the generated table only. The surrounding
  prose was rewritten to be qualitative, so there is no second copy to drift. This is why the README
  hero line no longer quotes a speedup ratio: an ungenerated number in prose is a future defect, and
  automating a hero sentence is worse than not making it numeric.
- **Generated regions are addressed by marker pairs**, not by line number or table detection. The
  README is Markdown and takes `<!-- generated:NAME:start -->`; the docs site is MDX, where an HTML
  comment is a parse error, so it takes `{/* generated:NAME:start */}`.

`--check` re-renders from the committed artifact and exits non-zero if the committed tables disagree
with it, naming each stale file. It reads **only files that are in git** — never the gitignored
measurement report — so it works on a fresh clone with no benchmark run behind it. That constraint
is load-bearing rather than incidental: a gate that needed the report could not run in CI at all,
which is how the first version of this job failed.

### Consequences

- Good: a published figure is traceable to the sceptre build, ONNX Runtime, EasyOCR and torch
  versions, host, and corpus subset that produced it.
- Good: the drift gate is a few hundred milliseconds of JSON and string work, so it runs on every PR
  rather than only when someone remembers.
- Good: measuring and publishing decouple. Numbers can be re-measured on a quiet machine and
  published later, or published from a report produced elsewhere via `--from`.
- Bad: the published numbers are only as current as the last deliberate `task python:publish`. The
  gate proves the docs match the artifact, **not** that the artifact matches today's code. Catching
  a stale artifact needs a re-measurement, which is a human decision by design.
- Bad: one more schema to version. `schema_version` is pinned in the artifact so a future shape
  change is detectable rather than silent.
- Neutral: the numbers still come from one machine. Publishing from a CI-produced report is
  possible through `--from`, and is the natural next step once CI can run the benchmark — the
  harness's LFS corpus submodule currently prevents that.

## More Information

- [ADR 0021](0021-benchmark-methodology-and-gate.md) — how the numbers are measured and gated.
- [ADR 0029](0029-cli-provisioning-default-and-runtime-scoped-parity.md) — why a number without its
  runtime is not a claim.
