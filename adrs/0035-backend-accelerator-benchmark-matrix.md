---
status: accepted
date: 2026-08-07
deciders: Na'aman Hirschfeld
---

# Backend x accelerator benchmark matrix (amends ADR 0032)

## Context and Problem Statement

sceptre now has three backends (`ort`, `tract`, `candle`) and, per backend, a small set of
hardware accelerators (ADR 0032). ADR 0031 measured `candle` against `ort` once, by hand, on
one machine, and recorded the numbers as prose in its own text. That is not a repeatable
instrument: there is no harness, no committed methodology, and no CI leg that keeps the
numbers from going stale the next time either backend's hot path changes.

The gap is not the head-to-head EasyOCR comparison — ADR 0021 and ADR 0030 already own that,
end to end, with a gate. It is the *intra-sceptre* question those ADRs deliberately do not
answer: given a backend and an accelerator that both register successfully, how much does
choosing one over another actually cost or save? ADR 0032 also left a `Neutral` clause that
is no longer accurate on its own terms: "CUDA is compiled and never executed" was true when
no CI job ran a GPU test at all; it says nothing about what a benchmark leg would mean once
one exists.

## Decision Drivers

- Legs must be comparable to each other, not just individually plausible. The single biggest
  threat to comparability already has a name: tract's fixed-canvas CRAFT (ADR 0027) makes the
  detection cost backend-dependent unless every leg fixes the same canvas.
- A benchmark leg is evidence of *liveness and throughput*, not of *correctness*. Conflating
  the two would quietly weaken the one correctness bar that exists (`backend_agreement.rs`)
  by implying a fast run is also a right one.
- The harness must run on hosted CI, including a real GPU, without becoming a merge gate —
  ADR 0015 already settled that criterion benches are report-only, and the same reasoning
  applies here a fortiori: GPU runners are scarce and a flaky external dependency should never
  block a PR.
- `build-gpu-test-binary@v1` (the org's mechanism for shipping a compiled test binary to a
  GPU runner) extracts a **test** binary via `cargo test --no-run`, not a benchmark binary.
- Runner choice is itself part of the measurement: a number from a shared, possibly
  contended host is not the same claim as one from a dedicated machine.

## Considered Options

- **Extend `backend_agreement.rs` to also assert on timing.** Rejected: it would conflate a
  correctness gate with a performance measurement, and correctness assertions must stay
  strict pass/fail while performance numbers are inherently noisy and report-only.
- **A criterion benchmark (`benches/`).** Rejected for the matrix specifically: criterion's
  ≥10-sample statistical model does not fit a ~20 s/iteration candle-CPU workload on a spot
  GPU runner — either the sample count is dishonestly reduced or the job times out. Criterion
  stays the right tool for the existing microbenchmarks (ADR 0015), which this ADR promotes
  from compile-check to a timed, uploaded run but otherwise leaves alone.
- **An `#[ignore]`d integration test, run explicitly with `--ignored`.** Chosen: it is a test
  binary, which is exactly what `build-gpu-test-binary@v1` knows how to extract and ship to
  `runner-gpu-l4`; it reuses `Reader` and the real model-loading path with no harness-specific
  reimplementation; and `#[ignore]` means the existing CI legs keep compiling it for free
  without ever running it by default.

## Decision Outcome

Chosen: `crates/sceptre/tests/backend_matrix.rs`, one `#[ignore]`d test per backend/accelerator
leg, gated behind `SCEPTRE_REQUIRE_MODELS` exactly like `backend_agreement.rs`, aggregated by
`python/sceptre_rs_tools/backend_matrix.py` into a workflow artifact and
`$GITHUB_STEP_SUMMARY`. It runs from a new `.github/workflows/benchmarks.yaml`, which also
re-homes the `bench` (criterion) and `benchmark`/`benchmark-drift` (EasyOCR head-to-head) jobs
dropped when `ci.yaml` split into `ci-lint`/`ci-rust`/`ci-python`.

### Methodology: what makes a leg comparable to another leg

- **`canvas_size = 1024` for every leg, no exceptions.** tract cannot shape-infer dynamic
  CRAFT (ADR 0027) and pads to a fixed square; at the 2560 default its optimization cost would
  dominate the measurement and the comparison would stop meaning anything. `ort` and `candle`
  both run the same fixed canvas so the number measures the backend, not a canvas mismatch.
- **One `Reader` per leg.** Model load happens once, matching how a long-lived process
  actually uses sceptre, and isolates `model_load_ms` as a distinct figure from steady-state
  inference — a backend that loads slowly but infers fast (or vice versa) must not average
  the two into one meaningless number.
- **Warm-up, then repeats, report the median.** One untimed call pages in models, backend
  device state, and OS caches; the median of several timed repeats is reported because a mean
  is dragged around by the rare stalled run, and there is no criterion-scale sample budget on
  a GPU runner to make a trimmed statistic worth computing.
- **A fixed image subset (`english.png`, `cyrillic.png`)**, the same two images
  `backend_agreement.rs` already uses, so a reader of both files does not have to hold two
  different corpora in their head.
- **Runner pinning is part of the measurement, not incidental to it.** A number from
  `runner-large-arm64` and a number from a contended shared host are different claims; the
  workflow records which runner produced which leg, and the first scheduled run of each new
  runner assignment is a deliberate re-baselining event, not a silent continuation.
- **The report names what registered, not what was requested.** Every leg captures
  `runtime_info_for(&config)` and writes `accelerator_registered` into its JSON, because a
  silently-degraded-to-CPU run reported as its requested accelerator would misattribute the
  number exactly the way ADR 0029/0032 already refuse to let provenance do.

### This amends ADR 0032's Neutral clause

ADR 0032 recorded: *"GPU numerics are validated on real hardware only... CUDA is compiled and
never executed anywhere."* That was accurate when written and is superseded by this ADR to
the extent a GPU benchmark leg now runs on `runner-gpu-l4`. It is **not** superseded on the
correctness question: a benchmark leg is a liveness and throughput measurement — it proves
the accelerator registers and produces output in bounded time — and is explicitly not parity
evidence. Golden fixtures stay CPU-generated; `backend_agreement.rs` remains the only
correctness bar for every backend/accelerator pairing, GPU included. Everything else in ADR
0032 — the per-backend accelerator table, the two-layer validate/load invariant, the Apple
CoreML/Metal distinction — stands unchanged.

### Runner assignment

| Lane | Runner | `--features` |
| --- | --- | --- |
| Linux CPU | `runner-large-arm64` | `ort-bundled,tract,candle,download` |
| macOS | `macos-latest` | `ort-bundled,ort-coreml,tract,candle-metal,download` |
| GPU build | `ubuntu-latest` | `ort-dynamic,ort-cuda,tract,candle-cuda,download` |
| GPU run | `runner-gpu-l4` | prebuilt artifact + CUDA runtime libs |

`ort-bundled` cannot do CUDA — the pyke prebuilt ships no CUDA execution provider — so the
CUDA leg is `ort-dynamic` + `ort-cuda`, with the runtime library supplied at run time by
`setup-onnx-runtime-gpu@v1`. Including the `tract`/`candle` CPU legs in the GPU job gives a
same-host amd64 baseline; without it, any "N times faster" claim about the GPU leg would have
no same-machine denominator to be N times faster *than*.

Only `runner-large-arm64`, `runner-medium`, `runner-gpu-l4`, `ubuntu-latest`, and
`macos-latest` are used. `runner-large`, `runner-small`, and `runner-large-spot` are allowed
by `.github/actionlint.yaml` but have no Terraform node pool or Helm scale-set match in
`infra/terraform`; a job targeting one sits `Pending` forever rather than failing loudly.

### `build-gpu-test-binary@v1` builds debug, worked around at the job level

The action runs `cargo test --no-run` with no `--release` and no profile input, so the GPU
build and run legs set `CARGO_PROFILE_DEV_{OPT_LEVEL,LTO,CODEGEN_UNITS,DEBUG,DEBUG_ASSERTIONS,
OVERFLOW_CHECKS}` at the job level (the `test` profile inherits `dev`) and confirm the flags
took effect with one `cargo test --no-run -v`. A `profile:` input on the action itself would
be the better fix; that is filed as follow-up against `xberg-io/actions`, not solved here.

### Kept separate from the published artifact

`benchmarks/published/backends.json` is a new, separately-committed artifact — not an
addition to `benchmarks/published/latest.json`. Three reasons already exist independently in
ADR 0029/0030/0031: every figure in `latest.json` is scoped to `ort` + CPU, `candle` is
explicitly excluded from that drift gate, and `latest.json` is a head-to-head *against
EasyOCR*, while this matrix is an intra-sceptre runtime comparison with a different audience
and a different update cadence. `backends.json` is promoted by a human task, not by an
automated `--check` gate — there is no drift invariant to enforce against a moving GPU-leg
number the way there is against a stable, CPU-only headline figure.

### Fixing the ADR 0021 schedule, not deciding it

ADR 0021 already states the EasyOCR head-to-head gate "runs via `workflow_dispatch` and a
weekly `schedule`." The old `ci.yaml` implementation carried a `~keep` comment asserting
dispatch-only and never added the schedule — code silently contradicting an accepted ADR.
This ADR's workflow adds the missing `schedule` trigger; that is conformance to 0021, not a
new decision. The job also moves from `ubuntu-latest` to `runner-medium` (amd64, non-spot),
which **does** change the numbers the published figures and `--assert` floors were measured
on. The first scheduled run on the new runner is deliberate re-baselining, and its provenance
block records the runner label so a future reader can tell which host produced it.

## Consequences

- Good: choosing a backend or accelerator is now an answerable question with a number behind
  it, refreshed by CI rather than hand-measured once per ADR.
- Good: the harness is a test binary, which is what the org's GPU tooling already knows how to
  build and ship — no bespoke benchmark-binary pipeline.
- Good: `backend_agreement.rs` keeps sole ownership of correctness; nothing about this matrix
  can be read as evidence that a fast backend is also a right one.
- Bad: one more CI workflow, one more committed artifact, one more thing a human has to
  remember to promote (`backends.json` has no automated drift gate by design).
- Neutral: the debug-binary workaround is job-level environment variables, not a real
  `--release` build; it narrows the gap but the reported numbers are not release-build
  numbers until `build-gpu-test-binary@v1` grows a `profile:` input.

## Open Questions

Left unresolved deliberately — to be settled on the first `workflow_dispatch` rather than
guessed at now:

1. Whether `macos-latest` exposes a Metal device at all. These are VMs, and the old `ci.yaml`
   already noted no Neural Engine. ADR 0032's fail-loud invariant means `bench_candle_metal`
   hard-fails if no Metal device registers — correct behavior, but reason enough to keep this
   lane off the weekly schedule until it is verified at least once.
2. The ORT version pin on the CUDA leg. Started at `1.24.2`, proven working in `xberg`'s
   `ci-gpu.yaml`; `RuntimeInfo.ort.version` records what actually loaded, so drift from the
   pin is observable rather than silent.
3. glibc compatibility between the `ubuntu-latest` build image and the `runner-gpu-l4` image
   the binary actually runs on.
4. criterion 0.8's `--save-baseline` CLI surface, relevant once the promoted `bench` job wants
   run-over-run comparison rather than a fresh report each time.
5. Contention if concurrent jobs land on the same `runner-large-arm64` host — the Linux CPU
   leg's numbers assume it does not, and nothing yet verifies that assumption.

## Related

- [ADR 0021](0021-benchmark-methodology-and-gate.md) — the EasyOCR head-to-head methodology
  and gate this ADR's workflow re-homes and brings into schedule conformance.
- [ADR 0027](0027-tract-fixed-canvas-craft.md) — why `canvas_size` must be fixed for every leg.
- [ADR 0029](0029-cli-provisioning-default-and-runtime-scoped-parity.md) and
  [ADR 0032](0032-per-backend-accelerator-support.md) — the accelerator vocabulary and the
  provisioning matrix this benchmark exercises; this ADR amends 0032's `Neutral` clause on GPU
  execution without reopening its accepted decision.
- [ADR 0030](0030-published-benchmark-artifact-and-drift-gate.md) — the published-artifact
  discipline this ADR mirrors with a separate, ungated `backends.json`. Its "LFS corpus
  submodule blocks CI publishing" caveat is resolved by the `test_documents` corpus migration
  (ADR 0034): the corpus is fetchable in CI with no credentials, which is what makes this
  benchmark matrix runnable there at all.
- [ADR 0015](0015-bench-seam-and-criterion.md) — the criterion microbenchmarks this ADR
  promotes from compile-check to a timed, report-only, uploaded run.
