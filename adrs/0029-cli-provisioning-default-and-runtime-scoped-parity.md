---
status: accepted
date: 2026-08-06
deciders: Na'aman Hirschfeld
---

# CLI provisioning default and runtime-scoped parity

## Context and Problem Statement

Two claims sceptre makes to its users turned out to be unbacked, for the same underlying
reason: a decision was left implicit where it needed to be explicit.

**Provisioning.** [ADR 0012](0012-ort-runtime-provisioning.md) established `ort-bundled` and
`ort-dynamic` as two provisioning strategies over a provisioning-agnostic `ort` feature, and
closed with "the CLI's per-target feature selection is deferred to the CLI pass." That pass
shipped `default = ["ort-dynamic", "download"]`. Cargo features are additive, so
`--features ort-bundled` did not replace `ort-dynamic` — it unioned with it, and
`load-dynamic` won at runtime. Every `cargo install sceptre-cli` on a machine without a
system `libonnxruntime` therefore produced a binary that panicked on first use with "Failed
to load ONNX Runtime dylib", and the documented workaround did not help. The CLI also never
exposed a `tract` feature, so `--backend tract` was a flag no shipped build could satisfy.

**Parity.** The README, the docs site, and the golden fixtures all assert parity with
EasyOCR and a ~2.8× speed advantage, without naming the runtime that produced those numbers.
`ort_backend` never called `with_execution_providers`, so every published figure was
implicitly CPU — but nothing said so, nothing let a user select anything else, and nothing
would have told them if their selection had silently not taken effect. ONNX Runtime's
`ExecutionProviderDispatch` defaults to `fail_silently`, so "asked for CoreML, got CPU"
looks exactly like success.

## Decision Drivers

- `cargo install sceptre-cli` must produce a binary that runs, with no follow-up steps.
- Switching provisioning strategy must stay expressible, and the escape hatch must work on
  the targets that need it — the ones ONNX Runtime publishes no prebuilt for.
- An accuracy or performance number is meaningless without the runtime that produced it;
  the runtime must be selectable, reportable, and scoped in the claim.
- A requested accelerator that does not take must be an error, not a quiet CPU run.
- No `compile_error!`-style mutual exclusion: cargo features are additive by design, and
  fighting that breaks feature unification for anyone depending on the crate.

## Considered Options

**Provisioning default:**

- `default = ["ort-bundled", "download"]` — self-contained binary; `--features ort-dynamic`
  remains available as an additive override.
- A mutually-exclusive selector enforced with `compile_error!` when both `ort-bundled` and
  `ort-dynamic` are enabled.
- `default = ["download"]` with no backend — force an explicit choice.
- A target-conditional default that picks `ort-bundled` where a prebuilt exists and
  `ort-dynamic` elsewhere.

**Accelerator selection:**

- No selection at all (status quo): CPU implicitly, forever.
- `Accelerator` defaulting to `Auto` — best available hardware out of the box.
- `Accelerator` defaulting to `Cpu`, with explicit opt-in and fail-fast registration.

## Decision Outcome

**Provisioning:** the CLI ships `default = ["ort-bundled", "download"]`, and additionally
exposes `tract`. This works *because* features are additive rather than in spite of it:
`ort-dynamic` implies `ort/load-dynamic`, which implies `ort-sys/disable-linking`, and the
`ort-sys` build script checks that flag before reaching its prebuilt-download branch. So
`cargo install sceptre-cli --features ort-dynamic` resolves to a pure `dlopen` build with no
build-time download, and neither direction needs `--no-default-features`. Only the
pure-Rust path does, because it has to drop `ort` from the graph entirely:
`--no-default-features --features tract,download`, then `--backend tract` at runtime.
`scripts/verify-features.sh` pins all four resolutions with `cargo tree --locked`, in the
pre-commit gate and in CI.

**Accelerator:** `ModelConfig::accelerator` (`--accelerator` on the CLI) takes `cpu`, `auto`,
`coreml`, `directml`, or `cuda`, backed by the feature-gated `ort-coreml` / `ort-directml` /
`ort-cuda` providers. The vocabulary is named for the backend seam rather than for ONNX
Runtime, so `tract` and `candle` answer the same field and reject a hardware accelerator at
config-validation time. The default is `Cpu`, not `Auto`: the published parity and benchmark
figures are CPU figures, and defaulting to `Auto` would silently change a user's results on
upgrade. An explicitly requested provider registers with `error_on_failure`; only `Auto`
registers candidates individually and settles for what takes. `runtime_info()` — surfaced as
`sceptre env --format json` — reports the backend, requested and registered accelerator,
provisioning strategy, ONNX Runtime version, and arch, so a parity artifact can record the
environment it came from.

**Scoping the claim:** every published parity and benchmark figure is stated as an `ort` +
CPU-execution-provider figure. CI runs a per-provider lane: CPU on Linux, macOS, and
Windows, plus CoreML on macOS. That CoreML lane proves registration and an end-to-end run,
not on-device numerics — GitHub's macOS runners are virtual machines with no Neural Engine.
CUDA and TensorRT are documented as unvalidated, because there is no GPU runner. DirectML is
deliberately not a lane at all: the Windows runner has no GPU, so DirectML would fall back
to the WARP software adapter and produce numbers describing neither CPU nor a real GPU.

### Rejected alternatives

- **`compile_error!` mutual exclusion.** It would turn the additive override into a build
  failure — the very mechanism that makes `--features ort-dynamic` work without
  `--no-default-features`. It also breaks feature unification for any consumer whose
  dependency graph enables both, which is a cost paid by third parties for our convenience.
- **A no-backend default (`default = ["download"]`).** Honest, and it never mispicks — but
  `cargo install sceptre-cli` would then produce a binary that cannot perform OCR at all,
  failing at runtime with a configuration error instead of a dlopen panic. Trading one
  broken default install for another is not an improvement.
- **A target-conditional default.** Cargo cannot express target-conditional *features* —
  `[target.'cfg(...)'.features]` does not exist (rust-lang/cargo#1197). The only
  approximation is a shim crate with target-specific dependencies, and it would have to pick
  `ort-dynamic` on precisely the targets that have no prebuilt — which are precisely the
  targets with no `libonnxruntime` to `dlopen` either. That reproduces the original runtime
  panic on exactly the machines the mechanism exists to help. A build-time failure naming
  the missing prebuilt, with two documented remedies, is strictly better than a runtime
  panic on first OCR.
- **`Accelerator::Auto` as the default.** Fastest out of the box, and wrong: the same ONNX
  graph produces different numeric output on different providers, so an upgrade would
  silently change a user's OCR results and invalidate their own regression baselines.

### Consequences

- Good: `cargo install sceptre-cli` produces a runnable, self-contained binary; both
  provisioning strategies stay expressible additively; `--backend tract` is real.
- Good: accuracy and performance claims now name the runtime that produced them, and the
  runtime is reportable from the binary itself.
- Good: a requested accelerator that cannot register fails loudly instead of quietly running
  on CPU and looking like a success.
- Bad: the default install now fails at *build* time on every target `ort` publishes no
  prebuilt for — `x86_64-apple-darwin`, all `*-unknown-linux-musl`, `armv7`, `riscv64gc`,
  `*-unknown-freebsd`, `i686`, `s390x`, `powerpc64le`. This is a deliberate trade: a build
  error naming the problem beats a binary that installs cleanly and panics later. The
  installation docs name the full target set and both remedies.
- Bad: the default install pulls a TLS/HTTP stack and a build-time binary download into
  every user's supply chain (see the status update on ADR 0012).
- Neutral: CoreML, DirectML, and CUDA support is compiled-in capability, not validated
  accuracy. The docs say so rather than implying the CPU numbers transfer.

## Related

- [ADR 0012](0012-ort-runtime-provisioning.md) — establishes `ort-bundled` / `ort-dynamic`
  and defers the CLI's per-target selection; this ADR resolves that deferral.
- [ADR 0016](0016-parity-harness-and-test-corpus.md) — the parity harness whose golden
  fixtures this ADR scopes to a named runtime.
- [ADR 0021](0021-benchmark-methodology-and-gate.md) — the performance-side equivalent of
  this decision: it made the speed and memory claim measurable and honest; this one does the
  same for the accuracy claim's runtime.
