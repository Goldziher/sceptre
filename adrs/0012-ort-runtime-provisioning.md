---
status: accepted
date: 2026-07-31
deciders: Na'aman Hirschfeld
---

# ONNX Runtime provisioning: `ort-bundled` + `ort-dynamic` (xberg pattern)

## Context and Problem Statement

[ADR 0009](0009-candle-evaluation-ort-primary.md) makes `ort` the primary native backend.
`ort` links the ONNX Runtime **C++ native library**, and our pin is
`default-features = false` (so `ort` never silently auto-downloads a prebuilt). That means a
build must explicitly choose how the native library is obtained, and no single strategy fits
every target — notably, ONNX Runtime dropped prebuilt `x86_64-apple-darwin` static binaries
after v2.0.0-rc.11, so `download-binaries` has no artifact there.

## Decision Drivers

- Zero-config native builds for contributors and CI on supported targets.
- A working path for targets with no prebuilt (e.g. macОS x86_64) and for release packaging.
- Explicit, per-target control — provisioning must not be an implicit default.
- Consistency with the sibling xberg project, which already solved this.

## Considered Options

- Single `download-binaries` — simplest, but no artifact for some targets, and forces a
  build-time binary fetch everywhere.
- Single `load-dynamic` — no build-time fetch, but nothing runs until a compatible
  `libonnxruntime` is present, pushing provisioning onto every environment.
- **Two feature flags (`ort-bundled` / `ort-dynamic`) over a provisioning-agnostic `ort`
  backend feature** — the caller picks per target. This is the xberg pattern.

## Decision Outcome

Chosen option: **mirror xberg — a provisioning-agnostic `ort` feature plus two selectable
provisioning features**, none enabled by default:

- `ort` — compiles the backend against the `ort` API; provisioning-agnostic.
- `ort-bundled = ["ort", "ort?/download-binaries", "ort?/tls-rustls"]` — fetches a prebuilt
  ONNX Runtime at build time; zero-config native for contributors/CI on supported targets.
- `ort-dynamic = ["ort", "ort?/load-dynamic"]` — `dlopen`s `libonnxruntime` at runtime via
  `ORT_DYLIB_PATH`; the path for targets without a prebuilt and for shipping a dylib next to
  the release binary (no `ORT_DYLIB_PATH` needed when it sits beside the executable).

The workspace pins `ort = { version = "2.0.0-rc.13", default-features = false, features =
["std", "ndarray", "api-18"] }`, matching xberg (the `ndarray` feature backs the tensor
interop, `api-18` pins the C API version). The consumer (and, later, the CLI) selects
`ort-bundled` or `ort-dynamic` per target. Verified: `ort-bundled` builds, links, and runs
the backend's unit tests locally.

This refines ADR 0004/0009's "ort = native default" with a concrete provisioning mechanism;
it does not change the backend choice. The pure-Rust story is unaffected — that is `tract`
(no native library), not either `ort` feature.

### Consequences

- Good: zero-config native via `ort-bundled` where prebuilts exist; a real fallback via
  `ort-dynamic` where they do not, and for release bundling.
- Good: provisioning is explicit and per-target, not an implicit auto-download.
- Good: aligns with xberg, so target-specific handling (e.g. macOS x86_64 → `ort-dynamic`)
  can be copied rather than re-derived.
- Bad: `ort-bundled` pulls a TLS/HTTP stack (`tls-rustls` → `ureq`, `rustls`) and a build-time
  binary download into the supply chain (covered by `cargo-deny`).
- Neutral: the CLI's per-target feature selection is deferred to the CLI pass.

## Status update (2026-08-06)

[ADR 0029](0029-cli-provisioning-default-and-runtime-scoped-parity.md) resolves the deferred
item above: the CLI ships `default = ["ort-bundled", "download"]`, with `ort-dynamic`
available as an additive override that needs no `--no-default-features`, and `tract` exposed
for targets that have neither a prebuilt nor a system `libonnxruntime`.

Two claims made here were also weaker than stated. The "no artifact for some targets" note
named only macOS x86_64; the real set is `x86_64-apple-darwin`, every
`*-unknown-linux-musl`, `armv7-unknown-linux-gnueabihf`, `riscv64gc-*`, `*-unknown-freebsd`,
`i686-*`, `s390x`, and `powerpc64le`.

And the supply-chain consequence claimed the `ort-bundled` tree was "covered by
`cargo-deny`" — it was not. `deny.toml` audited only the default feature graph, which at the
time did not include `ort-bundled`, so that dependency tree had never been in the
license/advisory scan. Measured against the `ort-dynamic` graph it adds roughly ten
build-only dependencies (all already licence-allow-listed) — `byteorder`, `getrandom`,
`hmac-sha256`, `lzma-rust2`, `ring`, `socks`, `ureq`, `ureq-proto`, `utf8-zero`,
`webpki-roots` — plus a build-time fetch of the ONNX Runtime prebuilt from `cdn.pyke.io`.
The claim became true only when `deny.toml` moved to `all-features = true`.
