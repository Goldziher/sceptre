# Architecture Decision Records

This directory holds the project's Architecture Decision Records (ADRs) in
[MADR](https://adr.github.io/madr/) format.

## Conventions

- One decision per file, named `NNNN-kebab-title.md`, numbered sequentially.
- Each ADR carries a `status` (`proposed` | `accepted` | `rejected` | `superseded`),
  a `date`, and `deciders`.
- ADRs are immutable once accepted. To reverse a decision, add a new ADR that
  supersedes the old one; mark the old one `superseded by NNNN`.
- Add an ADR whenever a choice between real alternatives has lasting consequences
  (a dependency, backend, algorithm, public API shape, model source, or target
  strategy). See the `adr-discipline` rule in `.ai-rulez/rules/`.

## Index

| ADR | Title | Status |
| --- | --- | --- |
| [0000](0000-use-madr-for-adrs.md) | Record decisions as MADR ADRs | Accepted |
| [0001](0001-cargo-workspace-rlib-core-thin-cli.md) | Cargo workspace: rlib core + thin CLI | Accepted |
| [0002](0002-scope-gen2-recognizers-and-craft.md) | Scope to gen2 recognizers + CRAFT | Superseded by 0026 |
| [0003](0003-model-source-itext-onnx-runtime-download.md) | Model source: iText ONNX, runtime download | Superseded by 0025 |
| [0004](0004-inference-backend-seam.md) | Inference backend seam: ort / tract / candle | Accepted |
| [0005](0005-mcp-as-feature-module-and-subcommand.md) | MCP as a feature-gated module + subcommand | Accepted |
| [0006](0006-ai-rulez-for-agent-config.md) | Generate agent config with ai-rulez | Accepted |
| [0007](0007-tooling-scaffolding-from-basemind.md) | Tooling scaffolding adapted from basemind | Accepted |
| [0008](0008-uv-taskfile-tools-infra.md) | In-repo dev tooling: uv Python fallback + Rust `tools/` for model export | Accepted |
| [0009](0009-candle-evaluation-ort-primary.md) | Backend evaluation: `ort` primary, `candle` deferred off the critical path | Superseded by 0031 |
| [0010](0010-ocr-engine-seam-and-encapsulation-lockdown.md) | Public `OcrEngine` seam, internal DTO boundaries, and encapsulation lockdown | Accepted |
| [0011](0011-repointable-registry-owner.md) | Re-pointable model registry via a `registry_owner` override | Accepted |
| [0012](0012-ort-runtime-provisioning.md) | ONNX Runtime provisioning: `ort-bundled` + `ort-dynamic` (xberg pattern) | Accepted |
| [0013](0013-imageproc-craft-postprocess-geometry.md) | `imageproc` for CRAFT postprocess geometry | Accepted |
| [0014](0014-public-detect-recognize-line-and-model-provisioning.md) | Public detect / recognize-line methods and a model-provisioning surface | Accepted |
| [0015](0015-bench-seam-and-criterion.md) | Criterion microbenchmarks behind a `bench` feature seam | Accepted |
| [0016](0016-parity-harness-and-test-corpus.md) | Parity harness: dual golden fixtures, HF-cache model resolution, and the test corpus | Accepted |
| [0017](0017-hf-hub-native-cache.md) | Resolve models through the Hugging Face hub's native on-disk cache | Accepted |
| [0018](0018-cv2-exact-craft-dilation.md) | cv2-exact CRAFT dilation (amends ADR 0013) | Accepted |
| [0019](0019-parity-safe-perf-and-simd.md) | Parity-safe performance optimization and SIMD | Accepted |
| [0020](0020-release-strategy.md) | Release strategy: tag-triggered multi-platform build with crates.io dry-run | Superseded by 0023 |
| [0021](0021-benchmark-methodology-and-gate.md) | Head-to-head benchmark methodology and regression gate | Accepted |
| [0022](0022-input-image-format-coverage.md) | Input image format coverage: pure-Rust decoders only | Accepted |
| [0023](0023-publish-to-crates-io-via-trusted-publishing.md) | Publish to crates.io for real, via GitHub Actions trusted publishing | Accepted |
| [0024](0024-documentation-site-and-web-presence.md) | Documentation site, social assets, and CI docs deployment | Superseded by 0033 |
| [0025](0025-first-party-onnx-exports.md) | First-party ONNX exports on the `sceptre-ocr` Hugging Face org | Accepted |
| [0026](0026-extend-gen2-scope-telugu-kannada.md) | Extend gen2 scope to Telugu and Kannada | Accepted |
| [0027](0027-tract-fixed-canvas-craft.md) | Fixed-canvas CRAFT detection on the tract backend | Accepted |
| [0028](0028-host-supplied-model-artifacts-and-wasm-execution.md) | Host-supplied model artifacts and sequential browser-WASM execution | Accepted |
| [0029](0029-cli-provisioning-default-and-runtime-scoped-parity.md) | CLI provisioning default and runtime-scoped parity | Accepted |
| [0030](0030-published-benchmark-artifact-and-drift-gate.md) | Published benchmark artifact and drift gate | Accepted |
| [0031](0031-hand-written-candle-networks.md) | Hand-written `candle` networks as the GPU and no-C++-runtime backend | Accepted |
| [0032](0032-per-backend-accelerator-support.md) | Per-backend accelerator support (amends ADR 0029) | Accepted |
| [0033](0033-org-migration-docs-domain-and-shared-theme.md) | Docs on `docs.sceptre.xberg.io` with the shared `@xberg-io/docs-theme` | Accepted |
| [0034](0034-test-documents-corpus-via-content-addressed-fetch.md) | Test corpus via the `test_documents` submodule, content-addressed fetch (supersedes ADR 0016 corpus decision) | Accepted |
| [0035](0035-backend-accelerator-benchmark-matrix.md) | Backend × accelerator benchmark matrix (amends ADR 0032) | Accepted |
| [0036](0036-opt-in-ctc-beam-search-decoding.md) | Opt-in CTC beam-search decoding, greedy stays default | Accepted |
| [0037](0037-opt-in-whole-page-orientation-pre-pass.md) | Opt-in whole-page orientation pre-pass, disabled by default | Accepted |
