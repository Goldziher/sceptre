---
priority: high
---

# Crate Layout

A Cargo workspace: one rlib core plus a thin CLI. Shared metadata, dependency pins, lints, and profiles live in the root `Cargo.toml`; members inherit with `.workspace = true`.

## `crates/sceptre` (library, rlib)

- `lib.rs` — crate docs, feature-gated `pub mod`s, curated flat re-exports.
- `error.rs` — `OcrError` + `Result`.
- `types.rs` — `Point`, `BBox`, `Quad`, `TextLine`, `OcrResult`.
- `config/` — `detection.rs`, `recognition.rs`, `concurrency.rs`, `model.rs`, aggregated in `OcrConfig`.
- `engine/` — `Reader` + `ReaderBuilder`; `engine/seams/` holds the injectable `ModelProvider` / `ProgressSink` traits and defaults.
- `imaging.rs` — decode to RGB + grayscale.
- `detect/` — `preprocess`, `craft`, `postprocess`, `group` (CRAFT).
- `recognize/` — `crop`, `preprocess`, `crnn`, `ctc`, `charset`, `contrast` (CRNN + CTC).
- `inference/` — `ModelBackend` seam + `ort_backend` / `tract_backend` impls, and `candle/` (hand-written `craft_net` / `crnn_net` over ONNX initializers, plus `device` selection).
- `models/` — `registry` (gen2 model table → Hugging Face) + `download` (fetch/cache/verify).
- `mcp/` — rmcp server (`mcp` feature).

## `crates/sceptre-cli` (binary `sceptre`)

- `main.rs` (thin), `cli.rs` (clap `Cli`/`Commands` + dispatch), `overrides.rs` (flattened flag group → `OcrConfig`), `style.rs` (anstyle palette).

## Repo root

- `adrs/` — MADR decision records. `tools/` (Rust crate) + `python/sceptre_rs_tools/` — model export / golden-fixture tooling. `poly.toml`, `rustfmt.toml`, `deny.toml`, `Taskfile.yaml`, `.github/workflows/ci.yaml`.
