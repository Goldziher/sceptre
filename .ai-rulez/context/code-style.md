---
priority: high
---

# Code Style

Conventions baked into context so they ship into every AI tool's config.

## Module layout

- One concern per file. Folder modules get a thin `mod.rs` (declare submodules + re-export); logic lives in named siblings.
- 1000-line cap on `crates/*/src/**/*.rs`, enforced by the `rust-max-lines` poly check. Refactor by extraction, never by lifting the cap.
- The library re-exports a curated flat surface from `lib.rs` (`pub use error::{OcrError, Result}`, `pub use types::*`, the config types), so consumers write `use easyocr::{Reader, OcrConfig, OcrResult}`.

## Errors

- Library errors: one `thiserror` enum (`OcrError`) plus `pub type Result<T>`. System errors (`Io`) bubble via `#[from]`; application variants carry `message` + an optional `#[source]`.
- CLI errors: `anyhow` with `.with_context(...)`.

## Config

- Aggregate struct (`OcrConfig`) of per-stage structs, each `#[serde(deny_unknown_fields)]` + `#[serde(default)]` with a `Default` impl. Precedence: defaults < file < env < flags.

## Dependency injection

- Extension points are traits with in-crate default impls stored as `Arc<dyn Trait>` "seams" on the `Reader`, built via `Reader::builder()`. No global mutable state.

## Output & concurrency

- `tracing` → stderr; structured data → stdout. Raw `print_stdout`/`print_stderr` are denied at the workspace level; the CLI opts back in locally with a justified `#[expect(...)]`.
- Terminal colour uses `anstyle` rendered through `anstream` (honors `NO_COLOR`).
- Parallelism is Rayon `par_iter` bounded by the single `ConcurrencyConfig` budget.

## Commits

- Conventional Commit prefixes (`feat:`, `fix:`, `perf:`, `chore:`, `refactor:`). Body explains *why*.
