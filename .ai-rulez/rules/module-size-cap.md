---
priority: critical
---

# Module Size Cap

- Every file under `crates/*/src/**/*.rs` is capped at **1000 lines** by the poly check `rust-max-lines` (see `poly.toml`).
- When a file approaches the cap, refactor by extracting helpers, types, or submodules — never raise the cap.
- The codebase is file-per-module: a folder module gets a thin `mod.rs` that only declares submodules and re-exports; real logic lives in named sibling files (`detect/preprocess.rs`, `detect/postprocess.rs`, …). Match that shape when a module area grows.
