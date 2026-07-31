---
status: accepted
date: 2026-07-31
deciders: Na'aman Hirschfeld
---

# In-repo dev tooling: uv Python fallback + Rust `tools/` for model export

## Context and Problem Statement

Model export and conversion (CRAFT + gen2 CRNN `.pth` weights into the ONNX/safetensors
artifacts the library loads) and golden-fixture generation for parity testing are
recurring dev tasks. They need a permanent, reproducible home in the repo rather than
ad-hoc scripts. The reference conversion path lives in Python EasyOCR (torch), but the
project is Rust-first and we prefer to keep tooling in Rust where practical.

## Decision Outcome

Chosen option: **scaffold both paths now, defer the logic**.

- A Rust `tools/` crate (`sceptre-tools`, `publish = false`) is the preferred model
  export/conversion path (candle-based). It is a workspace member but excluded from
  `default-members`, so a plain `cargo build` skips it.
- A Python package (`sceptre_rs_tools`, uv-managed via root `pyproject.toml`) is the
  fallback export path and the golden-fixture generator. Heavy dependencies (torch,
  easyocr, onnx, numpy) live in an opt-in `export` dependency group; the light `dev`
  group (ruff, pytest) is what CI installs.
- The Taskfile `setup`/`update`/`upgrade` targets drive uv for the Python side.
- Both entry points are stubs that exit informatively until the conversion logic lands.

### Consequences

- Good: a permanent, reproducible tooling home with a clear Rust-preferred / Python-
  fallback split; contributors do not install torch to run lint/test.
- Good: `default-members` keeps `sceptre-tools` out of the normal build/test loop.
- Bad: two parallel toolchains (cargo + uv) to keep current.
- Bad: the export/golden logic is deferred; the scaffolding does nothing useful yet.
