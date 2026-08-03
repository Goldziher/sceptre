# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Exposed the crate version as `sceptre::VERSION` for adapters and diagnostics.

### Fixed

- Each `Reader` now owns its configured Rayon worker pool instead of attempting to initialize the
  process-global pool, so readers with different thread budgets remain isolated and embedding
  Sceptre cannot override an application's Rayon configuration.

## [0.1.0] - 2026-08-01

### Added

- CRAFT text detection and gen2 CRNN recognition with CTC decoding, running over ONNX — a
  from-scratch Rust reimplementation of EasyOCR's pipeline.
- Validated EasyOCR parity across six scripts: English, Latin, Chinese (simplified), Japanese,
  Korean, and Cyrillic.
- Two inference backends behind one seam: native ONNX Runtime (`ort`) and pure-Rust `tract` for
  WASM / Android targets.
- Rust library API centered on a single `Reader` handle, configured through a backend-agnostic
  `OcrConfig`.
- `sceptre` CLI with `run`, `detect`, `models`, `mcp`, and `completions` subcommands, including
  multi-image batch mode (models load once) and multi-language recognition.
- MCP server exposing a `readtext` tool for agent integrations.
- Offline-first model provisioning: models download from Hugging Face on first use, cache locally,
  and are sha256-verified on download — every run thereafter reads the cache with no network.

[0.1.0]: https://github.com/Goldziher/sceptre/releases/tag/v0.1.0
