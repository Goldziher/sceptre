# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-08-03

### Added

- Telugu (`telugu_g2`) and Kannada (`kannada_g2`) gen2 recognizers, completing EasyOCR's full
  eight-model gen2 recognizer family. Select them with `--lang telugu` / `--lang kannada`.
- A first-party `.pth -> ONNX` model export pipeline (`sceptre_rs_tools.export`) that converts
  EasyOCR's CRAFT and gen2 CRNN weights to the ONNX artifacts the library loads.
- Full pure-Rust `tract` pipeline: the recognizers run under `tract` (the export replaces gen2's
  `AdaptiveAvgPool` with an equivalent `ReduceMean`, which upstream exports lack), and CRAFT runs on a
  fixed square canvas under `tract`, with a cross-backend test asserting `tract` recognizes the same
  words as `ort`.

### Changed

- Model source is now first-party: the registry points at the [`sceptre-ocr`](https://huggingface.co/sceptre-ocr)
  Hugging Face org instead of `itextresearch`, with fresh sha256 pins for our own exports.

## [0.1.1] - 2026-08-03

### Added

- Exposed the crate version as `sceptre::VERSION` for adapters and diagnostics.

### Fixed

- Detection and recognition preprocessing now borrow their source pixels, copy axis-aligned crops
  by row, reuse immutable recognizer/CTC state, and bypass parallel dispatch for singleton batches,
  reducing warm-path allocations and scheduling overhead without changing OCR output.
- Each `Reader` now owns its configured Rayon worker pool instead of attempting to initialize the
  process-global pool, so readers with different thread budgets remain isolated and embedding
  Sceptre cannot override an application's Rayon configuration.
- `recognition.filter_ths` now validates its range and removes recognition results below the
  configured confidence threshold instead of being accepted without affecting output; its default
  is now `0.1`, which improves the measured DocLayNet and TextOCR quality without regressing the
  receipt and scanned-PDF fixtures.
- The Hugging Face cache root now falls back to `%USERPROFILE%` when `$HOME` is unset, so
  `sceptre models` works on Windows instead of failing to locate the cache.

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

[Unreleased]: https://github.com/Goldziher/sceptre/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Goldziher/sceptre/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Goldziher/sceptre/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Goldziher/sceptre/releases/tag/v0.1.0
