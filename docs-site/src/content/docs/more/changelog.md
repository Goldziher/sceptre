---
title: "Changelog"
description: Release history for sceptre.
---

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] - 2026-08-08

### Added

- **Opt-in whole-page orientation detection.** `DetectionConfig::detect_orientation` (default
  `false`, plus a CLI flag and an MCP parameter) probes a page at 0°/90°/180°/270° by running a
  reduced-canvas CRAFT pass in each rotation and scoring region- and link-head activation, then
  detects and recognizes the winning rotation and maps the output quads back to the caller's
  frame. On the six known-rotated corpus images this takes token-F1 from ~0.00 to 0.911 and CER
  from 0.858 to 0.134. The default stays `false` because the scorer false-positives on five
  upright images (four dense tables/receipts plus `kannada.png`), where a wrong rotation is a
  total loss rather than a degradation — see
  [ADR 0037](https://github.com/xberg-io/sceptre/blob/main/adrs/0037-opt-in-whole-page-orientation-pre-pass.md) for the measured before/after
  tables and the false-positive analysis.
- **The orientation pre-pass no longer rotates upright pages.** A rotation is applied only when the
  combined region+link score and the link-only score independently select the same one, each
  clearing `orientation_margin`. The two CRAFT heads fail in opposite directions — the region head
  responds to stroke density rather than glyph orientation and drifts on dense tables and receipts,
  while the link head discriminates orientation but is noisy on photographed scenes — so a rotation
  one proposes and the other refuses is the signature of a false positive. Over the 23
  orientation-labeled corpus images this goes from 18/23 to **23/23**: all five wrong rotations are
  dropped, all six correcting rotations are kept, and no new false positive appears. No extra CRAFT
  pass and no new config field. `detect_orientation` still defaults to `false`. See
  [ADR 0038](https://github.com/xberg-io/sceptre/blob/main/adrs/0038-orientation-requires-two-agreeing-scores.md).
- **Opt-in CTC beam-search decoding.** `RecognitionConfig::decoder` accepts `Decoder::BeamSearch`,
  a faithful port of EasyOCR's `ctcBeamSearch` (`recognize/beam.rs`), reusing the greedy path's
  probability matrix and `custom_mean` confidence so confidence stays comparable across decoders.
  Greedy remains the default: measured against the tier-2 golden corpus, beam search is a mixed,
  net-negative change (one win, two regressions, Σdelta −0.138 over 8 images), and the pattern —
  dropping a character mid-word rather than substituting one — does not improve with a wider
  beam. `Decoder::WordBeamSearch` remains a config error; it needs per-language dictionaries and
  word segmentation sceptre does not have. See
  [ADR 0036](https://github.com/xberg-io/sceptre/blob/main/adrs/0036-opt-in-ctc-beam-search-decoding.md).
- Benchmark quality metrics gained a CJK bigram tokenizer (word-level F1 now scores CJK text on
  overlapping character pairs instead of treating a whole line as one token), line-level
  detection precision/recall/F1, and a reading-order score (anchor longest-increasing-subsequence
  over exactly-once tokens), ported from xberg's `benchmark-harness` quality module.
- The benchmark harness reports R-7 percentiles (p95, p99) with sample-count suppression — a
  percentile computed from fewer than 20 (p95) or 100 (p99) samples is reported as absent rather
  than as the maximum wearing a statistical label — plus CPU core-seconds per run (summed
  user+system time from `/usr/bin/time`, not sampled) and, per corpus image, the `corpus.lock.json`
  sha256 that image was fetched against, so a report cites exactly which corpus snapshot it
  measured.
- The benchmark comparison report carries a `schema_version`, and `validate_report` rejects a
  malformed report before it is written to disk instead of failing whoever reads it back later.
- The corpus-mean quality gate was replaced with per-image guardrails: `derive_guardrails` builds
  a floor per labeled image from a baseline run (ported from xberg's `split_benchmark.rs`
  boundary guardrails), and `--assert --guardrails` fails on the specific image that regressed
  instead of on a bimodal corpus average that can hide a large per-image regression behind an
  unrelated improvement.
- A PDF rasterization dev script, `task python:rasterize`, for turning a PDF page into a raster
  image ahead of benchmarking or fixture generation.
- A backend × accelerator benchmark matrix: `crates/sceptre/tests/backend_matrix.rs` runs one
  `#[ignore]`d test per backend/accelerator pairing at a fixed `canvas_size` so legs are
  comparable, aggregated into `benchmarks/published/backends.json` and a new `Benchmarks` CI
  workflow that also re-homes the criterion microbenchmarks and the EasyOCR head-to-head jobs.
  This is a liveness/throughput measurement, not a correctness bar — `backend_agreement.rs`
  remains the only correctness gate for every backend/accelerator pairing. See
  [ADR 0035](https://github.com/xberg-io/sceptre/blob/main/adrs/0035-backend-accelerator-benchmark-matrix.md).
- **The `candle` backend runs.** `--backend candle` (feature `candle`) executes CRAFT and the gen2
  recognizers with no ONNX Runtime at all. It does not interpret the ONNX graph: the two networks
  are written out against `candle_nn` and their weights are read from the initializers of the same
  ONNX files the other backends load, so there is no new model artifact, no change to the registry
  or its sha256 pins, and no `protoc` build dependency. Validated against `ort` at the tensor level
  (within `1e-4` over the shapes the export pipeline checks) and end to end (exact sorted
  word-multiset equality on `english.png` and `cyrillic.png`). See
  [ADR 0031](https://github.com/xberg-io/sceptre/blob/main/adrs/0031-hand-written-candle-networks.md).
- **GPU execution on the `candle` backend**, via the `candle-metal` and `candle-cuda` features and
  `--accelerator metal` / `--accelerator cuda`. This reaches a GPU without provisioning a
  provider-specific ONNX Runtime build. Metal is validated against `ort` on real hardware; neither
  path is exercised in CI, which has no GPU runner.
- `Accelerator::Metal`, and `Backend::hardware_accelerators()` / `Backend::supports()` reporting
  which accelerators each backend can run on.
- **`run --timings` now reports the stage breakdown in the `--format json` payload**, not only as a
  human line on stderr. A single image gains a `timings` key alongside `lines`; a batch, whose
  payload is an array, is wrapped in a `{ "images", "timings" }` envelope. Times are milliseconds:
  `setup_ms` (model load + decode), `detect_ms`, `recognize_ms`, `total_ms`.

### Changed

- **Models are hosted under the `xberg-io` Hugging Face org.** The nine ONNX repos moved from
  `sceptre-ocr/<model>` to `xberg-io/sceptre-<model>`, consolidating them with the rest of the
  stack's model artifacts. The exports are byte-identical and every sha256 pin is unchanged, so
  download verification is unaffected; the on-disk hub cache directory changes name with the repo
  id, so the first run after upgrading re-downloads once. Hugging Face serves redirects from the
  old ids, so 0.2.0–0.4.0 keep resolving models. The `sceptre-` prefix keeps
  [ADR 0011](https://github.com/xberg-io/sceptre/blob/main/adrs/0011-repointable-registry-owner.md)'s `registry_owner` override a pure
  owner-segment swap. See [ADR 0040](https://github.com/xberg-io/sceptre/blob/main/adrs/0040-models-hosted-under-the-xberg-io-hf-org.md).
- **Accelerator validation is a per-backend table** instead of a test for `ort`. `ort` takes
  `coreml`, `directml`, `cuda`; `candle` takes `metal`, `cuda`; `tract` remains CPU-only. `cuda` is
  shared because it names hardware rather than an execution provider, while `coreml` and `metal`
  stay distinct — same Apple GPU, different frameworks and different numerics — and a wrong pairing
  is rejected with the equivalent for your backend named in the error. Amends
  [ADR 0029](https://github.com/xberg-io/sceptre/blob/main/adrs/0029-cli-provisioning-default-and-runtime-scoped-parity.md); see
  [ADR 0032](https://github.com/xberg-io/sceptre/blob/main/adrs/0032-per-backend-accelerator-support.md).
- `runtime_info_for` (and `sceptre env`) no longer reports a fixed `cpu` for the `candle` backend.
  It resolves the device that would actually be opened, and reports the accelerator as undetermined
  when the backend is not compiled in, rather than naming a run that cannot happen.
- **The project moved to the `xberg-io` organisation.** The repository is now
  `github.com/xberg-io/sceptre` and the documentation is served from
  `https://docs.sceptre.xberg.io` on the shared Xberg theme, replacing the
  `goldziher.github.io/sceptre` project path. The old repository URL redirects; the old
  documentation URLs do not. Crate names, the CLI binary name and the public API are unchanged. See
  [ADR 0033](https://github.com/xberg-io/sceptre/blob/main/adrs/0033-org-migration-docs-domain-and-shared-theme.md).
- **The test fixture corpus moved to the `test_documents` git submodule**, matching the
  `xberg`/`xberg-enterprise` idiom (a plain submodule plus a `TEST_DOCUMENTS_DIR` env var with a
  repo-relative fallback), replacing the images and transcripts vendored directly into
  `crates/sceptre/tests/data/`. `task setup` now fetches the corpus
  (`python3 test_documents/scripts/fetch_corpus.py --include 'images/**'`); a missing or unfetched
  corpus skips the tests that need it rather than substituting a different image. See
  [ADR 0034](https://github.com/xberg-io/sceptre/blob/main/adrs/0034-test-documents-corpus-via-content-addressed-fetch.md).

### Fixed

- **Rotated text boxes now match OpenCV's geometry.** `imageproc`'s `min_area_rect` snapped every
  rectangle corner outward with a per-corner `floor`/`ceil`; OpenCV's `minAreaRect`/`boxPoints`,
  which EasyOCR uses, never does. Axis-aligned boxes were unaffected — they already matched
  bit-for-bit — but a rotated box inflated by up to ~1px per corner in heat-map space and ~2px
  after the `x2` scale-up, enough to flip borderline line merges: `french.jpg`'s `LOUVRE` box came
  out at slope 0.129 against cv2's 0.096, just over `slope_ths`, splitting one reference line into
  two. Replaced with in-crate rotating calipers over `imageproc`'s (unaffected) convex hull,
  keeping corners in `f64` with a single final cast, and testing every hull edge including the
  closing one that `imageproc`'s `windows(2)` scan omits. Line recall, precision and F1 are now
  **1.000 on all eight parity images** (`french.jpg` from 0.833/0.625, `english.png` from
  1.000/0.923), with `word_f1` unchanged everywhere. See
  [ADR 0039](https://github.com/xberg-io/sceptre/blob/main/adrs/0039-opencv-faithful-min-area-rect.md).
- `wide` moved off the yanked 1.6.0.
- **Beam-search decoding was nondeterministic**: the same crop could recognize to different text
  across runs of the same binary. Beam pruning and the final labeling-selection step both broke
  ties over a `HashMap`, whose per-instance random hash seed made iteration order — and so which
  tied entry a `sort_by`/`max_by` landed on — vary run to run for identical input. Fixed by a
  total order over `(total mass, labeling)` that breaks every float tie on the labeling itself,
  used by both the pruning sort and the final selection.

### Notes

- `candle` is **not** the fast backend. On Apple Silicon its CPU path measures ~10× slower than
  `ort` on the CPU, and Metal is still ~3–4× slower than `ort` on the CPU. Choose it for what it
  removes — the ONNX Runtime dependency and its provisioning matrix — not for speed. It is excluded
  from the published benchmark drift gate.
- `candle` is not a pure-Rust backend: `candle-core` 0.10+ pulls `tokenizers`, which compiles
  oniguruma from C. `tract` remains the pure-Rust path. The 0.11 pin is deliberate — 0.9.x
  miscomputes one of CRAFT's convolutions.

### Breaking

- `Accelerator` gains a `Metal` variant, so exhaustive matches on it need a new arm.

## [0.4.0] - 2026-08-07

### Changed

- **`cargo install sceptre-cli` now produces a self-contained binary.** The CLI's default features
  are `["ort-bundled", "download"]` instead of `["ort-dynamic", "download"]`, so the documented
  install path no longer requires a system `libonnxruntime`. To keep loading ONNX Runtime at runtime,
  install with `cargo install sceptre-cli --features ort-dynamic` — that override is additive and
  needs no `--no-default-features`. On targets with no prebuilt ONNX Runtime (Intel macOS, musl,
  armv7, riscv64, FreeBSD) the default now fails at build time; use `--features ort-dynamic` or
  `--no-default-features --features tract,download`.
- `deny.toml` audits every feature (`all-features = true`); previously only the default graph was
  checked, so the `ort-bundled` dependency tree had never been reviewed.
- The benchmark corpus is vendored into the repository instead of resolved from a Git-LFS submodule,
  so tests and benchmarks no longer need a submodule checkout. `bench::load_corpus_image` now fails
  loudly on a missing fixture instead of silently substituting a stand-in image, which could make a
  benchmark measure the wrong picture and still look healthy. The images and their transcripts are
  excluded from the published crate — only the Python benchmark tooling reads them.

### Added

- `model.accelerator` / `--accelerator` selects the execution provider (`cpu`, `auto`, `coreml`,
  `directml`, `cuda`), with the `ort-coreml`, `ort-directml`, and `ort-cuda` features. An explicitly
  requested provider that cannot register is a hard error rather than a silent fall back to CPU.
- `sceptre env` reports the runtime behind a result — sceptre and ONNX Runtime versions, provisioning
  strategy, requested and registered accelerator, architecture, and the model digests.
- `sceptre models download --all` (and `models list --all`) covers every language, so pre-seeding a
  cache no longer needs eight `--lang` flags.
- A `tract` feature on the CLI. `--backend tract` was previously accepted but never compiled in.
- CI verifies the real `cargo install` paths end to end, runs parity per execution provider, and
  uploads the reports it used to discard.
- Published benchmark numbers are now generated, not transcribed. `task python:publish` distils a
  measured run into the committed `benchmarks/published/latest.json` — headline figures plus the
  sceptre, ONNX Runtime, EasyOCR and torch versions that produced them — and regenerates the tables
  in the README and on the docs site from it. A report carrying no provenance block is refused
  rather than published, and `task python:publish:check` fails CI when the committed tables drift
  from the artifact. See ADR 0030.

### Fixed

- The cache fast path no longer returns unreadable or zero-length artifacts, and a SHA-256 mismatch
  now evicts the artifact and its backing blob so a corrupt download self-heals instead of being
  served from cache forever.
- The published speed figures were measured on an unrecorded machine with an older harness and
  overstated the advantage: re-measured with full provenance, sceptre is ~2.3× faster warm and
  ~2.4× faster cold than EasyOCR, not the ~2.8× and ~4.4× previously claimed. The peak-RSS
  advantage (~3×) is unchanged. Numeric claims now live only in the generated table, so the
  surrounding prose can no longer drift away from a measurement.
- The golden parity fixtures were regenerated and now carry the `metadata` provenance block that
  the format has always documented; the committed ones predated it and recorded neither the
  EasyOCR/torch version nor the sceptre commit behind them. Every EasyOCR reference side reproduced
  byte-identically. `example.json` lost five low-confidence lines and gained two, because it
  predated the 0.1.1 change that made `recognition.filter_ths` actually filter.
- `task python:benchmark`'s flags (`--group`, `--limit`, `--repeats`, `--baseline`, `--assert`)
  reach the module instead of being silently dropped.

## [0.3.0] - 2026-08-04

### Added

- Filesystem-free model provisioning through `ModelDescriptor`, `model_descriptors`,
  `ModelArtifact::Bytes`, and the SHA-256-verifying `VerifiedModelProvider`.
- Paired `model.detector_path` and `model.recognizer_path` configuration for mobile or other
  host-managed local model assets.
- Eager, reusable model initialization through `Reader::warm_up` and
  `ReaderBuilder::build_warmed`.

### Changed

- `ModelProvider` now resolves `ModelArtifact::{Path, Bytes}` instead of paths only; initialized
  detector and recognizer plans are serialized, cached, and release their source provider.
- Browser WASM uses sequential crop and decoding loops without a Rayon worker pool, and the pure-Rust
  backend is aligned to tract 0.23.4.

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

[0.5.1]: https://github.com/xberg-io/sceptre/compare/v0.4.0...v0.5.1
[0.4.0]: https://github.com/xberg-io/sceptre/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/xberg-io/sceptre/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/xberg-io/sceptre/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/xberg-io/sceptre/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/xberg-io/sceptre/releases/tag/v0.1.0
