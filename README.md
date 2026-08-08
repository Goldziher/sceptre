<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://cdn.jsdelivr.net/gh/xberg-io/assets@v1/banner/readme-banner-dark.svg">
    <img alt="Xberg" width="420" src="https://cdn.jsdelivr.net/gh/xberg-io/assets@v1/banner/readme-banner-light.svg">
  </picture>
</p>

# sceptre

<div align="center" style="display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin: 20px 0;">
  <a href="https://crates.io/crates/sceptre">
    <img src="https://img.shields.io/crates/v/sceptre?label=Rust&color=007ec6" alt="Rust">
  </a>
  <a href="https://github.com/xberg-io/sceptre/actions/workflows/ci.yaml">
    <img src="https://img.shields.io/github/actions/workflow/status/xberg-io/sceptre/ci.yaml?label=CI&color=007ec6" alt="CI">
  </a>
  <a href="https://docs.rs/sceptre">
    <img src="https://img.shields.io/docsrs/sceptre?label=docs.rs&color=007ec6" alt="docs.rs">
  </a>
  <a href="https://github.com/xberg-io/sceptre/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-007ec6" alt="License">
  </a>
  <a href="https://docs.sceptre.xberg.io">
    <img src="https://img.shields.io/badge/Docs-sceptre-007ec6" alt="Documentation">
  </a>
</div>

<div align="center" style="display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; margin: 28px 0 24px;">
  <a href="https://discord.gg/xt9WY3GnKR">
    <img height="22" src="https://img.shields.io/badge/Discord-Chat-007ec6?logo=discord&logoColor=white" alt="Join Discord">
  </a>
</div>

EasyOCR's accuracy in a single Rust binary — CRAFT text detection and gen2 CRNN recognition over ONNX, with no Python
runtime.

## What and Why?

EasyOCR is accurate, but it is a PyTorch stack: a Python interpreter, a multi-gigabyte runtime, and a heavy process to
keep warm. sceptre is a from-scratch Rust reimplementation of the same pipeline — **CRAFT** text detection then **gen2
CRNN** recognition with CTC decoding, over ONNX — that keeps the accuracy and drops all of that.

It agrees with [EasyOCR](https://github.com/JaidedAI/EasyOCR)'s output across the scripts it supports, runs faster —
even a cold, one-shot run beats EasyOCR's warm reader ([measured figures](#benchmarks)) — and ships as one
self-contained executable with no Python runtime. Models download once, cache locally, and run offline thereafter.
Use it as a Rust library, a CLI, or an MCP server.

Every model call goes through one backend seam, so the deployment target is a build-time choice rather than a rewrite:
ONNX Runtime for native speed, pure-Rust ONNX for WASM and Android, and `candle` for GPU devices or a build that links
no ONNX Runtime at all.

### Features

| Feature | Description |
| ------- | ----------- |
| **Parity accuracy** | Validated against real EasyOCR output across the gen2 scripts — English, Latin, Simplified Chinese, Japanese, Korean, Cyrillic, Telugu, Kannada — on text (word/char-F1) and boxes (IoU), held to a per-image floor rather than character-for-character equality |
| **Faster, warm or cold** | Higher throughput than EasyOCR warm on the same corpus; even a cold, one-shot CLI run beats EasyOCR's already-loaded reader ([measured figures](#benchmarks)) |
| **No Python, no torch** | No interpreter, no multi-gigabyte runtime, no process to keep warm; peak memory is dominated by the CRAFT detector both engines share, so it scales with page size — see [benchmarks](#benchmarks) |
| **One binary, no Python** | A single self-contained executable. Models download once, cache locally, and run offline thereafter — nothing to `pip install`, no interpreter to ship |
| **Three surfaces** | The same engine as a Rust library, a CLI (`sceptre`), and an MCP server for agents |
| **Three backends, one seam** | ONNX Runtime (`ort`) for native speed, pure-Rust ONNX (`tract`) for WASM / Android, and `candle` for a GPU or a build with no ONNX Runtime at all |
| **Pure-Rust image decoding** | PNG, JPEG, BMP, GIF, TIFF, WebP, and NetPBM |

<p align="center"><strong>⭐ Star this repo to show your support — it helps others discover sceptre.</strong></p>

## Quick Start

### Install

```sh
cargo install sceptre-cli
```

That default build is self-contained: it links a prebuilt ONNX Runtime fetched at build time (`ort-bundled`) and enables
model download (`download`), so the binary needs no `ORT_DYLIB_PATH` and no separately installed `libonnxruntime`.

The models — CRAFT plus the gen2 recognizers — are fetched from Hugging Face on first use, cached under the standard HF
cache, and sha256-verified as they download. Every run after that reads the cached models with no network.

> **Targets with no prebuilt ONNX Runtime.** `ort` publishes prebuilts for a fixed target list.
> `x86_64-apple-darwin` (Intel macOS), every `*-unknown-linux-musl` (Alpine),
> `armv7-unknown-linux-gnueabihf`, `riscv64gc-*`, `*-unknown-freebsd`, `i686-*`, `s390x`, and
> `powerpc64le` have none, so the default install fails **at build time** with ort-sys's own
> `no prebuilt binaries available for target ...`. Two remedies:
>
> ```sh
> cargo install sceptre-cli --features ort-dynamic                          # bring your own libonnxruntime
> cargo install sceptre-cli --no-default-features --features tract,download  # pure Rust, then --backend tract
> ```
>
> `--features ort-dynamic` is additive, but it still wins over the bundled default: `load-dynamic`
> implies `ort-sys/disable-linking`, which the ort-sys build script checks before its download
> branch. So switching ONNX Runtime provisioning never needs `--no-default-features` — only the
> pure-Rust path does, because it has to drop `ort` from the graph entirely.

### CLI

One image or many in a single warm process:

```sh
sceptre run receipt.png --lang english --format json
sceptre run page1.png page2.png page3.png            # batch: models load once
sceptre run sign.jpg --lang english --lang korean    # multi-language
sceptre run page.png --lang english --timings        # per-stage load/detect/recognize breakdown
sceptre run huge-scan.png --canvas-size 1600         # trade some accuracy for lower peak memory
```

Decodes PNG, JPEG, BMP, GIF, TIFF, WebP, and NetPBM (all pure-Rust; HEIF/AVIF/JPEG-2000
are out of scope — see [`adrs/0022`](adrs/0022-input-image-format-coverage.md)).

### Library

```rust
use sceptre::{Reader, ReadOptions};

let reader = Reader::builder().build()?;
for line in reader.readtext("receipt.png".as_ref(), &ReadOptions::default())?.lines {
    println!("{} ({:.2})", line.text, line.confidence);
}
# Ok::<(), sceptre::OcrError>(())
```

The library ships `default = []`, so enable a backend and model download:
`sceptre = { version = "0.5", features = ["ort-bundled", "download"] }` (see [Feature flags](#feature-flags)).

WASM and mobile hosts can avoid filesystem and network assumptions by calling `model_descriptors`,
fetching the selected artifacts, constructing a `VerifiedModelProvider`, and passing it to
`Reader::builder().model_provider(...).build_warmed()`.

Android and iOS hosts that package ONNX assets as real files may instead set both
`model.detector_path` and `model.recognizer_path`; the default provider then bypasses Hugging Face.

### MCP server

Expose a `readtext` tool to an agent. The `mcp` subcommand is behind the `mcp` cargo feature, which is off by default:

```sh
cargo install sceptre-cli --features mcp
sceptre mcp --lang english
sceptre mcp --lang english --detect-orientation   # whole-page orientation pre-pass for rotated scans
```

| Tool | Parameter | Type | Description |
| --- | --- | --- | --- |
| `readtext` | `image_path` | string (required) | Filesystem path to the image to run OCR over. |
| `readtext` | `detail` | bool (optional, default `true`) | When `false`, only line text is returned (no box/confidence). |

`--lang`, `--backend`, `--threads`, and `--detect-orientation` (plus its `--orientation-probe-canvas-size`
/ `--orientation-margin` knobs) configure the `Reader` the server is started with — they are not
per-call `readtext` parameters, so every call made to one `sceptre mcp` process shares them.

## Benchmarks

Measured against upstream EasyOCR over a mixed corpus — documents, tables, rotated scans, scene
text and receipts, across the English, Latin, Chinese-simplified, Japanese and Korean recognizer
groups — on CPU. Both engines are measured **identically**: each a fresh subprocess per language
group under `/usr/bin/time`, at its native multi-threaded default, loading its model/reader once and
processing every image — so peak RSS is a like-for-like whole-process figure (EasyOCR's legitimately
includes the torch runtime). Numbers vary with hardware and load, and peak RSS is dominated by the
CRAFT detector both engines share, so its ratio narrows toward parity as page size grows — see the
[benchmarks page](https://docs.sceptre.xberg.io/reference/benchmarks/) for the full methodology.

The table below is generated from [`benchmarks/published/latest.json`](benchmarks/published/latest.json)
by `task python:publish`; CI fails if the two drift apart. Do not edit it by hand.

<!-- generated:benchmark-headline:start -->

| Engine | Throughput (img/s) | Peak RSS | Mean CER | Mean token-F1 |
|---|---|---|---|---|
| EasyOCR (warm/batch) | 0.12 | 4.8 GB | 0.569 | 0.337 |
| **sceptre** (warm/batch) | **0.23** (~1.9×) | **4.5 GB** (~1.4× lower, per-image median) | 0.569 | 0.358 |
| sceptre (cold CLI run) | 0.21 (~1.8×) | 4.5 GB | 0.569 | 0.358 |

Measured with sceptre 0.6.0 (ort/cpu, ONNX Runtime 1.28.0) against EasyOCR 1.7.2 (torch 2.13.0+cu130) on
runner-medium (Linux/x86_64), over 40 of 43 corpus entries. Regenerate with `task python:benchmark` then `task
python:publish`.

<!-- generated:benchmark-headline:end -->

The accuracy columns are the parity check — sceptre and EasyOCR land on the same reading of the
corpus; the speedup is the win. Even a cold, one-shot CLI run, which pays model load on every
invocation, still beats EasyOCR's already-warm reader. Peak memory is dominated by the CRAFT
detector both engines share, so it tracks page size rather than the runtime; sceptre's memory
advantage is having no Python/torch process at all, which shows up on smaller pages and in process
count rather than as a fixed ratio.

**Runtime scope.** Every figure above — speed, memory, and the parity claim — was produced on the
`ort` backend running the **CPU execution provider**, which is what `model.accelerator` defaults to.
The same ONNX graph produces different numeric output on a different provider, so an accelerated run
is a different measurement, not a faster one. Select one with `--accelerator` or
`model.accelerator`; which values a backend accepts differs, because `ort` reaches hardware through
ONNX Runtime execution providers while `candle` addresses devices directly:

| Backend | Accelerators | Cargo feature |
| --- | --- | --- |
| `ort` | `coreml`, `directml`, `cuda` | `ort-coreml`, `ort-directml`, `ort-cuda` |
| `tract` | — (CPU only) | — |
| `candle` | `metal`, `cuda` | `candle-metal`, `candle-cuda` |

For `ort` the linked ONNX Runtime build must also carry the provider. An **explicitly requested**
accelerator that cannot be used is a hard error, not a silent fall back to CPU — only `auto` is
allowed to settle for whatever it finds. The parity figures above cover only the `ort` backend's CPU
execution provider, validated against the golden fixtures under
[`crates/sceptre/tests/data/golden/`](crates/sceptre/tests/data/golden/): every accelerator is
**unvalidated** in CI, which has no GPU runner, so none of the numbers above should be assumed to
transfer. `candle` is a compatibility and deployment option, not a fast one — on a GPU it is still
several times slower than `ort` on the CPU (see [ADR 0031](adrs/0031-hand-written-candle-networks.md)).
Re-run the parity harness on your own target before relying on them.

`cargo bench` covers the internal hot paths; the head-to-head harness (`task python:benchmark`)
re-measures and writes the gitignored `benchmark-results/comparison.{json,md}`. Use
`--group labeled --limit 3 --repeats 1` for a fast inner-loop run, `--baseline <prior.json>` to see
per-image deltas, and `--assert` for the regression gate (see
[ADR 0021](adrs/0021-benchmark-methodology-and-gate.md)). `task python:publish` then distils that
report into the committed artifact and regenerates the table above; `task python:publish:check` is
the drift gate CI runs (see [ADR 0030](adrs/0030-published-benchmark-artifact-and-drift-gate.md)).
Parity fixtures live under
[`crates/sceptre/tests/data/golden/`](crates/sceptre/tests/data/golden/).

## How it works

Three stages behind one `Reader`, mirroring EasyOCR's latest pipeline:

1. **Detect** — CRAFT produces region/link heat-maps; thresholding, connected components, and
   min-area boxes become text lines (horizontal and rotated).
2. **Recognize** — each line is cropped, normalized, and run through a gen2 CRNN; CTC greedy decoding
   turns the logits into text and a confidence.
3. **Inference** — every model call goes through one backend seam: `ort` (native ONNX Runtime),
   `tract` (pure Rust), or `candle` (hand-written networks over the same ONNX weights, able to run
   on Metal or CUDA). `config`, `types`, and geometry stay backend-agnostic.

Design decisions live as [MADR records under `adrs/`](adrs/); conventions live in `.ai-rulez/`.

## Scope

Targets EasyOCR's current models — the **CRAFT** detector and all eight **gen2** (`*_g2`) recognizers
(English, Latin, Simplified Chinese, Japanese, Korean, Cyrillic, Telugu, Kannada).
Legacy gen1 models, DBNet, and beam-search decoding are out of scope
(see [`adrs/0002`](adrs/0002-scope-gen2-recognizers-and-craft.md) and
[`adrs/0026`](adrs/0026-extend-gen2-scope-telugu-kannada.md)). ONNX artifacts are first-party exports,
built from EasyOCR's weights and hosted on the [`xberg-io`](https://huggingface.co/xberg-io)
Hugging Face org (Apache-2.0; see [`adrs/0025`](adrs/0025-first-party-onnx-exports.md)).

## Feature flags

| Feature | Enables | Library | CLI |
| --- | --- | --- | --- |
| `ort` | Native ONNX Runtime backend (desktop / server), provisioning-agnostic | ✓ | — |
| `ort-bundled` | `ort` with a prebuilt runtime fetched at build time (zero-config) | ✓ | **default** |
| `ort-dynamic` | `ort` loading `libonnxruntime` at runtime via `ORT_DYLIB_PATH` (no build-time download) | ✓ | ✓ |
| `tract` | Pure-Rust ONNX backend (WASM / Android), selected at runtime with `--backend tract` | ✓ | ✓ |
| `download` | Runtime model download + cache from Hugging Face | ✓ | **default** |
| `mcp` | MCP (`rmcp`) server surface | ✓ | ✓ |
| `ort-coreml` / `ort-directml` / `ort-cuda` | Compile in the matching ONNX Runtime execution provider for `model.accelerator` | ✓ | ✓ |
| `candle` | Native-tensor backend needing no ONNX Runtime, selected at runtime with `--backend candle` (see ADR 0031) | ✓ | ✓ |
| `candle-metal` / `candle-cuda` | Compile in the matching GPU device for `model.accelerator` on the `candle` backend | ✓ | ✓ |
| `bench` | Exposes the crate's internal hot paths through a `bench` seam for criterion benchmarking (see ADR 0015) | ✓ | — |

The library ships `default = []`. The CLI ships `default = ["ort-bundled", "download"]`, so
`cargo install sceptre-cli` produces a self-contained binary; `--features ort-dynamic` overrides the
provisioning strategy without `--no-default-features` (see [Install](#install)).

Configuration (`OcrConfig`) is backend-agnostic. The CLI builds it from `OcrConfig::default()` plus
flags — there is no config-file loader and no `SCEPTRE_*` environment variable. The CLI does read one
env var, `EASYOCR_LOG` (sets `--log-level`), a name kept from the project's EasyOCR lineage.
`cache_dir`, `registry_owner`, `detector_path`, and `recognizer_path` are library-only fields with no
CLI flag.

## Development

```sh
task setup    # fetch deps, install git hooks
task check    # cargo fmt + clippy -D warnings + test + poly lint
task bench    # criterion microbenchmarks (--features bench)
```

Coding conventions are generated from `.ai-rulez/` into `CLAUDE.md` / `AGENTS.md` — edit the source,
then run `ai-rulez generate`.

## Documentation

Full guides, the configuration reference, backend and accelerator selection, and the benchmark
methodology live at **[docs.sceptre.xberg.io](https://docs.sceptre.xberg.io)**. The Rust API
reference is on [docs.rs](https://docs.rs/sceptre).

## Part of Xberg.io

- [Xberg](https://github.com/xberg-io/xberg) — the open-source content-intelligence engine: text, tables, and metadata from 101 formats (115 file extensions), with OCR, transcription, and code intelligence. MIT.
- [Xberg Pro](https://xberg.io) — a complete self-hosted content-intelligence backend in a single container. Commercial.
- [Xberg Enterprise](https://xberg.io) — the distributed, governed content-intelligence platform, scaled on Kubernetes with team governance and support. Commercial.
- [crawlberg](https://github.com/xberg-io/crawlberg) — web crawling and scraping with HTML→Markdown and headless-Chrome fallback.
- [html-to-markdown](https://github.com/xberg-io/html-to-markdown) — fast, lossless HTML→Markdown engine.
- [liter-llm](https://github.com/xberg-io/liter-llm) — universal LLM API client with native bindings for 14 languages and 165 providers.
- [tree-sitter-language-pack](https://github.com/xberg-io/tree-sitter-language-pack) — tree-sitter grammars and code-intelligence primitives.
- [alef](https://github.com/xberg-io/alef) — the polyglot binding generator that produces every per-language binding across the 5 polyglot repos.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and guidelines.

## License

[MIT License](LICENSE). Model weights are distributed under their own licenses — the gen2 EasyOCR
models and the first-party [`xberg-io/sceptre-*`](https://huggingface.co/xberg-io) ONNX exports derived
from them are Apache-2.0.
