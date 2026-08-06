<!-- markdownlint-disable MD033 MD041 -->
<div align="center">

<img src="https://raw.githubusercontent.com/Goldziher/sceptre/main/docs/assets/banner.svg" alt="sceptre" width="820">

**EasyOCR's accuracy. Rust's speed and footprint.**

sceptre is a from-scratch Rust reimplementation of [EasyOCR](https://github.com/JaidedAI/EasyOCR)'s
OCR pipeline — **CRAFT** text detection then **gen2 CRNN** recognition with CTC decoding, over ONNX.
It agrees with EasyOCR's output across the scripts it supports, runs **~2.8× faster on a fraction of
the memory** (and a cold one-shot run is ~4.4× faster than EasyOCR warm), and ships as one
self-contained binary with **no Python runtime**. Use it as a **library**, a **CLI**, or an **MCP
server**.

8 scripts · CRAFT + gen2 CRNN · ONNX Runtime **or** pure-Rust · library · CLI · MCP · offline-first

[![crates.io](https://img.shields.io/crates/v/sceptre?color=f5b301&style=flat-square)](https://crates.io/crates/sceptre)
[![CI](https://img.shields.io/github/actions/workflow/status/Goldziher/sceptre/ci.yaml?style=flat-square&color=f5b301)](https://github.com/Goldziher/sceptre/actions/workflows/ci.yaml)
[![docs.rs](https://img.shields.io/docsrs/sceptre?style=flat-square&color=f5b301)](https://docs.rs/sceptre)
[![Documentation](https://img.shields.io/badge/docs-goldziher.github.io%2Fsceptre-f5b301?style=flat-square)](https://goldziher.github.io/sceptre)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

[Documentation](https://goldziher.github.io/sceptre) · [Install](#install) · [Quickstart](#quickstart) · [Why sceptre](#why-sceptre) · [Benchmarks](#benchmarks) · [How it works](#how-it-works) · [Contributing](CONTRIBUTING.md)

</div>

---

## Why sceptre

EasyOCR is excellent and accurate — but it's a PyTorch stack: a Python interpreter, a multi-gigabyte
runtime, and a heavy process to keep warm. sceptre keeps the accuracy and drops all of that.

| | What you get | Why it matters |
|---|---|---|
| **Parity accuracy** | Validated against real EasyOCR output across the gen2 scripts — English, Latin, Chinese (simplified), Japanese, Korean, Cyrillic, plus Telugu and Kannada — agreeing on text (word/char-F1) and boxes (IoU), on the `ort` backend's CPU execution provider. | A faithful reimplementation, held to a per-image word-F1 and box-IoU floor rather than character-for-character equality. |
| **Substantially faster** | ~2.8× higher throughput than EasyOCR warm on the same corpus — and even a cold, one-shot CLI run (~4.4×) beats EasyOCR's warm, already-loaded reader. | More pages per second, less waiting, cheaper batch jobs. |
| **A fraction of the memory** | Peak RSS around **3× lower** than the Python + torch process, measured like-for-like (both whole-process peaks). | Runs where EasyOCR won't — small containers, edge boxes, many workers. |
| **One binary, no Python** | A single self-contained executable. Models download once, cache locally, and run **offline** thereafter. | `cargo install` and go — nothing to `pip install`, no interpreter to ship. |
| **Three surfaces** | The same engine as a Rust **library**, a **CLI** (`sceptre`), and an **MCP server** for agents. | Drop it into a service, a shell pipeline, or an AI tool without re-plumbing. |
| **Native or pure-Rust** | ONNX Runtime (`ort`) for native speed, or a pure-Rust backend (`tract`) for WASM / Android — behind one seam. | Portability when you need it, native performance when you don't. |

---

## Install

```sh
cargo install sceptre-cli
```

That default build is self-contained: it links a prebuilt ONNX Runtime fetched at build time
(`ort-bundled`) and enables model download (`download`), so the binary needs no `ORT_DYLIB_PATH` and
no separately installed `libonnxruntime`.

The models — CRAFT plus the gen2 recognizers — are fetched from Hugging Face on first use, cached
under the standard HF cache, and sha256-verified as they download. Every run after that reads the
cached models with no network.

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

## Quickstart

**CLI** — one image or many in a single warm process:

```sh
sceptre run receipt.png --lang english --format json
sceptre run page1.png page2.png page3.png            # batch: models load once
sceptre run sign.jpg --lang english --lang korean    # multi-language
sceptre run page.png --lang english --timings        # per-stage load/detect/recognize breakdown
sceptre run huge-scan.png --canvas-size 1600         # trade some accuracy for lower peak memory
```

Decodes PNG, JPEG, BMP, GIF, TIFF, WebP, and NetPBM (all pure-Rust; HEIF/AVIF/JPEG-2000
are out of scope — see [`adrs/0022`](adrs/0022-input-image-format-coverage.md)).

**Library:**

```rust
use sceptre::{Reader, ReadOptions};

let reader = Reader::builder().build()?;
for line in reader.readtext("receipt.png".as_ref(), &ReadOptions::default())?.lines {
    println!("{} ({:.2})", line.text, line.confidence);
}
# Ok::<(), sceptre::OcrError>(())
```

The library ships `default = []`, so enable a backend and model download:
`sceptre = { version = "0.4", features = ["ort-bundled", "download"] }` (see [Feature flags](#feature-flags)).

WASM and mobile hosts can avoid filesystem and network assumptions by calling `model_descriptors`,
fetching the selected artifacts, constructing a `VerifiedModelProvider`, and passing it to
`Reader::builder().model_provider(...).build_warmed()`.

Android and iOS hosts that package ONNX assets as real files may instead set both
`model.detector_path` and `model.recognizer_path`; the default provider then bypasses Hugging Face.

**MCP server** — expose a `readtext` tool to an agent. The `mcp` subcommand is behind the `mcp`
cargo feature, which is off by default:

```sh
cargo install sceptre-cli --features mcp
sceptre mcp --lang english
```

## Benchmarks

Measured against upstream EasyOCR over a 43-entry mixed corpus — 40 measured (documents, tables,
rotated scans, scene text, receipts, across five of the eight supported recognizer groups: English,
Latin, Chinese-simplified, Japanese, Korean) plus 3 capability-format gaps — on CPU. Both engines are
measured **identically** — each a fresh subprocess per language group under `/usr/bin/time`, at its
native multi-threaded default, loading its model/reader once and processing every image — so peak RSS
is a like-for-like whole-process figure (EasyOCR's legitimately includes the torch runtime). Numbers
vary with hardware and load; the **ratios** are the point.

| Engine | Throughput (img/s) | Peak RSS | Mean CER | Mean token-F1 |
|---|---|---|---|---|
| EasyOCR (warm/batch) | 0.14 | 22.6 GB | 0.554 | 0.348 |
| **sceptre** (warm/batch) | **0.39** (~2.8×) | **6.6 GB** (~3× lower) | 0.568 | **0.356** |
| sceptre (cold CLI run) | 0.60 (~4.4×) | 6.6 GB | 0.568 | 0.356 |

CER and token-F1 are at parity (sceptre is marginally ahead on token-F1); the win is speed and memory.
Even a cold, one-shot CLI run — which pays model load every invocation — is ~4.4× faster than
EasyOCR's already-warm reader.

**Runtime scope.** Every figure above — speed, memory, and the parity claim — was produced on the
`ort` backend running the **CPU execution provider**, which is what `model.accelerator` defaults to.
The same ONNX graph produces different numeric output on a different provider, so an accelerated run
is a different measurement, not a faster one. Select a provider with `--accelerator`
(`cpu` | `auto` | `coreml` | `directml` | `cuda`) or `model.accelerator`; the matching cargo feature
(`ort-coreml`, `ort-directml`, `ort-cuda`) must also be compiled in, and the ONNX Runtime build has
to carry the provider. An **explicitly requested** provider that cannot register is a hard error, not
a silent fall back to CPU — only `auto` is allowed to settle for whatever it finds. The parity
figures above cover only the `ort` backend's CPU execution provider, validated against the golden
fixtures under [`crates/sceptre/tests/data/golden/`](crates/sceptre/tests/data/golden/): CoreML,
DirectML and CUDA are all **unvalidated**, so none of the numbers above should be assumed to transfer.
Re-run the parity harness on your own target before relying on them.

`cargo bench` covers the internal hot paths; the head-to-head harness (`task python:benchmark`)
reproduces the table above and writes `benchmark-results/comparison.{json,md}`. Use
`--group labeled --limit 3 --repeats 1` for a fast inner-loop run, `--baseline <prior.json>` to see
per-image deltas, and `--assert` for the regression gate (see
[ADR 0021](adrs/0021-benchmark-methodology-and-gate.md)). Parity fixtures live under
[`crates/sceptre/tests/data/golden/`](crates/sceptre/tests/data/golden/).

## How it works

Three stages behind one `Reader`, mirroring EasyOCR's latest pipeline:

1. **Detect** — CRAFT produces region/link heat-maps; thresholding, connected components, and
   min-area boxes become text lines (horizontal and rotated).
2. **Recognize** — each line is cropped, normalized, and run through a gen2 CRNN; CTC greedy decoding
   turns the logits into text and a confidence.
3. **Inference** — every model call goes through one backend seam: `ort` (native ONNX Runtime) or
   `tract` (pure Rust). `config`, `types`, and geometry stay backend-agnostic.

Design decisions live as [MADR records under `adrs/`](adrs/); conventions live in `.ai-rulez/`.

## Scope

Targets EasyOCR's current models — the **CRAFT** detector and all eight **gen2** (`*_g2`) recognizers
(English, Latin, Simplified Chinese, Japanese, Korean, Cyrillic, Telugu, Kannada).
Legacy gen1 models, DBNet, and beam-search decoding are out of scope
(see [`adrs/0002`](adrs/0002-scope-gen2-recognizers-and-craft.md) and
[`adrs/0026`](adrs/0026-extend-gen2-scope-telugu-kannada.md)). ONNX artifacts are first-party exports,
built from EasyOCR's weights and hosted on the [`sceptre-ocr`](https://huggingface.co/sceptre-ocr)
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
| `candle` | Reserved for a future pure-Rust native-tensor backend (see ADR 0009) | ✓ | — |
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

## License

MIT. Model weights are distributed by third parties under their own licenses (the gen2 EasyOCR models
and the iText ONNX exports are Apache-2.0).
