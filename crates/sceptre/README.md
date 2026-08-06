# sceptre

**EasyOCR's accuracy. Rust's speed and footprint.**

A from-scratch Rust reimplementation of [EasyOCR](https://github.com/JaidedAI/EasyOCR)'s OCR
pipeline — **CRAFT** text detection then **gen2 CRNN** recognition with CTC decoding, over ONNX.
It matches EasyOCR's output across eight scripts (English, Latin, Chinese-simplified, Japanese,
Korean, Cyrillic, Telugu, Kannada) on the CPU execution provider, with no Python runtime, and runs on
native ONNX Runtime (`ort`) or a pure-Rust backend (`tract`) behind one seam.

Models download from Hugging Face on first use, cache locally, and are sha256-verified on download —
every run after that reads the cache with no network.

Embedding hosts can instead supply registry-described ONNX bytes through `VerifiedModelProvider`;
`build_warmed` verifies and initializes the detector and selected recognizer once.

## Usage

```rust
use sceptre::{Reader, ReadOptions};

let reader = Reader::builder().build()?;
for line in reader.readtext("receipt.png".as_ref(), &ReadOptions::default())?.lines {
    println!("{} ({:.2})", line.text, line.confidence);
}
# Ok::<(), sceptre::OcrError>(())
```

The crate ships `default = []`; enable a backend and model download:
`sceptre = { version = "0.3", features = ["ort-bundled", "download"] }`. `ort-bundled` fetches a
prebuilt ONNX Runtime at build time; on targets `ort` publishes no prebuilt for — Intel macOS,
musl/Alpine, armv7, riscv64, FreeBSD, i686, s390x, powerpc64le — use `ort-dynamic` (bring your own
`libonnxruntime`) or `tract` (pure Rust) instead.

For the CLI, install the [`sceptre-cli`](https://crates.io/crates/sceptre-cli) crate; its default
build is self-contained (`ort-bundled` + `download`).

Full documentation, benchmarks, and design notes live at the
[project repository](https://github.com/Goldziher/sceptre).

## License

MIT. Model weights are distributed by third parties under their own licenses (Apache-2.0).
