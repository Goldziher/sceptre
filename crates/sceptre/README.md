# sceptre

**EasyOCR's accuracy. Rust's speed and footprint.**

A from-scratch Rust reimplementation of [EasyOCR](https://github.com/JaidedAI/EasyOCR)'s OCR
pipeline — **CRAFT** text detection then **gen2 CRNN** recognition with CTC decoding, over ONNX.
It matches EasyOCR's output across six scripts (English, Latin, Chinese-simplified, Japanese,
Korean, Cyrillic) with no Python runtime, and runs on native ONNX Runtime (`ort`) or a pure-Rust
backend (`tract`) behind one seam.

Models download from Hugging Face on first use, cache locally, and are sha256-verified — every run
after that is fully offline.

## Usage

```rust
use sceptre::{Reader, ReadOptions};

let reader = Reader::builder().build()?;
for line in reader.readtext("receipt.png".as_ref(), &ReadOptions::default())?.lines {
    println!("{} ({:.2})", line.text, line.confidence);
}
# Ok::<(), sceptre::OcrError>(())
```

For the CLI, install the [`sceptre-cli`](https://crates.io/crates/sceptre-cli) crate.

Full documentation, benchmarks, and design notes live at the
[project repository](https://github.com/Goldziher/sceptre).

## License

MIT. Model weights are distributed by third parties under their own licenses (Apache-2.0).
