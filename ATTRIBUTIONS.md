# Attributions

sceptre is a from-scratch Rust reimplementation, not a fork: no upstream source is vendored into this
repository. It nonetheless stands on prior work — an algorithm it reproduces, network weights it
re-exports, and a research result it depends on. This file records those obligations.

sceptre itself is MIT licensed (see [`LICENSE`](LICENSE)).

## EasyOCR

- **Author**: JaidedAI
- **License**: Apache-2.0
- **Repository**: <https://github.com/JaidedAI/EasyOCR>

sceptre reimplements EasyOCR's current pipeline — CRAFT text detection followed by gen2 (`*_g2`)
CRNN recognition with CTC decoding. The stage behaviour, preprocessing constants, post-processing
geometry, box grouping, charsets, and default tunables are reproduced from EasyOCR's implementation
(`imgproc.py`, `craft_utils.py`, `utils.py`, `recognition.py`) so that output matches. The network
weights sceptre runs are EasyOCR's, re-exported to ONNX (see below).

Correctness is held to EasyOCR's own output: the parity harness compares sceptre against golden
fixtures generated from upstream EasyOCR
(see [`adrs/0016-parity-harness-and-test-corpus.md`](adrs/0016-parity-harness-and-test-corpus.md)).

## Model artifacts

- **Source**: the first-party `sceptre-ocr/<model>` repos on Hugging Face —
  <https://huggingface.co/sceptre-ocr>
- **License**: Apache-2.0

The CRAFT detector and the eight gen2 recognizers are exported from EasyOCR's PyTorch weights by the
`sceptre_rs_tools` export pipeline in this repository and published under the `sceptre-ocr` org, one
repo per model. They are downloaded at runtime, cached in the shared Hugging Face hub cache, and
sha256-verified against pins in `crates/sceptre/src/models/registry.rs`. Because they are derived
from EasyOCR's weights, the EasyOCR attribution above travels with them. See
[`adrs/0025-first-party-onnx-exports.md`](adrs/0025-first-party-onnx-exports.md).

Earlier releases sourced the same models from the third-party `itextresearch/itext-EasyOCR-*` ONNX
repos on Hugging Face (Apache-2.0), which remain the lineage of the exports and are credited on each
`sceptre-ocr` model card. That decision is recorded, and superseded, in
[`adrs/0003-model-source-itext-onnx-runtime-download.md`](adrs/0003-model-source-itext-onnx-runtime-download.md).

## CRAFT

The detection stage implements CRAFT (Character Region Awareness For Text detection):

> Youngmin Baek, Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee. "Character Region Awareness for
> Text Detection." CVPR 2019. <https://arxiv.org/abs/1904.01941>

The `craft_mlt_25k` weights sceptre serves reach it through EasyOCR, which distributes them for its
own detection stage.

## Test fixtures

The test images and their ground-truth transcripts come from the shared `test_documents`
submodule and carry their own, per-dataset licenses and citations — NDL PDM OCR, CORD,
TextOCR, DocLayNet, and libheif-rs fixtures among them. Those obligations are recorded next
to the files they cover and are not duplicated here:

- [`test_documents/ATTRIBUTIONS.md`](test_documents/ATTRIBUTIONS.md)
- [`test_documents/LICENSES.md`](test_documents/LICENSES.md)

## Dependencies

Rust dependencies are listed in `Cargo.toml` / `Cargo.lock` and their licenses are gated in CI by
`cargo-deny` (see [`deny.toml`](deny.toml)). Each retains its own license; none is vendored here.
