# Golden fixtures

Each `<image>.json` here is the expected parity result for the matching image
under `tests/data/images/<image>.{png,jpg}`, consumed by the tier-2 parity tests
in `crates/sceptre/tests/tier2_golden.rs`.

## Dual golden scheme

Every fixture carries two goldens for the same image (see ADR 0016):

- **`easyocr`** — the authoritative reference, produced by upstream Python
  EasyOCR (torch). Compared *fuzzily*: bag-of-words F1 over the joined text plus a
  per-line box-IoU (>= 0.5) against detected quads. This tracks parity with the
  reference implementation while tolerating minor tokenization differences.
- **`sceptre`** — a snapshot of this crate's own output. Compared for *exact*
  text equality, so unintended regressions in our pipeline fail loudly.

## Format

```json
{
  "placeholder": false,
  "easyocr": {
    "lines": [
      { "text": "first line", "quad": [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] }
    ]
  },
  "sceptre": {
    "lines": [
      { "text": "first line", "quad": [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] }
    ]
  }
}
```

`quad` is four `[x, y]` corners, clockwise from top-left. While `placeholder` is
`true`, the fixture has not been regenerated from real runs and the parity
assertions are skipped (the test only verifies the pipeline runs). The committed
fixtures are generated from real EasyOCR and sceptre runs (`placeholder: false`).

## Model gating

The real-model tests gate on the library's `model_manifest`, which resolves CRAFT
+ gen2 ONNX models from the shared Hugging Face hub cache (`HF_HUB_CACHE` /
`HF_HOME` / `~/.cache/huggingface/hub`). When the models are absent the tests
**skip** (and pass). Set `SCEPTRE_REQUIRE_MODELS=1` to
force them to run and fail if the models cannot be resolved — this is what CI uses
so a broken model cache is visible.

## Regenerating

The fixtures are regenerated in two independent steps, then committed. Do this
whenever a new image is added under `tests/data/images/`, the reference EasyOCR
version changes, or the sceptre pipeline output intentionally changes.

1. Populate the corpus and models (opt-in, heavy — deferred by default):
   - `git submodule update --init test_documents && git -C test_documents lfs pull`
   - download the CRAFT + gen2 models into the Hugging Face cache.
2. EasyOCR reference (`easyocr` side), via the Python fallback tool:
   - `uv sync --group export`
   - `task py:golden` (runs `python -m sceptre_rs_tools.golden`)
3. Sceptre snapshot (`sceptre` side), via the Rust tool:
   - `cargo run -p sceptre-tools --features ort -- snapshot`

Each generator writes only its own side of every `<image>.json`, so the two
steps can run in either order.
