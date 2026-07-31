# Golden fixtures

Each `<image>.json` file here is the expected parity result for the matching
image under `tests/data/images/<image>.{png,jpg}`, consumed by the
`#[ignore]`d tests in `crates/easyocr/tests/tier2_golden.rs`.

## Format

A single-line JSON object with the expected recognized text per line, in
reading order:

```json
{"lines": ["first recognized line", "second recognized line"]}
```

This intentionally has no nested structure (no quads yet) so the consuming
test can parse it without a JSON dependency. Once detection produces real
quads, extend the format with per-line boxes and update the parser in
`tier2_golden.rs` alongside it.

## Regenerating

`english.json` currently holds a placeholder value — it has not yet been
generated from a real EasyOCR run. These fixtures are regenerated from the
upstream Python EasyOCR reference by the opt-in `task py:golden` (see
`Taskfile.yaml` and `scripts/`), which runs EasyOCR over `tests/data/images/*`
and writes the recognized text back into this directory. Run it, then commit
the resulting `*.json` files, whenever:

- a new image is added under `tests/data/images/`, or
- the reference EasyOCR version used for parity changes.
