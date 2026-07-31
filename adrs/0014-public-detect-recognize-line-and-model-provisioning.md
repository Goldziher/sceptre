---
status: accepted
date: 2026-07-31
deciders: Na'aman Hirschfeld
---

# Public detect / recognize-line methods and a model-provisioning surface

## Context and Problem Statement

The CLI ([ADR 0001](0001-cargo-workspace-rlib-core-thin-cli.md)) exposes `detect`,
`recognize`, and `models list`/`models download` subcommands, mirroring EasyOCR's
`Reader.detect()` / `Reader.recognize()` split and its model management. The library's
public surface after [ADR 0010](0010-ocr-engine-seam-and-encapsulation-lockdown.md) is
deliberately minimal: `Reader::{readtext, recognize}` run the *whole* pipeline, the
`OcrEngine` seam has only `recognize`, and the model registry/download layer is entirely
`pub(crate)`. So the CLI cannot detect without recognizing, cannot recognize a
pre-cropped line without running detection, and cannot enumerate or prefetch models
without reaching private code. We need to decide what public surface to add, and whether
it can be added without breaking the encapsulation lockdown or existing consumers.

## Decision Drivers

- Mirror EasyOCR's user-facing capabilities (detect-only, recognize a cropped line, manage
  models) so the CLI can offer the same commands.
- Preserve the `OcrEngine` seam: adding capability must not break existing `OcrEngine`
  implementors (e.g. `FallbackEngine`, custom engines) or force them to implement more.
- Keep internals (`detect`/`recognize`/`inference`/`models`) `pub(crate)`; only add a
  curated flat surface, per the crate's re-export convention.

## Considered Options

- **Widen `OcrEngine` with required `detect`/`recognize_line` methods.** Uniform, but a
  breaking change: every implementor must add both, and not every engine can detect.
- **Add `detect`/`recognize_line` as *defaulted* `OcrEngine` methods; override them in the
  built-in engine.** Non-breaking (defaults derive from `recognize`), while the default
  `SceptreEngine` provides efficient true implementations.
- **Expose model provisioning as `Reader` methods.** But listing/prefetching models needs
  no engine or loaded backend — only a config — so hanging it off `Reader` is awkward.
- **Expose model provisioning as free functions over `&OcrConfig`.** Matches the "config in,
  data out" shape and needs no `Reader`.

## Decision Outcome

Chosen: **defaulted `OcrEngine` methods plus free provisioning functions.**

- `OcrEngine` gains two methods with default bodies, so every existing implementor keeps
  compiling unchanged and `FallbackEngine` inherits correct fallback behavior for free (the
  defaults call `self.recognize`, which is already the fallback chain):
  - `detect(&self, image, options) -> Result<Vec<Quad>>` — default derives quads from
    `recognize`; `SceptreEngine` overrides to run detection only (no recognizer load).
  - `recognize_line(&self, image, options) -> Result<TextLine>` — default runs `recognize`
    and merges the lines; `SceptreEngine` overrides to run the recognizer on the whole
    image as one crop, skipping detection.
  `Reader::detect` and `Reader::recognize_line` delegate to the injected engine.
- Model provisioning is exposed as free functions plus two small owned types, re-exported at
  the crate root: `model_manifest(&OcrConfig) -> Result<Vec<ModelInfo>>` (filesystem-only
  cache inspection, no network) and `download_models(&OcrConfig) -> Result<Vec<ModelInfo>>`
  (fetches missing artifacts; requires the `download` feature, else the underlying fetch
  returns an `OcrError::Model`). `ModelInfo`/`ModelRole` describe each artifact's name,
  effective repo id, role, and cache status.

The `detect`/`recognize`/`inference`/`models` submodules stay `pub(crate)`; only the curated
items above are lifted to the public API.

### Consequences

- Good: no existing `OcrEngine` implementor breaks; custom engines get detect/recognize-line
  for free and may override for efficiency.
- Good: `models list`/`download` work from a config alone, with cache inspection available
  even when the `download` feature is off (`cache_path` is no longer feature-gated).
- Neutral: the default `recognize_line`/`detect` derivations run the full pipeline, so they
  are only as cheap as `recognize` for non-built-in engines; the built-in engine avoids that.
- Bad: two more methods on the public `OcrEngine` seam widen the trait's contract slightly,
  though both are defaulted.
