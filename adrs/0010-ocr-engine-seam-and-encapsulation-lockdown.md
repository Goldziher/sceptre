---
status: accepted
date: 2026-07-31
deciders: Na'aman Hirschfeld
---

# Public `OcrEngine` seam, internal DTO boundaries, and encapsulation lockdown

## Context and Problem Statement

The library must be an *extensible OCR backend*, not a fixed EasyOCR clone: callers
should be able to plug in an alternative engine, compose fallbacks, and switch models,
while the crate keeps its internals private so the public surface stays small and stable.
This ADR records the public API shape and the encapsulation model chosen for that goal.

## Decision Drivers

- A single, obvious extension point for callers who want a different engine or fallbacks.
- Decoupled pipeline stages that can be reimplemented without churning the public API.
- Minimal, curated public surface — internals must not leak and become de-facto API.
- Backend-agnostic types: no runtime (`ort`/`tract`/`candle`) or `ndarray`/`image` types
  in public signatures or stage-boundary data.

## Considered Options

- **Concrete `Reader` only** — expose the EasyOCR pipeline directly, no engine trait.
- **Public `OcrEngine` trait seam** — `Reader` wraps `Arc<dyn OcrEngine>`; the EasyOCR
  pipeline is one internal implementation; stages sit behind internal traits + DTOs.
- **Fully generic pipeline** — expose detector/recognizer traits publicly too.

## Decision Outcome

Chosen option: **the public `OcrEngine` trait seam with internal stage traits and DTOs, and
a crate-wide encapsulation lockdown.**

- **Public seam.** `OcrEngine::recognize(&Image, &ReadOptions) -> Result<OcrResult>` is the
  one extension point. `FallbackEngine` is a public combinator that tries engines in order,
  advancing past an `Err` or an empty result. `Reader` holds `Arc<dyn OcrEngine>`;
  `ReaderBuilder::engine(...)` injects a custom engine (default: the internal
  `SceptreEngine`). `Image` is a public owned RGB8 input DTO, decoupled from the `image`
  crate.
- **Internal stage boundaries.** `SceptreEngine` composes internal `pub(crate)` traits
  `TextDetector` (→ `DetectedRegions`) and `TextRecognizer` (`&[RegionCrop]` →
  `Vec<RecognizedText>`). These DTOs carry only primitives/`String` (corners as raw
  `[[f32; 2]; 4]`, not the public `Quad`), so a stage can change independently and the
  engine maps to public types only at its boundary.
- **Encapsulation lockdown.** Every module is `pub(crate)` except the feature-gated `mcp`
  module; the entire public API is the curated re-export list in `lib.rs`
  (`Reader`/`ReaderBuilder`, `OcrEngine`/`FallbackEngine`, `Image`, config + result DTOs,
  errors, and the `ModelProvider`/`ProgressSink` builder seams). A black-box test tier that
  may only `use sceptre::…` enforces this by construction.

Detector and recognizer stay *internal* traits (not public) so we are not committed to their
shape as API; if a use case to swap a detector/recognizer independently emerges, it can be
promoted deliberately in a later ADR.

### Consequences

- Good: one clear seam for custom engines/fallbacks; the EasyOCR pipeline is swappable
  without touching the public API.
- Good: stages are decoupled by owned DTOs, so detection/recognition can be reimplemented
  (Pass 4/5) behind stable boundaries.
- Good: the public surface equals the re-export list — internals cannot be depended on.
- Bad: an extra DTO-mapping step at the engine boundary and two internal trait layers.
- Neutral: internal stage traits are `pub(crate)`; promoting them to public API later is a
  one-way door that requires its own ADR.
