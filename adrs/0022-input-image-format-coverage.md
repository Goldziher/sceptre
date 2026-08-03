---
status: accepted
date: 2026-08-03
deciders: Na'aman Hirschfeld
---

# Input image format coverage: pure-Rust decoders only

## Context and Problem Statement

sceptre decodes input images through the `image` crate (`Image::from_bytes` →
`image::load_from_memory`), which only decodes formats whose Cargo features are enabled.
The crate was built with `["png", "jpeg"]`, so every other container — BMP, TIFF, WebP,
GIF, NetPBM, and the JPEG 2000 / HEIF / AVIF families — failed to decode. The
sceptre-vs-EasyOCR benchmark surfaced this as a capability gap: EasyOCR (via Pillow/OpenCV)
reads those formats, sceptre did not. We need to decide how much format coverage to add and
where to draw the line.

## Decision Drivers

- Match EasyOCR's practical input coverage on common raster formats.
- Preserve the pure-Rust story: the `tract` backend targets WASM and Android (ADR 0004,
  backend-seam), and image decoding is backend-agnostic — it must not pull a C toolchain
  dependency that breaks those targets.
- Keep binary size and the dependency/audit surface modest.

## Considered Options

- Keep `png` + `jpeg` only.
- Add the pure-Rust decoders the `image` crate ships (BMP, GIF, TIFF, WebP, NetPBM, …).
- Additionally add the HEIF/AVIF/JPEG-2000 families via C libraries (`libheif`, `dav1d`,
  `openjpeg`) behind an optional feature.

## Decision Outcome

Enable the **pure-Rust** decoders `bmp`, `gif`, `tiff`, `webp`, and `pnm` alongside
`png`/`jpeg`. These cover the common raster containers in the test corpus, add no C
dependency, and keep the `tract` WASM/Android path intact.

**HEIF, AVIF, and the JPEG 2000 family (`jp2`/`j2k`/`jpx`/`jpm`/`mj2`/`j2c`) stay
unsupported.** Their only mature decoders are C libraries that would break the pure-Rust
build and inflate the audit surface, so they remain documented capability gaps in the
benchmark (the `capability` corpus group). If demand warrants, they can be added later
behind an optional, native-only feature that the WASM/Android builds omit — a reversible
extension of this decision, not a reversal.

### Consequences

- `Image::from_bytes` now decodes BMP/GIF/TIFF/WebP/NetPBM; the benchmark's BMP probe moves
  from a capability gap to a scored entry, and three gaps (heif/avif/jp2) remain, clearly
  attributed to the pure-Rust constraint.
- No `#[cfg(feature = ...)]` leaks into decode code — enabling a decoder is a Cargo feature
  flip on the `image` dependency (root `Cargo.toml`), consistent with `config`/`types`
  staying backend-agnostic.
