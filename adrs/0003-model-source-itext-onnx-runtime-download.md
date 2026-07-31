---
status: accepted
date: 2026-07-31
deciders: Na'aman Hirschfeld
---

# Model source: iText ONNX, runtime download

## Context and Problem Statement

EasyOCR ships PyTorch `.pth` weights and provides an ONNX export script only for
the CRAFT detector, not the recognizers. We need ONNX artifacts for CRAFT and the
six gen2 recognizers, and a way to deliver them to users.

## Decision Drivers

- Avoid maintaining a fragile PyTorch export toolchain if we can.
- Permissive license for redistribution.
- Dynamic-width recognizers (no letterbox distortion).
- Keep large binaries out of git.

## Considered Options

- Author our own `.pth → ONNX` export (extending EasyOCR's script to recognizers).
- Use the `itextresearch/itext-EasyOCR-*` ONNX exports on Hugging Face.
- Vendor exported ONNX into the repository (git-lfs).

## Decision Outcome

Chosen option: **use the `itextresearch` ONNX exports, downloaded at runtime**.
They cover CRAFT + all six gen2 recognizers, are Apache-2.0, full-precision, and
dynamic-width. Models are fetched via `hf-hub`, cached under
`~/.cache/sceptre`, and sha256-verified. A Python export script is kept only
as a fallback (e.g. to re-export at a different opset if a pure-Rust backend
rejects a model).

### Consequences

- Good: no export toolchain on the critical path; small repo.
- Good: one consistent, permissively-licensed source for every model.
- Bad: first run needs network access; we depend on an external host (mitigated
  by caching + checksums, and the fallback export path).
