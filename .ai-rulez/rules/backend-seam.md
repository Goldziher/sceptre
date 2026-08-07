---
priority: high
---

# Inference Backend Seam

- All model execution goes through the `inference::ModelBackend` trait. Never call `ort`, `tract`, or `candle` APIs directly from the detection, recognition, or engine code.
- Backends are feature-gated: `ort` (native ONNX Runtime) is the default; `tract` (pure-Rust ONNX) is the WASM/Android path; `candle` runs hand-written networks over the same ONNX weights and is the GPU / no-ONNX-Runtime path. A new native capability must keep a pure-Rust story.
- `ort` and `tract` interpret the ONNX graph; `candle` does not, so it takes a `NetworkKind` on `BackendOptions` and validates the graph against it. A change to either exported architecture — not merely its weights — breaks `candle` alone and must be reflected in `inference/candle/`.
- Accelerator support is a per-backend table (`Backend::hardware_accelerators`), never a special case for one backend. Config validates the vocabulary; load validates availability; an explicit request never silently degrades to the CPU.
- `config`, `types`, and the geometry code stay backend-agnostic — never `#[cfg(feature = "ort")]` a config or result type.
- Concurrency flows through a single `ConcurrencyConfig` budget: Rayon and the backend's intra-op threads draw from the same limit so nested parallelism cannot oversubscribe the CPU.
