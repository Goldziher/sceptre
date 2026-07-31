---
priority: high
---

# Inference Backend Seam

- All model execution goes through the `inference::ModelBackend` trait. Never call `ort`, `tract`, or `candle` APIs directly from the detection, recognition, or engine code.
- Backends are feature-gated and paired: `ort` (native) is the default; `tract` (pure-Rust) is the WASM/Android path; `candle` is the pure-Rust native-tensor fallback. A new native capability must keep a pure-Rust story.
- `config`, `types`, and the geometry code stay backend-agnostic — never `#[cfg(feature = "ort")]` a config or result type.
- Concurrency flows through a single `ConcurrencyConfig` budget: Rayon and the backend's intra-op threads draw from the same limit so nested parallelism cannot oversubscribe the CPU.
