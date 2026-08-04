---
status: accepted
date: 2026-08-04
deciders: Na'aman Hirschfeld
---

# Host-supplied model artifacts and sequential browser-WASM execution

## Context and Problem Statement

The default model provider resolves Hugging Face cache paths. Browser WASM has no conventional
filesystem, and mobile embedding hosts commonly manage app assets or downloads themselves. The
existing path-only provider forces those hosts to emulate files even though every inference backend
already parses ONNX bytes. Sceptre also creates a private Rayon pool unconditionally, while ordinary
browser WASM cannot assume shared-memory threads.

## Decision Drivers

- Let WASM and mobile hosts fetch, cache, and verify model assets outside Rust.
- Keep one model-execution pipeline for path-backed and memory-backed artifacts.
- Fail closed when host-supplied bytes do not match their registry digest.
- Avoid requiring `SharedArrayBuffer`, COOP/COEP headers, or a threaded WASM runtime.
- Initialize expensive inference plans once and reuse them across calls.

## Considered Options

- Preserve a path-only provider and write host bytes to temporary files.
- Add a second byte-only engine API alongside the path provider.
- Unify provider output as a path-or-bytes artifact and make browser WASM sequential.

## Decision Outcome

Chosen: **one `ModelProvider` returning `ModelArtifact::{Path, Bytes}`.**

- The default provider returns cached paths. `VerifiedModelProvider` accepts public
  `ModelDescriptor`/byte pairs, verifies each SHA-256 pin, and returns memory artifacts.
- Mobile hosts with real asset files can configure paired `model.detector_path` and
  `model.recognizer_path` values; the default provider uses them before cache/download resolution.
- `model_descriptors` exposes name, role, repository, revision, filename, and digest without
  filesystem or network access, so embedding hosts can securely provision assets.
- The engine reads only path artifacts; byte artifacts pass directly to the inference backend.
- Fallible once-cells serialize the first detector and recognizer initialization. `warm_up` and
  `build_warmed` expose eager initialization, and the engine releases its provider after both plans
  are cached so verified source buffers need not remain retained by Sceptre.
- Native targets keep their private Rayon pool. `wasm32` uses the same reader API through a
  sequential execution adapter and sequential crop/CTC loops. Applications should run OCR in a Web
  Worker when main-thread responsiveness matters.
- tract is aligned to 0.23.4 so embedders do not compile a second 0.22 inference stack.

### Consequences

- Good: browser and mobile hosts can supply verified models without filesystem or `hf-hub` access.
- Good: path and byte artifacts share backend initialization, caching, and OCR behavior.
- Good: baseline browser WASM compiles without Rayon or native threading assumptions.
- Bad: the `ModelProvider` return type changes; existing custom path providers must wrap paths in
  `ModelArtifact::Path`.
- Neutral: sequential WASM avoids unsafe deployment assumptions but does not prevent CPU-heavy OCR
  from blocking its current worker.
