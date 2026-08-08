---
status: accepted
date: 2026-08-03
deciders: Na'aman Hirschfeld
---

# First-party ONNX exports on the `sceptre-ocr` Hugging Face org

## Context and Problem Statement

[ADR 0003](0003-model-source-itext-onnx-runtime-download.md) sourced CRAFT and the gen2
recognizers from the third-party `itextresearch/itext-EasyOCR-*` Hugging Face repos, downloaded
at runtime, and deliberately kept a `.pth → ONNX` export path only as a fallback. Relying on a
third-party host for the entire model supply chain means our availability, licensing story, and
model provenance all depend on repos we do not control. We now want sceptre to own its models
end-to-end: export them ourselves and host them under an org we manage.

## Decision Drivers

- Own the full model supply chain — availability, provenance, and licensing under our control.
- Reproducible exports we can re-run (e.g. to add a language or re-export at a new opset).
- Preserve the runtime I/O contract exactly so the `ort` and `tract` backends keep working.
- Keep parity with EasyOCR (the golden harness of ADR 0016 must still pass).

## Considered Options

- **Keep sourcing from `itextresearch`** (status quo, ADR 0003) — zero work, no ownership.
- **Mirror the itextresearch ONNX bytes** into our org — ownership of hosting, but the exports
  still originate from a third party and we cannot regenerate them.
- **Build a first-party `.pth → ONNX` export pipeline** and host the results ourselves.

## Decision Outcome

Chosen option: **build a first-party `.pth → ONNX` export pipeline and host the exports under the
`sceptre-ocr` Hugging Face org**, superseding ADR 0003's third-party source.

- The export pipeline is implemented in the Python `sceptre_rs_tools` package (torch + easyocr +
  onnx, in the opt-in `export` dependency group). It loads each EasyOCR network, runs
  `torch.onnx.export`, and emits both the `.onnx` and its charset. This **promotes the export path
  from ADR 0003's "fallback" to the primary source of truth**, and it **inverts ADR 0008's
  "Rust/candle preferred"** ordering for export: the reference conversion is torch-based, and
  [ADR 0009](0009-candle-evaluation-ort-primary.md) already found candle cannot load these models,
  so the Rust `tools/` export path stays deferred. ADR 0008 carries a status note recording this.
- Exports must satisfy the runtime contract verbatim (single f32 input, single f32 output, no
  baked-in normalization, no softmax on the recognizer, class 0 = CTC blank, dynamic batch+width
  for recognizers / dynamic H+W for CRAFT) and must survive tract's `.into_optimized()`. The export
  tool validates every artifact through both `ort` and `tract` before it is published.
- Models are hosted as **per-model repos under `sceptre-ocr/<model>`** (e.g.
  `sceptre-ocr/english_g2`, `sceptre-ocr/craft_mlt_25k`), each with an Apache-2.0 license and a
  README crediting EasyOCR (Apache-2.0, JaidedAI) and the itextresearch ONNX lineage.
- The registry (`crates/sceptre/src/models/registry.rs`) is repointed: every entry's `hf_repo`,
  `file`, and pinned `sha256` change to the first-party artifacts. Because the repo names change
  from `itext-EasyOCR-<model>` to `sceptre-ocr/<model>`, this goes **beyond ADR 0011's owner-only
  override** — 0011's own "revisit if a host with a different repo-naming scheme is chosen" clause.
  The baked `hf_repo` ids are edited directly; the `registry_owner` override remains for downstream
  re-pointing/mirroring.

### Consequences

- Good: sceptre owns every model it ships — hosting, provenance, and a re-runnable export.
- Good: adding a language or re-exporting at a new opset is now a first-party operation.
- Bad: we maintain a torch export toolchain and re-validate parity on every re-export.
- Bad: the first run still needs network access to fetch from `sceptre-ocr` (mitigated, as before,
  by the HF cache + sha256 pins per ADR 0017).
- Neutral: fresh exports produce different bytes than itextresearch's, so pins and the `sceptre`
  snapshot side of the golden fixtures are regenerated.

## Supersedes

Supersedes [ADR 0003](0003-model-source-itext-onnx-runtime-download.md). See also the status notes
on [ADR 0008](0008-uv-taskfile-tools-infra.md) (export path ordering) and
[ADR 0011](0011-repointable-registry-owner.md) (repo-name change beyond the owner override).

## Status update (2026-08-08)

The hosting location is **amended by [ADR 0040](0040-models-hosted-under-the-xberg-io-hf-org.md)**:
the exports now live under the `xberg-io` Hugging Face org as `xberg-io/sceptre-*` rather than
`sceptre-ocr/*`. Everything else in this ADR — the first-party export pipeline, the provenance and
licensing story, and the sha256 pinning — is unchanged, and the artifact bytes are identical.
