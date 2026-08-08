---
status: accepted
date: 2026-08-08
deciders: Na'aman Hirschfeld
---

# Model artifacts move to the `xberg-io` Hugging Face org (amends ADR 0025)

## Context and Problem Statement

[ADR 0025](0025-first-party-onnx-exports.md) took ownership of the model supply chain by building
a first-party `.pth → ONNX` export pipeline and hosting the results under a dedicated
`sceptre-ocr` Hugging Face org. That decision — own the exports, own the hosting — is unchanged
and still correct.

What changed is the surrounding context. sceptre moved from a solo project to
`github.com/xberg-io/sceptre`, and the rest of the stack already publishes its model artifacts
under the `xberg-io` Hugging Face org (`paddleocr-onnx-models`, `layout-models`,
`embedding-models`, and others). A sceptre-only org is now one more namespace to administer,
with its own membership and access story, hosting nine repos that belong to the same
organization as everything around them.

## Decision Drivers

- One org to administer, with the same membership and access controls as the rest of the stack.
- Do not break already-published crates. Versions 0.2.0 through 0.4.0 are live on crates.io with
  `sceptre-ocr/*` compiled into their registries; those binaries must keep resolving models.
- Preserve [ADR 0011](0011-repointable-registry-owner.md)'s registry-owner override, which swaps
  only the owner segment of a repo id so a mirror or air-gapped copy stays a one-setting change.
- Names must stay unambiguous inside a shared, multi-project org.

## Decision

**The nine model repos move to the `xberg-io` Hugging Face org, each prefixed with `sceptre-`:**

| before | after |
|---|---|
| `sceptre-ocr/craft_mlt_25k` | `xberg-io/sceptre-craft_mlt_25k` |
| `sceptre-ocr/<lang>_g2` | `xberg-io/sceptre-<lang>_g2` |

The `sceptre-` prefix is load-bearing rather than cosmetic. `xberg-io` hosts models for the whole
stack, so a bare `english_g2` or `korean_g2` says nothing about which project owns it or which
runtime contract it satisfies. The prefix also keeps ADR 0011 intact: `effective_repo` swaps the
owner segment and keeps the repo-name segment verbatim, so a mirror at `acme/sceptre-english_g2`
still works with a single `registry_owner` setting.

The exports themselves are untouched — same bytes, same sha256 pins, same charsets. This ADR
changes *where the artifacts live*, not what they are or how they are produced; ADR 0025's export
pipeline, provenance story, and licensing all stand.

## Considered and rejected

**A single bundled `xberg-io/sceptre-onnx-models` repo** holding all nine ONNX files, matching the
`paddleocr-onnx-models` / `layout-models` convention used elsewhere in the org. Rejected on two
counts. It cannot be done as a server-side move, so it would mean re-uploading ~215 MB and
breaking the digest-verified continuity of the existing repos. More importantly it would collapse
`ModelEntry::hf_repo` to a single shared constant, so the ADR 0011 override would no longer select
a per-model location and the on-disk cache layout (one hub directory per repo) would change for
every user. The per-model repos also let a consumer fetch exactly one language rather than the
whole family.

**Leaving the repos on `sceptre-ocr` and mirroring into `xberg-io`.** Two copies of the same bytes
drift, and it doubles the surface that has to be kept in sync at each re-export. The org-count
problem this ADR exists to solve would remain.

## Consequences

- Good: one org, one set of access controls, consistent with the rest of the stack.
- Good: published 0.2.0–0.4.0 keep working. Hugging Face serves a redirect from a moved repo, and
  this was verified end to end against the `resolve` endpoint the downloader actually uses — a
  throwaway repo was moved between namespaces and the old path still returned the file. The move
  was not performed on the model repos until that held.
- Neutral: the sha256 pins in `models::registry` are unchanged, so `models::download`'s
  verification is unaffected and existing caches stay valid. The on-disk hub cache directory does
  change name with the repo id, so the next run re-downloads once.
- Bad: the redirect is a Hugging Face behavior, not a contract we control. If it is ever withdrawn,
  crates 0.2.0–0.4.0 lose model resolution with no recourse short of a patch release. Anything
  that must not depend on it should pin `registry_owner` explicitly (ADR 0011).
- Bad: the `sceptre-ocr` org is now empty rather than deleted. Leaving it in place is deliberate —
  deleting it would free the namespace for re-registration by someone else, which is a worse
  outcome than an idle org.

## Related

- Amends [ADR 0025](0025-first-party-onnx-exports.md); does not supersede it. The first-party
  export decision, the pipeline, and the provenance story are unchanged.
- Depends on [ADR 0011](0011-repointable-registry-owner.md)'s owner-segment override, which the
  `sceptre-` prefix was chosen to preserve.
- [ADR 0003](0003-model-source-itext-onnx-runtime-download.md) records the original third-party
  `itextresearch` source, already superseded by ADR 0025.
