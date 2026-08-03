---
status: accepted
date: 2026-07-31
deciders: Na'aman Hirschfeld
---

# Re-pointable model registry via a `registry_owner` override

## Context and Problem Statement

[ADR 0003](0003-model-source-itext-onnx-runtime-download.md) sources the gen2 + CRAFT
ONNX models from the upstream `itextresearch/itext-EasyOCR-*` Hugging Face repos. We intend
to publish our own exports later (self-hosted or under a different HF org — undecided), and
users may want to mirror or vendor the models on a private account. The download layer must
let the host change **without a code change or a new build**, while keeping the default
behavior identical to ADR 0003.

## Decision Drivers

- Swap the model host with configuration, not a code edit — the host choice is still open.
- Zero behavior change by default (must resolve to today's `itextresearch/...` ids).
- No path-traversal / arbitrary-write risk from a config-controlled value reaching cache paths.

## Considered Options

- **Hardcode the org** — simplest, but a host change means editing the registry table and
  cutting a release.
- **`registry_owner` owner-segment override on `ModelConfig`** — swap only the owner part of
  each repo id; the `itext-EasyOCR-<model>` repo-name segment is preserved.
- **Full repo-prefix / base-URL override** — maximum flexibility, but more surface to
  validate and easy to misconfigure.

## Decision Outcome

Chosen option: **an optional `registry_owner` owner-segment override on `ModelConfig`.**
`registry::effective_repo(entry, registry_owner)` maps the base entry to an effective repo
id: `None` returns the entry's own `hf_repo` verbatim (exactly the ADR 0003 ids), `Some(owner)`
replaces only the owner segment. `DefaultModelProvider` reads `model.registry_owner` from
config and threads it into `download::ensure`. The mechanism is deliberately narrow (owner
swap, not an arbitrary URL) so the repo-name/file layout — and therefore the cache layout —
stays fixed and predictable.

Because `registry_owner` flows from the config precedence chain (file < env < flags) into an
on-disk cache path, it is **validated at the composition boundary**: `effective_repo` rejects
any owner outside `[A-Za-z0-9_-]+` with an `OcrError::Config`, so a value like `../../etc`
can never inject a path separator or `..` traversal into the cache directory (which would
otherwise be an arbitrary file write when the artifact is persisted).

This extends ADR 0003's model-source decision; it does not supersede it. The default source
is unchanged.

### Consequences

- Good: the model host is a config change; we can move to our own HF org later with no code
  edit and no release.
- Good: default resolves to the exact upstream ids — no behavior change for existing users.
- Good: the override is validated before touching the filesystem, closing an arbitrary-write
  path-traversal vector.
- Bad/limited: only the owner segment is overridable — a host with a different repo-naming
  scheme or a non-HF backend would need a broader mechanism (revisit if that host is chosen).
- Neutral: artifacts served under a different owner cache to a distinct path; identical
  content still satisfies the same sha256 pin once pins are populated.

## Status update (2026-08-03)

[ADR 0025](0025-first-party-onnx-exports.md) moves the models to first-party repos named
`sceptre-ocr/<model>` — a different repo-name scheme, not just a different owner — which is exactly
the "revisit if that host is chosen" limitation flagged above. The baked `hf_repo` ids in the
registry are therefore edited directly rather than via this owner-only override. The
`registry_owner` mechanism itself is retained unchanged for downstream mirroring/re-pointing of the
new `sceptre-ocr` ids.
