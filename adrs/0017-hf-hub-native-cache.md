---
status: accepted
date: 2026-08-01
deciders: Na'aman Hirschfeld
---

# Resolve models through the Hugging Face hub's native on-disk cache

## Context and Problem Statement

The library downloaded models into a bespoke flat layout,
`~/.cache/sceptre/<owner>/<name>/<file>`, fetching the bytes in memory via
`hf-hub` and writing them itself. The parity harness, meanwhile, read models from
Hugging Face's standard hub cache
(`~/.cache/huggingface/hub/models--<owner>--<name>/snapshots/<rev>/<file>`). The
two stores never agreed: a model downloaded by the library was invisible to the
harness and vice-versa, which blocked parity testing and meant the CLI `models
download`, the `sceptre-tools snapshot` tool, and the tests could each populate a
different cache.

## Decision Drivers

- One cache store shared by the library, the CLI, the snapshot tool, and the tests.
- Reuse `hf-hub`'s battle-tested cache (atomic writes, snapshot revisions, refs)
  rather than reimplementing atomic download + layout.
- Honor the ecosystem-standard cache location and its environment overrides
  (`HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME`).
- Keep model manifest/inspection dependency-free (usable without the `download`
  feature).

## Considered Options

- Keep the bespoke `~/.cache/sceptre` layout and teach the harness to read it.
- Adopt `hf-hub`'s native hub cache in the library and delete the harness resolver.
- Point both at a third, custom shared layout.

## Decision Outcome

Chosen option: **adopt `hf-hub`'s native hub cache in the library**. Downloads go
through `HFClientSync`'s `download_file().send()`, which populates the hub cache
and returns the on-disk snapshot path (a cache hit returns the path without
re-downloading). A dependency-free `std`-only resolver
(`hf_cache_root` / `repo_cache_dir_name` / `resolve_cached`) inspects the same
layout for the offline "is it cached, and where" check that backs
`model_manifest`. The cache root resolves from `HF_HUB_CACHE` →
`HUGGINGFACE_HUB_CACHE` → `$HF_HOME/hub` → `~/.cache/huggingface/hub`, overridable
via `ModelConfig::cache_dir` (now a hub-cache-root override). The bespoke
`HfCacheModelProvider` and its resolver were removed from `tests/helpers`; the
tests now gate on the library's `model_manifest` and build the reader with the
default provider.

This **extends ADR 0003** (which recorded the `~/.cache/sceptre` location and the
`itextresearch` runtime-download source — the source is unchanged). ADR 0011's
owner-segment path-safety validation still applies: `registry_owner` is validated
to a safe form before it reaches the `models--<owner>--<name>` cache directory
name.

### Consequences

- Good: the library, CLI, snapshot tool, and harness share one cache store, so a
  model downloaded once is seen everywhere — unblocking parity.
- Good: less code — `hf-hub` owns atomic download, revisions, and layout; the
  harness resolver duplication is deleted.
- Good: standard location and env overrides; sha256 pins can still be enforced by
  streaming the returned file (real pins can be added later).
- Neutral: the flat `~/.cache/sceptre` layout is abandoned; a previously cached
  copy there is simply re-downloaded into the hub cache once.
