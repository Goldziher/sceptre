---
status: accepted
date: 2026-08-03
deciders: Na'aman Hirschfeld
supersedes: 0020
---

# Publish to crates.io for real, via GitHub Actions trusted publishing

## Context and Problem Statement

ADR 0020 wired the tag-triggered release pipeline but deliberately **deferred the real
crates.io publish**, running only `cargo publish --dry-run` and leaving "turning on a real
publish later" as an explicit future decision. That decision is now made: the repository is
going public and `sceptre` + `sceptre-cli` are published starting at `v0.1.0`. We need a
publish mechanism that is secure (no long-lived registry token in the repo), correct in
dependency order, and idempotent across release re-runs.

## Decision Drivers

- No long-lived `CARGO_REGISTRY_TOKEN` stored as a GitHub secret.
- Publish order is load-bearing: `sceptre-cli` depends on `sceptre`, so the library must be
  on the index before the CLI resolves it (ADR 0020's original constraint).
- crates.io publishes are irreversible; a partial or premature publish must be avoidable and
  a re-run must not fail on an already-published version.
- Reuse the proven pattern from the sibling `basemind` release workflow.

## Considered Options

- A classic `CARGO_REGISTRY_TOKEN` GitHub secret.
- **crates.io trusted publishing** (GitHub Actions OIDC) via `rust-lang/crates-io-auth-action`.

## Decision Outcome

Publish for real from the tag-triggered release workflow using **trusted publishing**. The
`publish-dry-run` job is replaced by a `publish-crates` job that:

- Requests `id-token: write` and calls `rust-lang/crates-io-auth-action@v1`, which exchanges
  the workflow's OIDC identity for a short-lived registry token — no stored secret.
- Publishes `sceptre` then `sceptre-cli`. `cargo publish` waits for each crate to land in the
  index before returning, so the CLI resolves its just-published `sceptre` dependency.
- Wraps each publish in an idempotent helper: a version already uploaded (a re-dispatch of the
  same tag) is treated as a skip, not a failure, so a partial publish heals on re-run.
- Runs only after `finalize` (the GitHub release promoted with all platform binaries), and is
  skipped when the `meta` job detects this version is already on crates.io. Gating on the
  finalized release keeps the source and binary surfaces atomic across a partial failure.

The trusted publisher must be configured on crates.io for **both** `sceptre` and `sceptre-cli`
(owner `Goldziher`, repo `sceptre`, workflow `publish.yaml`) before the first tag; the
workspace version already equals the tag (`meta` verifies this). The local
`task release:dry-run` remains for packaging validation off the release path.

### Consequences

- Cutting a `v*` tag now publishes both crates to crates.io with no stored token.
- The first release requires the one-time trusted-publisher setup on crates.io; until then the
  `publish-crates` job fails at the auth step (the binary release still succeeds).
- Supersedes ADR 0020's dry-run-only stance; the multi-platform build, versioning, and
  release-promotion mechanics from 0020 are retained unchanged.
