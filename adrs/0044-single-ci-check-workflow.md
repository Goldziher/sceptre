---
status: accepted
date: 2026-08-08
deciders: Na'aman Hirschfeld
---

# One CI check workflow, gated by a paths filter

## Context and Problem Statement

CI *checks* (as opposed to the deploy in `ci-docs.yaml`, the scheduled `benchmarks.yaml`, and the
release-triggered `publish.yaml`) were split across three files: `ci-rust.yaml` (test, install,
parity, gpu-backends, bench, deny), `ci-lint.yaml` (validate, changelog), and `ci-python.yaml`
(python). Each carried its own `on:` block, its own `concurrency` group, and — for `ci-rust.yaml`
and `ci-python.yaml` — its own `paths:` filter to skip the workflow when nothing relevant changed.

The split was a per-language convention (`cicd-pipeline-standards`: "workflows split by domain"),
not a decision this repo made for a reason of its own. For an eight-workflow product with one
Rust core and one small Python tooling package, it produced eight separate check runs on the PR
status list for what a contributor experiences as a single "did CI pass" question, and it
duplicated the `if: github.repository == ... && github.actor != 'dependabot[bot]'` guard and the
concurrency-group boilerplate three times over.

## Decision Drivers

- CI status should read as one signal for one product, not require summing across files.
- The cost the per-language split bought — skipping the three-OS Rust matrix, the install lanes,
  and the parity lanes on a docs-only change — is real and must not be lost by merging files.
- `ci-docs.yaml` deploys to Pages and needs `pages: write` + `id-token: write`; no other job may
  inherit those permissions.
- A path filter that fails must fail toward running too much, never toward silently running
  nothing.

## Considered Options

- Keep the three-file split as-is.
- Merge everything, including `ci-docs.yaml`, into one workflow with per-job path conditions.
- Merge the three *check* workflows into one `ci.yaml` with a `changes` job gating the Rust and
  Python jobs, keeping `ci-docs.yaml` separate.
- Drop path filtering entirely and run every job on every push.

## Decision Outcome

**Merge `ci-rust.yaml`, `ci-lint.yaml`, and `ci-python.yaml` into a single `ci.yaml`.** `validate`
and `changelog` (from `ci-lint.yaml`) run unconditionally, exactly as before. The six Rust jobs and
the `python` job each gain `needs: changes` and an `if:` that ANDs the existing guard with
`needs.changes.outputs.{rust,python} == 'true'`.

**A `changes` job is required, not path filters dropped.** The Rust matrix is three OS times
several feature variants plus separate install and parity lanes — running all of it on a
docs-only push is exactly the cost the original per-language split existed to avoid. Folding the
files together without replacing that mechanism would have silently reintroduced it. `dorny/
paths-filter@v4` reproduces the same `paths:` lists inside one job (`rust`: `crates/**`,
`tools/**`, `scripts/**`, `Cargo.toml`, `Cargo.lock`, `rustfmt.toml`, `deny.toml`, `.task/**`,
`Taskfile.yaml`, and `ci.yaml` itself; `python`: `python/**`, `pyproject.toml`, `uv.lock`, and
`ci.yaml`), so the buy-back is inside the merged file rather than outside it.

**The filter fails open.** `paths-filter` needs a base SHA, and a tracking-ref name such as
`origin/main` makes it try to `git fetch` that name as a refspec, which fails — it must be given
a resolved SHA (`xberg-enterprise`'s `docker.yaml` documents the same trap).

The base is `github.event.pull_request.base.sha` on a pull request and **`github.event.before`**
on a push — deliberately not `HEAD~1`. A push can deliver several commits at once, and diffing
only its last commit would miss a `crates/**` edit made earlier in the same push, skipping the
Rust matrix behind a green check. `before` is the ref's tip prior to the push, so it spans every
commit the push delivered. It is all-zeros on a ref's first push and absent on
`workflow_dispatch`; both, plus a `before` that is not present in the checkout, resolve to an
empty base, which skips the filter step. In every one of those cases the `changes` job's outputs
default to `'true'` for both `rust` and `python`, so the matrix runs. A false "nothing changed" here would look like
a green CI while skipping the entire Rust matrix — silently reporting success on unverified code
is worse than an unnecessary run, so the fallback direction is not negotiable.

**`ci-docs.yaml` stays a separate workflow.** It is a deploy, not a check: it needs
`pages: write` and `id-token: write` to publish to GitHub Pages. Folding it into `ci.yaml` would
grant those permissions to the whole workflow file, including the Rust matrix and the Python job
— a least-privilege regression with no offsetting benefit, since `ci-docs.yaml` already has its
own `docs-site/**`-scoped trigger and never needed the `changes` job's Rust/Python filters.

**Python CI is not a binding-parity check and stays.** `python/sceptre_rs_tools` is not a
language binding for sceptre; it is the tooling that exports the ONNX models from EasyOCR's torch
weights, generates the golden fixtures the Rust `tier2_golden` test reads, drives the EasyOCR
head-to-head, and renders every published benchmark number. Concretely,
`crates/sceptre/tests/data/metrics_vectors.json` is a numeric contract asserted at 1e-6 by *both*
`crates/sceptre/tests/helpers/mod.rs` (Rust) and `python/tests/test_metrics_vectors.py` (Python).
Dropping the `python` CI job would leave that cross-implementation check enforced on only one
side, which is exactly the failure mode a shared fixture is meant to prevent.

### Consequences

- One workflow file, `ci.yaml`, carries every check job except the Pages deploy; the check list on
  a PR reads as one product's status rather than three files' worth of guards and concurrency
  groups.
- `main` currently has no branch protection, so there are no required-status-check names to
  migrate as part of this change. If branch protection is added later, the required check names
  will be the new `ci.yaml` job names (`test`, `install`, `parity`, `gpu-backends`, `bench`,
  `deny`, `python`, `validate`, `changelog`), not the old per-workflow names — that migration step
  is not automatic and must be done by hand when protection is configured.
- `README.md`'s CI badge now points at `ci.yaml` instead of `ci-rust.yaml`.
- ADR 0035, which references the pre-consolidation per-workflow split historically (naming
  `ci-rust.yaml` and its matrix), is left unchanged — that reference remains accurate as a
  description of the CI shape at the time it was written.

## Related

- ADR 0035 (backend x accelerator benchmark matrix) — references the pre-consolidation workflow
  split as history; not amended by this decision.
