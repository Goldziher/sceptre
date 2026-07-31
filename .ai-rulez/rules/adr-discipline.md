---
priority: high
---

# ADR Discipline

- Architecturally significant decisions are recorded as MADR ADRs under `adrs/` (`NNNN-title.md`, numbered sequentially).
- Add a new ADR when you choose between real alternatives with lasting consequences: a dependency or backend, a pipeline algorithm, a public API shape, a model source, a build/target strategy.
- Do not rewrite the history of an accepted ADR. To reverse a decision, add a new ADR with status `Accepted` that `Supersedes` the old one, and set the old one's status to `Superseded by NNNN`.
- Keep code comments free of decision rationale that belongs in an ADR — link to the ADR instead.
