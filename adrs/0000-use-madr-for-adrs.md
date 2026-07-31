---
status: accepted
date: 2026-07-31
deciders: Na'aman Hirschfeld
---

# Record decisions as MADR ADRs

## Context and Problem Statement

This is a from-scratch reimplementation with many architectural choices (scope,
model source, backends, packaging). We need a durable, reviewable record of why
each choice was made, so later contributors do not relitigate settled questions
or lose the rationale.

## Considered Options

- MADR (Markdown Any Decision Records) under `adrs/`.
- Rationale embedded in code comments and commit messages.
- A single running design document.

## Decision Outcome

Chosen option: **MADR under `adrs/`**, because it gives one immutable file per
decision with an explicit status lifecycle, is greppable and diff-friendly, and
keeps rationale out of code comments (which get stripped and do not carry
context well).

### Consequences

- Good: decisions are discoverable, versioned, and supersedable.
- Good: an `adr-discipline` rule keeps the practice enforced for agents.
- Bad: a small authoring overhead per significant decision.
