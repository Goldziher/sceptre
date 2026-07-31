---
status: accepted
date: 2026-07-31
deciders: Na'aman Hirschfeld
---

# Generate agent config with ai-rulez

## Context and Problem Statement

The repo targets multiple coding-agent harnesses (Claude Code, Codex). Hand-
maintaining `CLAUDE.md`, `AGENTS.md`, `.mcp.json`, and per-harness agent/command
files in parallel is error-prone and drifts.

## Considered Options

- Hand-write a committed `CLAUDE.md` (and duplicate for other harnesses).
- Use `ai-rulez`: edit a single `.ai-rulez/` source of truth and generate the
  per-harness files (which are gitignored).

## Decision Outcome

Chosen option: **ai-rulez**. `.ai-rulez/` (config, rules, context, commands) is
the source of truth; `ai-rulez generate` produces `CLAUDE.md`, `AGENTS.md`,
`.mcp.json`, and `.claude/` / `.codex/` agents, commands, and skills, all
gitignored. `builtins = ["rust", "cicd", "documentation", "default-commands"]`
supplies the implementation subagents.

### Consequences

- Good: one edit point; harness files stay consistent and regenerable.
- Good: the generated subagents drive later implementation sessions.
- Bad: contributors must run `ai-rulez generate` (via the poly hook) rather than
  editing the generated files directly.
