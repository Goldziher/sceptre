---
priority: high
---

# TDD + poly Workflow

- Practice red-green-refactor. Write a failing test first when the change is observable from the public API, the CLI, or the MCP surface.
- Before every commit, run the local check triad:
  - `cargo fmt --all`
  - `cargo clippy --workspace --all-targets --tests -- -D warnings`
  - `cargo test --workspace`
  - `poly lint .` (typos, markdown line length, cargo-deny, and the `uncomment` `~keep` check)
- Clippy is strict (`-D warnings`). Do not silence with `#[allow(...)]` unless the warning is genuinely incorrect — and add a one-line `//` justification ending with `~keep` when you do.
- After adding or changing any dependency, run `cargo upgrade --incompatible` so pins stay current, then rebuild.
- Commits use Conventional Commit prefixes (`feat:`, `fix:`, `perf:`, `chore:`, `refactor:`). Match the style in `git log`.
