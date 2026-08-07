---
priority: high
---

# Rust Check Discipline

- Write a failing test first when the change is observable from the public API, the CLI, or the MCP
  surface. Those three are sceptre's contract; everything else is an implementation detail.
- Clippy is strict (`-D warnings`). Do not silence a lint with `#[allow(...)]` unless the warning is
  genuinely incorrect — and add a one-line `//` justification ending with `~keep` when you do.
- After adding or changing any dependency, run `cargo upgrade --incompatible` so pins stay current,
  then rebuild.
- `cargo test --workspace` is the release gate, not a per-commit gate. Run it when the change
  touches the pipeline, before a release, or when asked.
