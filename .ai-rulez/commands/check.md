---
priority: high
aliases: [c]
usage: "/check"
description: "Run the local lint + test triad before committing"
---

# Check

Run the commit triad. Use this before every commit.

1. `cargo fmt --all`
2. `cargo clippy --workspace --all-targets --tests -- -D warnings`
3. `cargo test --workspace`
4. `poly lint .`

Each step gates the next. On failure:

- `cargo fmt` — re-stage formatted files.
- clippy — fix, or justify with a one-line `//` comment ending in `~keep`.
- tests — diagnose the failure; never `#[ignore]` to bypass.
- poly — fix the reported typo / markdown / deny / rust-max-lines issue.
