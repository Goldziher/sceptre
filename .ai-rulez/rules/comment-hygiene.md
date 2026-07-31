---
priority: high
---

# Comment Hygiene

- poly's `uncomment` hook strips non-doc comments unless they end with the `~keep` marker. Any plain `//` or TOML `#` comment that must survive a commit must end with `~keep`.
- Prefer Rust doc comments (`///`, `//!`) for explanatory content — they are preserved automatically and are the mature way to document.
- Never put session numbers, milestone tags, or progress notes ("Session 1", "M1", "wired later", "TODO next") in comments or in `todo!()` messages. Describe what the code does or the behavior that is missing, not when it is scheduled.
