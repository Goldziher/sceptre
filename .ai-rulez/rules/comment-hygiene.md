---
priority: high
---

# Comment Hygiene

Extends the shared `comment-discipline` rule with two sceptre-specific habits.

- Prefer Rust doc comments (`///`, `//!`) for explanatory content. They are preserved without a
  `~keep` marker and are the mature way to document a public surface.
- Never put session numbers, milestone tags, or progress notes ("Session 1", "M1", "wired later",
  "TODO next") in comments or in `todo!()` messages. Describe what the code does or the behavior
  that is missing, not when it is scheduled.
