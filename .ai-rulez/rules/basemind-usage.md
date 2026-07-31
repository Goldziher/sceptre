---
priority: high
---

# basemind usage

## basemind — prefer it over grep / read / git

basemind is this repo's indexed context layer. Prefer it BEFORE grep, before reading files to find structure, and before naked `git` — it's the default, not a preference. basemind returns paths, lines, and signatures at a fraction of the tokens of reading source.

### Routing

| Reach for | Instead of |
|---|---|
| `search_symbols` / `goto_definition` / `find_references` / `find_callers` / `find_implementations` / `call_graph` / `workspace_grep` | `grep` / `rg` / opening files to find a symbol |
| `outline` / `architecture_map` | reading whole files to learn their shape |
| `recent_changes` / `blame_symbol` / `commits_touching` / `diff_file` | `git log` / `git blame` / `git diff` |
| `find_files` (fuzzy path search) | `find` / `fd` / `ls -R` to locate a file by name |
| `thread_post` / `inbox_read` / `thread_list` | assuming you're the only agent in the repo |
| `workspaces` / `worktrees` / `worktree_claim` | editing a worktree another session may already own |
| `search_documents` / `web_scrape` / `web_crawl` / `web_map` | manually reading PDFs / docs or ad-hoc fetching |
| semantic code search over the index | keyword-only guessing at where a concept lives |

### Red flags — stop and re-route

- About to `grep` / `rg`? → `workspace_grep`.
- About to open a file just to find a symbol? → `outline` / `search_symbols`.
- About to `git log` / `git blame`? → `recent_changes` / `blame_symbol`.
- Already mapped a file with basemind? Don't re-read it.

### Setup & maintenance

- Install the basemind Claude Code plugin from its marketplace (`/plugin marketplace add Goldziher/basemind`, then install `basemind`).
- Keep basemind current: enable plugin auto-update, or update the binary regularly so the index format and tools stay in sync.
- Re-run `basemind init` (or `/bm-init`) after enabling new capabilities to refresh this block.
