#!/usr/bin/env bash
set -euo pipefail

# Regenerate the docs-site changelog page from the canonical CHANGELOG.md. ~keep
# The page is a Starlight content file, so it needs frontmatter that CHANGELOG.md must not ~keep
# carry; everything below the frontmatter is copied. That is what keeps the two from ~keep
# drifting: the docs copy is generated, never hand-edited. With --check, verify the copy is ~keep
# current and fail if it is not, so CI catches a CHANGELOG.md edit that skipped regeneration. ~keep

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT/CHANGELOG.md"
TARGET="$ROOT/docs-site/src/content/docs/more/changelog.md"
# CHANGELOG.md's repo-relative links resolve on GitHub but 404 on the docs site, which serves ~keep
# from a different root and does not publish adrs/. Absolutize them on the way out. ~keep
REPO_BLOB="https://github.com/xberg-io/sceptre/blob/main"

render() {
	cat <<'FRONTMATTER'
---
title: "Changelog"
description: Release history for sceptre.
---

FRONTMATTER
	# Drop the H1 and the blank line under it: Starlight renders the title from frontmatter, ~keep
	# so keeping the heading would print it twice. ~keep
	tail -n +2 "$SOURCE" | sed '/./,$!d' | sed "s#](adrs/#](${REPO_BLOB}/adrs/#g"
}

if [ "${1:-}" = "--check" ]; then
	if ! diff -u "$TARGET" <(render) >/dev/null; then
		echo "error: $TARGET is stale; run 'task docs:changelog'" >&2
		diff -u "$TARGET" <(render) >&2 || true
		exit 1
	fi
	echo "docs changelog is in sync with CHANGELOG.md"
	exit 0
fi

render >"$TARGET"
echo "wrote $TARGET"
