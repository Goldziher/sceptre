#!/usr/bin/env bash
set -euo pipefail

# Packaging dry-run for `sceptre-cli`, which cannot always run locally. ~keep
# ~keep
# `cargo publish --dry-run -p sceptre-cli` resolves its `sceptre` dependency from the crates.io ~keep
# index rather than the workspace path, and does so while preparing the package — before any ~keep
# verification build, so `--no-verify` does not avoid it either. On the version-bump commit the ~keep
# matching library version is by definition not published yet, so the check cannot pass on the ~keep
# one commit where a release is actually being cut. ~keep
# ~keep
# The release workflow publishes `sceptre` first and waits for it to appear in the index, so the ~keep
# ordering is sound there. This script therefore skips that single unavoidable cause, loudly, and ~keep
# lets every other packaging error fail as normal. ~keep

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

version="$(grep -E '^version = "' Cargo.toml | head -1 | cut -d'"' -f2)"

if [ -z "$version" ]; then
	echo "error: could not read the workspace version from Cargo.toml" >&2
	exit 1
fi

# crates.io rejects a request with no User-Agent with 403, so an unidentified probe reads as ~keep
# "not published" for every version. Distinguish the three answers rather than treating any ~keep
# non-200 as absent: a 403 or a network failure must not masquerade as a skippable release. ~keep
status="$(curl -sS -o /dev/null -w '%{http_code}' \
	-H 'User-Agent: sceptre release check (https://github.com/xberg-io/sceptre)' \
	"https://crates.io/api/v1/crates/sceptre/${version}" || echo 000)"

case "$status" in
200)
	cargo publish --dry-run -p sceptre-cli --no-default-features --features ort-dynamic,download
	exit 0
	;;
404) ;;
*)
	echo "error: crates.io returned HTTP ${status} for sceptre ${version}; cannot tell whether" >&2
	echo "       it is published, so refusing to guess. Re-run when crates.io is reachable." >&2
	exit 1
	;;
esac

cat >&2 <<EOF
skip: sceptre-cli packaging is unverifiable locally at this version.

  sceptre ${version} is not on crates.io, and sceptre-cli depends on it by version, so cargo
  cannot prepare the package regardless of flags. This is expected on a version-bump commit.

  The release workflow publishes sceptre first and waits for the index, then packages the CLI,
  so the real publish still validates it. To check it by hand, re-run this after sceptre
  ${version} is live.
EOF
