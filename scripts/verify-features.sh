#!/usr/bin/env bash
# Guards the CLI's ONNX Runtime provisioning matrix at the dependency-resolution level. ~keep
# Cargo features are additive, so a wrong default silently unions `load-dynamic` into ~keep
# every build and the binary only fails at runtime with "Failed to load ONNX Runtime ~keep
# dylib". `cargo tree --locked` resolves without compiling or hitting the network, so ~keep
# this stays cheap enough to run in the pre-commit gate. ~keep
set -euo pipefail

cd "$(dirname "$0")/.."

failures=0

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  failures=$((failures + 1))
}

pass() {
  printf 'ok: %s\n' "$1"
}

# Resolves the sceptre-cli graph under the given cargo feature flags and sets ~keep
# `resolved` to the `ort` crate's comma-delimited feature list, or the empty string ~keep
# when the `ort` crate is absent from the graph entirely. A cargo failure (a missing ~keep
# feature, a stale lockfile) is reported rather than mistaken for "ort is absent". ~keep
ort_features() {
  local tree
  if ! tree="$(cargo tree --locked -p sceptre-cli -e normal,build -f '{p} {f}' --prefix none "$@" 2>&1)"; then
    printf 'FAIL: `cargo tree %s` did not resolve:\n%s\n' "$*" "${tree}" >&2
    failures=$((failures + 1))
    resolved=""
    return 1
  fi
  resolved="$(printf '%s\n' "${tree}" | awk '$1 == "ort" { print $3 }' | sort -u | paste -sd, - | tr -d '[:space:]')"
}

# Substring-matches on a comma-fenced feature list so `tls-rustls` cannot satisfy a ~keep
# lookup for `tls-rustls-no-provider` or vice versa. ~keep
has_feature() {
  local resolved="$1" feature="$2"
  [[ ",${resolved}," == *",${feature},"* ]]
}

assert_has() {
  local label="$1" feature="$2" resolved="$3"
  if has_feature "${resolved}" "${feature}"; then
    pass "${label}: ort resolves with '${feature}'"
  else
    fail "${label}: expected the ort crate to resolve with '${feature}', got '${resolved:-<no ort crate in graph>}'"
  fi
}

assert_lacks() {
  local label="$1" feature="$2" resolved="$3"
  if has_feature "${resolved}" "${feature}"; then
    fail "${label}: expected the ort crate NOT to resolve with '${feature}', got '${resolved}'"
  else
    pass "${label}: ort does not resolve with '${feature}'"
  fi
}

printf 'Resolving sceptre-cli feature matrix (cargo tree --locked; no build, no network)\n\n'

resolved=""

# 1. The published default must link a prebuilt ONNX Runtime. If `load-dynamic` leaks in ~keep
#    here, `cargo install sceptre-cli` produces a binary that panics on first use. ~keep
if ort_features; then
  assert_has "default features" "download-binaries" "${resolved}"
  assert_lacks "default features" "load-dynamic" "${resolved}"
fi

# 2. Opting into `ort-dynamic` is additive on top of the default; `load-dynamic` implies ~keep
#    `ort-sys/disable-linking`, which short-circuits the ort-sys build script before the ~keep
#    prebuilt download, so this works without --no-default-features. ~keep
if ort_features --features ort-dynamic; then
  assert_has "--features ort-dynamic" "load-dynamic" "${resolved}"
fi

# 3. The explicit bundled selection must stay equivalent to the default. ~keep
bundled_label="--no-default-features --features ort-bundled,download"
if ort_features --no-default-features --features ort-bundled,download; then
  assert_has "${bundled_label}" "download-binaries" "${resolved}"
  assert_lacks "${bundled_label}" "load-dynamic" "${resolved}"
fi

# 4. The pure-Rust path must not drag ort (or its build-time native download) in at all. ~keep
tract_label="--no-default-features --features tract,download"
if ort_features --no-default-features --features tract,download; then
  if [[ -n "${resolved}" ]]; then
    fail "${tract_label}: expected no ort crate in the graph, got ort with '${resolved}'"
  else
    pass "${tract_label}: no ort crate in the graph"
  fi
fi

printf '\n'
if ((failures > 0)); then
  printf '%d feature-resolution expectation(s) failed.\n' "${failures}" >&2
  exit 1
fi
printf 'All feature-resolution expectations hold.\n'
