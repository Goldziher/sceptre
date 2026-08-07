#!/usr/bin/env bash
# Guards the CLI's backend and accelerator feature matrix at the dependency-resolution ~keep
# level. Cargo features are additive, so a wrong default silently unions `load-dynamic` ~keep
# into every build and the binary only fails at runtime with "Failed to load ONNX Runtime ~keep
# dylib" — and, in the other direction, quietly compiles candle's kernels into an install ~keep
# that never asked for them. `cargo tree --locked` resolves without compiling or hitting ~keep
# the network, so this stays cheap enough to run in the pre-commit gate. ~keep
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
# `resolved` to `crate`'s comma-delimited feature list, or the empty string when that ~keep
# crate is absent from the graph entirely. A cargo failure (a missing feature, a stale ~keep
# lockfile) is reported rather than mistaken for "the crate is absent". ~keep
crate_features() {
  local crate="$1" tree
  shift
  if ! tree="$(cargo tree --locked -p sceptre-cli -e normal,build -f '{p} {f}' --prefix none "$@" 2>&1)"; then
    printf 'FAIL: `cargo tree %s` did not resolve:\n%s\n' "$*" "${tree}" >&2
    failures=$((failures + 1))
    resolved=""
    return 1
  fi
  resolved="$(printf '%s\n' "${tree}" | awk -v crate="${crate}" '$1 == crate { print $3 }' | sort -u | paste -sd, - | tr -d '[:space:]')"
}

ort_features() {
  crate_features ort "$@"
}

# Fails when `crate` appears in the graph at all, whatever features it carries. ~keep
assert_absent() {
  local label="$1" crate="$2" resolved="$3"
  if [[ -n "${resolved}" ]]; then
    fail "${label}: expected no ${crate} crate in the graph, got ${crate} with '${resolved}'"
  else
    pass "${label}: no ${crate} crate in the graph"
  fi
}

# Substring-matches on a comma-fenced feature list so `tls-rustls` cannot satisfy a ~keep
# lookup for `tls-rustls-no-provider` or vice versa. ~keep
has_feature() {
  local resolved="$1" feature="$2"
  [[ ",${resolved}," == *",${feature},"* ]]
}

assert_has() {
  local label="$1" crate="$2" feature="$3" resolved="$4"
  if has_feature "${resolved}" "${feature}"; then
    pass "${label}: ${crate} resolves with '${feature}'"
  else
    fail "${label}: expected the ${crate} crate to resolve with '${feature}', got '${resolved:-<no ${crate} crate in graph>}'"
  fi
}

assert_lacks() {
  local label="$1" crate="$2" feature="$3" resolved="$4"
  if has_feature "${resolved}" "${feature}"; then
    fail "${label}: expected the ${crate} crate NOT to resolve with '${feature}', got '${resolved}'"
  else
    pass "${label}: ${crate} does not resolve with '${feature}'"
  fi
}

printf 'Resolving sceptre-cli feature matrix (cargo tree --locked; no build, no network)\n\n'

resolved=""

# 1. The published default must link a prebuilt ONNX Runtime. If `load-dynamic` leaks in ~keep
#    here, `cargo install sceptre-cli` produces a binary that panics on first use. ~keep
if ort_features; then
  assert_has "default features" ort "download-binaries" "${resolved}"
  assert_lacks "default features" ort "load-dynamic" "${resolved}"
fi

# 2. Opting into `ort-dynamic` is additive on top of the default; `load-dynamic` implies ~keep
#    `ort-sys/disable-linking`, which short-circuits the ort-sys build script before the ~keep
#    prebuilt download, so this works without --no-default-features. ~keep
if ort_features --features ort-dynamic; then
  assert_has "--features ort-dynamic" ort "load-dynamic" "${resolved}"
fi

# 3. The explicit bundled selection must stay equivalent to the default. ~keep
bundled_label="--no-default-features --features ort-bundled,download"
if ort_features --no-default-features --features ort-bundled,download; then
  assert_has "${bundled_label}" ort "download-binaries" "${resolved}"
  assert_lacks "${bundled_label}" ort "load-dynamic" "${resolved}"
fi

# 4. The pure-Rust paths must not drag ort (or its build-time native download) in at all. ~keep
for pure in tract candle; do
  pure_label="--no-default-features --features ${pure},download"
  if ort_features --no-default-features --features "${pure},download"; then
    assert_absent "${pure_label}" ort "${resolved}"
  fi
done

# 5. The reverse containment: candle carries its own kernels and is opt-in, so the ~keep
#    default install must not compile them. ~keep
if crate_features candle-core; then
  assert_absent "default features" candle-core "${resolved}"
fi

# 6. Each execution-provider feature must reach the matching ort feature. Without it the ~keep
#    EP struct is not compiled in and `--accelerator <ep>` can only ever fail. ~keep
for ep in coreml directml cuda; do
  ep_label="--features ort-${ep}"
  if ort_features --features "ort-${ep}"; then
    assert_has "${ep_label}" ort "${ep}" "${resolved}"
  fi
done

# 7. Same contract on the candle side: the device feature must reach candle-core, or ~keep
#    `--accelerator metal` compiles to a build that can only refuse it. ~keep
for device in metal cuda; do
  device_label="--features candle-${device}"
  if crate_features candle-core --features "candle-${device}"; then
    assert_has "${device_label}" candle-core "${device}" "${resolved}"
  fi
done

printf '\n'
if ((failures > 0)); then
  printf '%d feature-resolution expectation(s) failed.\n' "${failures}" >&2
  exit 1
fi
printf 'All feature-resolution expectations hold.\n'
