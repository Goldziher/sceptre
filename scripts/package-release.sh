#!/usr/bin/env bash
set -euo pipefail

# Package the release CLI binary for one target triple into a distributable archive. ~keep
# The `sceptre` binary is built with `--features ort-bundled`, which statically links ONNX ~keep
# Runtime (ort's `download-binaries`), so the executable is self-contained: the archive holds ~keep
# just the binary (plus any dynamic ONNX Runtime library a platform emits alongside it, copied ~keep
# defensively). Contents sit at the archive root so an extract drops `sceptre` straight into PATH. ~keep

if [ $# -ne 1 ]; then
	echo "Usage: $0 <target-triple>" >&2
	exit 1
fi

TRIPLE="$1"

case "$TRIPLE" in
x86_64-unknown-linux-gnu | aarch64-unknown-linux-gnu)
	SYSTEM="linux"
	BINEXT=""
	;;
aarch64-apple-darwin | x86_64-apple-darwin)
	SYSTEM="macos"
	BINEXT=""
	;;
x86_64-pc-windows-msvc)
	SYSTEM="windows"
	BINEXT=".exe"
	;;
*)
	echo "Unknown target triple: $TRIPLE" >&2
	exit 1
	;;
esac

RELEASE_DIR="target/${TRIPLE}/release"
BINARY_PATH="${RELEASE_DIR}/sceptre${BINEXT}"

if [ ! -f "$BINARY_PATH" ]; then
	echo "Binary not found at $BINARY_PATH" >&2
	exit 1
fi

STAGING_DIR="sceptre-staging-${TRIPLE}"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"
cp "$BINARY_PATH" "$STAGING_DIR/sceptre${BINEXT}"

# Defensive: ort-bundled links ONNX Runtime statically, so normally nothing else ships. If a ~keep
# platform build ever emits a dynamic libonnxruntime next to the binary, bundle it so the archive ~keep
# stays runnable rather than silently missing a runtime dependency. ~keep
for candidate in "$RELEASE_DIR"/libonnxruntime.* "$RELEASE_DIR"/onnxruntime.dll "$RELEASE_DIR"/deps/libonnxruntime.* "$RELEASE_DIR"/deps/onnxruntime.dll; do
	[ -f "$candidate" ] || continue
	cp -L "$candidate" "$STAGING_DIR/" 2>/dev/null || true
done

if [ "$SYSTEM" = "windows" ]; then
	ARCHIVE="sceptre-${TRIPLE}.zip"
	(cd "$STAGING_DIR" && {
		7z a -tzip "../${ARCHIVE}" . >/dev/null 2>&1 ||
			zip -q -r "../${ARCHIVE}" . ||
			powershell -Command "Compress-Archive -Path '*' -DestinationPath '../${ARCHIVE}' -Force"
	})
else
	ARCHIVE="sceptre-${TRIPLE}.tar.gz"
	tar czf "$ARCHIVE" -C "$STAGING_DIR" .
fi

rm -rf "$STAGING_DIR"
echo "Created ${ARCHIVE}"
