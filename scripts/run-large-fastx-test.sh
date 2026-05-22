#!/usr/bin/env bash
# Generate and process a FASTA at CUSBF_LARGE_FASTX_GB|_MB|_BYTES through cuSBF.
# The generated file lives under build/test_artifacts/ and is gitignored.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
GB="${CUSBF_LARGE_FASTX_GB:-16}"

export CUSBF_LARGE_FASTX_GB="$GB"
export CUSBF_LARGE_FASTX_DIR="${CUSBF_LARGE_FASTX_DIR:-$BUILD_DIR/test_artifacts}"

if [[ ! -x "$BUILD_DIR/tests/test-large-fastx" ]]; then
  echo "Building test-large-fastx ..."
  meson setup "$BUILD_DIR" -Dtests=enabled >/dev/null 2>&1 || meson setup "$BUILD_DIR" -Dtests=enabled
  ninja -C "$BUILD_DIR" tests/test-large-fastx
fi

echo "Running large FASTX test: ${GB} GiB target under ${CUSBF_LARGE_FASTX_DIR}"
"$BUILD_DIR/tests/test-large-fastx" "$@"
