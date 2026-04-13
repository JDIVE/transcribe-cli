#!/usr/bin/env bash

set -euo pipefail

VERSION="${1:-dev}"
OUTDIR="${2:-release}"

mkdir -p "${OUTDIR}"
rm -f "${OUTDIR}"/transcribe-cli-* "${OUTDIR}"/checksums.txt

targets=(
  "bun-linux-x64-baseline:transcribe-cli-${VERSION}-linux-x64-baseline"
  "bun-linux-arm64:transcribe-cli-${VERSION}-linux-arm64"
  "bun-darwin-x64:transcribe-cli-${VERSION}-darwin-x64"
  "bun-darwin-arm64:transcribe-cli-${VERSION}-darwin-arm64"
  "bun-windows-x64-baseline:transcribe-cli-${VERSION}-windows-x64-baseline.exe"
)

for item in "${targets[@]}"; do
  target="${item%%:*}"
  name="${item#*:}"
  echo "Building ${name} (${target})"
  bun build --compile --minify --target="${target}" ./src/cli.ts --outfile "${OUTDIR}/${name}"
done

(
  cd "${OUTDIR}"
  sha256sum transcribe-cli-* > checksums.txt
)
