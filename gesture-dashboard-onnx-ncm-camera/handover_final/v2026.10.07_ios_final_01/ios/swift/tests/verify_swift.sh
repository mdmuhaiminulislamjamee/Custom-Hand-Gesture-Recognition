#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
output="$(mktemp -d)"
# Only the temporary directory created above is removed.
trap 'rm -f "$output/gesture-parity"; rmdir "$output"' EXIT
xcrun swiftc -O GestureFeatureExtractor.swift GestureGeometryResolver.swift \
  GestureTemporalGate.swift GestureActionLatch.swift GestureDecisionPipeline.swift \
  tests/main.swift -o "$output/gesture-parity"
"$output/gesture-parity" tests/parity_vectors.json
