#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
output="$(mktemp -d)"
trap 'rm -f "$output/gesture-features"; rmdir "$output"' EXIT
xcrun swiftc -O GestureFeatureExtractor.swift GestureRuntimePolicy.swift \
  tests/mobile_features.swift -o "$output/gesture-features"
"$output/gesture-features" ../tests/parity_vectors.json
