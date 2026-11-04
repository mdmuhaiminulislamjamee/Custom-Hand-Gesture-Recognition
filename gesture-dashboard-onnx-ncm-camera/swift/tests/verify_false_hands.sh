#!/bin/sh
set -eu
base=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
fixture=${1:?Supply the path to false_hand_policy.json}
task_tmp=$(mktemp -d)
trap 'rm -f "$task_tmp/main.swift" "$task_tmp/check"; rmdir "$task_tmp"' EXIT
cp "$base/tests/false_hand_main.swift" "$task_tmp/main.swift"
xcrun swiftc "$base/GestureFalseHandPolicy.swift" "$task_tmp/main.swift" -o "$task_tmp/check"
"$task_tmp/check" "$fixture"
