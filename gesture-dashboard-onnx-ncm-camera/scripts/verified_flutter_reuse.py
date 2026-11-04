"""Prove a native-only update has identical Dart inputs to a previously verified APK."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def verify(report):
    report = Path(report).resolve()
    prior = json.loads((report / 'verification.json').read_text())
    build = json.loads((report / 'result.json').read_text())
    assert prior['passed'] and build['passed']
    apk = ROOT / build['apk']
    assert hashlib.sha256(apk.read_bytes()).hexdigest() == prior['sha256'] == build['sha256']
    checks = prior['host_checks']
    assert len(checks) == 6 and all(row['exit_code'] == 0 for row in checks)
    test = next(row for row in checks if row['project'] == 'tester' and row['check'] == 'tests')
    line = next(line for line in test['tail'] if 'loading ' in line)
    tested_app = Path(re.search(r'loading (.*?)/test/', line)[1])
    tested_root = tested_app.parent
    hashes = {}
    for folder in ('flutter_gesture_tester/lib', 'flutter_gesture_tester/test',
                   'static_gesture_runtime/flutter/lib', 'static_gesture_runtime/flutter/third_party/hand_detection/lib'):
        current = {s.relative_to(ROOT).as_posix(): s for s in (ROOT / folder).rglob('*.dart')}
        previous = {s.relative_to(tested_root).as_posix(): s for s in (tested_root / folder).rglob('*.dart')}
        assert current.keys() == previous.keys(), folder
        for name, source in current.items():
            assert source.read_bytes() == previous[name].read_bytes(), name
            hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    def normalized(path):
        return '\n'.join(line for line in path.read_text().splitlines()
                         if not line.startswith('version:') and line.strip() != 'skip_build: true')
    for folder in ('flutter_gesture_tester', 'static_gesture_runtime/flutter',
                   'static_gesture_runtime/flutter/third_party/hand_detection'):
        assert normalized(ROOT / folder / 'pubspec.yaml') == normalized(tested_root / folder / 'pubspec.yaml'), folder
    for folder in ('flutter_gesture_tester', 'static_gesture_runtime/flutter'):
        assert (ROOT / folder / 'pubspec.lock').read_bytes() == (tested_root / folder / 'pubspec.lock').read_bytes(), folder
    # Check against the recorded release source hashes as well as the actual successful test inputs.
    for name, sha in prior['source_sha256'].items():
        if name.endswith('.dart'):
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    return dict(reused=True, previous_version=prior['version'], report_dir=str(report),
                apk=str(apk), apk_sha256=prior['sha256'], dart_source_sha256=hashes,
                host_checks=checks, note='Dart code/tests/dependencies identical; current native code is tested separately')
