"""Verify that the mobile tracking update preserves its deployed model and rules."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

from finalize_current_model_release import input_files
from package_boss_handover import ROOT, source_models


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'handover_static/Boss_Static_Gesture_Handover_20261111_feedback_refresh.zip')
    parser.add_argument('--output', type=Path, default=ROOT / 'static_gesture_runtime/validation/two_hand_stability_release.json')
    args = parser.parse_args()
    allowed = {
        'static_gesture_runtime/flutter/lib/src/detector_adapter.dart',
        'static_gesture_runtime/flutter/test/detector_adapter_test.dart',
        'flutter_gesture_tester/pubspec.yaml',
        'flutter_gesture_tester/android/app/src/main/kotlin/com/gesturelab/gesture_model_tester/NcmNetworkBridge.kt',
    }
    current = input_files(ROOT)
    for path in (ROOT / 'flutter_gesture_tester/android/app/src/main/kotlin').rglob('*.kt'):
        current[path.relative_to(ROOT).as_posix()] = path
    unchanged, changed = {}, {}
    with zipfile.ZipFile(args.baseline) as archive:
        manifest = next(name for name in archive.namelist()
                        if name.count('/') == 1 and name.endswith('/release_manifest.json'))
        prefix = manifest.removesuffix('release_manifest.json')
        sums = json.loads(archive.read(prefix + 'SHA256SUMS.json'))
        for relative, path in sorted(current.items()):
            before, after = archive.read(prefix + relative), path.read_bytes()
            if digest(before) != sums[relative]:
                raise ValueError('Baseline checksum failed: ' + relative)
            if before == after:
                unchanged[relative] = digest(after)
                continue
            if relative not in allowed:
                raise ValueError('Unrelated model/rule/dependency change: ' + relative)
            if relative.endswith('pubspec.yaml'):
                normalize = lambda data: re.sub(r'^version:.*$', 'version: release-id', data.decode(), flags=re.M)
                if normalize(before) != normalize(after):
                    raise ValueError('Tester dependencies changed')
            if relative.endswith('NcmNetworkBridge.kt'):
                normalize = lambda data: re.sub(rb'Transport revision: v[0-9.]+', b'Transport revision: release-id', data)
                if normalize(before) != normalize(after):
                    raise ValueError('Native transport behavior changed')
            changed[relative] = dict(before_sha256=digest(before), after_sha256=digest(after))
    if changed.keys() != allowed:
        raise ValueError('Unexpected tracking release change inventory')
    models = source_models(args.baseline)
    static_manifest = json.loads((ROOT / 'static_gesture_runtime/flutter/assets/models/gesture_static_float32_manifest.json').read_text())
    if digest(models['gesture_mlp_production.onnx']) != static_manifest['source_onnx_sha256']:
        raise ValueError('Source snapshot does not match the preserved mobile model')
    result = dict(
        passed=True, release_revision='two_hand_stability_20261114',
        baseline_archive=args.baseline.name, baseline_sha256=digest(args.baseline.read_bytes()),
        changed_inputs=changed, unchanged_input_count=len(unchanged), unchanged_sha256=unchanged,
        preserved_source_model_sha256={name: digest(data) for name, data in models.items()},
        behavior='Established, spatially separated hand tracks retain their confirmed physical side during weak handedness observations; current frame landmarks remain required.',
        new_identity_minimum_handedness_confidence=.75, track_expiry_ms=350,
        model_weights_changed=False, feature_extraction_changed=False,
        geometry_rules_changed=False, command_timing_changed=False,
        detector_settings_changed=False, additional_inference_passes=0,
        scope='Source/asset preservation and synthetic tracking regressions; live phone/NCM improvement requires device acceptance.',
        physical_ios_tested=False, physical_ncm_tested=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('passed', 'release_revision', 'unchanged_input_count', 'changed_inputs', 'preserved_source_model_sha256')}, indent=2))


if __name__ == '__main__':
    main()
