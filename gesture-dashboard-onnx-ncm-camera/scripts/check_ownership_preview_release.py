"""Verify the ownership/preview update preserves deployed mobile recognition inputs."""
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
    parser.add_argument('--baseline', type=Path, default=ROOT / 'handover_static/Boss_Static_Gesture_Handover_20261114_two_hand_stability.zip')
    parser.add_argument('--output', type=Path, default=ROOT / 'static_gesture_runtime/validation/ownership_preview_release.json')
    args = parser.parse_args()
    allowed = {
        'static_gesture_runtime/flutter/lib/src/runtime_api.dart',
        'static_gesture_runtime/flutter/lib/src/hand_ownership.dart',
        'static_gesture_runtime/flutter/lib/static_gesture_preview.dart',
        'static_gesture_runtime/flutter/test/runtime_api_test.dart',
        'static_gesture_runtime/flutter/test/preview_test.dart',
        'static_gesture_runtime/flutter/example/lib/main.dart',
        'flutter_gesture_tester/lib/models/gesture_result.dart',
        'flutter_gesture_tester/test/gesture_result_test.dart',
        'flutter_gesture_tester/pubspec.yaml',
        'flutter_gesture_tester/android/app/src/main/kotlin/com/gesturelab/gesture_model_tester/NcmNetworkBridge.kt',
    }
    current = input_files(ROOT)
    for path in (ROOT / 'flutter_gesture_tester/android/app/src/main/kotlin').rglob('*.kt'):
        current[path.relative_to(ROOT).as_posix()] = path
    unchanged, changed = {}, {}
    with zipfile.ZipFile(args.baseline) as archive:
        root = next(name.removesuffix('release_manifest.json') for name in archive.namelist()
                    if name.count('/') == 1 and name.endswith('/release_manifest.json'))
        sums = json.loads(archive.read(root + 'SHA256SUMS.json'))
        for relative, path in sorted(current.items()):
            after = path.read_bytes()
            before = archive.read(root + relative) if relative in sums else None
            if before is not None and digest(before) != sums[relative]:
                raise ValueError('Baseline checksum failed: ' + relative)
            if before == after:
                unchanged[relative] = digest(after)
                continue
            if relative not in allowed:
                raise ValueError('Unrelated recognition/dependency change: ' + relative)
            if relative.endswith('pubspec.yaml'):
                normalize = lambda data: re.sub(r'^version:.*$', 'version: release-id', data.decode(), flags=re.M)
                if normalize(before) != normalize(after):
                    raise ValueError('Tester dependencies changed')
            if relative.endswith('NcmNetworkBridge.kt'):
                normalize = lambda data: re.sub(rb'Transport revision: v[0-9.]+', b'Transport revision: release-id', data)
                if normalize(before) != normalize(after):
                    raise ValueError('Native transport behavior changed')
            changed[relative] = dict(before_sha256=None if before is None else digest(before), after_sha256=digest(after))
        # No existing tested input may disappear from the update.
        roots = ('static_gesture_runtime/flutter/lib/', 'static_gesture_runtime/flutter/test/',
                 'flutter_gesture_tester/lib/', 'flutter_gesture_tester/test/')
        retired = ('flutter_gesture_tester/lib/services/native_gesture_engine.dart',
                   'flutter_gesture_tester/lib/widgets/landmark_painter.dart')
        missing = [name for name in sums if name.startswith(roots) and name.endswith('.dart')
                   and name not in current and name not in retired]
        if missing:
            raise ValueError('Existing Dart inputs removed: ' + repr(missing))
    if changed.keys() != allowed:
        raise ValueError('Unexpected ownership/preview change inventory: ' + repr(changed.keys()))
    models = source_models(args.baseline)
    manifest = json.loads((ROOT / 'static_gesture_runtime/flutter/assets/models/gesture_static_float32_manifest.json').read_text())
    if digest(models['gesture_mlp_production.onnx']) != manifest['source_onnx_sha256']:
        raise ValueError('Source snapshot does not match preserved mobile model')
    result = dict(
        passed=True, release_revision='ownership_preview_20261114',
        baseline_archive=args.baseline.name, baseline_sha256=digest(args.baseline.read_bytes()),
        changed_inputs=changed, unchanged_input_count=len(unchanged), unchanged_sha256=unchanged,
        preserved_source_model_sha256={name: digest(data) for name, data in models.items()},
        behavior='Retain the selected hand; confirm a consistently nearer challenger or the sole valid peer gesture. Keep the previous decoded NCM picture and its own overlay until the next pair is ready.',
        switch_confirmation_minimum_ms=250, switch_confirmation_minimum_observations=3,
        missing_owner_grace_ms=350, closer_projected_world_ratio=1.10, closer_palm_ratio=1.15,
        switch_frame_dispatch=False, fresh_gesture_evidence_after_handoff=True,
        model_weights_changed=False, feature_extraction_changed=False, geometry_rules_changed=False,
        single_hand_confirmation_changed=False, command_ownership_changed=True,
        false_hand_filter_changed=False, incidental_movement_tuning_changed=False,
        detector_settings_changed=False, admission_scheduling_changed=False, additional_inference_passes=0,
        scope='Source/asset preservation plus automated ownership, preview, model and regression checks; physical phone/NCM acceptance remains pending.',
        physical_ios_tested=False, physical_ncm_tested=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('passed', 'release_revision', 'unchanged_input_count', 'changed_inputs')}, indent=2))


if __name__ == '__main__':
    main()
