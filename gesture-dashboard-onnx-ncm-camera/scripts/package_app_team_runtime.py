"""Build a standalone static_gesture_runtime handover without the tester app.

Runtime inputs come from the immutable, qualified ownership/preview archive.
Only packaging/documentation changes. Every executable input is checked against
that archive before publication, and the extracted model fixtures are executed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'handover_static/app_team_docs'
BASELINE_SHA = 'f18ece4b77ca2a09514a1bf00c2a7ec1e673171c43c3dff3eda6b05abf7315df'
PREFIX = 'static_gesture_runtime/'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode('utf-8')


def versions(lock: bytes) -> dict[str, str]:
    result = {}
    package = None
    for line in lock.decode('utf-8').splitlines():
        match = re.fullmatch(r'  ([a-zA-Z0-9_]+):', line)
        if match:
            package = match[1]
        match = re.fullmatch(r'    version: "([^"]+)"', line)
        if match and package:
            result[package] = match[1]
    return result


def write_payload(destination: Path, payload: dict[str, bytes]) -> None:
    for name, data in payload.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
            raise ValueError('Unsafe output entry: ' + name)
        target = destination.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'handover_static/Boss_Static_Gesture_Handover_20261114_ownership_preview.zip')
    parser.add_argument('--output', type=Path, default=ROOT / 'handover_static/static_gesture_runtime_20261114_app_team_final.zip')
    parser.add_argument('--folder', type=Path, default=ROOT / 'handover_static/app_team_final_20261114/static_gesture_runtime')
    parser.add_argument('--model-python', type=Path, default=ROOT / '.runtime/static-conversion-venv/Scripts/python.exe')
    args = parser.parse_args()
    output, folder = args.output.resolve(), args.folder.resolve()
    if not output.is_relative_to(ROOT) or not folder.is_relative_to(ROOT):
        raise ValueError('Delivery must stay inside workspace')
    if output.exists() or output.with_suffix('.zip.sha256').exists() or folder.exists():
        raise FileExistsError('Use fresh delivery paths; previous releases remain immutable')
    if digest(args.source.read_bytes()) != BASELINE_SHA:
        raise ValueError('The source is not the qualified ownership/preview archive')
    payload: dict[str, bytes] = {}
    original: dict[str, bytes] = {}
    helper_origins = {}
    with zipfile.ZipFile(args.source) as archive:
        roots = [name.removesuffix('SHA256SUMS.json') for name in archive.namelist()
                 if name.count('/') == 1 and name.endswith('/SHA256SUMS.json')]
        if len(roots) != 1:
            raise ValueError('Expected one source archive root')
        base = roots[0]
        sums = json.loads(archive.read(base + 'SHA256SUMS.json'))
        for relative, expected in sums.items():
            data = archive.read(base + relative)
            if digest(data) != expected:
                raise ValueError('Source checksum failed: ' + relative)
            original[relative] = data
        for relative, data in original.items():
            if relative.startswith(PREFIX):
                name = relative.removeprefix(PREFIX)
                if name not in ('README.md', 'SHA256SUMS.json'):
                    payload[name] = data
        for name in ('phone_static_gesture_engine.dart', 'phone_frame_gate.dart'):
            source = 'flutter_gesture_tester/lib/services/' + name
            target = 'integration/phone_camera/' + name
            payload[target] = original[source]
            helper_origins[target] = source
        for name in ('gesture_mlp_production.onnx', 'gesture_mlp_onnx_metadata.json',
                     'gesture_mobile_runtime_config.json', 'gesture_artifact_manifest.json'):
            payload['reference/source_model/' + name] = original['source_model/' + name]
        for name in ('HOST_CHECKS.json', 'RELEASE_NOTES.md'):
            payload['reference/qualified_release/' + name] = original[name]
        payload['DEVICE_ACCEPTANCE.md'] = original['DEVICE_ACCEPTANCE.md']
        payload['reference/qualified_release/phone_camera_host.pubspec.lock'] = original['flutter_gesture_tester/pubspec.lock']

    # Documentation additions may not overwrite a runtime input.
    for name in ('START_HERE.md', 'README.md', 'INTEGRATION_GUIDE.md', 'MODEL_GUIDE.md',
                 'PERFORMANCE.md', 'verify_handover.py'):
        if name in payload:
            raise ValueError('Documentation would overwrite an input: ' + name)
        payload[name] = (DOCS / name).read_bytes()
    for path in sorted((DOCS / 'component_readmes').rglob('README.md')):
        name = 'component_readmes/' + path.relative_to(DOCS / 'component_readmes').as_posix()
        if name in payload:
            raise ValueError('Documentation would overwrite an input: ' + name)
        payload[name] = path.read_bytes()
    payload['integration/phone_camera/README.md'] = (DOCS / 'PHONE_CAMERA.md').read_bytes()
    payload['reference/source_model/README.md'] = (
        '# Matching mobile source snapshot\n\n'
        'These ONNX/configuration/metadata files identify the exact source of the '
        'delivered FP32 TFLite model. They are reference files, not app dependencies. '
        'The app uses flutter/assets/models/gesture_static_float32.tflite through '
        'the complete runtime. No conversion or training is required.\n'
    ).encode('utf-8')
    payload['reference/qualified_release/README.md'] = (
        '# Evidence from the qualified executable release\n\n'
        'HOST_CHECKS.json and RELEASE_NOTES.md are immutable evidence from the '
        '14 November 2026 ownership/preview delivery. They mention the historical '
        'tester app/APK and original repository paths because that build tested '
        'the shared runtime. The tester app/APK is deliberately not included here. '
        'No tester installation is required for integration.\n\n'
        'All flutter/ files and the two optional phone helpers in this package '
        'are byte-identical to that checked release. The main integration guide '
        'describes the simplified folder layout. The dependency inventory and '
        'phone-camera host lockfile are references for merging into an existing '
        'app; the consuming app must review its own resolved dependency lock.\n'
    ).encode('utf-8')
    dependencies = {
        'runtime': versions(payload['flutter/pubspec.lock']),
        'ncm_example': versions(payload['flutter/example/pubspec.lock']),
        'phone_camera_host': versions(payload['reference/qualified_release/phone_camera_host.pubspec.lock']),
    }
    payload['reference/qualified_release/dependency_versions.json'] = encoded(dependencies)

    runtime_hashes = {name: digest(data) for name, data in sorted(payload.items())
                      if name.startswith('flutter/')}
    for name, expected in runtime_hashes.items():
        if digest(original[PREFIX + name]) != expected:
            raise ValueError('Executable runtime changed: ' + name)
    original_runtime_names = {name.removeprefix(PREFIX) for name in original
                              if name.startswith(PREFIX + 'flutter/')}
    if runtime_hashes.keys() != original_runtime_names:
        raise ValueError('Executable runtime input removed or added')
    helper_hashes = {target: digest(payload[target]) for target in helper_origins}
    original_checks = json.loads(original['HOST_CHECKS.json'])
    preserved = dict(
        passed=True, release_revision=folder.parent.name,
        qualified_source_archive=args.source.name, qualified_source_archive_sha256=BASELINE_SHA,
        runtime_flutter_sha256=runtime_hashes, phone_helper_sha256=helper_hashes,
        phone_helper_source_paths=helper_origins, executable_files_changed=0,
        model_assets_changed=False, dependency_specifications_changed=False,
        dependency_lockfiles_changed=False, detector_settings_changed=False,
        feature_geometry_rules_changed=False, ownership_and_confirmation_changed=False,
        frame_admission_and_scheduling_changed=False, additional_inference_passes=0,
        tester_application_included=False,
        previous_release_checks=original_checks['flutter_tests'],
        scope='Packaging-only comparison; existing executable qualification is retained. Host app device integration/performance still requires acceptance.',
    )
    payload['validation/app_team_package_preservation.json'] = encoded(preserved)
    model_files = {
        'gesture_static_float32': 'flutter/assets/models/gesture_static_float32.tflite',
        'gesture_static_int8': 'flutter/assets/models/gesture_static_int8.tflite',
        **{name: 'flutter/third_party/hand_detection/assets/models/' + name + '.tflite'
           for name in ('hand_detection', 'hand_landmark_full', 'blaze_face_short_range')},
    }
    manifest = dict(
        schema_version=1, release=output.stem,
        assembled_at_utc=datetime.now(timezone.utc).isoformat(),
        source_archive=dict(name=args.source.name, sha256=BASELINE_SHA),
        entry_guide='START_HERE.md', flutter_path_dependency='flutter/',
        default_model='gesture_static_float32',
        models={name: dict(path=relative, sha256=digest(payload[relative]), bytes=len(payload[relative]))
                for name, relative in model_files.items()},
        runtime_code_changed=False, additional_inference_passes=0,
        tester_app_included=False, apk_included=False,
        future_motion_and_ghost_tuning_included=False,
        physical_ios_tested=False, ios_compile_tested=False,
    )

    # Execute the unchanged portable model verifier in the extracted layout.
    # That checks native LiteRT fixture outputs, not just file presence/hashes.
    (ROOT / '.runtime').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='app-team-package-', dir=ROOT / '.runtime') as temporary:
        temp = Path(temporary)
        extracted = temp / 'static_gesture_runtime'
        write_payload(extracted, payload)
        run = subprocess.run([str(args.model_python.resolve()),
                              str(extracted / 'tools/verify_static_runtime.py'),
                              '--package-root', str(extracted)],
                             cwd=ROOT, capture_output=True, text=True, check=True)
        model_check = json.loads(run.stdout[run.stdout.index('{'):])
        if not model_check['passed']:
            raise ValueError('Native model fixture verification failed')
        payload['validation/app_team_model_verification.json'] = encoded(model_check)
        payload['release_manifest.json'] = encoded(manifest)
        payload['HOST_CHECKS.json'] = encoded(dict(
            status='passed', scope='Standalone package preservation, extracted native model fixtures and payload integrity; executable code unchanged from qualified release.',
            qualified_runtime_tests_passed=original_checks['flutter_tests']['runtime_passed'],
            prior_qualification_reference='reference/qualified_release/HOST_CHECKS.json',
            package_preservation='validation/app_team_package_preservation.json',
            packaged_native_model_verification=model_check,
            tester_app_included=False, physical_ios_tested=False, ios_compile_tested=False,
        ))
        payload['SHA256SUMS.json'] = encoded({name: digest(data) for name, data in sorted(payload.items())})
        write_payload(extracted, payload)
        verification = subprocess.run([sys.executable, str(extracted / 'verify_handover.py')],
                                      cwd=extracted, capture_output=True, text=True, check=True)
        verified = json.loads(verification.stdout)
        if not verified['passed']:
            raise ValueError('Extracted payload verification failed')
        pending = temp / 'delivery.zip'
        with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
            for name, data in sorted(payload.items()):
                archive.writestr(PREFIX + name, data)
        with zipfile.ZipFile(pending) as archive:
            if archive.testzip() is not None:
                raise ValueError('ZIP CRC check failed')
            for name, data in payload.items():
                if archive.read(PREFIX + name) != data:
                    raise ValueError('ZIP payload differs: ' + name)
        folder.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(folder))
        output.parent.mkdir(parents=True, exist_ok=True)
        pending.replace(output)
    archive_sha = digest(output.read_bytes())
    output.with_suffix('.zip.sha256').write_text(archive_sha + '  ' + output.name + '\n', encoding='utf-8')
    print(json.dumps(dict(archive=str(output.relative_to(ROOT)), folder=str(folder.relative_to(ROOT)),
                         bytes=output.stat().st_size, sha256=archive_sha,
                         extracted_verification=verified, native_model_verification=model_check), indent=2))


if __name__ == '__main__':
    main()
