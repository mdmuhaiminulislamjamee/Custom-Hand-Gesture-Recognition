"""Build two immutable current mobile deliveries from one verified desktop snapshot."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_mobile_handover import MODELS, digest, verify, verify_onnx

QUALIFICATION_FILES = [
    'data/ten_gesture/hagrid_train.npz', 'data/ten_gesture/sources.json',
    'data/stop_inverted/hagrid_train.npz', 'data/stop_inverted/sources.json',
    'artifacts/ten_gesture/public_test.npz',
    'artifacts/twelve_gesture/candidate_contract.json', 'artifacts/twelve_gesture/evaluation.json',
    'artifacts/twelve_gesture/runtime_comparison.json',
    'artifacts/stop_inverted/train.npz', 'artifacts/stop_inverted/val.npz', 'artifacts/stop_inverted/test.npz',
    'artifacts/stop_inverted/data_audit.json', 'artifacts/stop_inverted/evaluation.json',
    'artifacts/stop_inverted/candidate.onnx',
    'artifacts/stop_inverted/reverse_palm_review/overlay_traces.json',
    'artifacts/left_video_review/overlay_traces.json',
    'artifacts/directional_feedback_20261027/evaluation.json',
]
ROOT_FILES = ['package.json', 'pnpm-lock.yaml', 'pnpm-workspace.yaml', 'requirements.txt',
              'requirements.lock.txt', 'requirements-handover-checks.txt', 'pytest.ini',
              'tsconfig.json', 'eslint.config.mjs', 'next-env.d.ts', 'next.config.ts',
              'vite.config.ts', 'vite.local.config.ts', '.env.example',
              'MODEL_CARD.md', 'DATA_COLLECTION.md', 'HAND_DISTANCE.md', 'MULTIVIEW_TRAINING.md']
CACHES = ['gesture_artifact_manifest.json', 'gesture_online_replay_cache.npz',
          'gesture_online_validation_cache.npz', 'gesture_online_untouched_test_cache.npz']
IOS_ADAPTERS = ['GestureNCMClient.swift', 'GestureHandLandmarker.swift', 'GestureONNXClassifier.swift',
                'GestureONNXParity.swift', 'GestureDiagnosticEngine.swift', 'Gesture-Bridging-Header.h',
                'Podfile.example', 'Info.plist.fragment']
LEGACY = ['GestureGeometryResolver.swift', 'GestureTemporalGate.swift',
          'GestureActionLatch.swift', 'GestureDecisionPipeline.swift']


def copy(source, destination):
    source = Path(source)
    if not source.is_file() or source.is_symlink():
        raise ValueError('Missing/linked source: ' + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def tree(source, destination, suffixes):
    for file in sorted(source.rglob('*')):
        relative = file.relative_to(source)
        if file.is_file() and file.suffix in suffixes and not any(p.startswith('.') or p == '__pycache__' for p in relative.parts):
            copy(file, destination / relative)


def archive(source, destination, files=None):
    with ZipFile(destination, 'x', ZIP_DEFLATED, compresslevel=6) as output:
        for file in sorted(source.rglob('*') if files is None else files):
            if file.is_file():
                output.write(file, source.name + '/' + file.relative_to(source).as_posix())
    with ZipFile(destination) as output:
        if output.testzip() is not None:
            raise ValueError('ZIP CRC failed: ' + str(destination))


def write_json(file, value):
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run(command, cwd, log):
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    environment.pop('GESTURE_MODELS_DIR', None)
    environment.pop('GESTURE_FEEDBACK_DIR', None)
    log.parent.mkdir(parents=True, exist_ok=True)
    print('Checking: ' + ' '.join(str(value) for value in command), flush=True)
    with log.open('w', encoding='utf-8') as output:
        result = subprocess.run(command, cwd=cwd, env=environment, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text(encoding='utf-8', errors='replace')[-6000:], flush=True)
        raise RuntimeError('Check failed; see ' + str(log))


def readme(platform, release, metadata, vectors):
    ios = platform == 'ios'
    title, directory = ('iOS', 'ios') if ios else ('Android', 'android')
    integration = 'IOS_INTEGRATION.md' if ios else 'ANDROID_INTEGRATION.md'
    native = ('`ios/swift/`: current feature/angle sources, NCM/MediaPipe/ONNX diagnostic adapters and Mac tests'
              if ios else '`android/kotlin/`: current NCM, network selection, two-hand MediaPipe, 76-feature, 13-output ONNX, angle and parity sources')
    rows = '\n'.join(f'| {index} | `{name}` | {"Disabled — no action" if name == "dorsal" else metadata["actions"][name]} |'
                     for index, name in enumerate(metadata['output_class_order']))
    return f'''# {title} app-team handover

Release **{release}**, prepared {datetime.now(timezone.utc).date()} (Asia/Dhaka).

Use this complete package with the existing {title} app. The input is the USB-NCM board's JLIP/JPEG stream; inference runs on the phone. This is an integration handover, not a signed app, complete native project or qualified device build.

**Current status:** the delivered desktop reference passes its isolated software tests and model/fixture checks. Native model adapters and diagnostic sources are supplied. The older native decision implementation does not contain all current Python behavior, so it is separated into `reference/legacy_swift/`. Full current native decision/frame parity, native compilation and physical USB-NCM acceptance remain app-team tasks. Follow [RUNTIME_PORTING.md](RUNTIME_PORTING.md) before enabling actions.

## Start here

1. Extract the entire ZIP into a writable directory; retain the companion `.zip.sha256` file.
2. Run `python verify_package.py` with Python 3.11/3.12 before editing delivered files. This checks all payload hashes, the 13-output contract, angles, source lineage and matching desktop/mobile assets using the standard library.
3. With NumPy and ONNX Runtime installed, run `python verify_package.py --onnx` to compare all {len(vectors['model_cases'])} model vectors. Use the included Python dependency files if needed.
4. Extract `desktop_reference.zip` separately and follow [WINDOWS_RUN.md](WINDOWS_RUN.md) to reproduce the Windows NCM/BAT workflow.
5. Follow [{integration}]({integration}), then complete [VALIDATION.md](VALIDATION.md).

Model: **{metadata['model_name']}**. SHA-256:

```text
{metadata['onnx_sha256']}
```

Input is float32 `landmark_features[1,76]`; outputs are `probabilities[1,13]` and `known_gesture_mass[1,1]`. There are **12 enabled commands**. Keep all 13 tensor indices, including the disabled Dorsal output. `no_gesture` means rejection/no action; it is not a fourteenth output. Copying the ONNX file alone does not copy current behavior.

## Folder and file guide

| Path | Purpose | In the native app? |
|---|---|---|
| `README.md` | Start, identity, file guide and action contract | No |
| `{integration}` | Platform resources, dependencies, NCM, diagnostics, parity and remaining port work | No; follow it |
| `RUNTIME_PORTING.md` | Current image/tensor/geometry/selection/temporal/action behavior | No; implementation guide |
| `NCM_PROTOCOL.md` | Network route, 24-byte JLIP header, CRC/JPEG, heartbeat and reconnect | No; protocol guide |
| `VALIDATION.md` | Host evidence and physical-device acceptance record | No |
| `CHANGES_SINCE_PREVIOUS.md` | Differences from the October 8 packages | No |
| `MODEL_CARD.md` | Current model provenance, performance scope and limitations | No |
| `MOBILE_RUNTIME_CONTRACT.json` | Machine-readable current taxonomy, actions, features, runtime JSON, angle limits and source identity | Keep as engineering contract |
| `RELEASE_MANIFEST.json`, `SHA256SUMS.json` | Exact snapshot, fixture counts, readiness flags and per-file hashes | No; retain with delivery |
| `verify_package.py` | Read-only checksum/contract and optional ONNX checks | No |
| `verify_reference_fixtures.py` | Re-evaluate vectors against an extracted Python desktop snapshot | No |
| `validation/` | Isolated backend test logs/summary, preflight, fixture and frontend evidence | No |
| `desktop_reference.zip` | Complete runnable current Windows source/model/lockfile/qualification snapshot, identical in both mobile bundles | No |
| `{directory}/models/` | Required ONNX/metadata/runtime/hand-landmarker resources; legacy unused face asset also retained | Required assets, except unused face model |
| `{directory}/tests/parity_vectors.json` | {len(vectors['geometry_cases'])} geometry/feature cases, {len(vectors['temporal_transition_cases'])} temporal transition sequences and {len(vectors['model_cases'])} ONNX cases | Debug/test resources |
| `{directory}/reference/backend/` | Current Python behavior and regression sources | No; porting oracle |
| `{directory}/reference/legacy_swift/` | Older resolver/gate/latch/pipeline, explicitly not qualified for this release | No |
| `{directory}/reference/protocol/` | C++ board receiver reference | No |

{native}.

## Required current behavior

- **Open Palm:** image XY **−160° to −20°**, inclusive (up ±70°), palm visible.
- **Stop inverted:** image XY **−145° to −35°**, inclusive (up ±55°), back of hand visible, fingers together or spread.
- Outside the command's band is immediately no gesture; return requires a fresh hold. Use the wrist-to-knuckle source-pixel angle, not the HUD world-plane angle.
- Analysis remains unmirrored. Positive index x resolves to Left; negative x resolves to Right. Preserve matching image dimensions and two-hand ownership.
- Preserve strict low-mass Peace recovery and all geometry/temporal/latch checks. A raw score or angle alone never dispatches an action.
- Face/body/clothing positions do not veto an otherwise valid hand gesture in the current policy.

## Fixed probability order and action contract

| Index | Output | Action |
|---:|---|---|
{rows}

## Verification and scope

For Python reference parity after extracting the desktop ZIP:

```text
python verify_reference_fixtures.py {directory}/tests/parity_vectors.json --reference-root /path/to/desktop_reference
```

That command needs the included reference Python dependencies. Native feature/angle/ONNX helpers are documented in the platform guide; the app team must implement and compare the full geometry/temporal/action outputs and all transition sequences as well.

The model was not retrained for packaging. Both platforms receive the same current approved source/model/configuration snapshot. Previous deliveries are unchanged. Local environments, node_modules, credentials, generated caches, feedback photos, private videos and mutable online-adapter state are excluded. Derived/public landmark qualification inputs are included so the desktop tests run; this is not a complete retraining corpus. Native builds, full decision-port completion and physical-device accuracy are not claimed.
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default=datetime.now(timezone.utc).strftime('%Y.%m.%d'))
    parser.add_argument('--sequence', default='01')
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--frontend-log', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'\d{4}\.\d{2}\.\d{2}', args.date) or not re.fullmatch(r'\d{2}', args.sequence):
        raise ValueError('Use YYYY.MM.DD and a two-digit sequence')
    versions = dict(ios=f'v{args.date}_ios_ncm_final_{args.sequence}', android=f'v{args.date}_android_ncm_{args.sequence}')
    destinations = {}
    for platform, version in versions.items():
        parent = ROOT / ('handover_final' if platform == 'ios' else 'handover_android')
        prefix = 'gesture-final-handover-' if platform == 'ios' else 'gesture-android-handover-'
        final, zipped = parent / version, parent / (prefix + version + '.zip')
        if any(p.exists() for p in (final, zipped, zipped.with_suffix('.zip.sha256'))):
            raise FileExistsError('Release exists; choose a new sequence. Existing deliveries are immutable.')
        destinations[platform] = (final, zipped)
    vectors = json.loads(args.fixtures.read_text(encoding='utf-8'))
    for name, expected in vectors['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Regenerate fixtures after source changes: ' + name)
    metadata = json.loads((ROOT / 'models/gesture_mlp_onnx_metadata.json').read_text(encoding='utf-8'))
    runtime = json.loads((ROOT / 'models/gesture_mobile_runtime_config.json').read_text(encoding='utf-8'))
    if vectors['model_sha256'] != digest(ROOT / 'models/gesture_mlp_production.onnx'):
        raise ValueError('Fixture model mismatch')
    stage = ROOT / '.runtime' / ('mobile-handover-build-' + uuid.uuid4().hex)
    desktop, checks = stage / 'desktop_reference', stage / 'validation'
    desktop.mkdir(parents=True)
    checks.mkdir()
    print('STAGE=' + str(stage), flush=True)
    for directory, suffixes in [('backend', {'.py'}), ('scripts', {'.py', '.ps1'}), ('research', {'.py'}),
                                ('app', {'.ts', '.tsx', '.css', '.mjs'}), ('public', {'.svg', '.png'}),
                                ('reference', {'.cpp', '.h'}), ('examples', {'.py'}),
                                ('swift', {'.swift', '.h', '.example', '.fragment', '.json', '.sh', '.md'}),
                                ('android_handover_src', {'.kt', '.xml', '.fragment', '.md'}), ('handover_docs', {'.md'})]:
        tree(ROOT / directory, desktop / directory, suffixes)
    for name in ROOT_FILES + QUALIFICATION_FILES:
        copy(ROOT / name, desktop / name)
    for file in ROOT.glob('*.bat'):
        copy(file, desktop / file.name)
    for name in ['package-lock.json', 'LICENSE', 'LICENSE.md']:
        if (ROOT / name).is_file():
            copy(ROOT / name, desktop / name)
    for name in MODELS + CACHES:
        copy(ROOT / 'models' / name, desktop / 'models' / name)
    copy(ROOT / 'handover_docs/mobile/WINDOWS_RUN.md', desktop / 'README.md')
    copy(args.fixtures, desktop / 'swift/tests/parity_vectors.json')
    contract = dict(schema_version=1, model_sha256=metadata['onnx_sha256'], class_names=runtime['class_names'],
                    active_class_names=[name for name in runtime['class_names'] if name not in runtime['disabled_gestures']],
                    disabled_gestures=runtime['disabled_gestures'], gesture_to_action=runtime['gesture_to_action'],
                    feature_names=runtime['feature_names'], input_shape=[1, 76], probability_shape=[1, 13],
                    known_gesture_mass_shape=[1, 1], reject_label='no_gesture', runtime_config=runtime,
                    source_sha256=vectors['source_sha256'], face_body_vetoes_enabled=False,
                    full_native_decision_port_complete=False)
    write_json(desktop / 'MOBILE_RUNTIME_CONTRACT.json', contract)
    inventory = {file.relative_to(desktop).as_posix(): digest(file)
                 for file in sorted(desktop.rglob('*')) if file.is_file()}
    write_json(desktop / 'SOURCE_SNAPSHOT.json', inventory)
    desktop_files = [desktop / name for name in inventory] + [desktop / 'SOURCE_SNAPSHOT.json']
    run([sys.executable, '-m', 'scripts.onnx_preflight'], desktop, checks / 'onnx_preflight.log')
    run([sys.executable, str(ROOT / 'scripts/verify_mobile_fixtures.py'), str(args.fixtures.resolve()),
         '--reference-root', str(desktop)], desktop, checks / 'fixture_reference.log')
    run([sys.executable, str(desktop / 'scripts/verify_backend_suite.py'), '--output', str(checks / 'backend')],
        desktop, checks / 'backend_suite.log')
    summary = json.loads((checks / 'backend/summary.json').read_text(encoding='utf-8'))
    if summary['failed_files'] or summary['completed'] != summary['files']:
        raise ValueError('Incomplete isolated desktop tests')
    frontend_bytes = args.frontend_log.read_bytes()
    frontend_text = frontend_bytes.decode('utf-16' if frontend_bytes.startswith(b'\xff\xfe') else 'utf-8-sig', errors='replace')
    if 'Build complete' not in frontend_text:
        raise ValueError('Frontend evidence does not contain a completed production build')
    (checks / 'frontend_build.log').write_text(frontend_text, encoding='utf-8')
    for name, expected in inventory.items():
        if digest(desktop / name) != expected:
            raise ValueError('Verification changed a frozen source artifact: ' + name)
    desktop_zip = stage / 'desktop_reference.zip'
    archive(desktop, desktop_zip, desktop_files)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    source_hashes = {file.relative_to(desktop).as_posix(): digest(file)
                     for file in sorted((desktop / 'backend').rglob('*.py')) if '__pycache__' not in file.parts}
    reports = []
    for platform, version in versions.items():
        package = stage / version
        package.mkdir()
        for name in MODELS:
            copy(desktop / 'models' / name, package / platform / 'models' / name)
        tree(desktop / 'backend', package / platform / 'reference/backend', {'.py'})
        for name in LEGACY:
            copy(ROOT / 'swift' / name, package / platform / 'reference/legacy_swift' / name)
        copy(ROOT / 'swift/ios/GestureLandmarkEngine.swift', package / platform / 'reference/legacy_swift/GestureLandmarkEngine.swift')
        (package / platform / 'reference/legacy_swift/README.md').write_text(
            'Older decision implementation. Not in parity with this release; reference only. '
            'Port the current Python snapshot using RUNTIME_PORTING.md before app actions.\n', encoding='utf-8')
        copy(ROOT / 'reference/NcmCameraTester.cpp', package / platform / 'reference/protocol/NcmCameraTester.cpp')
        copy(args.fixtures, package / platform / 'tests/parity_vectors.json')
        if platform == 'ios':
            for name in ['GestureFeatureExtractor.swift', 'GestureRuntimePolicy.swift']:
                copy(ROOT / 'swift' / name, package / 'ios/swift' / name)
            for name in IOS_ADAPTERS:
                copy(ROOT / 'swift/ios' / name, package / 'ios/swift/ios' / name)
            for name in ['mobile_features.swift', 'verify_features.sh']:
                copy(ROOT / 'swift/tests' / name, package / 'ios/swift/tests' / name)
        else:
            tree(ROOT / 'android_handover_src/kotlin', package / 'android/kotlin', {'.kt'})
            for name in ['AndroidManifest.fragment.xml', 'app-build.gradle.kts.fragment']:
                copy(ROOT / 'android_handover_src' / name, package / 'android' / name)
        for name in ['RUNTIME_PORTING.md', 'NCM_PROTOCOL.md', 'VALIDATION.md', 'WINDOWS_RUN.md', 'CHANGES_SINCE_PREVIOUS.md',
                     'IOS_INTEGRATION.md' if platform == 'ios' else 'ANDROID_INTEGRATION.md']:
            copy(ROOT / 'handover_docs/mobile' / name, package / name)
        copy(ROOT / 'MODEL_CARD.md', package / 'MODEL_CARD.md')
        copy(ROOT / 'scripts/verify_mobile_handover.py', package / 'verify_package.py')
        copy(ROOT / 'scripts/verify_mobile_fixtures.py', package / 'verify_reference_fixtures.py')
        copy(desktop_zip, package / 'desktop_reference.zip')
        tree(checks, package / 'validation', {'.log', '.json'})
        write_json(package / 'MOBILE_RUNTIME_CONTRACT.json', contract)
        doc_metadata = dict(metadata, actions=runtime['gesture_to_action'])
        (package / 'README.md').write_text(readme(platform, version, doc_metadata, vectors), encoding='utf-8')
        manifest = dict(schema_version=2, release=version, platform=platform,
                        prepared_date=datetime.now(timezone.utc).date().isoformat(), timezone='Asia/Dhaka',
                        created_utc=datetime.now(timezone.utc).isoformat(), source_base_commit=revision,
                        source_snapshot_includes_uncommitted_changes=True,
                        model_sha256=metadata['onnx_sha256'], class_names=runtime['class_names'],
                        gesture_to_action=runtime['gesture_to_action'],
                        paired_release=versions['android' if platform == 'ios' else 'ios'],
                        desktop_reference_sha256=digest(desktop_zip), model_retrained_for_handover=False,
                        native_build_validated=False, full_native_decision_port_complete=False,
                        physical_device_validated=False, mutable_online_adapter_included=False,
                        image_source='USB-NCM JLIP/JPEG; built-in phone camera excluded',
                        geometry_fixture_count=len(vectors['geometry_cases']),
                        temporal_transition_count=len(vectors['temporal_transition_cases']),
                        onnx_fixture_count=len(vectors['model_cases']),
                        isolated_backend_tests=summary['passed'], reference_source_sha256=source_hashes,
                        native_source_sha256={file.relative_to(package).as_posix(): digest(file)
                            for file in sorted((package / platform).rglob('*')) if file.suffix in {'.swift', '.kt', '.h'}},
                        scope='Current runnable desktop reference, model, native diagnostics, angle helpers, porting contract and parity evidence; native decision/device acceptance pending')
        write_json(package / 'RELEASE_MANIFEST.json', manifest)
        sums = {file.relative_to(package).as_posix(): dict(bytes=file.stat().st_size, sha256=digest(file))
                for file in sorted(package.rglob('*')) if file.is_file()}
        write_json(package / 'SHA256SUMS.json', sums)
        report = verify(package)
        report.update(verify_onnx(package))
        final, zipped = destinations[platform]
        final.parent.mkdir(exist_ok=True)
        shutil.copytree(package, final)
        archive(final, zipped)
        verify(final)
        zipped.with_suffix('.zip.sha256').write_text(digest(zipped) + '  ' + zipped.name + '\n', encoding='utf-8')
        reports.append(dict(directory=str(final), archive=str(zipped), archive_bytes=zipped.stat().st_size,
                            archive_sha256=digest(zipped), isolated_backend_tests=summary['passed'], **report))
    write_json(stage / 'build_report.json', reports)
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == '__main__':
    main()
