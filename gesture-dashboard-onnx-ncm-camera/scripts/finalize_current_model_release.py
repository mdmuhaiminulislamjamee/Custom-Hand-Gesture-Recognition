"""Bind a new Boss handover to freshly tested model, Flutter and APK inputs.

No historical pass record is reused. Run the host/native checks and APK build
first; this command checks their retained inputs and creates HOST_CHECKS.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

from verify_ncm_apk import verify as verify_apk
from package_boss_handover import source_models

ROOT = Path(__file__).resolve().parents[1]
RETIRED = {'flutter_gesture_tester/lib/services/native_gesture_engine.dart',
           'flutter_gesture_tester/lib/widgets/landmark_painter.dart'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def input_files(base: Path) -> dict[str, Path]:
    result = {}
    for folder in ('flutter_gesture_tester/lib', 'flutter_gesture_tester/test',
                   'static_gesture_runtime/flutter/lib', 'static_gesture_runtime/flutter/test',
                   'static_gesture_runtime/flutter/third_party/hand_detection/lib'):
        for path in sorted((base / folder).rglob('*.dart')):
            relative = path.relative_to(base).as_posix()
            if relative not in RETIRED:
                result[relative] = path
    for folder in ('flutter_gesture_tester', 'static_gesture_runtime/flutter',
                   'static_gesture_runtime/flutter/third_party/hand_detection'):
        for name in ('pubspec.yaml', 'pubspec.lock'):
            path = base / folder / name
            if path.is_file():
                result[path.relative_to(base).as_posix()] = path
    for folder in ('static_gesture_runtime/flutter/assets/models',
                   'static_gesture_runtime/flutter/third_party/hand_detection/assets/models'):
        for path in sorted((base / folder).glob('*')):
            if path.is_file():
                result[path.relative_to(base).as_posix()] = path
    for name in ('python_reference.json', 'model_reference.json', 'false_hand_policy.json', 'direction_pose_reference.json', 'right_pose_reference.json'):
        path = base / 'static_gesture_runtime/validation' / name
        result[path.relative_to(base).as_posix()] = path
    return result


def check_host_inputs(stage: Path) -> dict[str, str]:
    current, tested = input_files(ROOT), input_files(stage)
    require(current.keys() == tested.keys(), 'Host-test input file inventory changed')
    for relative, path in current.items():
        a, b = path.read_bytes(), tested[relative].read_bytes()
        if relative.endswith('pubspec.yaml'):
            def normalize(data):
                text = data.decode('utf-8').replace('\r\n', '\n')
                appended = '\nhooks:\n  user_defines:\n    dartcv4:\n      skip_build: true\n'
                if text.endswith(appended):
                    text = text[:-len(appended)]
                return '\n'.join(line for line in text.splitlines()
                                 if line.strip() != 'skip_build: true').rstrip()
            require(normalize(a) == normalize(b), 'Host-test configuration changed: ' + relative)
        else:
            require(a == b, 'Input changed since host tests: ' + relative)
    return {relative: sha(path) for relative, path in current.items()}


def check_native(report: Path, passed_keys: tuple[str, ...]):
    result = read(report / 'result.json')
    require(all(result.get(key) is True for key in passed_keys), 'Native checks failed: ' + str(report))
    require(bool(result.get('source_sha256')), 'Native source hashes are missing')
    for relative, expected in result['source_sha256'].items():
        require(sha(ROOT / relative) == expected, 'Native source changed since checks: ' + relative)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apk-report-dir', type=Path, default=ROOT / '.runtime/current_model_v208_build')
    parser.add_argument('--host-report-dir', type=Path, default=ROOT / '.runtime/current_model_v208_host')
    parser.add_argument('--host-stage', type=Path)
    parser.add_argument('--native-report-dir', type=Path, default=ROOT / '.runtime/current_model_v208_native_usb')
    parser.add_argument('--native-filter-report-dir', type=Path, default=ROOT / '.runtime/current_model_v208_native_filter')
    parser.add_argument('--model-python', type=Path, default=ROOT / '.runtime/static-conversion-venv/Scripts/python.exe')
    parser.add_argument('--output', type=Path, default=ROOT / 'handover_static/final_docs/HOST_CHECKS.json')
    parser.add_argument('--date', default='2026-11-11')
    parser.add_argument('--release-revision', default='current_model_20261111')
    parser.add_argument('--source-handover', type=Path, help='Keep the deployed mobile model source snapshot for a tracking-only update')
    args = parser.parse_args()

    host_report = args.host_report_dir.resolve()
    checks = read(host_report / 'results.json')
    expected = {(p, c) for p in ('runtime', 'tester') for c in ('dependencies', 'analysis', 'tests')}
    require(len(checks) == 6 and {(row['project'], row['check']) for row in checks} == expected,
            'Host report must contain each runtime/tester check exactly once')
    require(all(row['exit_code'] == 0 for row in checks), 'A Flutter host check failed')
    test_counts = {}
    for project in ('runtime', 'tester'):
        log = (host_report / f'{project}_tests.log').read_text(encoding='utf-8', errors='replace')
        matches = re.findall(r'\+(\d+): All tests passed!', log)
        require(bool(matches), 'No completed test suite for ' + project)
        test_counts[project] = int(matches[-1])
    stage = args.host_stage
    if stage is None:
        log = (host_report / 'tester_tests.log').read_text(encoding='utf-8', errors='replace')
        match = re.search(r'loading (.*?)[/\\]test[/\\]', log)
        require(match is not None, 'Pass --host-stage when test logs have no loading path')
        stage = Path(match[1]).parent
    source_hashes = check_host_inputs(stage.resolve())

    native_usb = check_native(args.native_report_dir, ('passed', 'independent_python_wire_oracle'))
    native_filter = check_native(args.native_filter_report_dir, ('android_filter_compile_passed', 'policy_test_passed'))
    apk_verification = verify_apk(args.apk_report_dir, host_report, args.native_report_dir, '2.0.8')
    build = read(args.apk_report_dir / 'result.json')
    require(not build.get('flutter_reuse'), 'Current-model release requires a freshly compiled Flutter library')
    source_hashes.update({key: value for key, value in apk_verification['source_sha256'].items()
                          if key not in RETIRED})

    package = ROOT / 'static_gesture_runtime'
    model_source = source_models(args.source_handover)
    production_sha = hashlib.sha256(model_source['gesture_mlp_production.onnx']).hexdigest()
    config_sha = hashlib.sha256(model_source['gesture_mobile_runtime_config.json']).hexdigest()
    conversion = read(package / 'validation/conversion_report.json')
    require(conversion['source_onnx_sha256'] == production_sha, 'Conversion is for another production model')
    require(conversion['runtime_config_sha256'] == config_sha, 'Conversion scalar gates use another runtime config')
    require(conversion['models']['float32']['passed'] is True, 'FP32 model conversion did not pass')
    for name in ('python_reference.json', 'model_reference.json', 'model_parity.json'):
        require(read(package / 'validation' / name)['model_sha256'] == production_sha,
                'Fixture belongs to another model: ' + name)
    geometry = {}
    for name in ('dart_geometry_parity.json', 'dart_cache_geometry_parity.json'):
        report = read(package / 'validation' / name)
        require(report['passed'] is True, 'Dart geometry parity failed: ' + name)
        geometry[name] = report
    model_run = subprocess.run([str(args.model_python.resolve()), str(ROOT / 'scripts/verify_static_runtime.py')],
                               cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, check=True)
    model_check = json.loads(model_run.stdout[model_run.stdout.index('{'):])
    require(model_check['passed'] is True, 'Native LiteRT model verification failed')
    (args.apk_report_dir / 'model_verification.json').write_text(json.dumps(model_check, indent=2) + '\n', encoding='utf-8')
    for path in sorted((package / 'validation').glob('*.json')):
        source_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in ('gesture_mlp_production.onnx', 'gesture_mlp_onnx_metadata.json',
                 'gesture_mobile_runtime_config.json', 'gesture_artifact_manifest.json'):
        source_hashes['source_model/' + name] = hashlib.sha256(model_source[name]).hexdigest()

    artifact = {key: value for key, value in build.items() if key not in ('stage', 'apk', 'flutter_reuse')}
    artifact.update(model_assets_matching_runtime=len(apk_verification['model_sha256']), apksigner_verify='passed')
    result = dict(status='passed', date=args.date, release_revision=args.release_revision,
                  generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  scope=('Preserved mobile-model verification and geometry parity, fresh Flutter host tests, native Android filter and USB-NCM checks, newly compiled Android APK'
                         if args.source_handover else
                         'Current-model conversion and geometry parity, fresh Flutter host tests, native Android filter and USB-NCM checks, newly compiled Android APK'),
                  toolchain=dict(flutter='3.47.8', dart='3.13.7', android_build_jdk='17', mediapipe_native_android='1.1.1'),
                  flutter_tests=dict(runtime_passed=test_counts['runtime'], tester_passed=test_counts['tester'],
                                     policy_cases=native_filter['policy_cases'],
                                     opencv_host_hook='Skipped only in isolated tests; normal native builds in APK and delivery'),
                  dart_analysis=dict(runtime='No issues found', android_tester='No issues found'),
                  native_android_filter=native_filter, native_usb_ncm=native_usb,
                  python_native_model_verification=model_check, conversion=conversion,
                  geometry_parity=geometry, source_model_sha256=production_sha,
                  runtime_config_sha256=config_sha, android_artifact=artifact,
                  source_sha256=dict(sorted(source_hashes.items())), physical_ios_tested=False,
                  physical_ncm_tested=False, ios_compile_tested=False,
                  remaining_acceptance='DEVICE_ACCEPTANCE.md; native_false_hand_update/README.md')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'date', 'release_revision', 'flutter_tests', 'source_model_sha256', 'android_artifact')}, indent=2))


if __name__ == '__main__':
    main()
