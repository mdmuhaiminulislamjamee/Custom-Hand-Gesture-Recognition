"""Verify the standalone app-team handover with Python 3.10+ standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


BASELINE_SHA256 = 'f18ece4b77ca2a09514a1bf00c2a7ec1e673171c43c3dff3eda6b05abf7315df'
FP32_SHA256 = 'cbd53a5c6874172ec8b16cb07c96c228a3063252f0156a7ee9232b4861d3f192'


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def checked(root: Path, relative: str) -> Path:
    part = PurePosixPath(relative)
    if part.is_absolute() or '..' in part.parts or '\\' in relative or ':' in relative:
        raise ValueError('Unsafe payload path: ' + relative)
    path = root.joinpath(*part.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Payload escapes handover: ' + relative)
    return path


def verify(root: Path) -> dict:
    root = root.resolve()
    sums = json.loads((root / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    for relative, expected in sums.items():
        path = checked(root, relative)
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Missing or changed file: ' + relative)
    if any(name.startswith('flutter_gesture_tester/') or
           name.endswith(('.apk', '.ipa', '.jks', '.keystore')) for name in sums):
        raise ValueError('Tester application or signing material in runtime delivery')
    manifest = json.loads((root / 'release_manifest.json').read_text(encoding='utf-8'))
    preservation = json.loads((root / 'validation/app_team_package_preservation.json').read_text(encoding='utf-8'))
    if manifest['source_archive']['sha256'] != BASELINE_SHA256 or not preservation['passed']:
        raise ValueError('Unexpected qualified baseline')
    for group in ('runtime_flutter_sha256', 'phone_helper_sha256'):
        for relative, expected in preservation[group].items():
            if sums.get(relative) != expected or digest(checked(root, relative)) != expected:
                raise ValueError('Executable input differs from qualified release: ' + relative)
    models = manifest['models']
    for name, entry in models.items():
        if digest(checked(root, entry['path'])) != entry['sha256']:
            raise ValueError('Model mismatch: ' + name)
    if models['gesture_static_float32']['sha256'] != FP32_SHA256:
        raise ValueError('Production FP32 model changed')
    fp32 = json.loads((root / 'flutter/assets/models/gesture_static_float32_manifest.json').read_text(encoding='utf-8'))
    int8 = json.loads((root / 'flutter/assets/models/gesture_static_int8_manifest.json').read_text(encoding='utf-8'))
    if not fp32['qualification']['passed'] or int8['qualification']['passed']:
        raise ValueError('Model loading policy changed')
    if digest(root / 'reference/source_model/gesture_mlp_production.onnx') != fp32['source_onnx_sha256']:
        raise ValueError('Source snapshot belongs to a different classifier')
    provenance = json.loads((root / 'contracts/detector_provenance.json').read_text(encoding='utf-8'))
    detector = {**provenance['files'], **provenance.get('local_patches', {})}
    for relative, expected in detector.items():
        if digest(checked(root, relative)) != expected:
            raise ValueError('Detector provenance mismatch: ' + relative)
    native = json.loads((root / 'validation/app_team_model_verification.json').read_text(encoding='utf-8'))
    if not native['passed']:
        raise ValueError('Packaged model verification did not pass')
    if 'skip_build: true' in (root / 'flutter/example/pubspec.yaml').read_text(encoding='utf-8'):
        raise ValueError('Native builds disabled in delivered example')
    return dict(
        passed=True, payload_files_verified=len(sums),
        preserved_runtime_flutter_files=len(preservation['runtime_flutter_sha256']),
        preserved_phone_helpers=len(preservation['phone_helper_sha256']),
        detector_provenance_files_verified=len(detector), model_assets_verified=len(models),
        default_classifier='qualified FP32 TFLite', tester_app_included=False,
        executable_changes_from_qualified_release=0, physical_ios_tested=False,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parent)
    print(json.dumps(verify(parser.parse_args().package_root), indent=2))
