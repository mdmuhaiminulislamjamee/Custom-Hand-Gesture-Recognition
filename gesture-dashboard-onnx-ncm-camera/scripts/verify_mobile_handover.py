"""Read-only mobile handover checks. Hash checks use only Python's standard library."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

CLASSES = ['left', 'right', 'up', 'down', 'open_palm', 'like', 'dorsal', 'ok',
           'fist', 'thumb_down', 'peace', 'rock', 'stop_inverted']
MODELS = ['gesture_mlp_production.onnx', 'gesture_mlp_onnx_metadata.json',
          'gesture_mobile_runtime_config.json', 'hand_landmarker.task', 'blaze_face_short_range.tflite']


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def safe(name):
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and '..' not in path.parts and '\\' not in name and ':' not in name


def verify(root):
    root = Path(root).resolve()
    listing = json.loads((root / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p != root / 'SHA256SUMS.json'}
    if actual != set(listing):
        raise ValueError(f'Package file set mismatch: missing={set(listing)-actual}; extra={actual-set(listing)}')
    for name, record in listing.items():
        file = root / name
        if not safe(name) or file.is_symlink() or not file.resolve().is_relative_to(root):
            raise ValueError(f'Unsafe package path: {name}')
        if file.stat().st_size != record['bytes'] or digest(file) != record['sha256']:
            raise ValueError(f'Hash/size mismatch: {name}')
    manifest = json.loads((root / 'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    platform = manifest['platform']
    if platform not in {'ios', 'android'}:
        raise ValueError('Unsupported platform')
    model_root = root / platform / 'models'
    metadata = json.loads((model_root / 'gesture_mlp_onnx_metadata.json').read_text(encoding='utf-8'))
    runtime = json.loads((model_root / 'gesture_mobile_runtime_config.json').read_text(encoding='utf-8'))
    contract = json.loads((root / 'MOBILE_RUNTIME_CONTRACT.json').read_text(encoding='utf-8'))
    model_hash = digest(model_root / 'gesture_mlp_production.onnx')
    if any(order != CLASSES for order in (metadata['output_class_order'], runtime['class_names'],
                                          manifest['class_names'], contract['class_names'])):
        raise ValueError('13-output class order mismatch')
    if any(value != model_hash for value in (metadata['onnx_sha256'], manifest['model_sha256'], contract['model_sha256'])):
        raise ValueError('Model hash mismatch')
    if metadata['input']['shape'] != [None, 76] or runtime['input_shape'] != [1, 76] or len(metadata['feature_names']) != 76:
        raise ValueError('76-feature input contract mismatch')
    if runtime['disabled_gestures'] != ['dorsal'] or 'dorsal' in runtime['gesture_to_action'] or len(runtime['gesture_to_action']) != 12:
        raise ValueError('Retired Dorsal/action contract mismatch')
    if runtime['gesture_to_action']['stop_inverted'] != 'Return to main position':
        raise ValueError('Stop inverted action mismatch')
    for name, limits in [('open_palm_rotation_validation', (-160., -20., 70.)),
                         ('stop_inverted_validation', (-145., -35., 55.))]:
        policy = runtime[name]
        if tuple(policy[k] for k in ('xy_minimum_degrees', 'xy_maximum_degrees', 'maximum_tilt_from_up_degrees')) != limits:
            raise ValueError(f'Wrong angle policy: {name}')
        if not policy['endpoints_inclusive'] or policy['angle_space'] != 'source image XY pixels':
            raise ValueError('Wrong angle coordinate space or endpoints')
    if contract['runtime_config'] != runtime or contract['gesture_to_action'] != runtime['gesture_to_action']:
        raise ValueError('Runtime contract JSON drift')
    vectors = json.loads((root / platform / 'tests/parity_vectors.json').read_text(encoding='utf-8'))
    if vectors['model_sha256'] != model_hash or vectors['class_names'] != CLASSES:
        raise ValueError('Wrong model parity fixtures')
    if len(vectors['geometry_cases']) != manifest['geometry_fixture_count'] or len(vectors['model_cases']) != manifest['onnx_fixture_count']:
        raise ValueError('Fixture count mismatch')
    desktop = root / 'desktop_reference.zip'
    if digest(desktop) != manifest['desktop_reference_sha256']:
        raise ValueError('Wrong desktop snapshot')
    with ZipFile(desktop) as archive:
        entries = [i.filename for i in archive.infolist() if not i.is_dir()]
        if len(entries) != len(set(entries)) or not all(safe(name) for name in entries) or archive.testzip() is not None:
            raise ValueError('Invalid desktop ZIP paths or CRC')
        prohibited = ('/.venv/', '/node_modules/', '/.runtime/', '/feedback/', '/.git/',
                      'gesture_online_adapter_state', 'online_adapter_backups')
        if any(any(token in name for token in prohibited) for name in entries):
            raise ValueError('Local/private mutable state found in desktop ZIP')
        for name in MODELS:
            if archive.read('desktop_reference/models/' + name) != (model_root / name).read_bytes():
                raise ValueError(f'Desktop/mobile asset mismatch: {name}')
        for name, expected in vectors['source_sha256'].items():
            data = archive.read('desktop_reference/' + name)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f'Stale fixtures: {name}')
        for name, expected in manifest['reference_source_sha256'].items():
            data = (root / platform / 'reference' / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected or data != archive.read('desktop_reference/' + name):
                raise ValueError(f'Desktop/mobile reference source mismatch: {name}')
    return dict(verified_files=len(listing), platform=platform, model_sha256=model_hash,
                desktop_reference_sha256=manifest['desktop_reference_sha256'], desktop_mobile_assets_match=True,
                geometry_cases=len(vectors['geometry_cases']), temporal_transitions=len(vectors['temporal_transition_cases']),
                onnx_cases=len(vectors['model_cases']), native_build_validated=manifest['native_build_validated'],
                full_native_decision_port_complete=manifest['full_native_decision_port_complete'])


def verify_onnx(root):
    import numpy as np
    import onnxruntime as ort
    root = Path(root)
    manifest = json.loads((root / 'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    platform = manifest['platform']
    vectors = json.loads((root / platform / 'tests/parity_vectors.json').read_text(encoding='utf-8'))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(root / platform / 'models/gesture_mlp_production.onnx'),
                                   sess_options=options, providers=['CPUExecutionProvider'])
    rows = vectors['model_cases']
    if not rows:
        raise ValueError('Empty ONNX fixture set')
    maximum = 0.
    for row in rows:
        probabilities, mass = session.run(['probabilities', 'known_gesture_mass'],
                                         {'landmark_features': np.asarray([row['features']], np.float32)})
        if probabilities.shape != (1, 13) or mass.shape != (1, 1):
            raise ValueError('Wrong ONNX output shape')
        error = max(float(np.max(np.abs(probabilities[0] - row['probabilities']))),
                    abs(float(mass[0, 0]) - row['known_gesture_mass']))
        if error > vectors['probability_tolerance']:
            raise ValueError('ONNX fixture mismatch')
        maximum = max(maximum, error)
    return dict(onnx_parity_passed=True, maximum_absolute_error=maximum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', nargs='?', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--onnx', action='store_true')
    args = parser.parse_args()
    result = verify(args.directory)
    if args.onnx:
        result.update(verify_onnx(args.directory))
    print(json.dumps(result, indent=2))
