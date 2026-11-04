"""Verify the NCM tester APK against its build inputs and host-test reports."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import zipfile
from verified_flutter_reuse import verify as verify_flutter_reuse

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,default=ROOT / '.runtime/ncm_apk_build')
    parser.add_argument('--host-report-dir',type=Path,default=ROOT / '.runtime/mobile_false_hand_host')
    args = parser.parse_args()
    report = args.report_dir.resolve()
    build = json.loads((report / 'result.json').read_text())
    assert build['passed']
    apk = ROOT / build['apk']
    assert digest(apk.read_bytes()) == build['sha256']
    stage = Path(build['stage'])
    version = re.search(r'^version:\s*(\S+)\+(\d+)',
                        (ROOT / 'flutter_gesture_tester/pubspec.yaml').read_text(), re.M)
    metadata = (report / 'apk_metadata.log').read_text()
    assert f"versionCode='{version[2]}'" in metadata
    assert f"versionName='{version[1]}'" in metadata
    source_hashes = {}
    for folder, pattern in [
        ('flutter_gesture_tester/lib', '*.dart'),
        ('flutter_gesture_tester/android/app/src/main/kotlin', '*.kt'),
        ('static_gesture_runtime/flutter/lib', '*.dart'),
    ]:
        for source in (ROOT / folder).rglob(pattern):
            relative = source.relative_to(ROOT)
            assert source.read_bytes() == (stage / relative).read_bytes(), str(relative)
            source_hashes[relative.as_posix()] = digest(source.read_bytes())
    model_hashes = {}
    with zipfile.ZipFile(apk) as archive:
        assert archive.testzip() is None
        for name in ('blaze_face_short_range', 'hand_detection', 'hand_landmark_full',
                     'gesture_static_float32', 'gesture_static_int8'):
            package = 'static_gesture_runtime' if name.startswith('gesture_static') else 'hand_detection'
            source = ROOT / ('static_gesture_runtime/flutter/assets/models' if package == 'static_gesture_runtime'
                             else 'static_gesture_runtime/flutter/third_party/hand_detection/assets/models') / (name + '.tflite')
            data = archive.read(f'assets/flutter_assets/packages/{package}/assets/models/{name}.tflite')
            assert data == source.read_bytes(), name
            model_hashes[name] = digest(data)
        assert any(b'gesture_lab/ncm_network' in archive.read(name)
                   for name in archive.namelist() if name.endswith('.dex'))
        if version[1] in ('2.0.4', '2.0.5', '2.0.6', '2.0.7'):
            assert any(b'USB_DESCRIPTOR' in archive.read(name) and b'describeUsb' in archive.read(name)
                       for name in archive.namelist() if name.endswith('.dex'))
            dart_app = archive.read('lib/arm64-v8a/libapp.so')
            assert b'USB diagnostics' in dart_app and b'Copy report' in dart_app
        if version[1] in ('2.0.5', '2.0.6', '2.0.7'):
            assert any(b'UsbNcmStream' in archive.read(name) and b'GET_NTB_PARAMETERS failed' in archive.read(name)
                       for name in archive.namelist() if name.endswith('.dex'))
            native = json.loads((ROOT / '.runtime/native_usb_ncm_check/result.json').read_text())
            assert native['passed'] and native['independent_python_wire_oracle']
            for path, sha in native['source_sha256'].items():
                assert digest((ROOT / path).read_bytes()) == sha
        if version[1] in ('2.0.6', '2.0.7'):
            dex = b''.join(archive.read(name) for name in archive.namelist() if name.endswith('.dex'))
            assert ('Transport revision: v' + version[1]).encode() in dex and b'with force=true' in dex
            transport = (ROOT / 'flutter_gesture_tester/android/app/src/main/kotlin/com/gesturelab/gesture_model_tester/UsbNcmStream.kt').read_text()
            assert 'conn.claimInterface(ctl, true)' in transport
            assert 'conn.claimInterface(active, true)' in transport
    host_report = args.host_report_dir.resolve()
    tests = json.loads((host_report / 'results.json').read_text())
    reused = build.get('flutter_reuse')
    if reused:
        proof = verify_flutter_reuse(reused['report_dir'])
        assert proof['dart_source_sha256'] == reused['dart_source_sha256']
        with zipfile.ZipFile(proof['apk']) as old, zipfile.ZipFile(apk) as current:
            assert old.read('lib/arm64-v8a/libapp.so') == current.read('lib/arm64-v8a/libapp.so')
        tests = proof['host_checks']
    else:
        assert len(tests) == 6 and all(row['exit_code'] == 0 for row in tests)
        for project in ('runtime', 'tester'):
            assert 'All tests passed!' in (host_report / f'{project}_tests.log').read_text()
    result = dict(passed=True, version=version[1], apk=build['apk'], sha256=build['sha256'],
                  source_sha256=source_hashes, model_sha256=model_hashes,
                  host_checks=tests, flutter_test_reuse=reused, physical_realme_ncm_tested=False)
    (report / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, version=version[1], sources=len(source_hashes),
                         model_assets=len(model_hashes), sha256=build['sha256']), indent=2))


if __name__ == '__main__':
    main()
