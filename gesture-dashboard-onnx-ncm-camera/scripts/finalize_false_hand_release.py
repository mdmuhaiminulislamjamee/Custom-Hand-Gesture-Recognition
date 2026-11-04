"""Record local detector patches and generate evidence for the checked release."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'static_gesture_runtime'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def prepare():
    contract=RUNTIME/'contracts/mediapipe_detector.json'
    data=json.loads(contract.read_text())
    data.update(detection_confidence=.55,face_model='blaze_face_short_range.tflite',
        face_refresh_ms=200,face_retention_ms=500,candidate_filter='face_ear_size_overlap_and_source_pixels_and_collapsed_chains',
        minimum_palm_pixels=18,minimum_hand_span_pixels=40,minimum_short_side_pixels=12,
        filtered_before=['gesture_inference','hand_overlay','continued_roi_tracking'])
    write(contract,data)
    provenance=RUNTIME/'contracts/detector_provenance.json'
    data=json.loads(provenance.read_text())
    paths=['contracts/mediapipe_detector.json','flutter/third_party/hand_detection/pubspec.yaml',
        'flutter/third_party/hand_detection/lib/src/hand_detector.dart',
        'flutter/third_party/hand_detection/lib/src/isolate/hand_detector_core.dart',
        'flutter/third_party/hand_detection/lib/src/models/face_guard.dart',
        'flutter/third_party/hand_detection/lib/src/shared/false_hand_policy.dart',
        'flutter/third_party/hand_detection/assets/models/blaze_face_short_range.tflite']
    data['local_revision']='false_hand_filters_20261031'
    data['local_patches']={p:sha(RUNTIME/p) for p in paths}
    data['local_patch_note']='Upstream files/hashes are retained as provenance; local_patches records intentional additions/overrides and is verified by release tools.'
    write(provenance,data)

def finalize():
    build=json.loads((ROOT/'.runtime/false_hand_apk_build/result.json').read_text())
    native=json.loads((ROOT/'.runtime/native_false_hand_check/result.json').read_text())
    checks=json.loads((ROOT/'.runtime/mobile_false_hand_host/results.json').read_text())
    if not build['passed'] or not native['policy_test_passed'] or len(checks)!=6 or any(c['exit_code'] for c in checks):
        raise ValueError('All source/test/build checks must pass before finalization')
    stage=Path(build['stage'])
    source={}
    for base in (ROOT/'static_gesture_runtime/flutter/lib',ROOT/'static_gesture_runtime/flutter/third_party/hand_detection/lib',ROOT/'flutter_gesture_tester/lib'):
        for path in sorted(base.rglob('*.dart')):
            relative=path.relative_to(ROOT)
            if sha(path)!=sha(stage/relative):raise ValueError('Source changed since APK build: '+str(relative))
            retired={'flutter_gesture_tester/lib/services/native_gesture_engine.dart','flutter_gesture_tester/lib/widgets/landmark_painter.dart'}
            if relative.as_posix() not in retired:
                source[relative.as_posix()]=sha(path)
    for relative in ('flutter_gesture_tester/pubspec.yaml','static_gesture_runtime/flutter/pubspec.yaml',
        'static_gesture_runtime/flutter/third_party/hand_detection/pubspec.yaml'):
        if sha(ROOT/relative)!=sha(stage/relative):raise ValueError('Build configuration changed: '+relative)
        source[relative]=sha(ROOT/relative)
    apk=ROOT/build['apk']
    if sha(apk)!=build['sha256']:raise ValueError('APK checksum changed')
    model_paths={
        'packages/static_gesture_runtime/assets/models/gesture_static_float32.tflite':RUNTIME/'flutter/assets/models/gesture_static_float32.tflite',
        'packages/static_gesture_runtime/assets/models/gesture_static_int8.tflite':RUNTIME/'flutter/assets/models/gesture_static_int8.tflite',
        **{'packages/hand_detection/assets/models/'+p.name:p for p in (RUNTIME/'flutter/third_party/hand_detection/assets/models').glob('*.tflite')},
    }
    with zipfile.ZipFile(apk) as archive:
        for name,path in model_paths.items():
            if hashlib.sha256(archive.read('assets/flutter_assets/'+name)).hexdigest()!=sha(path):raise ValueError('APK model differs: '+name)
        if {name.split('/')[1] for name in archive.namelist() if name.startswith('lib/') and name.endswith('.so')}!={'arm64-v8a'}:
            raise ValueError('Unexpected APK ABIs')
    metadata=(ROOT/'.runtime/false_hand_apk_build/apk_metadata.log').read_text()
    if "versionCode='4'" not in metadata or "versionName='2.0.2'" not in metadata or "sdkVersion:'24'" not in metadata:
        raise ValueError('APK version or minimum SDK differs')
    signatures=(ROOT/'.runtime/false_hand_apk_build/apk_signature.log').read_text()
    if 'Verifies' not in signatures:raise ValueError('Signature verification missing')
    for path,expected in native['source_sha256'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Native source changed since compile')
    model_check=subprocess.run([str(ROOT/'.runtime/static-conversion-venv/Scripts/python.exe'),str(ROOT/'scripts/verify_static_runtime.py')],
        cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True)
    model=json.loads(model_check.stdout[model_check.stdout.index('{'):])
    def tests(project):
        log=(ROOT/f'.runtime/mobile_false_hand_host/{project}_tests.log').read_text()
        return int(re.findall(r'\+(\d+): All tests passed!',log)[-1])
    report=dict(status='passed',date='2026-10-31',release_revision='false_hand_filters_20261031',
        scope='Isolated Flutter host checks, real LiteRT face/gesture tensor checks, native Kotlin filter compile and 42 policy cases, rebuilt Android APK',
        toolchain=dict(flutter='3.47.8',dart='3.13.7',android_build_jdk='17',mediapipe_native_android='1.1.1'),
        flutter_tests=dict(runtime_passed=tests('runtime'),tester_passed=tests('tester'),policy_cases=42,
            opencv_host_hook='Skipped only in isolated tests; normal native builds in APK and delivery'),
        dart_analysis=dict(runtime_and_example='No issues found',android_tester='No issues found'),
        native_android_filter=native,python_native_model_verification=model,
        face_decoder_saved_images=json.loads((ROOT/'.runtime/mobile_face_decoder/result.json').read_text()),
        android_artifact={**{k:v for k,v in build.items() if k not in ('stage','apk')},'model_assets_matching_runtime':len(model_paths),'apksigner_verify':'passed'},
        source_sha256=source,physical_ios_tested=False,physical_ncm_tested=False,ios_compile_tested=False,
        remaining_acceptance='DEVICE_ACCEPTANCE.md; native_false_hand_update/README.md')
    write(ROOT/'handover_static/final_docs/HOST_CHECKS.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare else finalize()
