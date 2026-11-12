"""Bind the feedback refresh to model, benchmark, test and mobile artifacts."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,shutil,subprocess

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def failures(directory):
    result=set()
    for p in directory.glob('test_*.log'):
        result.update(re.findall(r'^FAILED (\S+)',p.read_text(encoding='utf-8',errors='replace'),re.M))
    return result


def main():
    out=ROOT/'artifacts/feedback_refresh_20261111';validation=ROOT/'static_gesture_runtime/validation'
    evaluation=read(out/'evaluation.json');digest=sha(ROOT/'models/gesture_mlp_production.onnx')
    assert evaluation['deployed'] and evaluation['qualified'] and all(evaluation['release_checks'].values())
    assert digest==evaluation['model_sha256']
    before_dir=ROOT/'.runtime/feedback_refresh_20261111_baseline_backend';after_dir=ROOT/'.runtime/feedback_refresh_20261111_candidate_backend'
    before,after=read(before_dir/'summary.json'),read(after_dir/'summary.json')
    baseline_failures,current_failures=failures(before_dir),failures(after_dir)
    assert before['completed']==before['files'] and after['completed']==after['files']
    assert not current_failures-baseline_failures and after['passed']>=before['passed']
    host_dir=ROOT/'.runtime/feedback_refresh_20261111_host';host=read(host_dir/'results.json')
    assert len(host)==6 and all(r['exit_code']==0 for r in host)
    counts={k:int(re.findall(r'\+(\d+): All tests passed!',(host_dir/f'{k}_tests.log').read_text())[-1]) for k in ('runtime','tester')}
    native=read(ROOT/'.runtime/feedback_refresh_20261111_native_filter/result.json')
    usb=read(ROOT/'.runtime/feedback_refresh_20261111_native_usb/result.json')
    assert native['policy_test_passed'] and native['android_filter_compile_passed'] and usb['passed']
    apk=read(ROOT/'.runtime/feedback_refresh_20261111_build/verification.json')
    assert apk['passed'] and apk['version']=='2.0.11'
    conversion=read(validation/'conversion_report.json');latency=read(validation/'latency_comparison.json')
    assert conversion['source_onnx_sha256']==digest and conversion['models']['float32']['passed']
    assert latency['current_model_sha256']==conversion['models']['float32']['model_sha256']
    assert latency['current_model_bytes']==latency['previous_model_bytes']
    assert latency['models']['current']['p95_ms']-latency['models']['previous']['p95_ms']<=.01
    assert read(validation/'dart_geometry_parity.json')['passed'] and read(validation/'dart_cache_geometry_parity.json')['passed']
    prefix=subprocess.check_output(['git','rev-parse','--show-prefix'],cwd=ROOT,text=True).strip()
    sources={}
    for name in ('backend/geometry.py','backend/runtime.py','backend/model_runtime.py','backend/hand_selection.py','backend/direction_stability.py'):
        previous=subprocess.check_output(['git','show','HEAD:'+prefix+name],cwd=ROOT)
        assert previous.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n'),'Runtime policy changed: '+name
        sources[name]=sha(ROOT/name)
    shutil.copy2(out/'onnx_latency.json',validation/'onnx_latency.json')
    image_before,image_after=read(out/'baseline_images.json'),read(out/'candidate_images.json')
    for k,v in image_before['per_class'].items():
        assert image_after['per_class'][k]['correct']>=v['correct'] and image_after['per_class'][k]['actions']>=v['actions']
    report=dict(schema_version=1,qualified=True,generated_utc=datetime.now(timezone.utc).isoformat(),model_sha256=digest,
        previous_model_sha256=evaluation['audit']['baseline_model_sha256'],
        baseline_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        training=evaluation['audit'],selection=evaluation['selection'],fitted_parameters=evaluation['fitted_parameters'],
        frozen_encoder_and_unaffected_outputs=evaluation['frozen_weights_identical'],graph_unchanged=True,
        rejected_alternatives=evaluation['rejected_alternatives'],
        benchmark_scope=evaluation['acceptance_scope'],heldout_metrics=evaluation['splits'],release_checks=evaluation['release_checks'],
        jpeg_replay=dict(scope=image_after['scope'],selection_sha256=image_after['selection_sha256'],rows=image_after['total'],
            baseline_correct=image_before['correct'],candidate_correct=image_after['correct'],baseline_actions=image_before['actions'],
            candidate_actions=image_after['actions'],per_class=image_after['per_class']),
        backend=dict(files=after['files'],passed=after['passed'],baseline_passed=before['passed'],
            baseline_failures=sorted(baseline_failures),current_failures=sorted(current_failures),new_failures=[],all_tests_passed=not current_failures),
        flutter_tests=counts,native_filter_cases=native['policy_cases'],native_usb=dict(groups=len(usb['tests']),independent_wire_oracle=usb['independent_python_wire_oracle']),
        android_apk=dict(version=apk['version'],sha256=apk['sha256'],model_assets=len(apk['model_sha256'])),
        conversion=conversion['models']['float32'],onnx_latency=read(out/'onnx_latency.json'),tflite_latency=latency,
        runtime_source_sha256=sources,runtime_sources_match_baseline_after_line_endings=True,
        ghost_policy_changed=False,physical_device_tested=False,ios_compile_tested=False,
        limitations=['Reviewed landmarks and images are training regressions, not independent subjects or live-camera accuracy.',
            'Fixed regression benchmarks were reused for release acceptance; no validation/test data supplied training gradients.',
            'Larger fits were rejected. The accepted training-fit improvement is modest: 628 to 630 of 990.',
            'Forty-two new JPEGs were deleted; their saved landmarks remain training inputs. The image replay is a deterministic sample of 48 available photos.',
            'The six baseline fist-orientation/hand-ownership test failures remain. Physical phone/NCM latency, new-camera accuracy and iOS builds need acceptance.'])
    target=validation/'feedback_refresh_release.json';target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('qualified','model_sha256','backend','flutter_tests','android_apk')},indent=2))


if __name__=='__main__':main()
