"""Bind the Right-only release to training, regression and build evidence."""
from pathlib import Path
import csv
import hashlib
import json
import re
import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'artifacts/right_stability_20261111'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def log_text(path):
    data = path.read_bytes()
    return data.decode('utf-16' if data.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig', errors='replace')


def main():
    evaluation = read(DIRECTORY / 'evaluation.json')
    replay = read(DIRECTORY / 'candidate_feedback.json')
    preservation = read(DIRECTORY / 'preservation.json')
    comparison = read(DIRECTORY / 'baseline_comparison.json')
    model = ROOT / 'models/gesture_mlp_production.onnx'
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    assert evaluation['deployed'] and all(evaluation['release_checks'].values())
    assert digest == evaluation['artifact']['sha256'] == replay['model_sha256'] == preservation['candidate_sha256']
    before = {v.name: numpy_helper.to_array(v) for v in onnx.load(DIRECTORY / 'baseline/gesture_mlp_production.onnx').graph.initializer}
    after = {v.name: numpy_helper.to_array(v) for v in onnx.load(model).graph.initializer}
    for name in before:
        a, b = before[name], after[name]
        if name == 'w2':
            a, b = np.delete(a, 1, axis=1), np.delete(b, 1, axis=1)
        elif name == 'b2':
            a, b = np.delete(a, 1), np.delete(b, 1)
        assert np.array_equal(a, b), 'Non-Right weights changed: ' + name
    for split in preservation['splits'].values():
        for mode in ('scalar', 'runtime'):
            assert split[mode]['non_right_label_changes'] == 0
    backend_dir = ROOT / '.runtime/right_stability_v210_backend'
    backend = read(backend_dir / 'summary.json')
    assert backend['completed'] == backend['files']
    baseline_failures = set(re.findall(r'^FAILED (\S+)', log_text(DIRECTORY / 'baseline_known_failures.log'), re.M))
    failures = set()
    for path in backend_dir.glob('test_*.log'):
        failures.update(re.findall(r'^FAILED (\S+)', log_text(path), re.M))
    assert baseline_failures and not failures - baseline_failures, str(failures - baseline_failures)
    host_dir = ROOT / '.runtime/right_stability_v210_host'
    host = read(host_dir / 'results.json')
    assert len(host) == 6 and not any(r['exit_code'] for r in host)
    counts = {p: int(re.findall(r'\+(\d+): All tests passed!', log_text(host_dir / (p + '_tests.log')))[-1]) for p in ('runtime', 'tester')}
    sources = {}
    for relative, expected in replay['runtime_sources'].items():
        if relative == 'models/gesture_mobile_runtime_config.json':
            continue  # Installation updates the release record, not scalar gates.
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, 'Runtime changed after image replay: ' + relative
        sources[relative] = actual
    image = replay['image_replay']
    jpeg = {}
    for label in ('left', 'right'):
        rows = [r for r in image['rows'] if r['actual'] == label]
        jpeg[label] = dict(rows=len(rows), correct=sum(r['predicted'] == label for r in rows), actions=sum(r['action_seen'] for r in rows))
    with (ROOT / 'artifacts/direction_stability_20261111/feedback_snapshot.csv').open(encoding='utf-8-sig', newline='') as stream:
        old_ids = {r['feedback_id'] for r in csv.DictReader(stream)}
    fresh = [r for r in image['rows'] if r['actual'] == 'right' and r['id'] not in old_ids]
    jpeg['new_right'] = dict(rows=len(fresh), correct=sum(r['predicted'] == 'right' for r in fresh), actions=sum(r['action_seen'] for r in fresh))
    diagnostic = read(DIRECTORY / 'diagnostic_baseline.json')['image_replay']
    diagnostic_after = read(DIRECTORY / 'diagnostic_rules.json')['image_replay']
    report = dict(schema_version=1, qualified=True, model_sha256=digest,
        previous_model_sha256=preservation['previous_sha256'],
        training=dict(**evaluation['audit']['feedback'], new_right_records=331, new_hard_negative_records=90,
            fitted_parameters=65, non_right_weights_bit_identical=True,
            source='Reviewed 21x2 landmarks train the MLP; saved RGB images test the detector/runtime',
            feedback_is_independent_validation=False),
        policy=dict(right_side_view_requires_metric_world=True, confidence=.95, known_mass=.70,
            index_straightness=.92, axis_dominance=.86, index_lead=.50,
            relative_index_reach=.75, maximum_other_projected_reach=.15,
            missing_depth_cannot_use_entry_exception=True, continuation_grace_ms=200,
            invalid_frames_to_release=2, grace_emits_command=False,
            target_fps=10, fixed_action_cooldown_ms=0, added_model_or_detector_passes=0),
        preservation=preservation, heldout_metrics=comparison,
        jpeg_replay=dict(total=image['total'], correct=image['correct'], actions=image['correct_action'], per_class=jpeg,
            sampled_new_right_before=dict(correct=diagnostic['correct'], actions=diagnostic['correct_action'], rows=diagnostic['total']),
            sampled_new_right_rule_diagnosis=dict(correct=diagnostic_after['correct'], actions=diagnostic_after['correct_action'], rows=diagnostic_after['total']),
            scope='Reviewed training JPEG static holds; not independent live-camera accuracy'),
        feedback_runtime=dict(metric_world_rows=replay['metric_world_rows'], summary=replay['summary'], stress_counts=replay['stress']['counts'],
            scope='Matching JPEG redetection supplies metric world for annotated XY; metric world XY is reflected/rotated with synthetic stress variants'),
        left_stress_preservation=replay['left_stress_preservation'],
        release_checks=evaluation['release_checks'],
        backend=dict(passed=backend['passed'], current_failures=sorted(failures), baseline_failures=sorted(baseline_failures),
            new_failures=sorted(failures - baseline_failures), all_tests_passed=not failures),
        flutter_tests=counts,
        conversion=read(ROOT / 'static_gesture_runtime/validation/conversion_report.json')['models']['float32'],
        classifier_timing=read(ROOT / 'static_gesture_runtime/validation/latency_comparison.json'),
        rule_timing=read(ROOT / 'static_gesture_runtime/validation/direction_rule_latency.json'),
        runtime_source_sha256=sources,
        limits=['Saved feedback, JPEG holds and synthetic transforms are training regressions, not new people or live accuracy.',
            'The Right entry exception needs metric world landmarks; missing/contradictory depth remains rejected.',
            'Existing backend failures are retained; physical phone/NCM latency and iOS build acceptance remain pending.'])
    destination = ROOT / 'static_gesture_runtime/validation/right_stability_release.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('qualified', 'model_sha256', 'jpeg_replay', 'backend', 'flutter_tests')}, indent=2))


if __name__ == '__main__':
    main()
