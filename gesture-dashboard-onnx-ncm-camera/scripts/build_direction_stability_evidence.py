"""Record current direction training, qualification and baseline evidence."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def log_text(path):
    data = path.read_bytes()
    return data.decode('utf-16' if data.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig', errors='replace')


def main():
    directory = Path('artifacts/direction_stability_20261111')
    evaluation = read(directory / 'evaluation.json')
    comparison = read(directory / 'baseline_comparison.json')
    replay = read(directory / 'candidate_feedback.json')
    initial = read(directory / 'first_audit.json')
    backend_dir = ROOT / '.runtime/direction_stability_v209_installed_backend'
    backend = json.loads((backend_dir / 'summary.json').read_text())
    if backend['completed'] != backend['files']:
        raise ValueError('Installed-model backend checks are still running')
    baseline_text = log_text(ROOT / '.runtime/direction_stability_baseline_failures.log')
    baseline_failures = set(re.findall(r'^FAILED (\S+)', baseline_text, re.M))
    failures = set()
    for path in backend_dir.glob('test_*.log'):
        failures.update(re.findall(r'^FAILED (\S+)', log_text(path), re.M))
    new_failures = sorted(failures - baseline_failures)
    if new_failures or not baseline_failures:
        raise ValueError('New regressions or missing baseline evidence: ' + str(new_failures))
    host = read('.runtime/direction_stability_v209_host/results.json')
    if len(host) != 6 or any(row['exit_code'] for row in host):
        raise ValueError('Flutter host checks have not all passed')
    model_sha = hashlib.sha256((ROOT / 'models/gesture_mlp_production.onnx').read_bytes()).hexdigest()
    if (evaluation['artifact']['sha256'] != model_sha or replay['model_sha256'] != model_sha
            or not evaluation.get('deployed') or not all(evaluation['release_checks'].values())):
        raise ValueError('The installed model lacks current release qualification')
    source_hashes = {}
    for relative, expected in replay['runtime_sources'].items():
        # Installation changes the configuration's release record/model hash.
        # The behavior sources must still be exactly those used for JPEG replay.
        if relative == 'models/gesture_mobile_runtime_config.json':
            continue
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Runtime changed since JPEG replay: ' + relative)
        source_hashes[relative] = actual
    conversion = read('static_gesture_runtime/validation/conversion_report.json')
    timing = read('static_gesture_runtime/validation/latency_comparison.json')
    rule_timing = read('static_gesture_runtime/validation/direction_rule_latency.json')
    metrics = {}
    for split, values in comparison['splits'].items():
        before, rule_only, after = (values[k] for k in (
            'previous_model_previous_rules', 'previous_model_current_rules', 'candidate_model_current_rules'))
        if before['confusion'] != rule_only['confusion']:
            raise ValueError('Stateless rules regressed the preserved-model landmark contract')
        metrics[split] = {name: {key: row[key] for key in (
            'macro_f1', 'accepted_precision', 'unknown_false_accept_rate', 'per_class')}
            for name, row in [('previous_scalar', values['previous_model_scalar']),
                              ('retrained_scalar', values['candidate_model_scalar']),
                              ('previous_runtime', before), ('retrained_runtime', after)]}
    counts = {}
    for project in ('runtime', 'tester'):
        text = log_text(ROOT / f'.runtime/direction_stability_v209_host/{project}_tests.log')
        counts[project] = int(re.findall(r'\+(\d+): All tests passed!', text)[-1])
    report = dict(schema_version=1, qualified=True, model_sha256=model_sha,
        previous_model_sha256=comparison['model_sha256']['previous'],
        training=dict(usable_feedback_records=evaluation['audit']['feedback']['rows'],
            skipped_feedback_records=len(evaluation['audit']['feedback']['skipped']),
            feedback_sha256=evaluation['audit']['feedback']['sha256'],
            per_class=evaluation['audit']['feedback']['class_counts'],
            source='Saved reviewed 21x2 landmarks; the MLP does not train on RGB pixels',
            feedback_used_only_for_training=True),
        policy=dict(entry_axis_dominance=.73, continuation_signed_axis_alignment=.65,
            marginal_grace_ms=200, invalid_observations_to_release_direction=2,
            grace_can_emit_new_command=False, missing_hand_clears_grace=True,
            existing_positive_3d_turn_preserved=True, extra_prediction_panels=False,
            target_fps=10, added_detector_or_classifier_passes=0),
        release_checks=evaluation['release_checks'],
        jpeg_replay=dict(total=replay['image_replay']['total'],
            previous_correct=initial['image_replay']['correct'], correct=replay['image_replay']['correct'],
            previous_actions=initial['image_replay']['correct_action'], actions=replay['image_replay']['correct_action'],
            scope='Reviewed training JPEGs replayed as static holds; not independent live accuracy'),
        preserved_model_stateless_rule_predictions_unchanged=True, holdout_metrics=metrics,
        backend=dict(passed=backend['passed'], current_failures=sorted(failures),
            baseline_failures=sorted(baseline_failures), new_failures=new_failures,
            resolved_baseline_failures=sorted(baseline_failures-failures), all_tests_passed=not failures),
        flutter_tests=counts, conversion=conversion['models']['float32'],
        classifier_timing=timing, rule_timing=rule_timing, runtime_source_sha256=source_hashes,
        limits=['Per-class metrics can move by a few samples after retraining; all metrics are retained here.',
                'Existing backend failures remain listed; no new backend failures were observed.',
                'Physical phone/NCM latency, lighting/angle coverage and iOS build acceptance remain pending.'])
    destination = ROOT / 'static_gesture_runtime/validation/direction_stability_release.json'
    destination.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('qualified','model_sha256','training','jpeg_replay','backend','flutter_tests')}, indent=2))


if __name__ == '__main__':
    main()
