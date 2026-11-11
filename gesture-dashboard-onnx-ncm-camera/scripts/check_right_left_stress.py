"""Check unchanged Left behavior on the expanded Right mirror stress set."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_directional_feedback import stress_audit
from scripts.compare_direction_release import original_module


def key(row):
    return (row['id'], row['mirror'], row['tilt'], tuple(row['shape']))


def main():
    directory = ROOT / 'artifacts/right_stability_20261111'
    path = directory / 'candidate_feedback.json'
    current = json.loads(path.read_text())
    baseline_model = directory / 'baseline/gesture_mlp_production.onnx'
    old_resolver = original_module('backend.geometry').GeometryResolver
    previous = stress_audit(baseline_model, {r['id']: r for r in current['image_replay']['rows']},
        target_labels={'left'}, resolver_type=old_resolver)
    before_failures = {key(r) for r in previous['failures']}
    after_failures = {key(r) for r in current['stress']['failures'] if r['label'] == 'left'}
    baseline_comparison = json.loads((directory / 'baseline_comparison.json').read_text())
    result = dict(previous_model_sha256=hashlib.sha256(baseline_model.read_bytes()).hexdigest(),
        candidate_model_sha256=current['model_sha256'],
        baseline_geometry_sha256=baseline_comparison['baseline_geometry_sha256'],
        baseline_counts=previous['counts'], current_counts={k: v for k, v in current['stress']['counts'].items() if k.startswith('left_')},
        additional_failed_left_cases=len(after_failures - before_failures),
        resolved_left_cases=len(before_failures - after_failures),
        scope='Expanded synthetic Left set includes mirrors of the new Right-only feedback; require preservation of every previously correct case')
    current['left_stress_preservation'] = result
    path.write_text(json.dumps(current, indent=2) + '\n')
    (directory / 'left_stress_baseline.json').write_text(json.dumps(previous, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
