"""Compare exact qualified labels with the pre-Right model and geometry."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_direction_release import original_module
from scripts.train_right_feedback import labels
from research.v18_20 import load_npz, predict_onnx
from backend.config import load_runtime_config
from backend.geometry import GeometryResolver
from scripts.train_stop_inverted import load_feedback


def main():
    directory = ROOT / 'artifacts/right_stability_20261111'
    previous = original_module('backend.geometry').GeometryResolver
    config = load_runtime_config()
    feedback, _ = load_feedback()
    with (ROOT / 'artifacts/direction_stability_20261111/feedback_snapshot.csv').open(encoding='utf-8-sig', newline='') as stream:
        old_ids = {'feedback:' + row['feedback_id'] for row in csv.DictReader(stream)}
    data_sets = {s: load_npz(directory / f'{s}.npz') for s in ('val', 'test')}
    data_sets['previous_feedback'] = {k: v[np.isin(feedback['source_id'], list(old_ids))] for k, v in feedback.items()}
    data_sets['new_feedback'] = {k: v[~np.isin(feedback['source_id'], list(old_ids))] for k, v in feedback.items()}
    report = dict(previous_sha256=hashlib.sha256((directory / 'baseline/gesture_mlp_production.onnx').read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256((directory / 'candidate.onnx').read_bytes()).hexdigest(), splits={})
    for split, data in data_sets.items():
        outputs = {}
        for name, model, resolver_type in (
            ('previous', directory / 'baseline/gesture_mlp_production.onnx', previous),
            ('candidate', directory / 'candidate.onnx', GeometryResolver)):
            p, m = predict_onnx(model, data['X'])
            scalar = labels(p, m)
            adjusted, valid = p.copy(), np.ones(len(p), bool)
            resolver = resolver_type(config)
            for i, points in enumerate(data['landmarks']):
                side = str(data['handedness'][i]) or None
                adjusted[i], details = resolver.resolve(p[i], points,
                    handedness=side, handedness_confidence=.99 if side else None)
                valid[i] = details['pose_validation']['valid']
            runtime = np.where(valid, labels(adjusted, m), 13)
            outputs[name] = dict(scalar=scalar, runtime=runtime)
        row = {}
        truth = np.where(data['y'] == 6, 13, data['y'])
        for mode in ('scalar', 'runtime'):
            before, after = outputs['previous'][mode], outputs['candidate'][mode]
            non_right = data['y'] != 1
            changed = np.flatnonzero(non_right & (before != after))
            regressions = np.flatnonzero(non_right & (before == truth) & (after != truth))
            row[mode] = dict(rows=len(truth), non_right_rows=int(non_right.sum()),
                non_right_label_changes=len(changed), non_right_correct_to_wrong=len(regressions),
                right_before_correct=int(np.sum((truth == 1) & (before == 1))),
                right_after_correct=int(np.sum((truth == 1) & (after == 1))),
                changes=[dict(id=str(data['source_id'][i]), actual=int(truth[i]), previous=int(before[i]), candidate=int(after[i])) for i in changed])
        report['splits'][split] = row
        print(split, json.dumps(row), flush=True)
    destination = directory / 'preservation.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
