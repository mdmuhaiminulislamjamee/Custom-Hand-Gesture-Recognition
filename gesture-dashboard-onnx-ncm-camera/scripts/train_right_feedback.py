"""Fit only the Right output of the released MLP; freeze all other weights.

Reviewed landmarks are training data, never independent evaluation evidence.
The existing participant-held-out validation split selects the update strength;
the untouched test split is scored only after selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import onnx
from onnx import numpy_helper
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax
from threadpoolctl import threadpool_limits

import scripts.train_stop_inverted as trainer
from research.v18_20 import load_npz, predict_onnx, feature_keys


def hidden(x, weights):
    z = (x - weights['mean']) / weights['scale']
    for i in range(2):
        z = np.maximum(z @ weights[f'w{i}'] + weights[f'b{i}'], 0)
    return z


def labels(p, m):
    ordered = np.sort(p, axis=1)
    top = p.argmax(1)
    accepted = (top != 6) & (m >= .70) & (ordered[:, -1] >= .75) & (ordered[:, -1] - ordered[:, -2] >= .10)
    return np.where(accepted, top, 13)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/right_stability_20261111')
    parser.add_argument('--previous', type=Path, default=ROOT / 'artifacts/direction_stability_20261111')
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    trainer.OUTPUT = directory
    baseline = directory / 'baseline/gesture_mlp_production.onnx'
    graph = onnx.load(baseline)
    weights = {v.name: numpy_helper.to_array(v).copy() for v in graph.graph.initializer}
    groups = {split: load_npz(args.previous / f'{split}.npz') for split in ('train', 'val', 'test')}
    feedback, feedback_audit = trainer.load_feedback()
    with (args.previous / 'feedback_snapshot.csv').open(encoding='utf-8-sig', newline='') as stream:
        previous_ids = {'feedback:' + row['feedback_id'] for row in csv.DictReader(stream)}
    extra = trainer.subset(feedback, ~np.isin(feedback['source_id'], list(previous_ids)))
    for split in ('val', 'test'):
        if set(feature_keys(groups[split]['X'])) & set(feature_keys(extra['X'])):
            raise ValueError('New feedback overlaps held-out features')
    groups['train'] = {k: np.concatenate([v, extra[k]]) for k, v in groups['train'].items()}
    for split, data in groups.items():
        np.savez_compressed(directory / f'{split}.npz', **data)
    shutil.copy2(ROOT / 'feedback/misclassification_log.csv', directory / 'feedback_snapshot.csv')
    audit = json.loads((args.previous / 'data_audit.json').read_text())
    audit['feedback'] = feedback_audit
    audit['right_only_update'] = dict(new_records=len(extra['y']), new_right=int(np.sum(extra['y'] == 1)),
        new_negatives=int(np.sum(extra['y'] == 13)), frozen='normalization, hidden layers, all non-Right output columns',
        training_source='Previous training split plus new reviewed landmarks; no test rows used for fitting/selection')
    trainer.write_json(directory / 'data_audit.json', audit)
    # Add the reviewed poses as weighted corrections, without changing or
    # resampling the other outputs' representation.
    x = np.concatenate([groups['train']['X'], feedback['X']])
    y = np.concatenate([groups['train']['y'], feedback['y']])
    h = hidden(x, weights).astype(np.float64)
    design = np.column_stack([h, np.ones(len(h))])
    logits = h @ weights['w2'] + weights['b2']
    offset = logsumexp(np.delete(logits, 1, axis=1), axis=1)
    original = np.r_[weights['w2'][:, 1], weights['b2'][1]].astype(np.float64)
    normalization = np.sqrt(np.mean(design ** 2, axis=0))
    normalization = np.maximum(normalization, 1.)
    a = design / normalization
    sample_weight = np.ones(len(y))
    sample_weight[len(groups['train']['y']):] = 20.
    sample_weight[y == 1] *= 2.
    sample_weight /= sample_weight.sum()
    target = (y == 1).astype(float)
    base_right = design @ original
    anchors = y != 1
    anchor_weight = anchors / max(1, anchors.sum())
    right_anchor_weight = (y == 1) / max(1, np.sum(y == 1))

    def objective(delta):
        change = a @ delta
        binary = base_right + change - offset
        loss = np.dot(sample_weight, np.logaddexp(0, binary) - target * binary)
        gradient = a.T @ (sample_weight * (expit(binary) - target))
        # Preserve other poses, strongly penalizing any extra Right evidence.
        positive = np.maximum(change, 0.)
        negative = np.minimum(change, 0.)
        loss += 2. * np.dot(anchor_weight, positive ** 2) + .002 * np.dot(delta, delta)
        gradient += 4. * a.T @ (anchor_weight * positive) + .004 * delta
        # Existing Right poses should not lose evidence while adding coverage.
        loss += 5. * np.dot(right_anchor_weight, negative ** 2)
        gradient += 10. * a.T @ (right_anchor_weight * negative)
        return loss, gradient

    result = minimize(objective, np.zeros(65), jac=True, method='L-BFGS-B', options={'maxiter': 250, 'ftol': 1e-12})
    print('Right-head optimization', result.message, 'iterations', result.nit, flush=True)
    delta = result.x / normalization
    val = groups['val']
    old_p, old_m = predict_onnx(baseline, val['X'])
    old_labels = labels(old_p, old_m)
    val_h = hidden(val['X'], weights)
    val_logits = val_h @ weights['w2'] + weights['b2']
    trials = []
    selected = None
    for fraction in (1., .5, .25, .125, .0625, .03125, .015625, .0078125, .00390625):
        proposed = val_logits.copy()
        proposed[:, 1] += fraction * (val_h @ delta[:-1] + delta[-1])
        all_p = softmax(proposed, axis=1)
        mass = all_p[:, :13].sum(1)
        conditional = all_p[:, :13] / mass[:, None]
        predicted = labels(conditional, mass)
        non_right_changes = int(np.sum((predicted != old_labels) & (val['y'] != 1)))
        right_regressions = int(np.sum((old_labels == 1) & (val['y'] == 1) & (predicted != 1)))
        trials.append(dict(fraction=fraction, non_right_changes=non_right_changes, right_regressions=right_regressions))
        if not non_right_changes and not right_regressions:
            selected = fraction
            break
    if selected is None:
        raise ValueError('No Right-head update preserves validation decisions: ' + str(trials))
    updated = original + selected * delta
    weights['w2'][:, 1] = updated[:-1]
    weights['b2'][1] = updated[-1]
    for value in graph.graph.initializer:
        if value.name in ('w2', 'b2'):
            value.CopyFrom(numpy_helper.from_array(weights[value.name], value.name))
    candidate = directory / 'candidate.onnx'
    onnx.checker.check_model(graph)
    onnx.save(graph, candidate)
    actual, mass = predict_onnx(candidate, val['X'])
    expected = softmax(hidden(val['X'], weights) @ weights['w2'] + weights['b2'], axis=1)
    expected_mass = expected[:, :13].sum(1)
    conditional = expected[:, :13] / expected_mass[:, None]
    evaluation = dict(artifact=dict(bytes=candidate.stat().st_size, sha256=hashlib.sha256(candidate.read_bytes()).hexdigest()),
        parity=dict(label_agreement=float(np.mean(conditional.argmax(1) == actual.argmax(1))),
            max_probability_error=float(np.max(np.abs(conditional - actual))), max_mass_error=float(np.max(np.abs(expected_mass - mass)))),
        audit=audit, deployed=False, history=[], right_head_training=dict(iterations=result.nit, success=bool(result.success),
            selected_fraction=selected, validation_trials=trials, fitted_parameters=65,
            all_non_right_weights_bit_identical=True))
    for split in ('val', 'test'):
        for runtime in (False, True):
            key = split + ('_runtime' if runtime else '')
            evaluation[key] = trainer.score(candidate, groups[split], runtime=runtime)
            print(key, evaluation[key]['macro_f1'], flush=True)
        before = labels(*predict_onnx(baseline, groups[split]['X']))
        after = labels(*predict_onnx(candidate, groups[split]['X']))
        evaluation['right_head_training'][split + '_non_right_decision_changes'] = int(np.sum((before != after) & (groups[split]['y'] != 1)))
    evaluation['feedback_training_fit'] = trainer.score(candidate, feedback)
    evaluation['feedback_training_fit_runtime'] = trainer.score(candidate, feedback, runtime=True)
    sources = load_npz(ROOT / 'data/ten_gesture/hagrid_train.npz')
    calls = sources['source_id'][sources['variant'] == 'source_call']
    evaluation['retired_call_test'] = trainer.score(candidate, trainer.subset(groups['test'], np.isin(groups['test']['source_id'], calls)))
    trainer.write_json(directory / 'evaluation.json', evaluation)
    print(json.dumps(evaluation['right_head_training'], indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
