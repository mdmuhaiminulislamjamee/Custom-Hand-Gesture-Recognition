"""Fit unused reviewed feedback while retaining the released encoder and graph.

Only affected output columns are fitted. Existing participant-held-out validation
selects a preservation-constrained step; test data is scored after selection.
Reviewed bursts remain training data and do not establish live-camera accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
from scipy.special import logsumexp, softmax
from threadpoolctl import threadpool_limits

from backend.config import load_runtime_config
from backend.geometry import landmarks_to_feature
from research.v18_20 import feature_keys, load_npz, predict_onnx
import scripts.train_stop_inverted as trainer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hidden(x, w):
    h = (x - w['mean']) / w['scale']
    for i in range(2):
        h = np.maximum(h @ w[f'w{i}'] + w[f'b{i}'], 0)
    return h


def gated(p, mass):
    config = load_runtime_config()
    rank = np.sort(p, axis=1)
    top = p.argmax(1)
    accept = ((top != 6) & (mass >= config.known_mass_floor)
              & (rank[:, -1] >= config.confidence_floor)
              & (rank[:, -1] - rank[:, -2] >= config.probability_margin_floor))
    return np.where(accept, top, 13)


def from_logits(logits):
    p = softmax(logits, axis=1)
    mass = p[:, :13].sum(1)
    return p[:, :13] / np.maximum(mass[:, None], 1e-30), mass


def preserves_metrics(truth, before, after):
    truth = np.where(truth == 6, 13, truth)
    if np.any((before == truth) & (after != truth)):
        return False
    for label in range(14):
        if label == 6: continue
        old_tp = int(np.sum((before == label) & (truth == label)))
        new_tp = int(np.sum((after == label) & (truth == label)))
        old_precision = old_tp / max(1, int(np.sum(before == label)))
        new_precision = new_tp / max(1, int(np.sum(after == label)))
        if new_tp < old_tp or new_precision + 1e-12 < old_precision:
            return False
    return True


def feedback(path):
    records = {}
    with path.open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            fid = row['feedback_id']
            if fid in records and records[fid] != row:
                raise ValueError('Conflicting duplicate feedback ID: ' + fid)
            records[fid] = row
    poses, labels, ids, skipped = [], [], [], []
    for fid, row in records.items():
        try:
            points = np.asarray(json.loads(row['landmarks_json']), np.float32)
            if points.shape != (21, 2) or not np.isfinite(points).all():
                raise ValueError('missing finite 21x2 landmarks')
            landmarks_to_feature(points)
            label = 'no_gesture' if row['actual_label'] == 'dorsal' else row['actual_label']
            target = trainer.LABELS.index(label)
        except (ValueError, TypeError, KeyError) as error:
            skipped.append({'id': fid, 'reason': str(error)})
            continue
        poses.append(points); labels.append(target); ids.append('feedback:' + fid)
    data = trainer.make_data(poses, labels, ['local-feedback:unidentified-session'] * len(poses), ids)
    return records, data, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/feedback_refresh_20261111')
    parser.add_argument('--previous', type=Path, default=ROOT / 'artifacts/right_stability_20261111')
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    baseline_dir = out / 'baseline'; baseline_dir.mkdir(exist_ok=True)
    for name in ('gesture_mlp_production.onnx', 'gesture_mlp_onnx_metadata.json',
                 'gesture_mobile_runtime_config.json', 'gesture_artifact_manifest.json',
                 'gesture_online_replay_cache.npz', 'gesture_online_validation_cache.npz',
                 'gesture_online_untouched_test_cache.npz'):
        target = baseline_dir / name
        if target.exists() and sha(target) != sha(ROOT / 'models' / name):
            raise ValueError('Preserved baseline differs from current release: ' + name)
        if not target.exists(): shutil.copy2(ROOT / 'models' / name, target)
    snapshot = out / 'feedback_snapshot.csv'
    shutil.copy2(ROOT / 'feedback/misclassification_log.csv', snapshot)
    records, reviewed, skipped = feedback(snapshot)
    ledger = args.previous / 'used_feedback_ids.json'
    if ledger.exists():
        used_ids = {'feedback:' + fid for fid in json.loads(ledger.read_text())}
    else:
        _, previously_usable, _ = feedback(args.previous / 'feedback_snapshot.csv')
        used_ids = set(map(str, previously_usable['source_id']))
    new = trainer.subset(reviewed, ~np.isin(reviewed['source_id'], list(used_ids)))
    old = trainer.subset(reviewed, np.isin(reviewed['source_id'], list(used_ids)))
    if not len(new['y']): raise ValueError('No unused usable feedback')
    groups = {s: load_npz(args.previous / f'{s}.npz') for s in ('train', 'val', 'test')}
    heldout_hashes = {s: sha(args.previous / f'{s}.npz') for s in ('val', 'test')}
    for s in ('val', 'test'):
        if set(feature_keys(groups[s]['X'])) & set(feature_keys(new['X'])):
            raise ValueError('Unused feedback overlaps held-out features: ' + s)
        shutil.copy2(args.previous / f'{s}.npz', out / f'{s}.npz')
    combined_train = {k: np.concatenate([v, new[k]]) for k, v in groups['train'].items()}
    np.savez_compressed(out / 'train.npz', **combined_train)
    baseline = baseline_dir / 'gesture_mlp_production.onnx'
    graph = onnx.load(baseline)
    w = {v.name: numpy_helper.to_array(v).copy() for v in graph.graph.initializer}
    columns = sorted(set(map(int, new['y'])))
    if 6 in columns: raise ValueError('Retired Dorsal output must remain frozen')
    counts = Counter(map(int, new['y']))
    audit = {'feedback_sha256': sha(snapshot), 'baseline_model_sha256': sha(baseline),
             'previous_snapshot_sha256': sha(args.previous / 'feedback_snapshot.csv'),
             'valid_unique_feedback': len(reviewed['y']), 'unused_records': len(new['y']),
             'unused_class_counts': {trainer.LABELS[k]: v for k, v in counts.items()},
             'unused_ids': [str(v).removeprefix('feedback:') for v in new['source_id']],
             'missing_unused_jpegs': sum(not Path(records[str(v).removeprefix('feedback:')]['image_path']).is_file() for v in new['source_id']),
             'skipped': skipped, 'heldout_split_sha256': heldout_hashes,
             'fitted_columns': [trainer.LABELS[k] for k in columns],
             'heldout_overlap': 0, 'feedback_evaluation_use': 'training regression only',
             'test_used_for_selection': False, 'runtime_rules_changed': False}
    trainer.write_json(out / 'data_audit.json', audit)
    trainer.write_json(out / 'used_feedback_ids.json', sorted(str(v).removeprefix('feedback:') for v in reviewed['source_id']))
    with (out / 'new_feedback.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(records.values()))))
        writer.writeheader(); writer.writerows(records[str(v).removeprefix('feedback:')] for v in new['source_id'])
    print('Unused feedback:', audit['unused_class_counts'], flush=True)

    htrain = hidden(groups['train']['X'], w).astype(np.float64)
    normalization = np.maximum(np.sqrt(np.mean(np.column_stack([htrain, np.ones(len(htrain))]) ** 2, axis=0)), 1.)
    anchor = np.column_stack([htrain, np.ones(len(htrain))]) / normalization
    class_counts = Counter(map(int, groups['train']['y']))
    anchor_weights = np.asarray([1. / class_counts[int(y)] for y in groups['train']['y']])
    anchor_weights /= anchor_weights.sum()
    gram = anchor.T @ (anchor * anchor_weights[:, None])
    hn = hidden(new['X'], w).astype(np.float64)
    design = np.column_stack([hn, np.ones(len(hn))]) / normalization
    original_logits = hn @ w['w2'] + w['b2']
    weights = np.asarray([1. / counts[int(y)] for y in new['y']]); weights /= weights.sum()
    eval_data = {'val': groups['val'], 'old_feedback': old, 'new_feedback': new}
    eval_h = {s: hidden(d['X'], w).astype(np.float64) for s, d in eval_data.items()}
    eval_logits = {s: h @ w['w2'] + w['b2'] for s, h in eval_h.items()}
    eval_design = {s: np.column_stack([h, np.ones(len(h))]) / normalization for s, h in eval_h.items()}
    before = {s: gated(*predict_onnx(baseline, d['X'])) for s, d in eval_data.items()}
    guard = before['old_feedback'] == old['y']
    guard_logits = eval_logits['old_feedback'][guard]
    guard_design = eval_design['old_feedback'][guard]
    guard_y = old['y'][guard]
    guard_nll = logsumexp(guard_logits, axis=1) - guard_logits[np.arange(len(guard_y)), guard_y]
    guard_counts = Counter(map(int, guard_y))
    guard_weight = np.asarray([1. / guard_counts[int(y)] for y in guard_y]); guard_weight /= guard_weight.sum()
    other_design = eval_design['old_feedback'][~np.isin(old['y'], columns)]
    other_gram = other_design.T @ other_design / max(1, len(other_design))
    base_new_correct = int(np.sum(before['new_feedback'] == new['y']))
    trials, proposals = [], []
    for penalty in (.3, .03, .003):
        def objective(flat):
            delta = flat.reshape(65, len(columns))
            logits = original_logits.copy(); logits[:, columns] += design @ delta
            prob = softmax(logits, axis=1)
            loss = np.dot(weights, logsumexp(logits, axis=1) - logits[np.arange(len(logits)), new['y']])
            prob[np.arange(len(prob)), new['y']] -= 1.
            gradient = design.T @ (weights[:, None] * prob[:, columns])
            loss += penalty * np.sum(delta * (gram @ delta)) + .00005 * np.sum(delta ** 2)
            gradient += 2 * penalty * gram @ delta + .0001 * delta
            protected = guard_logits.copy(); protected[:, columns] += guard_design @ delta
            nll = logsumexp(protected, axis=1) - protected[np.arange(len(guard_y)), guard_y]
            hinge = np.maximum(nll - guard_nll, 0.)
            loss += 30. * np.dot(guard_weight, hinge ** 2) + .05 * np.sum(delta * (other_gram @ delta))
            guard_error = softmax(protected, axis=1)
            guard_error[np.arange(len(guard_y)), guard_y] -= 1.
            gradient += 60. * guard_design.T @ ((guard_weight * hinge)[:, None] * guard_error[:, columns])
            gradient += .1 * other_gram @ delta
            return float(loss), gradient.ravel()
        fit = minimize(objective, np.zeros(65 * len(columns)), jac=True, method='L-BFGS-B',
                       options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-7})
        delta = fit.x.reshape(65, len(columns))
        for power in range(14):
            fraction = 2. ** -power
            pred = {}
            for s in eval_data:
                logits = eval_logits[s].copy(); logits[:, columns] += fraction * (eval_design[s] @ delta)
                pred[s] = gated(*from_logits(logits))
            val_changes = int(np.sum(pred['val'] != before['val']))
            val_preserved = preserves_metrics(groups['val']['y'], before['val'], pred['val'])
            old_regressions = int(np.sum((before['old_feedback'] == old['y']) & (pred['old_feedback'] != old['y'])))
            old_other_changes = int(np.sum((pred['old_feedback'] != before['old_feedback']) & ~np.isin(old['y'], columns)))
            correct = int(np.sum(pred['new_feedback'] == new['y']))
            balanced = float(np.mean([np.mean(pred['new_feedback'][new['y'] == k] == k) for k in columns]))
            row = dict(penalty=penalty, fraction=fraction, validation_label_changes=val_changes, validation_metrics_preserved=val_preserved,
                       old_feedback_correct_to_wrong=old_regressions, old_unaffected_label_changes=old_other_changes,
                       unused_correct=correct, unused_balanced_recall=balanced, optimizer_success=bool(fit.success), iterations=fit.nit)
            trials.append(row)
            if val_preserved and old_regressions == old_other_changes == 0 and correct > base_new_correct:
                proposals.append((balanced, correct, row, delta * fraction / normalization[:, None]))
        print('Completed head fit', penalty, 'iterations', fit.nit, flush=True)
    proposals.sort(key=lambda r: (r[0], r[1]), reverse=True)
    selected = None
    for _, _, row, delta in proposals:
        candidate_graph = onnx.load(baseline)
        candidate_w = {k: v.copy() for k, v in w.items()}
        candidate_w['w2'][:, columns] += delta[:-1].astype(np.float32)
        candidate_w['b2'][columns] += delta[-1].astype(np.float32)
        for v in candidate_graph.graph.initializer:
            if v.name in ('w2', 'b2'): v.CopyFrom(numpy_helper.from_array(candidate_w[v.name], v.name))
        candidate = out / 'candidate.onnx'; onnx.checker.check_model(candidate_graph); onnx.save(candidate_graph, candidate)
        actual = {s: gated(*predict_onnx(candidate, d['X'])) for s, d in eval_data.items()}
        if (preserves_metrics(groups['val']['y'], before['val'], actual['val']) and
            not np.any((before['old_feedback'] == old['y']) & (actual['old_feedback'] != old['y'])) and
            not np.any((actual['old_feedback'] != before['old_feedback']) & ~np.isin(old['y'], columns))):
            selected = row; break
    trainer.write_json(out / 'validation_trials.json', trials)
    if selected is None: raise ValueError('No improving update preserved validation and old feedback; production unchanged')
    frozen = all(np.array_equal(w[k], candidate_w[k]) for k in w if k not in ('w2', 'b2'))
    untouched = [i for i in range(14) if i not in columns]
    frozen &= np.array_equal(w['w2'][:, untouched], candidate_w['w2'][:, untouched]) and np.array_equal(w['b2'][untouched], candidate_w['b2'][untouched])
    print('Selected', selected, flush=True)
    report = {'audit': audit, 'selection': selected, 'fitted_parameters': 65 * len(columns),
              'frozen_weights_identical': bool(frozen), 'model_sha256': sha(candidate), 'model_bytes': candidate.stat().st_size,
              'baseline_bytes': baseline.stat().st_size, 'same_graph_operations': [n.op_type for n in graph.graph.node] == [n.op_type for n in candidate_graph.graph.node],
              'splits': {}, 'deployed': False}
    for split in ('val', 'test', 'old_feedback', 'new_feedback'):
        data = groups[split] if split in groups else eval_data[split]
        report['splits'][split] = {}
        for mode in ('scalar', 'runtime'):
            a = trainer.score(baseline, data, runtime=mode == 'runtime')
            b = trainer.score(candidate, data, runtime=mode == 'runtime')
            report['splits'][split][mode] = {'baseline': a, 'candidate': b}
            print(split, mode, 'F1', a['macro_f1'], '->', b['macro_f1'], flush=True)
    checks = {'frozen_encoder_and_other_heads': bool(frozen), 'same_graph': report['same_graph_operations'],
              'same_model_bytes': report['model_bytes'] == report['baseline_bytes']}
    for mode in ('scalar', 'runtime'):
        a, b = (report['splits']['test'][mode][n] for n in ('baseline', 'candidate'))
        checks[mode + '_test_per_class_preserved'] = all(b['per_class'][name][key] + 1e-12 >= a['per_class'][name][key]
            for name in trainer.LABELS if name != 'dorsal' for key in ('precision', 'recall', 'f1-score'))
        checks[mode + '_test_accepted_precision_preserved'] = b['accepted_precision'] + 1e-12 >= a['accepted_precision']
        checks[mode + '_test_unknown_false_accept_preserved'] = b['unknown_false_accept_rate'] <= a['unknown_false_accept_rate'] + 1e-12
    p, mass = predict_onnx(candidate, groups['val']['X'])
    expected_p, expected_m = from_logits(eval_h['val'] @ candidate_w['w2'] + candidate_w['b2'])
    report['parity'] = {'label_agreement': float(np.mean(p.argmax(1) == expected_p.argmax(1))),
                        'max_probability_error': float(np.max(np.abs(p - expected_p))),
                        'max_mass_error': float(np.max(np.abs(mass - expected_m)))}
    checks['onnx_parity'] = report['parity']['label_agreement'] == 1. and report['parity']['max_probability_error'] < 1e-4
    checks['heldout_files_unchanged'] = all(sha(out / f'{s}.npz') == h for s, h in heldout_hashes.items())
    report['release_checks'] = checks; report['qualified'] = all(checks.values())
    trainer.write_json(out / 'evaluation.json', report)
    print('Qualification', checks, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
