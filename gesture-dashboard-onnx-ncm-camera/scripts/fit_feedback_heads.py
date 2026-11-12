"""Try individual output corrections using training data and validation only."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import onnx
from onnx import numpy_helper
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from threadpoolctl import threadpool_limits
from research.v18_20 import load_npz, predict_onnx
from scripts.train_feedback_refresh import feedback, from_logits, gated, hidden, preserves_metrics
import scripts.train_stop_inverted as trainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, default=ROOT / 'artifacts/feedback_refresh_20261111')
    args = p.parse_args(); out = args.directory
    audit = json.loads((out / 'data_audit.json').read_text())
    _, reviewed, _ = feedback(out / 'feedback_snapshot.csv')
    is_new = np.isin(reviewed['source_id'], ['feedback:' + s for s in audit['unused_ids']])
    data = {'new': trainer.subset(reviewed, is_new), 'old': trainer.subset(reviewed, ~is_new), 'val': load_npz(out / 'val.npz')}
    train = load_npz(ROOT / 'artifacts/right_stability_20261111/train.npz')
    baseline = out / 'baseline/gesture_mlp_production.onnx'
    original = {v.name: numpy_helper.to_array(v).copy() for v in onnx.load(baseline).graph.initializer}
    current = {k: v.copy() for k, v in original.items()}
    htrain = hidden(train['X'], original).astype(np.float64)
    scale = np.maximum(np.sqrt(np.mean(np.column_stack([htrain, np.ones(len(htrain))]) ** 2, axis=0)), 1.)
    a = np.column_stack([htrain, np.ones(len(htrain))]) / scale
    gram = a.T @ a / len(a)
    h = {k: hidden(d['X'], original).astype(np.float64) for k, d in data.items()}
    design = {k: np.column_stack([v, np.ones(len(v))]) / scale for k, v in h.items()}
    before = {k: gated(*predict_onnx(baseline, d['X'])) for k, d in data.items()}
    all_trials, selected = [], []
    final = out / 'coordinate_candidate.onnx'
    for column in (12, 9, 3, 5, 13):
        logits = {k: v @ current['w2'] + current['b2'] for k, v in h.items()}
        current_new = gated(*from_logits(logits['new']))
        target = (data['new']['y'] == column).astype(float)
        weight = np.where(target == 1, .5 / max(1, target.sum()), .5 / max(1, (1-target).sum()))
        offset = logsumexp(np.delete(logits['new'], column, axis=1), axis=1)
        old_sign = np.where(data['old']['y'] == column, -1., 1.)
        candidates = []
        for penalty in (.03, .003, .0003):
            def objective(delta):
                change = design['new'] @ delta
                binary = logits['new'][:, column] + change - offset
                loss = np.dot(weight, np.logaddexp(0, binary) - target * binary)
                grad = design['new'].T @ (weight * (expit(binary) - target))
                harmful = np.maximum(old_sign * (design['old'] @ delta), 0.)
                loss += penalty * np.dot(delta, gram @ delta) + .0001 * np.dot(delta, delta) + 20. * np.mean(harmful ** 2)
                grad += 2 * penalty * gram @ delta + .0002 * delta + 40. * design['old'].T @ (old_sign * harmful) / len(harmful)
                return float(loss), grad
            fit = minimize(objective, np.zeros(65), jac=True, method='L-BFGS-B', options={'maxiter':800,'ftol':1e-12})
            for power in range(14):
                fraction = 2. ** -power; delta = fit.x * fraction
                pred = {}
                for k in data:
                    proposal = logits[k].copy(); proposal[:, column] += design[k] @ delta
                    pred[k] = gated(*from_logits(proposal))
                valid = preserves_metrics(data['val']['y'], before['val'], pred['val'])
                old_ok = preserves_metrics(data['old']['y'], before['old'], pred['old'])
                new_ok = not np.any((current_new == data['new']['y']) & (pred['new'] != data['new']['y']))
                correct = int(np.sum(pred['new'] == data['new']['y']))
                row = dict(label=trainer.LABELS[column], penalty=penalty, fraction=fraction, validation_preserved=valid,
                    old_feedback_preserved=old_ok, new_correct_preserved=bool(new_ok), new_correct=correct,
                    iterations=fit.nit, optimizer_success=bool(fit.success))
                all_trials.append(row)
                if valid and old_ok and new_ok and correct > int(np.sum(current_new == data['new']['y'])):
                    candidates.append((correct,row,delta/scale))
        candidates.sort(key=lambda r:r[0], reverse=True)
        accepted = None
        for _, row, delta in candidates:
            w = {k:v.copy() for k,v in current.items()}
            w['w2'][:,column] += delta[:-1].astype(np.float32); w['b2'][column] += np.float32(delta[-1])
            graph = onnx.load(baseline)
            for tensor in graph.graph.initializer:
                if tensor.name in ('w2','b2'): tensor.CopyFrom(numpy_helper.from_array(w[tensor.name],tensor.name))
            onnx.checker.check_model(graph); onnx.save(graph,final)
            actual = {k:gated(*predict_onnx(final,d['X'])) for k,d in data.items()}
            if (preserves_metrics(data['val']['y'],before['val'],actual['val']) and
                preserves_metrics(data['old']['y'],before['old'],actual['old']) and
                not np.any((current_new==data['new']['y']) & (actual['new']!=data['new']['y']))):
                current=w; accepted=row;selected.append(row);break
        print(trainer.LABELS[column], accepted or 'No improving preservation-safe step', flush=True)
    graph=onnx.load(baseline)
    for tensor in graph.graph.initializer:
        if tensor.name in ('w2','b2'):tensor.CopyFrom(numpy_helper.from_array(current[tensor.name],tensor.name))
    onnx.save(graph,final)
    report={'selected':selected,'validation_only_selection':True,'trials':all_trials,
            'baseline_new_correct':int(np.sum(before['new']==data['new']['y'])),
            'candidate_new_correct':int(np.sum(gated(*predict_onnx(final,data['new']['X']))==data['new']['y']))}
    trainer.write_json(out/'coordinate_validation.json',report)
    print({k:v for k,v in report.items() if k!='trials'},flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
