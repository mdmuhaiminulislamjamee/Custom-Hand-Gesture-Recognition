"""Shared balanced training/evaluation code used by the notebook and CLI.

Ten commands plus an internal no_gesture class; subject-aware public splits;
training-only augmentation. No test-set threshold selection or model fitting.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from backend.config import CLASS_NAMES, RuntimeConfig
from backend.geometry import GeometryResolver, landmarks_to_feature

LABELS = CLASS_NAMES + ['no_gesture']
COMMAND_COUNT = len(CLASS_NAMES)
REJECT_INDEX = COMMAND_COUNT
INTERNAL_COUNT = len(LABELS)
ARTIFACT_DIRECTORY = 'artifacts/ten_gesture'
DATA_DIRECTORY = 'data/ten_gesture'


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def reconstruct_landmarks(feature):
    """Recover scale-free screen orientation from the exact 76-D contract."""
    canonical = np.asarray(feature[:42]).reshape(21, 2)
    target = feature[63:65]
    candidates = []
    for reflection in [1, -1]:
        points = canonical * [reflection, 1]
        vector = points[8] - points[5]
        angle = np.arctan2(target[1], target[0]) - np.arctan2(vector[1], vector[0])
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        points = (points @ rotation.T).astype(np.float32)
        error = np.max(np.abs(landmarks_to_feature(points) - feature))
        candidates.append((error, points))
    error, points = min(candidates, key=lambda item: item[0])
    if error > 0.003:
        raise ValueError(f'Feature/landmark reconstruction mismatch: {error}')
    return points


def subset(data, indices):
    return {k: v[indices] for k, v in data.items()}


def concatenate(parts):
    keys = set.intersection(*(set(p) for p in parts))
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def feature_keys(features):
    return [hashlib.sha256(np.round(row, 5).tobytes()).hexdigest() for row in features.astype(np.float32)]


def prepare_data(root, per_class=4096, seed=1820):
    import pandas as pd
    root = Path(root)
    directory = root / DATA_DIRECTORY
    output = root / ARTIFACT_DIRECTORY
    output.mkdir(parents=True, exist_ok=True)
    public = {s: load_npz(directory / f'hagrid_{s}.npz') for s in ['train', 'val', 'test']}
    dorsal = load_npz(directory / 'dorsal.npz')
    metadata = pd.read_csv(directory / 'HandInfo.csv', dtype={'id': str}).set_index('imageName')
    missing = [n for n in dorsal['source_id'] if n not in metadata.index]
    if missing:
        raise ValueError(f'Dorsal participant metadata missing for {len(missing)} images')
    dorsal['group'] = np.array(['11k:' + metadata.loc[n, 'id'] for n in dorsal['source_id']])
    # Split unique people before rotations or augmentation. Metadata are supplied
    # by the 11K authors, never inferred from the images here.
    groups = sorted(set(dorsal['group']))
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    train_end, val_end = int(len(groups) * .70), int(len(groups) * .85)
    group_sets = dict(train=set(groups[:train_end]), val=set(groups[train_end:val_end]), test=set(groups[val_end:]))
    for split in public:
        public[split] = concatenate([public[split], subset(dorsal, np.isin(dorsal['group'], list(group_sets[split])))])
    # Enforce both participant and exact-feature isolation even if upstream
    # partitions contain mistakes. Evaluation has precedence over training.
    excluded = {}
    seen_groups, seen_features = set(), set()
    for split in ['test', 'val', 'train']:
        data = public[split]
        keys = feature_keys(data['X'])
        keep, local = [], set()
        for i, key in enumerate(keys):
            if data['group'][i] in seen_groups or key in seen_features or key in local:
                continue
            keep.append(i)
            local.add(key)
        excluded[split] = len(keys) - len(keep)
        public[split] = subset(data, keep)
        seen_groups.update(public[split]['group'])
        seen_features.update(local)
    # Legacy replay is training data, but its unknown participant IDs mean old
    # cached test results cannot establish participant generalization.
    replay = load_npz(root / 'models/gesture_online_replay_cache.npz')
    legacy_points = np.array([reconstruct_landmarks(x) for x in replay['X']])
    legacy_y = np.where(replay['y'] < 0, REJECT_INDEX, replay['y']).astype(np.int64)
    if 'source_y' in replay and 'source_class_names' in replay:
        source_names = list(replay['source_class_names'])
        if 'fist' in source_names:
            legacy_y[replay['source_y'] == source_names.index('fist')] = CLASS_NAMES.index('fist')
    legacy = dict(X=replay['X'], landmarks=legacy_points, y=legacy_y,
                  group=np.array(['legacy_unknown'] * len(replay['X'])),
                  source_id=np.array([f'legacy:{i}' for i in range(len(replay['X']))]),
                  source=np.array(['legacy'] * len(replay['X'])), variant=np.array(['original'] * len(replay['X'])))
    # Remove exact copies of evaluation observations from replay before fitting.
    eval_keys = set(feature_keys(public['val']['X']) + feature_keys(public['test']['X']))
    keep = [i for i, key in enumerate(feature_keys(legacy['X'])) if key not in eval_keys]
    legacy = subset(legacy, keep)
    train = concatenate([public['train'], legacy])
    rows, labels, parents, augmented = [], [], [], []
    for label in range(INTERNAL_COUNT):
        available = np.flatnonzero(train['y'] == label)
        if not len(available):
            raise ValueError(f'No training examples for {LABELS[label]}')
        selected = np.resize(rng.permutation(available), per_class)
        for number, index in enumerate(selected):
            points = train['landmarks'][index].copy()
            augment = number >= len(available)
            if augment:
                points -= points[0]
                angle_limit = .35 if label in {
                    CLASS_NAMES.index('like'), CLASS_NAMES.index('fist'),
                    CLASS_NAMES.index('thumb_down'),
                } else .20
                angle = rng.uniform(-angle_limit, angle_limit)
                rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                points = points @ rotation.T
                scale_range = (.78, 1.12) if label in {
                    CLASS_NAMES.index('like'), CLASS_NAMES.index('fist'),
                    CLASS_NAMES.index('thumb_down'),
                } else (.92, 1.08)
                points *= rng.uniform(*scale_range, size=(1, 2))
                scale = np.linalg.norm(points[9])
                points += rng.normal(0, max(scale, 1e-6) * .006, points.shape)
            if label in {CLASS_NAMES.index('fist'), CLASS_NAMES.index('thumb_down')} and number % 2:
                points[:, 0] = 2 * points[0, 0] - points[:, 0]
            rows.append(landmarks_to_feature(points))
            labels.append(label)
            parents.append(str(train['source_id'][index]))
            augmented.append(augment)
    order = rng.permutation(len(rows))
    balanced = dict(X=np.array(rows, np.float32)[order], y=np.array(labels)[order],
                    parent_id=np.array(parents)[order], augmented=np.array(augmented)[order])
    np.savez_compressed(output / 'balanced_train.npz', **balanced)
    for split in ['val', 'test']:
        np.savez_compressed(output / f'public_{split}.npz', **public[split])
    counts = []
    for label, name in enumerate(LABELS):
        counts.append(dict(gesture=name, public_train=int((public['train']['y'] == label).sum()),
                           legacy_replay=int((legacy['y'] == label).sum()), balanced_train=per_class,
                           augmented=int(np.sum(balanced['augmented'] & (balanced['y'] == label))),
                           validation=int((public['val']['y'] == label).sum()), test=int((public['test']['y'] == label).sum())))
    pd.DataFrame(counts).to_csv(output / 'dataset_balance.csv', index=False)
    audit = dict(seed=seed, per_class=per_class, excluded_overlap=excluded,
                 subjects={s: len(set(d['group'])) for s, d in public.items()},
                 dorsal_subjects=len(groups), dorsal_metadata_matched=len(dorsal['y']),
                 subject_overlap={f'{a}_{b}': len(set(public[a]['group']) & set(public[b]['group']))
                                  for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]},
                 limitations=['Directional public examples are rotations of one, not new horizontal camera captures.',
                              'Legacy replay has no participant IDs; cross-source subject overlap cannot be proved absent.',
                              'No claim of universal demographic, child, disability, or lighting coverage.'])
    (output / 'data_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return balanced, public['val'], audit


def evaluate(probabilities, mass, data, config=None, geometry=True):
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    config = config or RuntimeConfig()
    raw = probabilities.argmax(1)
    resolved = probabilities.copy()
    valid = np.ones(len(raw), bool)
    geometry_supported = np.zeros(len(raw), bool)
    support_allowed = np.zeros(len(raw), bool)
    if geometry:
        resolver = GeometryResolver(config)
        for i, points in enumerate(data['landmarks']):
            handedness = (
                str(data['handedness'][i])
                if 'handedness' in data and str(data['handedness'][i])
                else None
            )
            handedness_confidence = (
                float(data['handedness_confidence'][i])
                if 'handedness_confidence' in data else None
            )
            resolved[i], details = resolver.resolve(
                probabilities[i], points,
                handedness=handedness,
                handedness_confidence=handedness_confidence,
            )
            pose = details['pose_validation']
            direction = details.get('directional') or {}
            valid[i] = pose['valid']
            geometry_supported[i] = pose.get('geometry_supported', False)
            name = config.class_names[int(np.argmax(resolved[i]))]
            recovery_mass_ok = bool(mass[i] >= config.geometry_recovery_mass_floor)
            directional_recovery = bool(
                geometry_supported[i]
                and direction.get('strong_geometry', False)
                and direction.get('model_supported_pose', False)
            )
            support_allowed[i] = bool(
                (name == 'dorsal' and geometry_supported[i] and recovery_mass_ok)
                or (
                    name in {'left', 'right', 'up', 'down'}
                    and directional_recovery
                    and (name == 'down' or recovery_mass_ok)
                )
                or (
                    name in {'open_palm', 'fist', 'thumb_down'}
                    and geometry_supported[i]
                    and recovery_mass_ok
                )
            )
    ordered = np.sort(resolved, axis=1)
    accepted = valid & ((mass >= config.known_mass_floor) | support_allowed) & (ordered[:, -1] >= config.confidence_floor) & ((ordered[:, -1] - ordered[:, -2]) >= config.probability_margin_floor)
    predicted = np.where(accepted, resolved.argmax(1), REJECT_INDEX)
    data_y = np.asarray(data['y'], dtype=np.int64).copy()
    if 'source_y' in data and 'source_class_names' in data:
        source_names = list(data['source_class_names'])
        if 'fist' in source_names:
            data_y[data['source_y'] == source_names.index('fist')] = CLASS_NAMES.index('fist')
    truth = np.where(data_y < 0, REJECT_INDEX, data_y)
    known = truth < COMMAND_COUNT
    correct_accept = accepted & (predicted == truth)
    metrics = dict(raw_known_accuracy=float(accuracy_score(truth[known], raw[known])),
                   raw_known_macro_f1=float(f1_score(truth[known], raw[known], labels=np.arange(COMMAND_COUNT), average='macro', zero_division=0)),
                   runtime_macro_f1=float(f1_score(truth, predicted, labels=np.arange(INTERNAL_COUNT), average='macro', zero_division=0)),
                   correct_command_rate=float(np.mean(correct_accept[known])),
                   accepted_precision=float(np.mean(predicted[accepted] == truth[accepted])) if accepted.any() else 0.,
                   unknown_false_accept_rate=float(np.mean(accepted[~known])) if (~known).any() else 0.,
                   known_rejection_rate=float(np.mean(~accepted[known])))
    return dict(metrics=metrics, predicted=predicted, accepted=accepted, truth=truth,
                report=classification_report(truth, predicted, labels=np.arange(INTERNAL_COUNT), target_names=LABELS, output_dict=True, zero_division=0),
                confusion=confusion_matrix(truth, predicted, labels=np.arange(INTERNAL_COUNT)), probabilities=resolved, mass=mass)


def conditional(probabilities):
    mass = probabilities[:, :COMMAND_COUNT].sum(1)
    known = probabilities[:, :COMMAND_COUNT] / np.maximum(mass[:, None], 1e-30)
    return known, mass


def train_candidate(root, epochs=80, seed=1820):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, log_loss
    from threadpoolctl import threadpool_limits
    root = Path(root)
    output = root / ARTIFACT_DIRECTORY
    train = load_npz(output / 'balanced_train.npz')
    validation = load_npz(output / 'public_val.npz')
    scaler = StandardScaler().fit(train['X'])
    x = scaler.transform(train['X']).astype(np.float32)
    val_x = scaler.transform(validation['X']).astype(np.float32)
    model = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=.003, batch_size=256,
                          learning_rate_init=.0007, random_state=seed, max_iter=1)
    best, best_score, stale, history = None, -1., 0, []
    with threadpool_limits(limits=1):
        for epoch in range(epochs):
            model.partial_fit(x, train['y'], classes=np.arange(INTERNAL_COUNT))
            probabilities = model.predict_proba(val_x)
            score = f1_score(validation['y'], probabilities.argmax(1), labels=np.arange(INTERNAL_COUNT), average='macro', zero_division=0)
            history.append(dict(epoch=epoch + 1, train_loss=float(model.loss_), val_log_loss=float(log_loss(validation['y'], probabilities, labels=np.arange(INTERNAL_COUNT))), val_macro_f1=float(score)))
            if score > best_score + .0001:
                best, best_score, stale = copy.deepcopy(model), score, 0
            else:
                stale += 1
            if epoch % 5 == 0:
                print('epoch', epoch + 1, 'validation macro F1', round(score, 4), flush=True)
            if stale >= 15:
                break
    import pandas as pd
    pd.DataFrame(history).to_csv(output / 'training_history.csv', index=False)
    export_onnx(root, best, scaler, output / 'candidate.onnx')
    return best, scaler, history


def export_onnx(root, model, scaler, path):
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    initializers = []
    nodes = []
    def constant(name, value):
        initializers.append(numpy_helper.from_array(np.asarray(value), name))
        return name
    constant('mean', scaler.mean_.astype(np.float32)); constant('scale', scaler.scale_.astype(np.float32))
    nodes += [helper.make_node('Sub', ['landmark_features', 'mean'], ['centered']), helper.make_node('Div', ['centered', 'scale'], ['scaled'])]
    current = 'scaled'
    for i, (weights, bias) in enumerate(zip(model.coefs_, model.intercepts_)):
        constant(f'w{i}', weights.astype(np.float32)); constant(f'b{i}', bias.astype(np.float32))
        nodes += [helper.make_node('MatMul', [current, f'w{i}'], [f'mat{i}']), helper.make_node('Add', [f'mat{i}', f'b{i}'], [f'add{i}'])]
        current = f'add{i}'
        if i < len(model.coefs_) - 1:
            nodes.append(helper.make_node('Relu', [current], [f'act{i}'])); current = f'act{i}'
    nodes.append(helper.make_node('Softmax', [current], ['all_probabilities'], axis=1))
    constant('known_indices', np.arange(COMMAND_COUNT, dtype=np.int64)); constant('axis', np.array([1], np.int64)); constant('epsilon', np.array([1e-30], np.float32))
    nodes += [helper.make_node('Gather', ['all_probabilities', 'known_indices'], ['known'], axis=1),
              helper.make_node('ReduceSum', ['known', 'axis'], ['known_gesture_mass'], keepdims=1),
              helper.make_node('Add', ['known', 'epsilon'], ['safe_known']),
              helper.make_node('ReduceSum', ['safe_known', 'axis'], ['denominator'], keepdims=1),
              helper.make_node('Div', ['safe_known', 'denominator'], ['probabilities']),
              helper.make_node('ArgMax', ['probabilities'], ['label'], axis=1, keepdims=0)]
    graph = helper.make_graph(nodes, 'ten_gesture_balanced_mlp', [helper.make_tensor_value_info('landmark_features', TensorProto.FLOAT, [None, 76])],
                              [helper.make_tensor_value_info('label', TensorProto.INT64, [None]),
                               helper.make_tensor_value_info('probabilities', TensorProto.FLOAT, [None, COMMAND_COUNT]),
                               helper.make_tensor_value_info('known_gesture_mass', TensorProto.FLOAT, [None, 1])], initializers)
    exported = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 16)], ir_version=8)
    onnx.checker.check_model(exported)
    onnx.save(exported, path)
    validation = load_npz(Path(root) / ARTIFACT_DIRECTORY / 'public_val.npz')['X']
    expected, expected_mass = conditional(model.predict_proba(scaler.transform(validation).astype(np.float32)))
    actual, actual_mass = predict_onnx(path, validation)
    parity = dict(prediction_agreement=float(np.mean(expected.argmax(1) == actual.argmax(1))),
                  maximum_absolute_probability_error=float(np.max(np.abs(actual - expected))),
                  maximum_absolute_mass_error=float(np.max(np.abs(actual_mass - expected_mass))))
    parity['passed'] = parity['prediction_agreement'] >= .999 and parity['maximum_absolute_probability_error'] < 1e-4 and parity['maximum_absolute_mass_error'] < 1e-4
    if not parity['passed']:
        raise RuntimeError(f'ONNX parity failed: {parity}')
    Path(path).with_suffix('.parity.json').write_text(json.dumps(parity, indent=2) + '\n')


def predict_onnx(path, features):
    import onnxruntime as ort
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    p, mass = session.run(['probabilities', 'known_gesture_mass'], {'landmark_features': np.asarray(features, np.float32)})
    return p, mass.reshape(-1)


def benchmark(path, features, repeats=300):
    import onnxruntime as ort
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    row = features[:1].astype(np.float32)
    for _ in range(20):
        session.run(None, {'landmark_features': row})
    times = []
    for _ in range(repeats):
        start = time.perf_counter(); session.run(None, {'landmark_features': row}); times.append((time.perf_counter() - start) * 1000)
    return dict(p50_ms=float(np.quantile(times, .5)), p95_ms=float(np.quantile(times, .95)), p99_ms=float(np.quantile(times, .99)),
                note='ONNX classifier only; full camera pipeline is benchmarked separately.')
