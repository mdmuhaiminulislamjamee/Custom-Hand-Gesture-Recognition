"""Export current model, pixel-angle, geometry, recovery and hold reference cases."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.config import CLASS_NAMES, load_runtime_config
from backend.geometry import GeometryResolver, landmarks_to_feature, open_hand_xy_orientation
from backend.hand_selection import gate_arguments
from backend.model_runtime import ModelManager
from backend.runtime import TemporalGate

SOURCE_FILES = ['backend/config.py', 'backend/geometry.py', 'backend/runtime.py',
                'backend/hand_selection.py', 'backend/model_runtime.py',
                'models/gesture_mobile_runtime_config.json', 'models/gesture_mlp_onnx_metadata.json']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decision_record(decision):
    row = asdict(decision)
    return {key: row[key] for key in ('execute', 'predicted_gesture', 'stable_frames', 'held_seconds', 'reason')}


def export(destination):
    from backend.tests.test_stop_inverted import stop_pose
    from backend.tests.test_reported_low_mass_peace import reported_pose, SHAPE as PEACE_SHAPE
    from backend.tests.test_reported_left_video import FRAMES, traced_pose, SHAPE as LEFT_SHAPE
    from backend.tests.test_gesture_orientation import rotated_closed_fist

    config, cases, transitions = load_runtime_config(), [], []
    resolver = GeometryResolver(config)

    def add(points, raw, source, *, shape=None, side=None, world=None, require_depth=False,
            probabilities=None):
        points = np.asarray(points, np.float32)
        probabilities = (np.eye(len(CLASS_NAMES))[CLASS_NAMES.index(raw)]
                         if probabilities is None else np.asarray(probabilities, np.float64))
        kwargs = dict(handedness=side, handedness_confidence=.99 if side else None,
                      world_landmarks=None if world is None else np.asarray(world, np.float32),
                      require_fist_depth=require_depth, image_shape=shape)
        row, details = resolver.resolve(probabilities, points, **kwargs)
        pose = details['pose_validation']
        temporal = []
        for mass in (1., .2, .01, .00499, 0.):
            gate = TemporalGate(config)
            arguments = gate_arguments(details, {'known_gesture_mass': mass})
            decisions = [decision_record(gate.update(row, now=10. + tick*.1, **arguments)) for tick in range(8)]
            temporal.append(dict(mass=mass, gate_arguments=arguments, execute=[r['execute'] for r in decisions],
                                 decisions=decisions))
        angle = open_hand_xy_orientation(points, shape)
        tilt = abs(angle['tilt_from_up_degrees'])
        case = dict(id=len(cases), source=source, probabilities=probabilities.tolist(), landmarks=points.tolist(),
                    handedness=side, handedness_confidence=kwargs['handedness_confidence'],
                    world_landmarks=None if world is None else np.asarray(world, np.float32).tolist(),
                    require_fist_depth=require_depth, image_shape=None if shape is None else list(shape),
                    features=landmarks_to_feature(points).tolist(),
                    expected=dict(probabilities=row.tolist(), gesture=pose['gesture'], valid=pose['valid'],
                                  geometry_supported=pose['geometry_supported'], pose_validation=pose,
                                  angle_policy=dict(xy_degrees=angle['xy_degrees'],
                                                    open_palm=tilt <= 70.0001, stop_inverted=tilt <= 55.0001),
                                  temporal=temporal))
        cases.append(case)
        return case, row, details

    angles = [-180., -160.01, -160., -150., -145.01, -145., -90., -35., -34.99, -30., -20., -19.99, 0., 90.]
    for gesture in ('open_palm', 'stop_inverted'):
        for mirror in (False, True):
            for shape in ((600, 600, 3), (540, 960, 3), (960, 540, 3)):
                for spread in (0., 22.):
                    for xy in angles:
                        points, side = stop_pose(xy+90., mirror, spread, shape)
                        if gesture == 'open_palm':
                            side = 'Left' if side == 'Right' else 'Right'
                        for raw in ('open_palm', 'stop_inverted', 'like'):
                            add(points, raw, f'synthetic/{gesture}/xy={xy}/mirror={mirror}/spread={spread}',
                                shape=shape, side=side)
            minimum, maximum = (-160., -20.) if gesture == 'open_palm' else (-145., -35.)
            for endpoint, sign in ((minimum, -1), (maximum, 1)):
                entries = []
                for xy in (endpoint-sign*.1, endpoint+sign*.01):
                    points, side = stop_pose(xy+90., mirror, image_shape=(540, 960, 3))
                    if gesture == 'open_palm':
                        side = 'Left' if side == 'Right' else 'Right'
                    entries.append(add(points, gesture, 'boundary hold transition', shape=(540, 960, 3), side=side))
                gate, steps = TemporalGate(config), []
                for index, timestamp in ((0, 10.), (0, 10.1), (0, 10.2), (0, 10.3),
                                         (1, 10.4), (0, 10.5), (0, 10.6), (0, 10.71)):
                    case, row, details = entries[index]
                    arguments = gate_arguments(details, {'known_gesture_mass': .99})
                    steps.append(dict(case_id=case['id'], timestamp=timestamp, gate_arguments=arguments,
                                      expected=decision_record(gate.update(row, now=timestamp, **arguments))))
                assert steps[3]['expected']['execute'] and not steps[4]['expected']['execute']
                assert not steps[5]['expected']['execute'] and steps[-1]['expected']['execute']
                transitions.append(dict(gesture=gesture, endpoint=endpoint, mirror=mirror, steps=steps))
    for mirror in (False, True):
        points = reported_pose(mirror)
        for raw in ('peace', 'up', 'like'):
            add(points, raw, 'approximate reported Peace overlay; not pixel replay', shape=PEACE_SHAPE)
        for mcp in (5, 9, 13, 17):
            fault = points.copy()
            fault[mcp+1:mcp+4] = (fault[9:12] if mcp in (5, 9) else fault[mcp] + points[6:9] - points[5])
            add(fault, 'peace', f'Peace negative finger={mcp}', shape=PEACE_SHAPE)
        for frame in FRAMES:
            for raw in ('left', 'like', 'peace', 'rock'):
                add(traced_pose(frame, mirror), raw, f"reviewed Left/Right overlay {frame['seconds']}s; not pixel replay",
                    shape=LEFT_SHAPE)
        for angle in (-90, -60, 0, 60, 90, 120):
            for pitch in (-30, 0):
                points, world = rotated_closed_fist(angle, pitch, 'x', mirror)
                add(points, 'fist', f'synthetic closed fist/angle={angle}/pitch={pitch}',
                    world=world, require_depth=True)

    # The reported side-on thumb was scored 98.3% Thumb Down despite pointing
    # upward. Check both thumb directions and both raw labels with world bends.
    from backend.tests.test_geometry import side_on_like_pose
    for mirror in (False, True):
        for thumb_up in (False, True):
            pixels = side_on_like_pose()
            if mirror:
                pixels[:, 0] = 2 * pixels[0, 0] - pixels[:, 0]
            if not thumb_up:
                pixels[:, 1] = 2 * pixels[0, 1] - pixels[:, 1]
            world = np.column_stack(((pixels - pixels[0]) / 1000., np.zeros(21)))
            for mcp in (5, 9, 13, 17):
                world[mcp+1:mcp+4] = world[mcp] + .45 * (world[mcp+1:mcp+4] - world[mcp])
            for raw in ('like', 'thumb_down'):
                probabilities = np.zeros(len(CLASS_NAMES))
                probabilities[CLASS_NAMES.index(raw)] = .983
                probabilities[CLASS_NAMES.index('thumb_down' if raw == 'like' else 'like')] = .017
                add(pixels / [1280., 720.], raw,
                    f'approximate reported side thumb/up={thumb_up}/mirror={mirror}/raw={raw}; not pixel replay',
                    shape=(720, 1280, 3), world=world, require_depth=True,
                    probabilities=probabilities)

    manager = ModelManager(config)
    assert manager.ready, manager.errors
    classifier, model_cases = manager.models['ONNX'], []
    with np.load(ROOT / 'models/gesture_online_untouched_test_cache.npz', allow_pickle=False) as cache:
        for label in sorted(set(cache['y'].tolist())):
            for index in np.flatnonzero(cache['y'] == label)[:2]:
                features = cache['X'][index].astype(np.float32)
                probabilities, mass = classifier.predict_with_quality(features[None])
                model_cases.append(dict(label=int(label), features=features.tolist(),
                                        probabilities=probabilities[0].tolist(),
                                        known_gesture_mass=float(np.asarray(mass).ravel()[0])))
    payload = dict(schema_version=2, model_sha256=sha(ROOT / 'models/gesture_mlp_production.onnx'),
                   class_names=CLASS_NAMES, feature_count=76, feature_tolerance=.0001,
                   probability_tolerance=.0001, angle_tolerance_degrees=.0002,
                   source_sha256={name: sha(ROOT / name) for name in SOURCE_FILES},
                   geometry_cases=cases, temporal_transition_cases=transitions, model_cases=model_cases,
                   limitations='Synthetic/overlay landmark regressions; not new image detection or physical-device accuracy.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return dict(geometry_cases=len(cases), temporal_transition_cases=len(transitions), model_cases=len(model_cases),
                model_sha256=payload['model_sha256'], fixture_sha256=sha(destination))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    print(json.dumps(export(parser.parse_args().destination), indent=2))
