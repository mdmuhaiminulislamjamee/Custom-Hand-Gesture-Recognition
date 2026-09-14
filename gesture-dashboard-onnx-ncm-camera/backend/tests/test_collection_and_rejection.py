from io import BytesIO
import json

import numpy as np
import pytest
from PIL import Image

from backend.config import RuntimeConfig
from backend.data_collection import CollectionStore, collection_plan, identifier
from backend.face_guard import overlaps_face_as_small_hand
from backend.geometry import GeometryResolver
from backend.runtime import TemporalGate
from backend.tests.test_multiview_training import directional_points
from backend.tests.test_geometry import representative_hand, side_view_pointing_hand


def jpeg_bytes():
    stream = BytesIO()
    Image.new('RGB', (80, 60), (120, 130, 140)).save(stream, 'JPEG')
    return stream.getvalue()


def test_save_raw_detector_failure_preserves_frozen_pair_and_restarts(tmp_path):
    store = CollectionStore(tmp_path)
    store.add_participant('person-001')
    jpeg = jpeg_bytes()
    store.observe(4, jpeg, {'status': 'no_hand', 'runtime_prediction': 'no_gesture'})
    frozen = store.freeze()
    store.observe(5, b'a different frame', {'runtime_prediction': 'left'})
    result = store.save(token=frozen['token'], participant_id='person-001', session_id='session-1', step_id='up/casual')
    assert result['has_features'] is False
    from pathlib import Path
    path = Path(result['image_path'])
    assert path.read_bytes() == jpeg
    record = json.loads(path.with_suffix('.json').read_text())
    assert record['frame_id'] == 4
    assert record['label'] == 'up'
    assert record['prediction']['runtime_prediction'] == 'no_gesture'
    assert CollectionStore(tmp_path).summary('person-001')['counts']['up/casual'] == 1
    with pytest.raises(ValueError, match='expired'):
        store.save(token=frozen['token'], participant_id='person-001', session_id='session-1', step_id='up/casual')


def test_collection_plan_keeps_54_sections_with_12_images_and_balanced_hand_angles():
    plan = collection_plan()
    assert len(plan) == 54
    assert all(step['target'] == 12 for step in plan)
    assert all(step['per_hand_target'] == 6 for step in plan if step['view'] not in {'background', 'face_ear'})
    assert all(step['per_orientation_target'] == 3 for step in plan if step['view'] not in {'background', 'face_ear'})
    assert all(step['per_hand_target'] is None for step in plan if step['view'] in {'background', 'face_ear'})
    assert all(step['per_orientation_target'] is None for step in plan if step['view'] in {'background', 'face_ear'})


def test_summary_counts_each_hand_angle_and_delete_last_removes_image_and_label(tmp_path):
    from pathlib import Path
    store = CollectionStore(tmp_path)
    store.add_participant('person-001')
    store.observe(1, jpeg_bytes(), {})
    first = store.save(token=store.freeze()['token'], participant_id='person-001', session_id='session-1',
                       step_id='up/front', handedness='right', finger_orientation='toward_camera')
    store.observe(2, jpeg_bytes(), {})
    second = store.save(token=store.freeze()['token'], participant_id='person-001', session_id='session-2',
                        step_id='up/front', handedness='left', finger_orientation='away_from_camera')
    summary = store.summary('person-001')
    assert summary['counts']['up/front'] == 2
    assert summary['counts_by_hand']['up/front'] == {'right': 1, 'left': 1}
    assert summary['counts_by_hand_orientation']['up/front'] == {
        'right': {'toward_camera': 1}, 'left': {'away_from_camera': 1},
    }
    latest = summary['recent'][0]
    deleted = store.delete_last('person-001')
    assert deleted['deleted']['sample_id'] == latest['sample_id']
    assert deleted['deleted']['handedness'] == latest['handedness']
    assert deleted['deleted']['finger_orientation'] == latest['finger_orientation']
    deleted_path = Path(first['image_path'] if latest['handedness'] == 'right' else second['image_path'])
    remaining_path = Path(second['image_path'] if latest['handedness'] == 'right' else first['image_path'])
    assert deleted['summary']['total'] == 1
    assert remaining_path.is_file()
    assert not deleted_path.exists()
    assert not deleted_path.with_suffix('.json').exists()
    empty_store = CollectionStore(tmp_path / 'empty')
    empty_store.add_participant('person-002')
    with pytest.raises(ValueError, match='no saved images'):
        empty_store.delete_last('person-002')


def test_list_preview_and_delete_selected_images(tmp_path):
    from pathlib import Path
    store = CollectionStore(tmp_path)
    store.add_participant('person-001')
    saved = []
    for frame_id, session_id, hand, orientation in (
        (1, 'session-1', 'right', 'toward_camera'),
        (2, 'session-2', 'left', 'away_from_camera'),
    ):
        store.observe(frame_id, jpeg_bytes(), {})
        saved.append(store.save(
            token=store.freeze()['token'], participant_id='person-001', session_id=session_id,
            step_id='left/casual', handedness=hand, finger_orientation=orientation,
        ))

    listed = store.list_samples('person-001', 'left/casual')['samples']
    assert len(listed) == 2
    assert {item['finger_orientation'] for item in listed} == {'toward_camera', 'away_from_camera'}
    assert store.sample_image('person-001', listed[0]['sample_ref']) == jpeg_bytes()

    removed_ref = listed[0]['sample_ref']
    removed_image = Path(tmp_path, *removed_ref.split('/')).with_suffix('.jpg')
    result = store.delete_selected('person-001', [removed_ref, removed_ref])
    assert len(result['deleted']) == 1
    assert result['summary']['total'] == 1
    assert not removed_image.exists()
    assert not removed_image.with_suffix('.json').exists()
    assert Path(saved[0]['image_path']).exists() != Path(saved[1]['image_path']).exists()

    with pytest.raises(ValueError, match='Invalid selected image reference'):
        store.delete_selected('person-001', ['../outside.json'])
    with pytest.raises(ValueError, match='between 1 and 100'):
        store.delete_selected('person-001', [])


@pytest.mark.parametrize('value', ['../escape', 'CON', 'x/y', 'x\\y', 'x:y', 'person.', ''])
def test_participant_paths_cannot_escape_or_use_reserved_names(value):
    with pytest.raises(ValueError):
        identifier(value, 'Participant')


def test_stale_camera_image_cannot_be_frozen(tmp_path):
    store = CollectionStore(tmp_path)
    store.observe(1, jpeg_bytes(), {})
    store._latest_at -= 3
    with pytest.raises(ValueError, match='fresh'):
        store.freeze()


def test_frozen_preview_includes_landmarks_from_the_same_frame(tmp_path):
    store = CollectionStore(tmp_path)
    landmarks = representative_hand().round(6).tolist()
    store.observe(7, jpeg_bytes(), {
        'landmarks': landmarks,
        'feature_vector': [0.0] * 76,
        'runtime_prediction': 'open_palm',
    })

    frozen = store.freeze()

    assert frozen['frame_id'] == 7
    assert frozen['landmarks'] == landmarks
    assert frozen['has_features'] is True


@pytest.mark.parametrize('raised', [9, 13, 17])
@pytest.mark.parametrize('direction,angle', [('up', -np.pi/2), ('down', np.pi/2), ('left', 0), ('right', np.pi)])
def test_other_fingers_cannot_be_directional_even_with_confident_model(raised, direction, angle):
    points = side_view_pointing_hand()
    points[raised:raised+4] = points[5:9]  # Another finger extended.
    points[5:9] = [[1, -.3], [1.1, -.3], [1.04, -.26], [.98, -.25]]  # Index folded.
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = points @ rotation.T
    probabilities = np.eye(8)[RuntimeConfig().class_to_idx[direction]]
    _, details = GeometryResolver(RuntimeConfig()).resolve(probabilities, points)
    assert not details['pose_validation']['valid']
    assert not details['directional']['index_only']


@pytest.mark.parametrize('direction', ['left', 'right', 'up', 'down'])
def test_real_index_has_strong_evidence_but_low_mass_still_rejected(direction):
    config = RuntimeConfig(known_mass_floor=.95)
    p, details = GeometryResolver(config).resolve(np.eye(8)[config.class_to_idx[direction]], directional_points(direction))
    assert details['pose_validation']['valid']
    gate = TemporalGate(config)
    for t in (1., 1.1, 1.2, 1.3, 1.4, 1.5):
        decision = gate.update(p, now=t, known_gesture_mass=.1, geometry_supported=True)
    assert not decision.execute


def test_open_palm_with_strong_shape_and_partial_mass_needs_longer_hold():
    config = RuntimeConfig(known_mass_floor=.95)
    p, details = GeometryResolver(config).resolve(
        np.eye(8)[4],
        representative_hand(),
        handedness='Left',
        handedness_confidence=.99,
    )
    assert details['pose_validation']['geometry_supported']
    gate = TemporalGate(config)
    for t in (1., 1.1, 1.2, 1.3):
        decision = gate.update(p, now=t, known_gesture_mass=.7, geometry_supported=True)
        assert not decision.execute
    assert gate.update(p, now=1.5, known_gesture_mass=.7, geometry_supported=True).execute
    assert not gate.update(p, now=1.6, known_gesture_mass=.7, geometry_supported=False).execute


def test_face_guard_rejects_small_ear_landmarks_but_allows_large_palm():
    shape = (400, 400, 3)
    boxes = [(150, 60, 90, 110)]
    tiny = representative_hand() * .15 + [.32, .12]
    assert overlaps_face_as_small_hand(tiny, shape, boxes)
    assert not overlaps_face_as_small_hand(representative_hand(), shape, boxes)


@pytest.mark.parametrize('label,view', [('up', 'casual'), ('open_palm', 'front'), ('no_gesture', 'middle_finger')])
def test_dashboard_dataset_loads_human_labels_for_training(tmp_path, label, view):
    from backend.geometry import landmarks_to_feature
    from research.collection_dataset import load_collection_dataset
    store = CollectionStore(tmp_path)
    store.add_participant('person-001')
    points = directional_points('up') if label == 'up' else representative_hand()
    store.observe(1, jpeg_bytes(), dict(landmarks=points.tolist(), feature_vector=landmarks_to_feature(points).tolist(), runtime_prediction='left'))
    store.save(token=store.freeze()['token'], participant_id='person-001', session_id='session-1', step_id=f'{label}/{view}')
    data, audit = load_collection_dataset(tmp_path)
    assert data['label'].tolist() == [label]
    assert data['y'].tolist() == [(RuntimeConfig().class_names + ['no_gesture']).index(label)]
    assert audit['usable_feature_rows'] == audit['raw_images'] == 1


def test_raw_only_images_are_reported_without_fabricating_training_features(tmp_path):
    from research.collection_dataset import load_collection_dataset
    store = CollectionStore(tmp_path)
    store.add_participant('person-001')
    store.observe(1, jpeg_bytes(), {})
    store.save(token=store.freeze()['token'], participant_id='person-001', session_id='session-1', step_id='no_gesture/face_ear')
    data, audit = load_collection_dataset(tmp_path)
    assert data['X'].shape == (0, 76)
    assert audit['raw_images'] == 1
    assert audit['raw_cases_for_review'][0]['reason'] == 'needs_landmark_extraction'


def test_tracking_gap_does_not_rearm_a_held_action_too_soon():
    from backend.model_runtime import RuntimeSession
    session = RuntimeSession(RuntimeConfig(target_fps=10))
    session.last_action_gesture = 'left'
    for _ in range(5):
        assert not session.observe_action_release(None, tracking_gap=True)
    assert session.last_action_gesture == 'left'
    assert session.observe_action_release(None, tracking_gap=True)


def test_collection_api_lifecycle_validation_and_storage_failure(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    import backend.app as module
    store = CollectionStore(tmp_path)
    monkeypatch.setattr(module, 'collection_store', store)
    monkeypatch.setattr(module, 'ncm_camera', SimpleNamespace(status=lambda: {'connected': True}))
    client = TestClient(module.app)
    assert client.post('/api/collection/participants', json={'participant_id': '../x'}).status_code == 409
    assert client.post('/api/collection/participants', json={'participant_id': 'person-001'}, headers={'origin': 'https://untrusted.example'}).status_code == 403
    assert client.post('/api/collection/participants', json={'participant_id': 'person-001'}).status_code == 200
    store.observe(7, jpeg_bytes(), {})
    token = client.post('/api/collection/preview').json()['token']
    payload = dict(participant_id='person-001', session_id='session-1', step_id='up/front', token=token,
                   handedness='right', finger_orientation='toward_camera')
    assert client.post('/api/collection/samples', json={**payload, 'step_id': 'bad'}).status_code == 409
    saved = client.post('/api/collection/samples', json=payload)
    assert saved.status_code == 200
    summary = client.get('/api/collection?participant_id=person-001').json()
    assert summary['total'] == 1
    assert summary['counts_by_hand']['up/front']['right'] == 1
    assert summary['counts_by_hand_orientation']['up/front']['right']['toward_camera'] == 1
    listed = client.get('/api/collection/samples', params={
        'participant_id': 'person-001', 'step_id': 'up/front',
    }).json()['samples']
    assert len(listed) == 1
    image = client.get('/api/collection/image', params={
        'participant_id': 'person-001', 'sample_ref': listed[0]['sample_ref'],
    })
    assert image.status_code == 200
    assert image.headers['content-type'] == 'image/jpeg'
    assert image.content == jpeg_bytes()
    assert client.post('/api/collection/samples', json=payload).status_code == 409
    selected = client.post('/api/collection/samples/delete-selected', json={
        'participant_id': 'person-001', 'sample_refs': [listed[0]['sample_ref']],
    })
    assert selected.status_code == 200
    assert selected.json()['summary']['total'] == 0
    assert client.get('/api/collection/image', params={
        'participant_id': 'person-001', 'sample_ref': listed[0]['sample_ref'],
    }).status_code == 409

    store.observe(8, jpeg_bytes(), {})
    payload['token'] = client.post('/api/collection/preview').json()['token']
    payload['session_id'] = 'session-2'
    assert client.post('/api/collection/samples', json=payload).status_code == 200
    deleted = client.post('/api/collection/samples/delete-last', json={'participant_id': 'person-001'})
    assert deleted.status_code == 200
    assert deleted.json()['summary']['total'] == 0
    assert client.post('/api/collection/samples/delete-last', json={'participant_id': 'person-001'}).status_code == 409
    def unavailable(*args, **kwargs):
        raise OSError('disk unavailable')
    monkeypatch.setattr(store, 'add_participant', unavailable)
    assert client.post('/api/collection/participants', json={'participant_id': 'person-002'}).status_code == 503
