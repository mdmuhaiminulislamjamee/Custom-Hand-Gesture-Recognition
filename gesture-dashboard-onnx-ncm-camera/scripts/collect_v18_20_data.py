"""Import public, labelled data; retain provenance and original subject splits.

HaGRID's published 2-D landmarks avoid a 119 GB image download. Directional
variants are explicitly rotations of the same 'one' image, never new people.
Fist and dislike (Thumbs Down) remain first-class commands; non-upward palms
are retained as rejection examples. Run from the project root:
python -m scripts.collect_v18_20_data
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np

from backend.geometry import landmarks_to_feature
from backend.config import CLASS_NAMES

HAGRID_URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip"
DORSAL_URL = "https://www.kaggle.com/api/v1/datasets/download/mdmuhaiminulislam/dorsal-hands-dataset"
SOURCES = {
    "hagrid": {"url": "https://github.com/hukenovs/hagrid", "download": HAGRID_URL,
               "license": "HaGRID custom CC BY-SA 4.0; see https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf",
               "landmarks": "Publisher's automatic MediaPipe annotations", "subjects": "Published user_id and train/val/test partitions"},
    "dorsal": {"url": "https://www.kaggle.com/datasets/mdmuhaiminulislam/dorsal-hands-dataset",
               "download": DORSAL_URL, "version": 1, "license": "CC BY-NC-SA 4.0",
               "landmarks": "Locally extracted using the bundled MediaPipe hand landmarker",
               "subjects": "Unknown unless original 11K HandInfo.csv is available"},
}
SOURCES['dorsal']['participant_metadata'] = '11K Hands / HandInfo.csv; https://github.com/mahmoudnafifi/11K-Hands'
SOURCES['dorsal']['metadata_mirror'] = 'https://www.kaggle.com/datasets/shyambhu/hands-and-palm-images-dataset'


def rotate_to(points, direction):
    points = np.asarray(points, dtype=np.float32).copy()
    vector = points[8] - points[5]
    angle = direction - np.arctan2(vector[1], vector[0])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return ((points - points[0]) @ rotation.T).astype(np.float32)


def rotate_palm_to(points, direction):
    """Rotate the wrist-to-MCP palm axis to a screen direction."""
    points = np.asarray(points, dtype=np.float32).copy()
    axis = points[[5, 9, 13, 17]].mean(axis=0) - points[0]
    angle = direction - np.arctan2(axis[1], axis[0])
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return ((points - points[0]) @ rotation.T).astype(np.float32)


def save_rows(path, rows):
    np.savez_compressed(path, X=np.array([landmarks_to_feature(r['points']) for r in rows]),
                        landmarks=np.array([r['points'] for r in rows], dtype=np.float32),
                        y=np.array([r['label'] for r in rows], dtype=np.int64),
                        group=np.array([r['group'] for r in rows]),
                        source_id=np.array([r['source_id'] for r in rows]),
                        source=np.array([r['source'] for r in rows]),
                        variant=np.array([r.get('variant', 'original') for r in rows]),
                        handedness=np.array([r.get('handedness', '') for r in rows]),
                        handedness_confidence=np.array([
                            r.get('handedness_confidence', 0.0) for r in rows
                        ], dtype=np.float32))


def _shape_label(name: str, points: np.ndarray) -> int:
    command = {
        'like': 'like',
        'ok': 'ok',
        'fist': 'fist',
        'dislike': 'thumb_down',
    }.get(name)
    if name == 'palm':
        axis = points[[5, 9, 13, 17]].mean(axis=0) - points[0]
        norm = max(float(np.linalg.norm(axis)), 1e-8)
        command = 'open_palm' if float(-axis[1] / norm) >= .70 else None
    return len(CLASS_NAMES) if command is None else CLASS_NAMES.index(command)


def _palmar_handedness(points: np.ndarray) -> str:
    """Return the handedness label that identifies this known palm as palmar."""
    index_axis = points[5] - points[0]
    pinky_axis = points[17] - points[0]
    winding = float(
        index_axis[0] * pinky_axis[1] - index_axis[1] * pinky_axis[0]
    )
    return 'Right' if winding < 0 else 'Left'


def collect_hagrid(
    destination: Path,
    train_limit=2000,
    eval_limit=400,
    *,
    force=False,
    splits=('train', 'val', 'test'),
    excluded_groups: set[str] | None = None,
):
    from remotezip import RemoteZip
    destination.mkdir(parents=True, exist_ok=True)
    # Several unknown shapes, including genuine neutral hands, are retained.
    names = [
        'one', 'palm', 'like', 'ok', 'fist', 'dislike',
        'rock', 'call', 'peace', 'no_gesture',
    ]
    with RemoteZip(HAGRID_URL, timeout=90) as archive:
        for split in splits:
            output = destination / f'hagrid_{split}.npz'
            if output.exists() and not force:
                continue
            rows = []
            for name in names:
                member = f'annotations/{split}/{name}.json'
                annotations = json.loads(archive.read(member))
                # Deterministic sampling across the whole file, at most 2 images
                # per participant per source gesture to increase subject breadth.
                keys = sorted(annotations, key=lambda key: hashlib.sha256(key.encode()).digest())
                limit = train_limit if split == 'train' else eval_limit
                count = 0
                participants = {}
                for key in keys:
                    record = annotations[key]
                    group = 'hagrid:' + str(record.get('user_id', key))
                    if excluded_groups and group in excluded_groups:
                        continue
                    if participants.get(group, 0) >= 2:
                        continue
                    for hand, (label, landmarks) in enumerate(zip(record['labels'], record.get('hand_landmarks', []))):
                        if label != name or landmarks is None:
                            continue
                        points = np.asarray(landmarks, dtype=np.float32)
                        if points.shape != (21, 2) or not np.isfinite(points).all():
                            continue
                        try:
                            landmarks_to_feature(points)
                        except ValueError:
                            continue
                        base = dict(group=group, source_id=f'{key}:{hand}', source='hagrid')
                        if name == 'one':
                            for target, angle in [(0, 0), (1, np.pi), (2, -np.pi / 2), (3, np.pi / 2)]:
                                rows.append(dict(base, points=rotate_to(points, angle), label=target, variant='direction_rotation'))
                        elif name == 'palm':
                            rows.append(dict(
                                base, points=points, label=_shape_label(name, points),
                                variant='source_palm',
                            ))
                            rejection_target = (0.0, np.pi, np.pi / 2)[count % 3]
                            rows.append(dict(
                                base,
                                source_id=f'{key}:{hand}:non_up',
                                points=rotate_palm_to(points, rejection_target),
                                label=len(CLASS_NAMES),
                                variant=(
                                    'palm_axis_right_rejection',
                                    'palm_axis_left_rejection',
                                    'palm_axis_down_rejection',
                                )[count % 3],
                            ))
                        else:
                            rows.append(dict(
                                base, points=points, label=_shape_label(name, points),
                                variant=f'source_{name}',
                            ))
                        count += 1
                        participants[group] = participants.get(group, 0) + 1
                        break
                    if count >= limit:
                        break
                print(split, name, count, 'source hands', flush=True)
            save_rows(output, rows)
    (destination / 'sources.json').write_text(json.dumps(SOURCES, indent=2) + '\n')


def upgrade_hagrid_evaluation_splits(
    destination: Path, legacy_directory: Path, eval_limit: int = 400
):
    """Extend the proven public eval splits with Fist and Thumbs Down.

    Some HaGRID hard-negative annotation members are extremely large when
    decompressed.  The earlier release already imported those exact official
    participant-separated members.  Reuse them, relabel non-upward palms as
    rejection, and read only the two new official command members.  This keeps
    validation/test untouched by fitting while avoiding a redundant high-RAM
    parse of the same negatives.
    """
    from remotezip import RemoteZip

    destination.mkdir(parents=True, exist_ok=True)
    with RemoteZip(HAGRID_URL, timeout=90) as archive:
        for split in ('val', 'test'):
            legacy_path = legacy_directory / f'hagrid_{split}.npz'
            with np.load(legacy_path, allow_pickle=False) as source:
                old = {key: source[key] for key in source.files}
            rows = []
            for index, points in enumerate(old['landmarks']):
                label = int(old['y'][index])
                if label == 8:  # prior internal no_gesture index
                    label = len(CLASS_NAMES)
                elif label == CLASS_NAMES.index('open_palm'):
                    label = _shape_label('palm', points)
                rows.append({
                    'points': points,
                    'label': label,
                    'group': str(old['group'][index]),
                    'source_id': str(old['source_id'][index]),
                    'source': str(old['source'][index]),
                    'variant': str(old['variant'][index]),
                })
            row_by_source_id = {
                str(row['source_id']): index for index, row in enumerate(rows)
            }
            for name in ('fist', 'dislike'):
                annotations = json.loads(
                    archive.read(f'annotations/{split}/{name}.json')
                )
                keys = sorted(
                    annotations,
                    key=lambda key: hashlib.sha256(key.encode()).digest(),
                )
                participants: dict[str, int] = {}
                count = 0
                for key in keys:
                    record = annotations[key]
                    group = 'hagrid:' + str(record.get('user_id', key))
                    if participants.get(group, 0) >= 2:
                        continue
                    for hand, (label_name, landmarks) in enumerate(
                        zip(record['labels'], record.get('hand_landmarks', []))
                    ):
                        if label_name != name or landmarks is None:
                            continue
                        points = np.asarray(landmarks, dtype=np.float32)
                        if points.shape != (21, 2) or not np.isfinite(points).all():
                            continue
                        try:
                            landmarks_to_feature(points)
                        except ValueError:
                            continue
                        new_row = {
                            'points': points,
                            'label': _shape_label(name, points),
                            'group': group,
                            'source_id': f'{key}:{hand}',
                            'source': 'hagrid',
                            'variant': 'original',
                        }
                        # The earlier release intentionally used a small Fist
                        # subset as rejection. Reclassify that exact official
                        # source row instead of retaining contradictory labels.
                        existing = row_by_source_id.get(new_row['source_id'])
                        if existing is None:
                            row_by_source_id[new_row['source_id']] = len(rows)
                            rows.append(new_row)
                        elif int(rows[existing]['label']) == len(CLASS_NAMES):
                            rows[existing] = new_row
                        elif int(rows[existing]['label']) != int(new_row['label']):
                            raise ValueError(
                                f"Conflicting labels for {new_row['source_id']}"
                            )
                        participants[group] = participants.get(group, 0) + 1
                        count += 1
                        break
                    if count >= eval_limit:
                        break
                print(split, name, count, 'source hands', flush=True)
            save_rows(destination / f'hagrid_{split}.npz', rows)
            print(split, 'saved', len(rows), 'rows', flush=True)
    (destination / 'sources.json').write_text(json.dumps(SOURCES, indent=2) + '\n')


def collect_hagrid_confirmation(
    dataset_directory: Path,
    output: Path,
    per_command: int = 400,
):
    """Create a final cohort from HaGRID test participants never used above."""
    from remotezip import RemoteZip

    used_groups: set[str] = set()
    for split in ('train', 'val', 'test'):
        with np.load(
            dataset_directory / f'hagrid_{split}.npz', allow_pickle=False
        ) as archive:
            used_groups.update(str(value) for value in archive['group'])
    rows: list[dict] = []
    confirmation_groups: set[str] = set()

    targets = (
        ('one', per_command), ('palm', per_command),
        ('like', per_command), ('ok', per_command),
        ('fist', per_command), ('dislike', per_command),
        ('rock', per_command // 2),
    )
    with RemoteZip(HAGRID_URL, timeout=90) as remote:
        for name, limit in targets:
            count = 0
            for source_split in ('test', 'val', 'train'):
                annotations = json.loads(
                    remote.read(f'annotations/{source_split}/{name}.json')
                )
                keys = sorted(
                    annotations,
                    key=lambda key: hashlib.sha256(
                        ('confirm:' + source_split + ':' + key).encode()
                    ).digest(),
                )
                for key in keys:
                    record = annotations[key]
                    group = 'hagrid:' + str(record.get('user_id', key))
                    if group in used_groups or group in confirmation_groups:
                        continue
                    for hand, (label_name, landmarks) in enumerate(
                        zip(record['labels'], record.get('hand_landmarks', []))
                    ):
                        if label_name != name or landmarks is None:
                            continue
                        points = np.asarray(landmarks, dtype=np.float32)
                        if points.shape != (21, 2) or not np.isfinite(points).all():
                            continue
                        try:
                            landmarks_to_feature(points)
                        except ValueError:
                            continue
                        base = {
                            'group': group,
                            'source_id': (
                                f'confirmation:{source_split}:{key}:{hand}'
                            ),
                            'source': 'hagrid_confirmation',
                        }
                        if name == 'one':
                            for label, angle in (
                                (0, 0), (1, np.pi),
                                (2, -np.pi / 2), (3, np.pi / 2),
                            ):
                                rows.append({
                                    **base, 'points': rotate_to(points, angle),
                                    'label': label,
                                    'variant': 'direction_rotation',
                                })
                        elif name == 'palm':
                            upward = rotate_palm_to(points, -np.pi / 2)
                            palm_base = {
                                **base,
                                'handedness': _palmar_handedness(upward),
                                'handedness_confidence': .99,
                            }
                            rows.append({
                                **palm_base, 'points': upward,
                                'label': CLASS_NAMES.index('open_palm'),
                                'variant': 'palm_axis_up',
                            })
                            if count % 2 == 0:
                                rows.append({
                                    **palm_base,
                                    'source_id': base['source_id'] + ':non_up',
                                    'points': rotate_palm_to(points, np.pi / 2),
                                    'label': len(CLASS_NAMES),
                                    'variant': 'palm_axis_down_rejection',
                                })
                        else:
                            rows.append({
                                **base, 'points': points,
                                'label': _shape_label(name, points),
                                'variant': 'original',
                            })
                        confirmation_groups.add(group)
                        count += 1
                        break
                    if count >= limit:
                        break
                del annotations
                gc.collect()
                if count >= limit:
                    break
            print('confirmation', name, count, 'new-subject hands', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_rows(output, rows)
    print('confirmation saved', len(rows), 'rows from', len(confirmation_groups), 'subjects', flush=True)


def collect_dorsal(destination: Path, model: Path, limit=2000):
    import cv2
    import mediapipe as mp
    output = destination / 'dorsal.npz'
    if output.exists():
        return
    archive_path = destination / 'dorsal-images.zip'
    if not archive_path.exists():
        with urllib.request.urlopen(DORSAL_URL, timeout=90) as response, archive_path.open('wb') as target:
            while block := response.read(1024 * 1024):
                target.write(block)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=1,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5)
    rows = []
    with zipfile.ZipFile(archive_path) as archive, mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        names = sorted([n for n in archive.namelist() if n.lower().endswith('.jpg')],
                       key=lambda key: hashlib.sha256(key.encode()).digest())
        for index, name in enumerate(names):
            image = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.resize(image, (640, round(640 * image.shape[0] / image.shape[1])))
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            if result.hand_landmarks:
                points = np.array([[p.x, p.y] for p in result.hand_landmarks[0]], np.float32)
                # Existing command contract: dorsal hand, four fingers down.
                points = rotate_to(points, np.pi / 2)
                try:
                    landmarks_to_feature(points)
                except ValueError:
                    continue
                rows.append(dict(points=points, label=6, source='dorsal', source_id=Path(name).name,
                                 group='dorsal_image:' + Path(name).name, variant='direction_rotation'))
            if index % 200 == 0:
                print('dorsal', index + 1, 'images inspected,', len(rows), 'hands extracted', flush=True)
            if len(rows) >= limit:
                break
    save_rows(output, rows)
    print('dorsal saved', len(rows), flush=True)


def collect_hand_metadata(destination):
    import requests
    from remotezip import RemoteZip
    path = destination / 'HandInfo.csv'
    if path.exists():
        return
    url = 'https://www.kaggle.com/api/v1/datasets/download/shyambhu/hands-and-palm-images-dataset'
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        final_url = response.url
    with RemoteZip(final_url, timeout=90) as archive:
        names = [n for n in archive.namelist() if Path(n).name == 'HandInfo.csv']
        if len(names) != 1:
            raise ValueError('Cannot identify the original participant metadata file')
        path.write_bytes(archive.read(names[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/ten_gesture'))
    parser.add_argument(
        '--source', choices=['all', 'hagrid', 'dorsal', 'confirmation'], default='all'
    )
    parser.add_argument('--force', action='store_true', help='Rebuild existing HaGRID NPZ files.')
    parser.add_argument(
        '--evaluation-base', type=Path,
        help='Reuse earlier official hard-negative val/test files and add the new commands.',
    )
    parser.add_argument(
        '--confirmation-output', type=Path,
        default=Path('artifacts/ten_gesture/independent_confirmation.npz'),
    )
    parser.add_argument(
        '--hagrid-split', choices=('all', 'train', 'val', 'test'), default='all'
    )
    parser.add_argument(
        '--exclude-groups-from', type=Path,
        help='NPZ whose participant groups must remain outside a rebuilt split.',
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.source == 'confirmation':
        collect_hagrid_confirmation(args.output, args.confirmation_output)
        return
    if args.source in ['all', 'hagrid']:
        if args.evaluation_base is not None:
            upgrade_hagrid_evaluation_splits(args.output, args.evaluation_base)
        else:
            excluded_groups = None
            if args.exclude_groups_from is not None:
                with np.load(args.exclude_groups_from, allow_pickle=False) as archive:
                    excluded_groups = set(str(value) for value in archive['group'])
            splits = (
                ('train', 'val', 'test')
                if args.hagrid_split == 'all'
                else (args.hagrid_split,)
            )
            collect_hagrid(
                args.output,
                force=args.force,
                splits=splits,
                excluded_groups=excluded_groups,
            )
    if args.source in ['all', 'dorsal']:
        collect_hand_metadata(args.output)
        collect_dorsal(args.output, Path('models/hand_landmarker.task'))
    (args.output / 'sources.json').write_text(json.dumps(SOURCES, indent=2) + '\n')


if __name__ == '__main__':
    main()
