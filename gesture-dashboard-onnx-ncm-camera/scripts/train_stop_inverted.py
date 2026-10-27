"""Train/evaluate a 13-output candidate, including reviewed local feedback.

The first twelve tensor indices remain stable. Dorsal stays disabled; the new
Stop inverted output is enabled only by its own surface/angle validator.
Run with --install only after reviewing the generated evaluation.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from backend.config import CLASS_NAMES, FEATURE_NAMES, FEEDBACK_LABELS, GESTURE_TO_ACTION
from backend.geometry import (GeometryResolver, hand_surface_orientation,
                              landmarks_to_feature, stop_inverted_geometry,
                              open_hand_xy_orientation, OPEN_HAND_MAX_TILT_FROM_UP_DEGREES)
from backend.config import RuntimeConfig
from research.v18_20 import feature_keys, load_npz, predict_onnx
from scripts.train_twelve_gesture_candidate import prepare as prepare_legacy, export_onnx, _partition

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "stop_inverted"
STOP = CLASS_NAMES.index("stop_inverted")
REJECT = len(CLASS_NAMES)
LABELS = [*CLASS_NAMES, "no_gesture"]
DORSAL = CLASS_NAMES.index("dorsal")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def subset(data, indices):
    return {key: value[indices] for key, value in data.items()}


def join(parts):
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def replay_indices(data, per_class=128):
    """Keep unique reviewed corrections alongside public Safe Learn anchors."""
    identifiers = data["source_id"].astype(str)
    feedback_mask = np.char.startswith(identifiers, "feedback:")
    original_feedback = feedback_mask & np.asarray([name.count(":") == 1 for name in identifiers])
    selected = []
    for label in range(len(LABELS)):
        reviewed = np.flatnonzero((data["y"] == label) & original_feedback)
        if len(reviewed):
            _, unique = np.unique(identifiers[reviewed], return_index=True)
            reviewed = reviewed[np.sort(unique)][:per_class // 2]
        public = np.flatnonzero((data["y"] == label) & ~feedback_mask)
        selected.extend(public[:per_class - len(reviewed)])
        selected.extend(reviewed)
    return np.asarray(selected, np.int64)


def dorsal_side(points):
    # Source label establishes the surface. This is NOT an independent
    # validation of MediaPipe's handedness prediction in a camera capture.
    return next(side for side in ("Right", "Left")
                if hand_surface_orientation(points, side, .99)["dorsal_visible"])


def make_data(poses, labels, groups, identifiers, sides=None):
    poses = np.asarray(poses, np.float32)
    return dict(X=np.asarray([landmarks_to_feature(p) for p in poses], np.float32),
                landmarks=poses, y=np.asarray(labels, np.int64), group=np.asarray(groups),
                source_id=np.asarray(identifiers),
                handedness=np.asarray(sides if sides is not None else [""] * len(poses)))


def rotate_tilt(points, tilt, mirror=False, spread=0.):
    points = points.copy() - points[0]
    if mirror:
        points[:, 0] *= -1
    # Splay each whole finger about its MCP, preserving all its joint angles.
    # No synthetic viewpoint/depth is claimed by these 2-D augmentations.
    for finger, offset in zip(((5, 6, 7, 8), (9, 10, 11, 12),
                              (13, 14, 15, 16), (17, 18, 19, 20)), (-1., -.3, .3, 1.)):
        radians = np.deg2rad(spread * offset * (-1 if mirror else 1))
        c, s = np.cos(radians), np.sin(radians)
        base = points[finger[0]].copy()
        points[list(finger)] = (points[list(finger)] - base) @ np.asarray([[c, -s], [s, c]]).T + base
    axis = points[[5, 9, 13, 17]].mean(0) - points[0]
    angle = np.deg2rad(tilt) - np.pi / 2 - np.arctan2(axis[1], axis[0])
    c, s = np.cos(angle), np.sin(angle)
    return (points @ np.asarray([[c, -s], [s, c]]).T).astype(np.float32)


def vary_image_aspect(points, rng, number):
    """Simulate normalized XY from square, 4:3 and 16:9 source frames.

    Features consume normalized coordinates, while runtime angles use pixels.
    Keeping this variation in training avoids rejecting a correctly spread hand
    simply because horizontal and vertical image dimensions differ.
    """
    points = points.copy()
    scale = float(rng.choice((1., 9. / 16., 3. / 4., 16. / 9.)))
    if number % 5 == 0:
        scale = rng.uniform(.5, 2.)
    points[:, 0] *= scale
    return points


def vary_pointing_thumb(points, rng):
    """Keep the source One finger/direction, varying only its free thumb.

    A C-shaped thumb can sit opposite three folded fingers. The four directional
    labels do not require a tucked thumb, so cover that shape using training
    participants rather than copying a reported-camera regression into training.
    """
    points = points.copy()
    axis = points[8] - points[5]
    axis /= max(float(np.linalg.norm(axis)), 1e-8)
    normal = np.asarray([-axis[1], axis[0]])
    folded_side = float(np.dot(points[[9, 13, 17]].mean(0) - points[5], normal))
    if folded_side > 0:
        normal *= -1
    scale = max(float(np.linalg.norm(points[9] - points[0])), 1e-8)
    reach = rng.uniform(.9, 1.8)
    arch = rng.uniform(.25, .8)
    for joint, along, away in zip((1, 2, 3, 4), (.22, .50, .80, 1.), (.55, 1., .7, .1)):
        points[joint] = points[0] + scale * (axis * reach * along + normal * arch * away)
    return points


def directional_feedback_augmentation(feedback, rng, per_direction=4096):
    """Preserve reviewed side-view shapes across both horizontal directions.

    Rotate in source-pixel space before normalizing for the destination aspect
    ratio. Reflection swaps Left/Right; rotation stays within the horizontal
    command sector. These are 2-D training variants, not additional people or
    measured depth/lighting conditions.
    """
    available = np.flatnonzero(np.isin(feedback["y"], [0, 1]))
    if not len(available):
        return None
    from PIL import Image
    source_aspects = {}
    for index in available:
        feedback_id = str(feedback["source_id"][index]).removeprefix("feedback:")
        with Image.open(ROOT / "feedback/images" / f"{feedback_id}.jpg") as image:
            source_aspects[index] = image.width / image.height
    poses, labels, people, identifiers = [], [], [], []
    aspects = (1., 4. / 3., 16. / 9., 3. / 4.)
    for label in (0, 1):
        chosen = np.resize(rng.permutation(available), per_direction)
        for number, index in enumerate(chosen):
            pose = (feedback["landmarks"][index] - feedback["landmarks"][index, 0]) * [source_aspects[index], 1.]
            mirrored = int(feedback["y"][index]) != label
            if mirrored:
                pose[:, 0] *= -1
            # Mild anisotropy/shear covers detector projection variation while
            # retaining each complete finger chain and its reviewed shape.
            affine = np.asarray([[rng.uniform(.85, 1.15), rng.uniform(-.08, .08)],
                                 [0., rng.uniform(.85, 1.15)]])
            pose = pose @ affine.T
            vector = pose[8] - pose[5]
            tilt = np.deg2rad(rng.uniform(-38., 38.))
            target = tilt if label == 0 else np.pi - tilt
            angle = target - np.arctan2(vector[1], vector[0])
            c, s = np.cos(angle), np.sin(angle)
            pose = pose @ np.asarray([[c, -s], [s, c]]).T
            aspect = aspects[number % len(aspects)]
            pose[:, 0] /= aspect
            poses.append(pose.astype(np.float32))
            labels.append(label)
            people.append(str(feedback["group"][index]))
            identifiers.append(str(feedback["source_id"][index]) + f":horizontal_variant:{label}:{number}")
    return make_data(poses, labels, people, identifiers)


def load_feedback():
    path = ROOT / "feedback/misclassification_log.csv"
    poses, labels, identifiers, skipped = [], [], [], []
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                label = row["actual_label"]
                if label == "dorsal":
                    label = "no_gesture"
                try:
                    points = np.asarray(json.loads(row["landmarks_json"]), np.float32)
                    if points.shape != (21, 2) or not np.isfinite(points).all():
                        raise ValueError("missing finite 21x2 landmarks")
                    # Recompute features from the reviewed pose; never trust the
                    # model's prediction as the ground-truth training label.
                    landmarks_to_feature(points)
                    target = LABELS.index(label)
                except (ValueError, TypeError, KeyError) as error:
                    skipped.append({"id": row.get("feedback_id"), "reason": str(error)})
                    continue
                poses.append(points)
                labels.append(target)
                identifiers.append("feedback:" + row["feedback_id"])
    if not poses:
        raise ValueError("No usable reviewed feedback landmarks were found")
    data = make_data(poses, labels, ["local-feedback:unidentified-session"] * len(poses), identifiers)
    audit = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(poses),
             "class_counts": {name: int(np.sum(data["y"] == i)) for i, name in enumerate(LABELS)},
             "skipped": skipped,
             "evaluation_use": "training only; participant/session metadata is absent"}
    return data, audit


def prepare(seed=18065, per_class=4096):
    train, validation, test, legacy_audit = prepare_legacy(per_class=per_class)
    groups = {}
    excluded = {}
    palm_relabels = {}
    legacy_source = load_npz(ROOT / "data/ten_gesture/hagrid_train.npz")
    palm_ids = set(map(str, legacy_source["source_id"][np.asarray([
        str(variant) == "source_palm" or str(variant).startswith("palm_axis_")
        for variant in legacy_source["variant"]])]))
    for split, original in (("train", train), ("val", validation), ("test", test)):
        data = dict(X=original["X"], landmarks=original["landmarks"], y=original["y"].copy(),
                    group=original["group"], source_id=original.get("source_id", original.get("parent_id")),
                    handedness=np.full(len(original["y"]), ""))
        data["y"][data["y"] == STOP] = REJECT  # old rejection sentinel was index 12
        palm_source = (data["y"] == CLASS_NAMES.index("open_palm")) | np.isin(data["source_id"], list(palm_ids))
        previous = data["y"].copy()
        for index in np.flatnonzero(palm_source):
            data["y"][index] = (CLASS_NAMES.index("open_palm")
                                if open_hand_xy_orientation(data["landmarks"][index])["allowed_direction"] else REJECT)
        palm_relabels[split] = int(np.sum(previous != data["y"]))
        # Some older 11K Dorsal negatives show the newly requested command.
        # Their category has no Stop-inverted review, so exclude ambiguous
        # upright extended poses instead of assigning contradictory labels.
        ambiguous = np.asarray([label == DORSAL and stop_inverted_geometry(p)["valid"]
                                for p, label in zip(data["landmarks"], data["y"])])
        excluded[split] = int(ambiguous.sum())
        groups[split] = subset(data, ~ambiguous)
    assigned = {str(group): split for split in ("train", "val", "test") for group in groups[split]["group"]}
    additions = {split: [] for split in groups}
    for official in ("train", "val", "test"):
        data = load_npz(ROOT / f"data/stop_inverted/hagrid_{official}.npz")
        for points, group, identifier in zip(data["landmarks"], data["group"], data["source_id"]):
            split = assigned.get(str(group), _partition(str(group)) if official == "train" else official)
            if official != "train" and split != official:
                raise ValueError("Official HaGRID partitions conflict with legacy participant isolation")
            assigned[str(group)] = split
            additions[split].append((points, str(group), str(identifier)))
    source_counts = {}
    for split, rows in additions.items():
        source_counts[split] = len(rows)
        extra = make_data([r[0] for r in rows], [STOP] * len(rows), [r[1] for r in rows],
                          [r[2] for r in rows], [dorsal_side(r[0]) for r in rows])
        # Source misdetections/poses outside the specified command shape are
        # retained as rejection evidence, not mislabeled positive commands.
        extra["y"] = np.asarray([STOP if stop_inverted_geometry(p)["valid"] else REJECT
                                 for p in extra["landmarks"]], np.int64)
        groups[split] = join([groups[split], extra])

    feedback, feedback_audit = load_feedback()
    feedback_keys = set(feature_keys(feedback["X"]))
    deduplicated = {}
    seen = set(feedback_keys)
    for split in ("test", "val", "train"):
        keys = feature_keys(groups[split]["X"])
        keep = np.asarray([key not in seen for key in keys])
        # Preserve training resampling/augmentation; only prevent cross-split
        # duplicates. Feedback is exclusively training evidence.
        deduplicated[split] = int((~keep).sum())
        groups[split] = subset(groups[split], keep)
        seen.update(keys)
    # Camera corrections have no independent people/session IDs: all belong
    # in train. Repetition gives a small set useful weight, never eval credit.
    feedback_repeated = subset(feedback, np.repeat(np.arange(len(feedback["y"])), 128))
    groups["train"] = join([groups["train"], feedback_repeated])

    rng = np.random.default_rng(seed)
    source = groups["train"]
    stop_indices = np.flatnonzero(source["y"] == STOP)
    poses, labels, people, ids, sides = [], [], [], [], []
    for number in range(per_class * 3):
        index = int(rng.choice(stop_indices))
        negative = number >= per_class * 2
        # A soft classifier needs shape evidence across the exact runtime
        # endpoints. The geometry gate, not softmax mass, imposes the hard
        # cutoff at +/-90 from up. A small synthetic guard band avoids placing
        # each allowed horizontal endpoint at a learned 50/50 boundary.
        tilt = (rng.choice((-1, 1)) * rng.uniform(110, 180) if negative
                else rng.choice((-90., 90.)) if number % 4 == 0 else rng.uniform(-100, 100))
        pose = rotate_tilt(source["landmarks"][index], tilt,
                           mirror=bool(number % 2), spread=float(rng.uniform(0, 24)))
        side = dorsal_side(pose)
        pose = vary_image_aspect(pose, rng, number)
        poses.append(pose)
        labels.append(REJECT if negative else STOP)
        people.append(str(source["group"][index]))
        ids.append(str(source["source_id"][index]) + ":train_augmentation")
        # Positive aspect scaling preserves the known hand side, even if its
        # squashed 2-D winding would be too weak for a live surface decision.
        sides.append(side)
    groups["train"] = join([groups["train"], make_data(poses, labels, people, ids, sides)])
    # Palm shares the same upper semicircle. Legacy horizontal Palm negatives
    # are corrected above using their known HaGRID Palm provenance.
    palm_indices = np.flatnonzero(source["y"] == CLASS_NAMES.index("open_palm"))
    palm_poses, palm_labels, palm_people, palm_identifiers = [], [], [], []
    for number in range(per_class * 2):
        index = int(rng.choice(palm_indices))
        negative = number >= per_class
        tilt = (rng.choice((-1, 1)) * rng.uniform(110, 180) if negative
                else rng.choice((-90., 90.)) if number % 4 == 0 else rng.uniform(-100, 100))
        palm_poses.append(vary_image_aspect(
            rotate_tilt(source["landmarks"][index], tilt, mirror=bool(number % 2)), rng, number))
        palm_labels.append(REJECT if negative else CLASS_NAMES.index("open_palm"))
        palm_people.append(str(source["group"][index]))
        palm_identifiers.append(str(source["source_id"][index]) + ":palm_rotation")
    groups["train"] = join([groups["train"], make_data(palm_poses, palm_labels, palm_people, palm_identifiers)])
    # Preserve the established rotation behavior of Like and Dislike while
    # retraining, rather than teaching the replacement only upright thumbs.
    thumb_poses, thumb_labels, thumb_people, thumb_ids = [], [], [], []
    for name in ("like", "thumb_down"):
        label = CLASS_NAMES.index(name)
        available = np.flatnonzero(source["y"] == label)
        for number in range(per_class):
            index = int(rng.choice(available))
            thumb_poses.append(rotate_tilt(source["landmarks"][index], rng.uniform(-180, 180),
                                           mirror=bool(number % 2)))
            thumb_labels.append(label)
            thumb_people.append(str(source["group"][index]))
            thumb_ids.append(str(source["source_id"][index]) + ":thumb_rotation")
    groups["train"] = join([groups["train"], make_data(thumb_poses, thumb_labels, thumb_people, thumb_ids)])
    # Give genuine non-command shapes the same aspect variation as commands.
    # Otherwise broadening only positive shapes can inflate unknown acceptance.
    rejection_indices = np.flatnonzero(
        (source["y"] == REJECT)
        & ~np.isin(source["source_id"], list(palm_ids))
        & ~np.char.startswith(source["source_id"].astype(str), "feedback:"))
    if not len(rejection_indices):
        raise ValueError("No participant-isolated non-command training poses")
    rejected_poses, rejected_people, rejected_ids = [], [], []
    for number in range(per_class):
        index = int(rng.choice(rejection_indices))
        rejected_poses.append(vary_image_aspect(source["landmarks"][index], rng, number))
        rejected_people.append(str(source["group"][index]))
        rejected_ids.append(str(source["source_id"][index]) + ":rejection_aspect")
    groups["train"] = join([groups["train"], make_data(
        rejected_poses, [REJECT] * len(rejected_poses), rejected_people, rejected_ids)])
    pointing_poses, pointing_labels, pointing_people, pointing_ids = [], [], [], []
    for label in range(4):
        available = np.flatnonzero((source["y"] == label) & (source["X"][:, 69] >= .7)
                                   & (source["X"][:, 70:73].max(1) <= .5))
        if not len(available):
            raise ValueError(f"No genuine single-index training shapes for {CLASS_NAMES[label]}")
        for number in range(per_class // 2):
            index = int(rng.choice(available))
            pointing_poses.append(vary_image_aspect(vary_pointing_thumb(source["landmarks"][index], rng), rng, number))
            pointing_labels.append(label)
            pointing_people.append(str(source["group"][index]))
            pointing_ids.append(str(source["source_id"][index]) + ":expanded_thumb")
    groups["train"] = join([groups["train"], make_data(
        pointing_poses, pointing_labels, pointing_people, pointing_ids)])
    directional_variants = directional_feedback_augmentation(feedback, rng, per_class)
    if directional_variants is not None:
        groups["train"] = join([groups["train"], directional_variants])
    people = {split: set(map(str, data["group"])) for split, data in groups.items()}
    overlap = {f"{a}/{b}": len(people[a] & people[b])
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlap.values()):
        raise ValueError(f"Participant leakage: {overlap}")
    audit = dict(seed=seed, per_class=per_class, labels=LABELS, legacy=legacy_audit,
                 open_hand_rotation=dict(xy_minimum=-180., xy_maximum=0., endpoints_inclusive=True,
                                         maximum_tilt_from_up=OPEN_HAND_MAX_TILT_FROM_UP_DEGREES,
                                         classifier_synthetic_guard_band_degrees=10.,
                                         normalized_xy_aspect_augmentation=[.5, 2.],
                                         runtime_still_rejects_entire_lower_half=True),
                 legacy_palm_rows_relabelled_for_upper_arc=palm_relabels,
                 expanded_thumb_direction_training_rows=len(pointing_poses),
                 reviewed_horizontal_augmentation=dict(
                     rows=0 if directional_variants is None else len(directional_variants["y"]),
                     source_aspect="read from each feedback JPEG", target_aspects=[1., 4. / 3., 16. / 9., 3. / 4.],
                     maximum_pixel_angle_from_horizontal=38., reflection_swaps_labels=True,
                     training_only=True),
                 source_stop_counts=source_counts, feedback=feedback_audit,
                 ambiguous_legacy_dorsal_excluded=excluded, cross_split_rows_removed=deduplicated,
                 participant_overlap=overlap,
                 counts={split: {name: int(np.sum(data["y"] == i)) for i, name in enumerate(LABELS)}
                         for split, data in groups.items()},
                 limitations=["Stop source includes official val/test and participant-held-out training annotations.",
                              "Finger spread, mirroring, angle and image-aspect coverage are 2-D augmentation, not new camera viewpoints.",
                              "Feedback is training only and cannot independently measure generalization.",
                              "Source Stop-inverted handedness is inferred from the known surface label, not a tested detector.",
                              "Live webcam verification at different distances and lighting is still required."])
    return groups, audit


def score(path, data, runtime=False):
    probabilities, mass = predict_onnx(path, data["X"])
    if probabilities.shape[1] == STOP:
        # Read-only comparison with the preserved previous model.
        probabilities = np.pad(probabilities, ((0, 0), (0, 1)))
    resolved, valid = probabilities.copy(), np.ones(len(mass), bool)
    if runtime:
        resolver = GeometryResolver(RuntimeConfig())
        for i, points in enumerate(data["landmarks"]):
            side = str(data["handedness"][i]) or None
            resolved[i], details = resolver.resolve(probabilities[i], points,
                                                    handedness=side, handedness_confidence=.99 if side else None)
            valid[i] = details["pose_validation"]["valid"]
    raw = resolved.argmax(1)
    ordered = np.sort(resolved, axis=1)
    accepted = valid & (raw != DORSAL) & (mass >= .70) & (ordered[:, -1] >= .70) & (ordered[:, -1] - ordered[:, -2] >= .08)
    predicted = np.where(accepted, raw, REJECT)
    truth = np.where(data["y"] == DORSAL, REJECT, data["y"])
    report = classification_report(truth, predicted, labels=np.arange(len(LABELS)),
                                   target_names=LABELS, output_dict=True, zero_division=0)
    active_labels = [i for i in range(len(LABELS)) if i != DORSAL]
    known = (truth != REJECT) & (truth != STOP)
    return dict(macro_f1=float(f1_score(truth, predicted, labels=active_labels, average="macro", zero_division=0)),
                accepted_precision=float(np.mean(predicted[accepted] == truth[accepted])) if accepted.any() else 0.,
                old_command_correct_accept_rate=float(np.mean(predicted[known] == truth[known])) if known.any() else 0.,
                unknown_false_accept_rate=float(np.mean(accepted[truth == REJECT])) if np.any(truth == REJECT) else 0.,
                stop_recall=report["stop_inverted"]["recall"], stop_precision=report["stop_inverted"]["precision"],
                per_class=report, confusion=confusion_matrix(truth, predicted, labels=np.arange(len(LABELS))).tolist())


def train(epochs=100, per_class=4096, prepared=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if prepared:
        audit = json.loads((OUTPUT / "data_audit.json").read_text(encoding="utf-8"))
        _, current_feedback = load_feedback()
        if current_feedback["sha256"] != audit["feedback"]["sha256"]:
            raise ValueError("Feedback changed since this training dataset was prepared")
        groups = {split: load_npz(OUTPUT / f"{split}.npz") for split in ("train", "val", "test")}
    else:
        groups, audit = prepare(per_class=per_class)
        write_json(OUTPUT / "data_audit.json", audit)
        for split, data in groups.items():
            np.savez_compressed(OUTPUT / f"{split}.npz", **data)
    shutil.copy2(ROOT / "feedback/misclassification_log.csv", OUTPUT / "feedback_snapshot.csv")
    print("Prepared", {split: len(data["y"]) for split, data in groups.items()}, flush=True)
    scaler = StandardScaler().fit(groups["train"]["X"])
    x = scaler.transform(groups["train"]["X"]).astype(np.float32)
    validation = scaler.transform(groups["val"]["X"]).astype(np.float32)
    model = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=.003, batch_size=256,
                          learning_rate_init=.0007, random_state=18065, max_iter=1)
    best, best_f1, stale, history = None, -1., 0, []
    training_rng = np.random.default_rng(18065)
    with threadpool_limits(limits=1):
        for epoch in range(epochs):
            # Bound sklearn's label-binarization and activation scratch arrays
            # instead of allocating them for the complete expanded dataset.
            order = training_rng.permutation(len(x))
            for start in range(0, len(order), 8192):
                indices = order[start:start + 8192]
                model.partial_fit(x[indices], groups["train"]["y"][indices], classes=np.arange(len(LABELS)))
            metric = float(f1_score(groups["val"]["y"], model.predict(validation),
                                    labels=np.arange(len(LABELS)), average="macro", zero_division=0))
            history.append({"epoch": epoch + 1, "validation_raw_macro_f1": metric})
            if metric > best_f1 + 1e-4:
                best, best_f1, stale = copy.deepcopy(model), metric, 0
            else:
                stale += 1
            if epoch % 5 == 0:
                print(f"Epoch {epoch + 1}: val macro F1 {metric:.5f}", flush=True)
            if stale >= 18:
                break
    path = OUTPUT / "candidate.onnx"
    artifact = export_onnx(best, scaler, path, commands=CLASS_NAMES)
    expected = best.predict_proba(validation)
    actual, mass = predict_onnx(path, groups["val"]["X"])
    expected_mass = expected[:, :REJECT].sum(1)
    conditional = expected[:, :REJECT] / np.maximum(expected_mass[:, None], 1e-30)
    parity = dict(label_agreement=float(np.mean(conditional.argmax(1) == actual.argmax(1))),
                  max_probability_error=float(np.max(np.abs(conditional - actual))),
                  max_mass_error=float(np.max(np.abs(expected_mass - mass))))
    evaluation = dict(artifact=artifact, parity=parity, audit=audit, history=history, deployed=False)
    for split in ("val", "test"):
        evaluation[split] = score(path, groups[split])
        evaluation[split + "_runtime"] = score(path, groups[split], runtime=True)
        print(split, {k: v for k, v in evaluation[split].items() if k not in {"per_class", "confusion"}}, flush=True)
        print(split + " runtime", {k: v for k, v in evaluation[split + "_runtime"].items() if k not in {"per_class", "confusion"}}, flush=True)
    feedback, _ = load_feedback()
    evaluation["feedback_training_fit"] = score(path, feedback)
    evaluation["feedback_training_fit_runtime"] = score(path, feedback, runtime=True)
    sources = load_npz(ROOT / "data/ten_gesture/hagrid_train.npz")
    calls = sources["source_id"][sources["variant"] == "source_call"]
    call_mask = np.isin(groups["test"]["source_id"], calls)
    evaluation["retired_call_test"] = score(path, subset(groups["test"], call_mask))
    write_json(OUTPUT / "evaluation.json", evaluation)
    return evaluation


def evaluate_saved_candidate():
    """Refresh post-model checks without fitting or selecting another epoch."""
    evaluation = json.loads((OUTPUT / "evaluation.json").read_text())
    candidate = OUTPUT / "candidate.onnx"
    if hashlib.sha256(candidate.read_bytes()).hexdigest() != evaluation["artifact"]["sha256"]:
        raise ValueError("Saved candidate differs from the fitted evaluation record")
    data = {split: load_npz(OUTPUT / f"{split}.npz") for split in ("val", "test")}
    for split in data:
        evaluation[split + "_runtime"] = score(candidate, data[split], runtime=True)
    feedback, _ = load_feedback()
    evaluation["feedback_training_fit_runtime"] = score(candidate, feedback, runtime=True)
    sources = load_npz(ROOT / "data/ten_gesture/hagrid_train.npz")
    calls = sources["source_id"][sources["variant"] == "source_call"]
    mask = np.isin(data["test"]["source_id"], calls)
    if not mask.any():
        raise ValueError("No held-out retired-call negatives")
    evaluation["retired_call_test"] = score(candidate, subset(data["test"], mask))
    legacy_candidates = sorted((OUTPUT / "previous_release").glob("*/gesture_mlp_production.onnx"))
    if legacy_candidates:
        comparison = {}
        for split in data:
            old_classes = subset(data[split], data[split]["y"] != STOP)
            comparison[split] = {"previous": score(legacy_candidates[0], old_classes, runtime=True),
                                  "candidate": score(candidate, old_classes, runtime=True)}
        evaluation["previous_runtime_comparison_on_same_old_classes"] = comparison
    write_json(OUTPUT / "evaluation.json", evaluation)
    print("Refreshed runtime, feedback and same-data previous-model comparison", flush=True)


def install():
    from backend.artifact_integrity import ArtifactRegistry
    models = ROOT / "models"
    evaluation = json.loads((OUTPUT / "evaluation.json").read_text())
    source = OUTPUT / "candidate.onnx"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != evaluation["artifact"]["sha256"]:
        raise ValueError("Candidate differs from the evaluated ONNX artifact")
    parity, test = evaluation["parity"], evaluation["test"]
    checks = dict(onnx_parity=parity["label_agreement"] >= .999 and parity["max_probability_error"] < 1e-4,
                  offline_macro_f1_at_least_0_97=test["macro_f1"] >= .97,
                  offline_accepted_precision_at_least_0_99=test["accepted_precision"] >= .99,
                  offline_unknown_false_accept_at_most_0_02=test["unknown_false_accept_rate"] <= .02,
                  old_command_correct_accept_at_least_0_98=test["old_command_correct_accept_rate"] >= .98,
                  new_command_recall_at_least_0_97=test["stop_recall"] >= .97,
                  retired_call_false_accept_at_most_0_02=evaluation["retired_call_test"]["unknown_false_accept_rate"] <= .02,
                  runtime_peace_recall_at_least_0_95=evaluation["test_runtime"]["per_class"]["peace"]["recall"] >= .95,
                  runtime_rock_recall_at_least_0_97=evaluation["test_runtime"]["per_class"]["rock"]["recall"] >= .97,
                  runtime_unknown_false_accept_at_most_0_02=evaluation["test_runtime"]["unknown_false_accept_rate"] <= .02,
                  stop_precision=test["stop_precision"] >= .97,
                  runtime_stop_precision=evaluation["test_runtime"]["stop_precision"] >= .97,
                  runtime_stop_recall=evaluation["test_runtime"]["stop_recall"] >= .95)
    if evaluation["audit"].get("reviewed_horizontal_augmentation", {}).get("rows", 0):
        report = json.loads((OUTPUT / "candidate_feedback.json").read_text(encoding="utf-8"))
        checks["horizontal_feedback_matches_model"] = report["model_sha256"] == digest
        checks["horizontal_feedback_matches_dataset"] = report["feedback_sha256"] == evaluation["audit"]["feedback"]["sha256"]
        for label in ("left", "right"):
            summary = report["summary"][label]
            checks[f"reviewed_{label}_classified_and_executed"] = (
                summary["rows"] > 0 and summary["raw_accepted_correct"] == summary["rows"]
                and summary["executed_correct"] == summary["rows"])
            counts = report["stress"]["counts"]
            checks[f"reviewed_{label}_angle_mirror_aspect_variants"] = (
                counts[label + "_total"] > 0 and counts[label + "_correct"] == counts[label + "_total"])
        replay = report["image_replay"]
        checks["reviewed_directional_jpeg_detection_and_actions"] = (
            replay["total"] > 0 and replay["correct"] == replay["total"]
            and replay["correct_action"] == replay["total"])
        evaluation["directional_feedback_evidence"] = dict(
            report=(OUTPUT / "candidate_feedback.json").relative_to(ROOT).as_posix(),
            feedback_sha256=report["feedback_sha256"],
            reviewed_rows=sum(report["summary"][label]["rows"] for label in ("left", "right")),
            transformed_rows=sum(report["stress"]["counts"][label + "_total"] for label in ("left", "right")),
            jpeg_actions=replay["correct_action"], training_evidence_only=True)
    if not all(checks.values()):
        raise ValueError(f"Candidate release checks failed: {checks}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = OUTPUT / "previous_release" / stamp
    backup.mkdir(parents=True, exist_ok=False)
    names = ["gesture_mlp_production.onnx", "gesture_mlp_onnx_metadata.json",
             "gesture_mobile_runtime_config.json", "gesture_artifact_manifest.json",
             "gesture_online_replay_cache.npz", "gesture_online_validation_cache.npz",
             "gesture_online_untouched_test_cache.npz", "gesture_online_adapter_state.npz"]
    for name in names:
        if (models / name).exists():
            shutil.copy2(models / name, backup / name)
    old = json.loads((models / "gesture_mlp_onnx_metadata.json").read_text())
    config = json.loads((models / "gesture_mobile_runtime_config.json").read_text())
    config.update(class_names=CLASS_NAMES, feedback_labels=FEEDBACK_LABELS, gesture_to_action=GESTURE_TO_ACTION,
                  model_format="ONNX float32, thirteen tensor outputs including retired Dorsal")
    config["stop_inverted_validation"] = dict(maximum_tilt_from_up_degrees=90.,
                                               angle_space="source image XY pixels", required_surface="dorsal",
                                               fingers_together_or_spread=True,
                                               xy_minimum_degrees=-180., xy_maximum_degrees=0.,
                                               above_limit="no_gesture")
    config["open_palm_rotation_validation"] = dict(xy_minimum_degrees=-180., xy_maximum_degrees=0.,
                                                    angle_space="source image XY pixels", endpoints_inclusive=True,
                                                    lower_half="no_gesture")
    config.setdefault("directional_resolution", {}).update(
        angle_space="source image XY pixels when dimensions are available",
        positive_index_dx_command="left", negative_index_dx_command="right")
    for key in ("twelve_gesture_release", "thirteen_gesture_release"):
        config.pop(key, None)
    config["stop_inverted_release"] = dict(model_sha256=digest, offline_checks=checks,
                                           evaluation=(OUTPUT / "evaluation.json").relative_to(ROOT).as_posix(),
                                           live_camera_validation_pending=True)
    metadata = dict(schema_version=4, created_utc=datetime.now(timezone.utc).isoformat(),
                    format="ONNX", model_name=("StopInvertedDirectionalFeedbackMLP" if "directional_feedback_evidence" in evaluation else "StopInvertedFeedbackMLP"), onnx_file="gesture_mlp_production.onnx",
                    onnx_bytes=source.stat().st_size, onnx_sha256=digest,
                    input=dict(name="landmark_features", dtype="float32", shape=[None, 76]),
                    probability_output="probabilities", known_mass_output="known_gesture_mass",
                    output_class_order=CLASS_NAMES, internal_class_order=LABELS, feature_names=FEATURE_NAMES,
                    source_class_mapping={**old["source_class_mapping"], "stop_inverted": "HaGRID stop_inverted"},
                    horizontal_direction_calibration=old["horizontal_direction_calibration"],
                    parity=dict(passed=True, prediction_agreement=parity["label_agreement"],
                                maximum_absolute_probability_error=parity["max_probability_error"],
                                maximum_absolute_mass_error=parity["max_mass_error"]),
                    quality=dict(macro_f1=test["macro_f1"], accepted_precision=test["accepted_precision"],
                                 unknown_false_acceptance_rate=test["unknown_false_accept_rate"],
                                 old_command_correct_accept_rate=test["old_command_correct_accept_rate"],
                                 new_command_recall={"stop_inverted": test["stop_recall"]}),
                    qualification=dict(offline_passed=True, live_camera_validation_pending=True,
                                       checks=checks, data_audit=evaluation["audit"],
                                       selection=dict(model_sha256=digest, **{
                                           name: config["temporal_smoothing"][name]
                                           for name in ("known_mass_floor", "confidence_floor", "probability_margin_floor")})),
                    runtime_note="Stop inverted plus feedback training; live camera validation pending.")
    for split, name in (("train", "replay"), ("val", "validation"), ("test", "untouched_test")):
        data = load_npz(OUTPUT / f"{split}.npz")
        if split == "train":
            indices = replay_indices(data)
            data = subset(data, indices)
            metadata["reviewed_feedback_replay_counts"] = {
                name: int(np.sum((data["y"] == label) & np.char.startswith(data["source_id"].astype(str), "feedback:")))
                for label, name in enumerate(LABELS)}
        data["y"] = np.where((data["y"] == REJECT) | (data["y"] == DORSAL), -1, data["y"])
        np.savez_compressed(models / f"gesture_online_{name}_cache.npz", **data, class_names=np.asarray(CLASS_NAMES))
    shutil.copy2(source, models / "gesture_mlp_production.onnx")
    write_json(models / "gesture_mobile_runtime_config.json", config)
    write_json(models / "gesture_mlp_onnx_metadata.json", metadata)
    ArtifactRegistry(models).build()
    evaluation.update(deployed=True, backup=str(backup.relative_to(ROOT)), release_checks=checks)
    write_json(OUTPUT / "evaluation.json", evaluation)
    print(f"Installed evaluated candidate {digest}; previous bundle: {backup}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--per-class", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Isolated candidate and evaluation directory")
    parser.add_argument("--prepared", action="store_true", help="Reuse the isolated, feedback-hash-checked prepared splits")
    parser.add_argument("--install", action="store_true", help="Install the already evaluated candidate; do not retrain")
    parser.add_argument("--evaluate", action="store_true", help="Refresh runtime comparisons without retraining")
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    install() if args.install else evaluate_saved_candidate() if args.evaluate else train(args.epochs, args.per_class, args.prepared)
