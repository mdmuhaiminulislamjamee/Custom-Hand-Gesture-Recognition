"""Compare saved-image hold regressions; these images are training evidence."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from backend.config import load_runtime_config
from backend.model_runtime import InferenceEngine, OnnxClassifier, RuntimeSession


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--directory', type=Path, default=ROOT / 'artifacts/feedback_refresh_20261111')
    parser.add_argument('--per-class', type=int, default=8)
    args = parser.parse_args()
    out = args.directory.resolve(); selection = out / 'image_replay_selection.csv'
    if selection.exists():
        records = read_csv(selection)
    else:
        fresh = read_csv(out / 'new_feedback.csv')
        all_rows = read_csv(out / 'feedback_snapshot.csv')
        new_ids = {r['feedback_id'] for r in fresh}
        groups = {}
        for r in fresh:
            if Path(r['image_path']).is_file(): groups.setdefault(r['actual_label'], []).append(r)
        for r in all_rows:
            if r['actual_label'] in ('left', 'right') and r['feedback_id'] not in new_ids and Path(r['image_path']).is_file():
                groups.setdefault('old_' + r['actual_label'], []).append(r)
        records = []
        for label, rows in sorted(groups.items()):
            rows.sort(key=lambda r: r['created_at_utc'])
            indices = np.linspace(0, len(rows) - 1, min(args.per_class, len(rows)), dtype=int)
            records.extend(rows[int(i)] for i in indices)
        with selection.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    config = load_runtime_config()
    metadata = json.loads((ROOT / 'models/gesture_mlp_onnx_metadata.json').read_text())
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    metadata['onnx_sha256'] = digest
    meta_path = args.output.with_suffix('.metadata.json')
    meta_path.write_text(json.dumps(metadata), encoding='utf-8')
    engine = InferenceEngine(config, load_models=False)
    engine.model_manager.models['ONNX'] = OnnxClassifier(args.model, meta_path, config)
    engine.model_manager.errors.clear()
    if not engine.detector.ready: raise RuntimeError(engine.detector.error)
    rows = []
    try:
        for number, record in enumerate(records):
            content = Path(record['image_path']).read_bytes()
            session = RuntimeSession(config)
            results = [engine.process_frame(content, session, now=1. + number * 4 + tick * .1) for tick in range(12)]
            last = results[-1]; target = record['actual_label']
            expected = config.gesture_to_action.get(target, 'Wait / No Action')
            action_ok = (all(r.get('runtime_action') in (None, 'Wait / No Action') for r in results) if target == 'no_gesture'
                         else any(r.get('runtime_action') == expected for r in results))
            rows.append(dict(id=record['feedback_id'], actual=target, predicted=last.get('runtime_prediction'),
                raw=last.get('raw_prediction'), status=last.get('status'), reason=last.get('action_reason'),
                action_correct=action_ok, feature_vector=last.get('feature_vector'), landmarks=last.get('landmarks'),
                world_landmarks=last.get('world_landmarks'), quality=last.get('quality')))
            print(f'{number+1}/{len(records)} {target} -> {last.get("runtime_prediction")}', flush=True)
    finally:
        engine.close()
    report = dict(model_sha256=digest, selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),
        scope='Deterministic sampled reviewed JPEG holds, 12 frames each; training regression, not independent live accuracy',
        total=len(rows), correct=sum(r['predicted'] == r['actual'] for r in rows),
        actions=sum(r['action_correct'] for r in rows),
        per_class={k: dict(rows=sum(r['actual'] == k for r in rows), correct=sum(r['actual'] == k and r['predicted'] == k for r in rows),
                          actions=sum(r['actual'] == k and r['action_correct'] for r in rows)) for k in sorted({r['actual'] for r in rows})}, rows=rows)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
