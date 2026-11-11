"""Compare an isolated direction candidate with the preserved production model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def original_module(name):
    prefix = subprocess.check_output(['git', 'rev-parse', '--show-prefix'], cwd=ROOT, text=True).strip()
    relative = name.replace('.', '/') + '.py'
    source = subprocess.check_output(['git', 'show', 'HEAD:' + prefix + relative], cwd=ROOT)
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / relative)
    module.__package__ = name.rpartition('.')[0]
    sys.modules[name] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-dir', type=Path, default=ROOT / 'artifacts/direction_stability_20261111')
    parser.add_argument('--baseline-tests', nargs='+')
    args = parser.parse_args()
    if args.baseline_tests:
        print('BASELINE_MODEL_SHA256=' + hashlib.sha256(
            (ROOT / 'models/gesture_mlp_production.onnx').read_bytes()).hexdigest(), flush=True)
        original_module('backend.geometry')
        original_module('backend.model_runtime')
        import pytest
        raise SystemExit(pytest.main(args.baseline_tests + ['-q', '-p', 'no:cacheprovider', '--tb=short']))
    import scripts.train_stop_inverted as trainer
    from research.v18_20 import load_npz
    current_resolver = trainer.GeometryResolver
    # Load the old resolver under a distinct module name so current imports and
    # the source checkout remain intact. Geometry has no mutable global state.
    prefix = subprocess.check_output(['git', 'rev-parse', '--show-prefix'], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(['git', 'show', 'HEAD:' + prefix + 'backend/geometry.py'], cwd=ROOT)
    baseline = types.ModuleType('backend.geometry_before_direction_stability')
    baseline.__package__ = 'backend'
    sys.modules[baseline.__name__] = baseline
    exec(compile(source, '<preserved geometry>', 'exec'), baseline.__dict__)
    directory = args.candidate_dir.resolve()
    models = {'previous': directory / 'baseline/gesture_mlp_production.onnx',
              'candidate': directory / 'candidate.onnx'}
    report = {'model_sha256': {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in models.items()},
              'baseline_geometry_sha256': hashlib.sha256(source).hexdigest(), 'splits': {}}
    for split in ('val', 'test'):
        data = load_npz(directory / (split + '.npz'))
        rows = {}
        trainer.GeometryResolver = baseline.GeometryResolver
        rows['previous_model_previous_rules'] = trainer.score(models['previous'], data, runtime=True)
        trainer.GeometryResolver = current_resolver
        rows['previous_model_current_rules'] = trainer.score(models['previous'], data, runtime=True)
        rows['candidate_model_current_rules'] = trainer.score(models['candidate'], data, runtime=True)
        rows['previous_model_scalar'] = trainer.score(models['previous'], data)
        rows['candidate_model_scalar'] = trainer.score(models['candidate'], data)
        report['splits'][split] = rows
        print(split, json.dumps({k: {n: v[n] for n in ('macro_f1', 'accepted_precision', 'unknown_false_accept_rate')}
                                for k, v in rows.items()}), flush=True)
    output = directory / 'baseline_comparison.json'
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(output, flush=True)


if __name__ == '__main__':
    main()
