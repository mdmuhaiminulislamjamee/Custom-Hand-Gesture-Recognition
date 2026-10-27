"""Run the reference tests in separate processes to bound native-library memory."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('.runtime/backend_verification'))
    parser.add_argument('--failed-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    temporary = root / '.runtime/backend_test_temp'
    temporary.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    previous = {}
    if args.failed_only:
        previous = {r['file']: r for r in json.loads((output / 'summary.json').read_text())['results']}
    tests = sorted((root / 'backend/tests').glob('test_*.py'))
    rows = []
    for number, test in enumerate(tests, 1):
        if previous.get(test.name, {}).get('exit_code') == 0:
            rows.append(previous[test.name])
        else:
            command = [sys.executable, '-m', 'pytest', str(test), '-o', 'addopts=', '-q',
                       '-p', 'no:cacheprovider', '--tb=short', '--basetemp=' + str(temporary / test.stem)]
            log = output / (test.stem + '.log')
            with log.open('w', encoding='utf-8') as stream:
                completed = subprocess.run(command, cwd=root, env=environment,
                                           stdout=stream, stderr=subprocess.STDOUT)
            content = log.read_text(encoding='utf-8', errors='replace')
            counts = re.findall(r'(\d+) passed', content)
            rows.append(dict(file=test.name, exit_code=completed.returncode,
                             passed=int(counts[-1]) if counts else 0))
            print(f'{number}/{len(tests)} {test.name}: {rows[-1]}', flush=True)
            if completed.returncode:
                print(content[-5000:], flush=True)
        summary = dict(files=len(tests), completed=len(rows), passed=sum(r['passed'] for r in rows),
                       failed_files=sum(r['exit_code'] != 0 for r in rows), results=rows)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return int(summary['failed_files'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
