"""Isolated Flutter host checks; native OpenCV builds remain enabled in delivery."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / '.runtime/mobile_false_hand_host'
STAGE = Path(tempfile.mkdtemp(prefix='gesture-filter-tests-'))
FLUTTER = [r'C:\src\flutter\bin\cache\dart-sdk\bin\dart.exe','--old_gen_heap_size=128',r'C:\src\flutter\bin\cache\flutter_tools.snapshot','--no-version-check']

def main():
    global REPORTS, STAGE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tester-only',action='store_true')
    parser.add_argument('--report-dir',type=Path,default=REPORTS)
    parser.add_argument('--stage',type=Path,help='Reuse a short test workspace and its compiler cache')
    args=parser.parse_args()
    REPORTS=args.report_dir.resolve()
    if args.stage: STAGE=args.stage.resolve()
    STAGE.mkdir(parents=True,exist_ok=True)
    REPORTS.mkdir(parents=True,exist_ok=True)
    print('Short test workspace: ' + str(STAGE),flush=True)
    ignored=shutil.ignore_patterns('build','.dart_tool','.gradle','.git','__pycache__','.cxx','releases')
    for name in ('static_gesture_runtime','flutter_gesture_tester'):
        shutil.copytree(ROOT/name,STAGE/name,ignore=ignored,dirs_exist_ok=True)
    results=[]
    if args.tester_only:
        results=[row for row in json.loads((REPORTS/'results.json').read_text()) if row['project']=='runtime']
    for name,project in [('runtime',STAGE/'static_gesture_runtime/flutter'),('tester',STAGE/'flutter_gesture_tester')]:
        if args.tester_only and name=='runtime':continue
        pubspec=project/'pubspec.yaml'
        text=pubspec.read_text(encoding='utf-8')
        if 'hooks:' in text:
            text=text.replace('      parallel_jobs: 1','      parallel_jobs: 1\n      skip_build: true')
        else:
            text+='\nhooks:\n  user_defines:\n    dartcv4:\n      skip_build: true\n'
        pubspec.write_text(text,encoding='utf-8')
        for kind,command_args in [('dependencies',['pub','get','--offline','--enforce-lockfile']),
                          ('analysis',['analyze','--no-pub','lib','test']),
                          ('tests',['test','--no-pub','--concurrency=1','--reporter=expanded'])]:
            log=REPORTS/f'{name}_{kind}.log'
            with log.open('w',encoding='utf-8') as stream:
                completed=subprocess.run(FLUTTER+command_args,cwd=project,stdout=stream,stderr=subprocess.STDOUT)
            lines=log.read_text(encoding='utf-8',errors='replace').splitlines()
            entry=dict(project=name,check=kind,exit_code=completed.returncode,tail=[v[:400] for v in lines[-12:]])
            results.append(entry)
            print(json.dumps(entry),flush=True)
            (REPORTS/'results.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
            if completed.returncode: return 1
    return 0

if __name__=='__main__':sys.exit(main())
