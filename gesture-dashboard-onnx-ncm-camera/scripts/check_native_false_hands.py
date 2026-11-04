"""Compile Android filter integration against cached real MediaPipe/Android APIs.

Run the pure policy against all exported Python cases without an Android device.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'.runtime/native_false_hand_check'
CACHE=Path(os.environ['USERPROFILE'])/'.gradle/caches/modules-2/files-2.1'
JAVA=Path(os.environ['TEMP'])/'gesture-apk-toolchain/jdk/jdk-17.0.21.0.2+1/bin/java.exe'

def jars(group,name):return sorted((CACHE/group/name).rglob('*.jar'))
def main():
    STAGE.mkdir(parents=True,exist_ok=True)
    libraries=[]
    for name in ('tasks-core','tasks-vision'):
        aar=next((CACHE/'com.google.mediapipe'/name/'1.1.1').rglob('*.aar'))
        output=STAGE/(name+'.jar')
        with zipfile.ZipFile(aar) as archive:output.write_bytes(archive.read('classes.jar'))
        libraries.append(output)
    onnx=sorted((CACHE/'com.microsoft.onnxruntime/onnxruntime-android').rglob('*.aar'))[-1]
    onnx_jar=STAGE/'onnxruntime.jar'
    with zipfile.ZipFile(onnx) as archive:onnx_jar.write_bytes(archive.read('classes.jar'))
    libraries.append(onnx_jar)
    for group,name in [('com.google.protobuf','protobuf-javalite'),('org.jetbrains','annotations'),('com.google.guava','listenablefuture')]:
        libraries+=jars(group,name)
    stdlib=jars('org.jetbrains.kotlin','kotlin-stdlib')
    compiler=[]
    for group,name in [('org.jetbrains.kotlin','kotlin-compiler-embeddable'),('org.jetbrains.kotlin','kotlin-stdlib'),
                       ('org.jetbrains.kotlin','kotlin-script-runtime'),('org.jetbrains.kotlin','kotlin-reflect'),
                       ('org.jetbrains.kotlin','kotlin-daemon-embeddable'),('org.jetbrains.intellij.deps','trove4j'),
                       ('org.jetbrains.kotlinx','kotlinx-coroutines-core-jvm'),('org.jetbrains','annotations')]:
        matches=jars(group,name)
        compiler.extend(matches[-1:])
    android=Path(os.environ['TEMP'])/'gesture-apk-toolchain/android-sdk/platforms/android-36/android.jar'
    if not android.exists():android=sorted((android.parent.parent).glob('*/android.jar'))[-1]
    cases=json.loads((ROOT/'static_gesture_runtime/validation/false_hand_policy.json').read_text())['cases']
    def f(x):return repr(float(x))+'f'
    def points(v):return 'listOf('+','.join('GesturePoint('+','.join(map(f,p))+')' for p in v)+')'
    def world(v):return 'null' if v is None else 'listOf('+','.join('GestureWorldPoint('+','.join(map(f,p))+')' for p in v)+')'
    lines=['package com.example.gesturehandover','fun main() {','var count=0']
    for c in cases:
        faces='listOf('+','.join('GestureFaceBox('+','.join(map(f,p))+')' for p in c['faces'])+')'
        expected='null' if c['reason'] is None else json.dumps(c['reason'])
        call=f'GestureFalseHandPolicy.rejection({points(c["normalized"])},{world(c["world"])},{c["width"]},{c["height"]},{faces})'
        lines.append('check('+call+' == '+expected+') { '+json.dumps(c['id'])+' }; count++')
    lines+=['println("PASS: $count Python/Kotlin candidate-policy cases")','}']
    main_file=STAGE/'FalseHandMain.kt'
    main_file.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    sources=sorted((ROOT/'android_handover_src/kotlin').glob('*.kt'))
    cp=os.pathsep.join(map(str,[android,*libraries,*stdlib]))
    command=[str(JAVA),'-Xmx512m','-cp',os.pathsep.join(map(str,compiler)),
        'org.jetbrains.kotlin.cli.jvm.K2JVMCompiler','-no-stdlib','-no-reflect','-jvm-target','17',
        '-classpath',cp,'-d',str(STAGE/'classes'),str(main_file),*map(str,sources)]
    done=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (STAGE/'compile.log').write_text(done.stdout,encoding='utf-8')
    print(done.stdout[-6000:],flush=True)
    if done.returncode:return done.returncode
    test=subprocess.run([str(JAVA),'-cp',os.pathsep.join(map(str,[STAGE/'classes',*stdlib])),
        'com.example.gesturehandover.FalseHandMainKt'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    print(test.stdout,flush=True)
    result=dict(android_filter_compile_passed=True,native_source_files_compiled=len(sources),mediapipe_version='1.1.1',onnxruntime_artifact=onnx.name,policy_cases=len(cases),
        policy_test_passed=test.returncode==0,source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        android_device_tested=False,ios_compile_tested=False)
    (STAGE/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return test.returncode

if __name__=='__main__':sys.exit(main())
