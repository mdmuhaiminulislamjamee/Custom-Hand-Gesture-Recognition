"""Build the release APK in a short Windows path and retain reproducible logs."""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from verified_flutter_reuse import verify as verify_flutter_reuse

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / '.runtime/false_hand_apk_build'
FLUTTER = [r'C:\src\flutter\bin\cache\dart-sdk\bin\dart.exe','--old_gen_heap_size=192',r'C:\src\flutter\bin\cache\flutter_tools.snapshot','--no-version-check']
TOOLCHAIN = Path(tempfile.gettempdir())/'gesture-apk-toolchain'

def main():
    global REPORT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stage',type=Path,help='Reuse an existing short build workspace')
    parser.add_argument('--report-dir',type=Path,default=REPORT)
    parser.add_argument('--debug',action='store_true',help='Build a debug APK without R8')
    parser.add_argument('--diagnostic',action='store_true',help='Release Dart with Android shrinking disabled')
    parser.add_argument('--native-only-from',type=Path,help='Reuse verified identical release Dart inputs from this build report')
    args=parser.parse_args()
    REPORT=args.report_dir.resolve()
    REPORT.mkdir(parents=True,exist_ok=True)
    context=REPORT/'context.json'
    if args.resume or args.stage:
        stage=args.stage or Path(json.loads(context.read_text())['stage'])
        # Refresh the source while retaining the expensive native build cache.
        for name in ('static_gesture_runtime/flutter/lib','static_gesture_runtime/flutter/third_party/hand_detection/lib','flutter_gesture_tester/lib'):
            shutil.copytree(ROOT/name,stage/name,dirs_exist_ok=True)
        shutil.copytree(ROOT/'flutter_gesture_tester/android',stage/'flutter_gesture_tester/android',
            dirs_exist_ok=True,ignore=shutil.ignore_patterns('.gradle','build','.cxx','local.properties'))
        for name in ('pubspec.yaml','pubspec.lock'):
            shutil.copyfile(ROOT/'flutter_gesture_tester'/name,stage/'flutter_gesture_tester'/name)
    else:
        stage=Path(tempfile.mkdtemp(prefix='gesture-filter-apk-'))
        ignored=shutil.ignore_patterns('build','.dart_tool','.gradle','.git','__pycache__','.cxx','releases','local.properties')
        for name in ('static_gesture_runtime','flutter_gesture_tester'):
            shutil.copytree(ROOT/name,stage/name,ignore=ignored)
        context.write_text(json.dumps(dict(stage=str(stage)),indent=2),encoding='utf-8')
    print('Android build workspace: '+str(stage),flush=True)
    app=stage/'flutter_gesture_tester'
    version=re.search(r'^version:\s*(\S+)\+(\d+)',(app/'pubspec.yaml').read_text(),re.M)
    if version is None: raise ValueError('Missing tester version')
    version_name,version_code=version.group(1),int(version.group(2))
    context.write_text(json.dumps(dict(stage=str(stage)),indent=2),encoding='utf-8')
    sdk=TOOLCHAIN/'android-sdk'
    jdk=TOOLCHAIN/'jdk/jdk-17.0.21.0.2+1'
    environment=dict(os.environ,JAVA_HOME=str(jdk),ANDROID_HOME=str(sdk))
    mode='debug' if args.debug else 'release'
    properties=f'flutter.sdk=C:/src/flutter\nsdk.dir={sdk.as_posix()}\nflutter.buildMode={mode}\nflutter.versionName={version_name}\nflutter.versionCode={version_code}\n'
    (app/'android/local.properties').write_text(properties,encoding='utf-8')
    def run(name,command,cwd=app):
        log=REPORT/(name+'.log')
        print('Running '+name,flush=True)
        with log.open('w',encoding='utf-8') as stream:
            done=subprocess.run(command,cwd=cwd,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        tail=log.read_text(encoding='utf-8',errors='replace').splitlines()[-20:]
        print(json.dumps(dict(step=name,exit_code=done.returncode,tail=[line[:400] for line in tail])),flush=True)
        if done.returncode: raise RuntimeError(name+' failed; see '+str(log))
    reused = verify_flutter_reuse(args.native_only_from) if args.native_only_from else None
    if reused:
        assert not args.debug, 'Verified release bundle reuse requires release mode'
        bundle = app/'build/app/intermediates/flutter/release'
        with zipfile.ZipFile(reused['apk']) as archive:
            for name in archive.namelist():
                if name.startswith('assets/flutter_assets/') and not name.endswith('/'):
                    relative = name.removeprefix('assets/')
                    assert (bundle/relative).read_bytes() == archive.read(name), relative
            # AGP strips the original app.so, so reuse the verified APK's actual
            # compiled library rather than assuming the cache has identical ELF metadata.
            (bundle/'arm64-v8a/app.so').write_bytes(archive.read('lib/arm64-v8a/libapp.so'))
        print('Verified identical Dart/test/dependency inputs; reusing '+str(reused['previous_version'])+' release bundle',flush=True)
    else:
        run('dependencies',FLUTTER+['pub','get','--offline','--enforce-lockfile'])
    defines=','.join(base64.b64encode(v.encode()).decode() for v in (f'FLUTTER_BUILD_NAME={version_name}',f'FLUTTER_BUILD_NUMBER={version_code}'))
    assemble_target='debug_android_application' if args.debug else 'android_aot_bundle_release_android-arm64'
    if not reused: run('flutter_assemble',FLUTTER+['assemble',f'--output=build/app/intermediates/flutter/{mode}',
        '-dTargetFile=lib/main.dart','-dTargetPlatform=android',f'-dBuildMode={mode}',
        '-dAndroidArchs=android-arm64','-dMinSdkVersion=24','-dTreeShakeIcons=false',
        '--DartDefines='+defines,assemble_target])
    run('gradle', [str(jdk/'bin/java.exe'),'-Xms16m','-Xmx64m','-XX:MaxMetaspaceSize=96m',
        '-XX:ReservedCodeCacheSize=16m','-XX:ActiveProcessorCount=1','-XX:+UseSerialGC',
        '-Dorg.gradle.appname=gradlew','-classpath',
        str(app/'android/gradle/wrapper/gradle-wrapper.jar'),'org.gradle.wrapper.GradleWrapperMain',
        '--offline','--no-daemon',
        '-Dorg.gradle.vfs.watch=false',
        ('-Dorg.gradle.jvmargs=-Xms32m -Xmx384m -XX:MaxMetaspaceSize=384m -XX:ReservedCodeCacheSize=32m -Xss512k -XX:TieredStopAtLevel=1 -XX:ActiveProcessorCount=1 -XX:+UseSerialGC' if args.diagnostic else
         '-Dorg.gradle.jvmargs=-Xmx768m -XX:MaxMetaspaceSize=512m -XX:ReservedCodeCacheSize=64m -XX:TieredStopAtLevel=1 -XX:ActiveProcessorCount=2 -XX:+UseSerialGC'),
        '-Ptarget-platform=android-arm64',
        *(['-PgestureDiagnosticBuild=true'] if args.diagnostic else []),
        ('assembleDebug' if args.debug else 'assembleRelease'),'-x',f':app:compileFlutterBuild{mode.title()}'],app/'android')
    apk=app/f'build/app/outputs/flutter-apk/app-{mode}.apk'
    if not apk.is_file(): apk=app/f'build/app/outputs/apk/{mode}/app-{mode}.apk'
    run('apk_signature',[str(jdk/'bin/java.exe'),'-jar',str(sdk/'build-tools/36.0.0/lib/apksigner.jar'),
        'verify','--verbose','--print-certs',str(apk)])
    run('apk_metadata',[str(sdk/'build-tools/36.0.0/aapt.exe'),'dump','badging',str(apk)])
    suffix='diagnostic-release' if args.diagnostic else ('debug' if args.debug else 'release')
    destination=ROOT/f'flutter_gesture_tester/releases/GestureLab-v{version_name}-litert-ncm-arm64-{suffix}.apk'
    destination.parent.mkdir(exist_ok=True)
    digest=hashlib.sha256(apk.read_bytes()).hexdigest()
    shutil.copyfile(apk,destination)
    destination.with_suffix('.apk.sha256').write_text(digest+'  '+destination.name+'\n',encoding='utf-8')
    result=dict(passed=True,apk=str(destination.relative_to(ROOT)),sha256=digest,bytes=destination.stat().st_size,
        version_name=version_name,version_code=version_code,mode=mode,abi='arm64-v8a',minimum_api=24,
        signing='Android Debug certificate (test distribution)',android_shrinking=not(args.diagnostic or args.debug),android_lint_run=not args.debug,physical_device_tested=False,stage=str(stage),flutter_reuse=reused)
    (REPORT/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
