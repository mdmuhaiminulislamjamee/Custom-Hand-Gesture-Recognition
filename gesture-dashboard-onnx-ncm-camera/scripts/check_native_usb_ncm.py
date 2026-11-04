"""Compile the Android NCM transport and exercise its wire/state-machine tests on the JVM."""
from pathlib import Path
import hashlib
import json
import os
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ['USERPROFILE']) / '.gradle/caches/modules-2/files-2.1'
STAGE = ROOT / '.runtime/native_usb_ncm_check'
JAVA = Path(os.environ['TEMP']) / 'gesture-apk-toolchain/jdk/jdk-17.0.21.0.2+1/bin/java.exe'


def jars(group, name):
    return sorted((CACHE / group / name).rglob('*.jar'))


def checksum(data):
    if len(data) % 2:
        data += b'\0'
    total = sum(struct.unpack('!' + 'H' * (len(data) // 2), data))
    while total >> 16:
        total = (total & 65535) + (total >> 16)
    return (~total) & 65535


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    compiler = []
    for group, name in [('org.jetbrains.kotlin', 'kotlin-compiler-embeddable'),
                        ('org.jetbrains.kotlin', 'kotlin-stdlib'), ('org.jetbrains.kotlin', 'kotlin-script-runtime'),
                        ('org.jetbrains.kotlin', 'kotlin-reflect'), ('org.jetbrains.kotlin', 'kotlin-daemon-embeddable'),
                        ('org.jetbrains.intellij.deps', 'trove4j'), ('org.jetbrains.kotlinx', 'kotlinx-coroutines-core-jvm'),
                        ('org.jetbrains', 'annotations')]:
        compiler += jars(group, name)[-1:]
    stdlib = jars('org.jetbrains.kotlin', 'kotlin-stdlib')
    sdk = Path(os.environ['TEMP']) / 'gesture-apk-toolchain/android-sdk/platforms/android-36/android.jar'
    embedding = jars('io.flutter', 'flutter_embedding_release')[-1:]
    assert embedding, 'Cached real Flutter embedding is required'
    source_folder = ROOT / 'flutter_gesture_tester/android/app/src/main/kotlin/com/gesturelab/gesture_model_tester'
    sources = [source_folder / name for name in ('UsbNcmProtocol.kt', 'UsbNcmStream.kt', 'NcmNetworkBridge.kt')]
    test = ROOT / 'flutter_gesture_tester/android/host_tests/NativeNcmMain.kt'
    command = [str(JAVA), '-Xmx256m', '-XX:ActiveProcessorCount=2', '-XX:+UseSerialGC', '-cp',
               os.pathsep.join(map(str, compiler)), 'org.jetbrains.kotlin.cli.jvm.K2JVMCompiler',
               '-no-stdlib', '-no-reflect', '-jvm-target', '17', '-classpath',
               os.pathsep.join(map(str, [sdk, *embedding, *stdlib])), '-d', str(STAGE / 'classes'),
               *map(str, sources), str(test)]
    done = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (STAGE / 'compile.log').write_text(done.stdout, encoding='utf-8')
    print(done.stdout[-6000:], flush=True)
    if done.returncode:
        return done.returncode
    run = subprocess.run([str(JAVA), '-Xmx128m', '-XX:ActiveProcessorCount=2', '-XX:+UseSerialGC',
                          '-cp', os.pathsep.join(map(str, [STAGE / 'classes', *stdlib])),
                          'com.gesturelab.gesture_model_tester.NativeNcmMainKt'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (STAGE / 'tests.log').write_text(run.stdout, encoding='utf-8')
    print(run.stdout, flush=True)
    if run.returncode:
        return run.returncode
    packets = dict(line.split('=', 1) for line in run.stdout.splitlines() if line.startswith('WIRE_'))
    syn = bytes.fromhex(packets['WIRE_SYN'])
    assert syn[12:14] == b'\x08\x00' and checksum(syn[14:34]) == 0
    ip_length = struct.unpack_from('!H', syn, 16)[0]
    tcp = syn[34:14 + ip_length]
    pseudo = syn[26:34] + bytes([0, 6]) + struct.pack('!H', len(tcp))
    assert checksum(pseudo + tcp) == 0 and tcp[13] == 2
    assert struct.unpack_from('!H', tcp, 2)[0] == 5000
    ntb = bytes.fromhex(packets['WIRE_NTB'])
    assert ntb[:4] == b'NCMH' and ntb[12:16] == b'NCM0'
    assert struct.unpack_from('<H', ntb, 8)[0] == len(ntb)
    start, count = struct.unpack_from('<HH', ntb, 20)
    assert ntb[start:start + count] == bytes(range(60))
    result = dict(passed=True, android_sources_compiled=len(sources),
                  tests=[line for line in run.stdout.splitlines() if line.startswith('PASS:')],
                  independent_python_wire_oracle=True,
                  source_sha256={s.relative_to(ROOT).as_posix(): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
                  physical_usb_tested=False)
    (STAGE / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, test_groups=len(result['tests']), independent_python_wire_oracle=True)), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
