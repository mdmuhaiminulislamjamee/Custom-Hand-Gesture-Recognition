"""Package the final runtime, iOS guide and current Android tester for Boss."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "static_gesture_runtime"
TESTER = ROOT / "flutter_gesture_tester"
DOCS = ROOT / "handover_static/final_docs"
BUNDLE = "Boss_Static_Gesture_Handover_20261111"
APK = "releases/GestureLab-v2.0.8-litert-ncm-arm64-diagnostic-release.apk"
IGNORED = {".dart_tool", "build", ".git", "__pycache__", ".idea", ".gradle", ".kotlin", ".cxx"}


def encoded(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def included(path: Path, base: Path) -> bool:
    relative = path.relative_to(base)
    return (
        path.is_file()
        and not IGNORED.intersection(relative.parts)
        and path.name not in {"SHA256SUMS.json", ".flutter-plugins-dependencies", "local.properties"}
        and path.suffix not in {".log", ".hprof", ".iml", ".pyc", ".jks", ".keystore"}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / f"handover_static/{BUNDLE}_current_model.zip")
    parser.add_argument("--bundle-name", default=BUNDLE)
    parser.add_argument("--release-revision", default="current_model_20261111")
    parser.add_argument("--apk", default=APK, help="APK path relative to flutter_gesture_tester")
    parser.add_argument("--retrained", action="store_true", help="The source classifier was retrained for this release")
    parser.add_argument("--supersedes", action="append", help="Earlier archive superseded by this release; repeat as needed")
    parser.add_argument("--native-output", type=Path, help="Optionally also export the unchanged native filter update")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--allow-pending-checks", action="store_true", help="Only for creating an isolated preflight archive")
    args = parser.parse_args()
    bundle = args.bundle_name
    if not bundle or any(c in bundle for c in "/\\:") or bundle in {".", ".."}:
        raise ValueError("Bundle name must be a single safe directory name")
    apk_path = (TESTER / args.apk).resolve()
    if not apk_path.is_relative_to(TESTER.resolve()) or apk_path.suffix != ".apk":
        raise ValueError("APK must be a .apk file inside flutter_gesture_tester")
    apk_relative = apk_path.relative_to(TESTER).as_posix()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Build output must remain inside the workspace")
    if (output.exists() or output.with_suffix(output.suffix + ".sha256").exists()) and not args.replace:
        raise FileExistsError(f"Use --replace to regenerate {output}")
    native_output = args.native_output.resolve() if args.native_output else None
    if native_output is not None:
        if not native_output.is_relative_to(ROOT) or native_output == output:
            raise ValueError("Native output must be a separate file inside the workspace")
        if native_output.exists() and not args.replace:
            raise FileExistsError(native_output)

    payload: dict[str, bytes] = {}
    for path in sorted(RUNTIME.rglob("*")):
        if included(path, RUNTIME):
            payload["static_gesture_runtime/" + path.relative_to(RUNTIME).as_posix()] = path.read_bytes()
    # Keep the portable verification tool aligned with the repository source.
    payload["static_gesture_runtime/tools/verify_static_runtime.py"] = (ROOT / "scripts/verify_static_runtime.py").read_bytes()

    tester_roots = (TESTER / "lib", TESTER / "test", TESTER / "android/app/src", TESTER / "android/gradle/wrapper", TESTER / "android/host_tests")
    retired = {"lib/services/native_gesture_engine.dart", "lib/widgets/landmark_painter.dart"}
    for base in tester_roots:
        for path in sorted(base.rglob("*")):
            relative = path.relative_to(TESTER).as_posix()
            if included(path, TESTER) and relative not in retired and "GeneratedPluginRegistrant" not in path.name:
                payload["flutter_gesture_tester/" + relative] = path.read_bytes()
    tester_files = (
        "README.md", "pubspec.yaml", "pubspec.lock", "analysis_options.yaml", ".gitignore", ".metadata",
        "android/.gitignore", "android/build.gradle.kts", "android/settings.gradle.kts",
        "android/gradle.properties", "android/gradlew", "android/gradlew.bat", "android/app/build.gradle.kts",
        apk_relative, apk_relative + ".sha256",
    )
    for relative in tester_files:
        path = TESTER / relative
        if path.is_file():
            payload["flutter_gesture_tester/" + relative] = path.read_bytes()
        elif relative != ".metadata":
            raise FileNotFoundError(path)
    for path in sorted(DOCS.iterdir()):
        if path.is_file() and path.suffix in {".md", ".py", ".json"}:
            payload[path.name] = path.read_bytes()
    payload["scripts/check_native_usb_ncm.py"] = (ROOT / "scripts/check_native_usb_ncm.py").read_bytes()
    payload["scripts/README.md"] = (
        "# Native transport host checks\n\n"
        "check_native_usb_ncm.py and android/host_tests/NativeNcmMain.kt are the "
        "Windows/JVM transport regression harness. The script expects the prepared "
        "Gradle cache and JDK/Android SDK paths used by the build host; adapt those "
        "paths on another workstation. Flutter builds/tests do not invoke this script. "
        "Run the root verify_handover.py for portable payload/APK verification.\n"
    ).encode("utf-8")
    for name in ("gesture_mlp_production.onnx", "gesture_mlp_onnx_metadata.json",
                 "gesture_mobile_runtime_config.json", "gesture_artifact_manifest.json"):
        payload["source_model/" + name] = (ROOT / "models" / name).read_bytes()
    payload["source_model/README.md"] = (
        "# Production classifier source snapshot\n\n"
        "These exact ONNX/configuration/metadata files identify the source used for "
        "the FP32 TFLite conversion. The Flutter runtime loads the converted model "
        "under static_gesture_runtime/flutter/assets/models/.\n\n"
        "The source JSON includes desktop controller settings. For Flutter timing, "
        "retain the supplied Dart controller: immediate strong predictions, two of "
        "three ordinary confirmation and no fixed action cooldown. Use the source "
        "JSON's current scalar gates (.75 confidence, .10 margin, .70 known mass) "
        "with the matching feature and geometry code.\n"
    ).encode("utf-8")

    for name in ('GestureFalseHandGuard.kt', 'GestureHandLandmarker.kt', 'GestureFrameAnalyzer.kt'):
        payload['native_false_hand_update/android/kotlin/' + name] = (ROOT/'android_handover_src/kotlin'/name).read_bytes()
    for name in ('GestureFalseHandPolicy.swift', 'ios/GestureFaceGuard.swift', 'ios/GestureHandLandmarker.swift',
                 'tests/false_hand_main.swift', 'tests/verify_false_hands.sh'):
        payload['native_false_hand_update/swift/' + name] = (ROOT/'swift'/name).read_bytes()
    payload['native_false_hand_update/models/blaze_face_short_range.tflite'] = (ROOT/'models/blaze_face_short_range.tflite').read_bytes()
    payload['native_false_hand_update/README.md'] = (ROOT/'handover_docs/FALSE_HAND_UPDATE.md').read_bytes()
    payload['native_false_hand_update/false_hand_policy.json'] = (RUNTIME/'validation/false_hand_policy.json').read_bytes()
    payload['native_false_hand_update/verify_update.py'] = (ROOT/'handover_docs/verify_false_hand_update.py').read_bytes()
    native_payload = {name.removeprefix('native_false_hand_update/'): data
                      for name,data in payload.items() if name.startswith('native_false_hand_update/')}
    native_payload['SHA256SUMS.json'] = encoded({name:digest(data) for name,data in sorted(native_payload.items())})
    payload['native_false_hand_update/SHA256SUMS.json'] = native_payload['SHA256SUMS.json']

    checks_path = DOCS / "HOST_CHECKS.json"
    checks = json.loads(checks_path.read_text(encoding="utf-8")) if checks_path.exists() else {"status": "pending"}
    if checks.get("status") != "passed" and not args.allow_pending_checks:
        raise ValueError("Final packaging requires a passed HOST_CHECKS.json report")
    for relative, expected in checks.get("source_sha256", {}).items():
        if relative not in payload or digest(payload[relative]) != expected:
            raise ValueError(f"Source differs from the host-checked revision: {relative}")
    model_manifest = json.loads(payload["static_gesture_runtime/flutter/assets/models/gesture_static_float32_manifest.json"])
    source_sha = digest(payload["source_model/gesture_mlp_production.onnx"])
    if model_manifest["source_onnx_sha256"] != source_sha:
        raise ValueError("Static classifier was not converted from the current production ONNX")
    if not model_manifest["qualification"]["passed"]:
        raise ValueError("The default FP32 model is not qualified")
    apk_data = payload["flutter_gesture_tester/" + apk_relative]
    apk_sha = digest(apk_data)
    expected = (TESTER / (apk_relative + ".sha256")).read_text(encoding="utf-8").split()[0].lower()
    if apk_sha != expected:
        raise ValueError("APK differs from the previously verified release checksum")
    runtime_sums = {
        name.removeprefix("static_gesture_runtime/"): digest(data)
        for name, data in sorted(payload.items()) if name.startswith("static_gesture_runtime/")
    }
    payload["static_gesture_runtime/SHA256SUMS.json"] = encoded(runtime_sums)
    payload["release_manifest.json"] = encoded({
        "schema_version": 1,
        "release": bundle,
        "release_revision": args.release_revision,
        "assembled_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_target": "Existing Flutter iOS application; owning team builds/signs on Mac",
        "default_classifier": "gesture_static_float32.tflite",
        "default_classifier_sha256": digest(payload["static_gesture_runtime/flutter/assets/models/gesture_static_float32.tflite"]),
        "int8_qualified": json.loads(payload["static_gesture_runtime/flutter/assets/models/gesture_static_int8_manifest.json"])["qualification"]["passed"],
        "source_onnx_sha256": source_sha,
        "runtime_config_sha256": digest(payload["source_model/gesture_mobile_runtime_config.json"]),
        "retrained": args.retrained,
        "dynamic_detector_content_version": "dynamic_gesture_runtime_api_v10_3",
        "toolchain": {"flutter": "3.47.8", "dart": "3.13.7", "android_build_jdk": "17"},
        "android_test_apk": {"path": "flutter_gesture_tester/" + apk_relative, "sha256": apk_sha, "bytes": len(apk_data), "version_name": checks["android_artifact"]["version_name"], "version_code": checks["android_artifact"]["version_code"], "abi": "arm64-v8a", "minimum_api": 24, "mode": "release", "signing": "Android Debug certificate (test distribution)"},
        "latency_policy": {"target_fps": 10, "pending_frames": 1, "strong_prediction": "immediate after scalar and geometry checks", "ordinary_prediction": "two matching observations among the latest three", "fixed_action_cooldown_ms": 0, "physical_latency_measured": False},
        "host_checks": checks,
        "physical_ios_tested": False,
        "physical_ncm_tested": False,
        "supersedes": args.supersedes or ["Boss_Static_Gesture_Handover_20261031_false_hands.zip"],
    })
    payload["SHA256SUMS.json"] = encoded({name: digest(data) for name, data in sorted(payload.items())})

    required = {
        "README.md", "START_HERE.md", "verify_handover.py", "DEVICE_ACCEPTANCE.md",
        "static_gesture_runtime/flutter/lib/static_gesture_runtime_api.dart",
        "flutter_gesture_tester/lib/services/phone_static_gesture_engine.dart",
        "flutter_gesture_tester/android/gradle/wrapper/gradle-wrapper.jar",
    }
    if not required.issubset(payload):
        raise ValueError(f"Missing required payload files: {sorted(required - payload.keys())}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="boss-package-check-", dir=ROOT / ".runtime") as temporary:
        temporary_path = Path(temporary)
        pending = temporary_path / "delivery.zip"
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
            for name, data in sorted(payload.items()):
                info = zipfile.ZipInfo(bundle + "/" + name)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100755 if name.endswith("/gradlew") else 0o100644) << 16
                archive.writestr(info, data, compresslevel=7)
        with zipfile.ZipFile(pending) as archive:
            if archive.testzip() is not None:
                raise ValueError("Archive CRC verification failed")
            archive.extractall(temporary_path / "extracted")
        extracted = temporary_path / "extracted" / bundle
        spec = importlib.util.spec_from_file_location("boss_verify", DOCS / "verify_handover.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        verification = module.verify(extracted)
        pending.replace(output)
    archive_sha = digest(output.read_bytes())
    output.with_suffix(output.suffix + ".sha256").write_text(archive_sha + "  " + output.name + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(output.relative_to(ROOT)), "bytes": output.stat().st_size, "sha256": archive_sha, "files": len(payload), "extracted_verification": verification}, indent=2))
    if native_output is not None:
        native_output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(native_output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as archive:
            for name,data in sorted(native_payload.items()):
                archive.writestr(native_output.stem+'/'+name,data)
        native_sha=digest(native_output.read_bytes())
        native_output.with_suffix('.zip.sha256').write_text(native_sha+'  '+native_output.name+'\n',encoding='utf-8')
        print(json.dumps({'native_update':str(native_output.relative_to(ROOT)),'sha256':native_sha,'files':len(native_payload)},indent=2))


if __name__ == "__main__":
    main()
