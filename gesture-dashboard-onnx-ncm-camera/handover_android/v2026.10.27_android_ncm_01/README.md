# Android app-team handover

Release **v2026.10.27_android_ncm_01**, prepared 2026-10-27 (Asia/Dhaka).

Use this complete package with the existing Android app. The input is the USB-NCM board's JLIP/JPEG stream; inference runs on the phone. This is an integration handover, not a signed app, complete native project or qualified device build.

**Current status:** the delivered desktop reference passes its isolated software tests and model/fixture checks. Native model adapters and diagnostic sources are supplied. The older native decision implementation does not contain all current Python behavior, so it is separated into `reference/legacy_swift/`. Full current native decision/frame parity, native compilation and physical USB-NCM acceptance remain app-team tasks. Follow [RUNTIME_PORTING.md](RUNTIME_PORTING.md) before enabling actions.

## Start here

1. Extract the entire ZIP into a writable directory; retain the companion `.zip.sha256` file.
2. Run `python verify_package.py` with Python 3.11/3.12 before editing delivered files. This checks all payload hashes, the 13-output contract, angles, source lineage and matching desktop/mobile assets using the standard library.
3. With NumPy and ONNX Runtime installed, run `python verify_package.py --onnx` to compare all 26 model vectors. Use the included Python dependency files if needed.
4. Extract `desktop_reference.zip` separately and follow [WINDOWS_RUN.md](WINDOWS_RUN.md) to reproduce the Windows NCM/BAT workflow.
5. Follow [ANDROID_INTEGRATION.md](ANDROID_INTEGRATION.md), then complete [VALIDATION.md](VALIDATION.md).

Model: **StopInvertedDirectionalFeedbackMLP**. SHA-256:

```text
11d6991cc555dfd5f490a647270c4f326dd9dfb14ce4ad1cf503300aea29c331
```

Input is float32 `landmark_features[1,76]`; outputs are `probabilities[1,13]` and `known_gesture_mass[1,1]`. There are **12 enabled commands**. Keep all 13 tensor indices, including the disabled Dorsal output. `no_gesture` means rejection/no action; it is not a fourteenth output. Copying the ONNX file alone does not copy current behavior.

## Folder and file guide

| Path | Purpose | In the native app? |
|---|---|---|
| `README.md` | Start, identity, file guide and action contract | No |
| `ANDROID_INTEGRATION.md` | Platform resources, dependencies, NCM, diagnostics, parity and remaining port work | No; follow it |
| `RUNTIME_PORTING.md` | Current image/tensor/geometry/selection/temporal/action behavior | No; implementation guide |
| `NCM_PROTOCOL.md` | Network route, 24-byte JLIP header, CRC/JPEG, heartbeat and reconnect | No; protocol guide |
| `VALIDATION.md` | Host evidence and physical-device acceptance record | No |
| `CHANGES_SINCE_PREVIOUS.md` | Differences from the October 8 packages | No |
| `MODEL_CARD.md` | Current model provenance, performance scope and limitations | No |
| `MOBILE_RUNTIME_CONTRACT.json` | Machine-readable current taxonomy, actions, features, runtime JSON, angle limits and source identity | Keep as engineering contract |
| `RELEASE_MANIFEST.json`, `SHA256SUMS.json` | Exact snapshot, fixture counts, readiness flags and per-file hashes | No; retain with delivery |
| `verify_package.py` | Read-only checksum/contract and optional ONNX checks | No |
| `verify_reference_fixtures.py` | Re-evaluate vectors against an extracted Python desktop snapshot | No |
| `validation/` | Isolated backend test logs/summary, preflight, fixture and frontend evidence | No |
| `desktop_reference.zip` | Complete runnable current Windows source/model/lockfile/qualification snapshot, identical in both mobile bundles | No |
| `android/models/` | Required ONNX/metadata/runtime/hand-landmarker resources; legacy unused face asset also retained | Required assets, except unused face model |
| `android/tests/parity_vectors.json` | 1134 geometry/feature cases, 8 temporal transition sequences and 26 ONNX cases | Debug/test resources |
| `android/reference/backend/` | Current Python behavior and regression sources | No; porting oracle |
| `android/reference/legacy_swift/` | Older resolver/gate/latch/pipeline, explicitly not qualified for this release | No |
| `android/reference/protocol/` | C++ board receiver reference | No |

`android/kotlin/`: current NCM, network selection, two-hand MediaPipe, 76-feature, 13-output ONNX, angle and parity sources.

## Required current behavior

- **Open Palm:** image XY **−160° to −20°**, inclusive (up ±70°), palm visible.
- **Stop inverted:** image XY **−145° to −35°**, inclusive (up ±55°), back of hand visible, fingers together or spread.
- Outside the command's band is immediately no gesture; return requires a fresh hold. Use the wrist-to-knuckle source-pixel angle, not the HUD world-plane angle.
- Analysis remains unmirrored. Positive index x resolves to Left; negative x resolves to Right. Preserve matching image dimensions and two-hand ownership.
- Preserve strict low-mass Peace recovery and all geometry/temporal/latch checks. A raw score or angle alone never dispatches an action.
- Face/body/clothing positions do not veto an otherwise valid hand gesture in the current policy.

## Fixed probability order and action contract

| Index | Output | Action |
|---:|---|---|
| 0 | `left` | Move Left |
| 1 | `right` | Move Right |
| 2 | `up` | Move Up |
| 3 | `down` | Move Down |
| 4 | `open_palm` | Enable / Disable Object Tracking |
| 5 | `like` | Play / Pause |
| 6 | `dorsal` | Disabled — no action |
| 7 | `ok` | Start / Stop Recording |
| 8 | `fist` | Mute |
| 9 | `thumb_down` | Volume Down |
| 10 | `peace` | Turn 90 Degrees |
| 11 | `rock` | Backup |
| 12 | `stop_inverted` | Return to main position |

## Verification and scope

For Python reference parity after extracting the desktop ZIP:

```text
python verify_reference_fixtures.py android/tests/parity_vectors.json --reference-root /path/to/desktop_reference
```

That command needs the included reference Python dependencies. Native feature/angle/ONNX helpers are documented in the platform guide; the app team must implement and compare the full geometry/temporal/action outputs and all transition sequences as well.

The model was not retrained for packaging. Both platforms receive the same current approved source/model/configuration snapshot. Previous deliveries are unchanged. Local environments, node_modules, credentials, generated caches, feedback photos, private videos and mutable online-adapter state are excluded. Derived/public landmark qualification inputs are included so the desktop tests run; this is not a complete retraining corpus. Native builds, full decision-port completion and physical-device accuracy are not claimed.
