# Current runtime porting contract

This delivery contains 13 ONNX probability outputs and 12 enabled commands. The Python snapshot is the behavior reference. The native diagnostic adapters and angle helpers are integration sources; they are not a completed native decision port.

## Image and tensor contract

Use unmirrored, upright NCM/JLIP JPEG pixels. The reference dashboard admits the newest valid frame at 10 FPS, applies the configured 0.92 central crop, and runs one serial inference stream. The standalone Windows viewer uses the full board frame; select one policy for comparison and record it. Do not compare different crops or transform landmarks without transforming the matching dimensions.

Preserve each hand's 21 normalized image XY landmarks, 21 world XYZ landmarks, detector handedness/confidence, timestamp, and prepared image width/height. Image z is not world depth. Select a hand after qualifying both hands; a valid gesture can take ownership from an idle hand regardless of distance. Ownership changes reset gesture history. Apparent hand distance is not measured depth.

The input is float32 `landmark_features[1,76]`. The exact feature names/order are in the metadata and runtime JSON. Read both named outputs: `probabilities[1,13]` and `known_gesture_mass[1,1]`. Keep tensor index 6 (`dorsal`) in place; it is disabled rejection evidence and has no action. `stop_inverted` is index 12. `no_gesture` is a runtime rejection state, not an additional ONNX probability.

Horizontal command calibration is **positive index screen dx → Left; negative dx → Right**. Angles and sectors use source-image pixels. Mirroring a preview must not introduce an additional semantic swap in inference.

## Latest angle limits

Compute `axis = mean(imageXY[5,9,13,17]) - imageXY[0]`, then multiply its x/y by the corresponding image width/height. `XY = atan2(axis.y, axis.x)` in degrees. Screen-right is 0°, up is −90°, screen-left is −180°. This is separate from the world-plane HUD angles.

| Command | Inclusive image XY range | Surface |
|---|---|---|
| Open Palm | −160° to −20° (up ±70°) | Palm visible |
| Stop inverted | −145° to −35° (up ±55°) | Back of hand visible |

Outside the corresponding range, an open hand is `no_gesture`, including when the raw model reports another class. Crossing a limit clears the hold immediately; returning requires a fresh hold. Endpoints are inclusive; the reference uses 0.0001° numerical tolerance. Finger spacing can be together or spread. An angle passing alone does not prove a command: shape, surface, depth, confidence and timing still apply. Preserve the narrowly verified single-index and folded-world-finger exceptions; those are pointing/thumb shapes rather than open hands.

## Port in this order

| Stage | Current Python source | Required behavior |
|---|---|---|
| NCM transport | `backend/ncm_camera.py` | JLIP framing, CRC, JPEG trimming/validation, heartbeat, bounded newest-frame buffer, reconnect |
| Frame preparation and detection | `backend/model_runtime.py`, `backend/hand_detection.py` | Crop/orientation, adaptive low light, primary/recovery streams, hand-size qualification, monotonic frame clocks |
| Landmark stabilization | `backend/landmark_smoothing.py` | Per-track adaptive state, matching image/world data and crop transforms |
| Hand ownership | `backend/hand_selection.py` | Qualify both hands, spatial continuity, handedness confirmation, valid-hand priority, reset on ownership change |
| Features | `backend/geometry.py::landmarks_to_feature` | Exact 76 floats; image-shape direction measurements do not change the model feature order |
| Geometry | `backend/geometry.py::GeometryResolver` | Surface/shape checks, directional recovery, retired Dorsal rejection, Fist world-depth checks, strict Peace pair, both open-hand angle limits |
| Recovery arguments | `backend/hand_selection.py::gate_arguments` | Forward all flags, including Peace, Stop inverted, horizontal/expanded-thumb direction and continuation; negative support disables independent recovery |
| Temporal gate | `backend/runtime.py::TemporalGate` | Known mass, confidence, margin, EMA, stable frames, hold, timestamp checks and immediate angle reset |
| Action latch/session | `backend/model_runtime.py::RuntimeSession` | One action per deliberate held gesture, release/cooldown, tracking-gap behavior and state reset |

The actual files in the delivery settle implementation details. The Python runtime currently disables face and body vetoes. Do not port the former face-contact/clothing/body-position exclusions from older deliveries. The legacy face asset is included only for continuity; it is unused by the current hand-only policy.

Default gates are known mass 0.70, confidence 0.70, probability margin 0.08, three stable frames, hold 0.18 s and cooldown 0.80 s. Read the shipped configuration. Strict geometry recovery has class-specific mass/hold exceptions in `TemporalGate`; do not replace them with a global lower threshold. The reported Peace pair may recover mass ≥0.005 only with its strict pair flag and a 0.50 s low-mass hold. Ordinary Peace geometry still requires the established mass evidence. Stop inverted and verified palmar hands retain their own recovery rules.

## Porting vectors

`parity_vectors.json` records source hashes, exact feature values, adjusted probabilities, all pose/recovery flags, angle policy and expected temporal decisions. `image_shape` is `[height,width,channels]`; retain it when invoking geometry. `temporal_transition_cases` reuse `geometry_cases` through `case_id` and specify timestamps, arguments and expected decisions. These include crossing each limit and returning to a fresh hold.

The Swift/Kotlin parity helpers check features, angle limits and ONNX tensors. They do **not** certify the pending full geometry, ownership, frame-recovery or action port. Compare the remaining fields and transition cases after implementing those stages. The fixtures are synthetic or manually traced landmark regressions, not raw camera-frame replays or new physical-device accuracy measurements.

Older Swift resolver/gate/pipeline files are retained under `reference/legacy_swift/` for engineering history only. They have not been brought into parity with this release and must not be copied into the production action path.
