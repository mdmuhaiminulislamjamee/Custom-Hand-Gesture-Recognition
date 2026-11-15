# Ownership and preview validation - 14 November 2026

`validation/ownership_preview_release.json` compares this release with the
previous `two_hand_stability` archive. It verifies the deployed model assets,
features, geometry, detector filters/settings, dependencies, frame scheduling
and single-hand confirmation are preserved. Only ownership, selected-hand
display, paired preview rendering, their regression tests and tester release
identifiers change among the protected mobile inputs.

Runtime regressions cover first-owner retention, changed detection order,
closer-hand confirmation using world projection and palm fallback, short
distance bursts, invalid-owner handoff, loss grace, returning ownership,
duplicate prevention, and all twelve single-hand commands on either hand with
strong and ordinary confirmation. Preview tests force uncached asynchronous
decodes and check old-pair retention, new-pair replacement and genuine empty
frames. The tester checks that a higher-confidence peer cannot take over the
displayed gesture. Fresh suite counts, native checks, model fixture results,
and exact APK/source identities are in delivery `HOST_CHECKS.json`.

These are host tests, not physical phone/NCM measurements. Camera flicker,
switching behavior, sustained latency and iOS builds still require device
acceptance. Ear/ghost and incidental-movement tuning remain deferred.

## Preserved model training evidence - 11 November 2026

Model evidence is `validation/feedback_refresh_release.json`. The historical
Right and direction stability JSON files describe previous models and policies.
The deployed ONNX is
`d8a37784b603fc9917ef5635f40c2b20b1032c7898a1fe98b3babc9b33c00044`.

All 990 previously unused valid reviewed landmark records participated in
training: 480 Down, 240 Thumb Down, 150 Stop Inverted, 90 Like and 30 rejection
examples. Their original labels were preserved. One old exact duplicate ID was
deduplicated for fitting, and 11 old invalid landmark records were excluded.
The source CSV has 4,838 rows; 4,826 unique records have usable landmarks.
Forty-two new JPEGs are missing; their saved landmarks remain valid training
inputs. Feedback and replay images are training regressions, not independent
subjects or new-camera accuracy evidence.

The selected model retains the encoder, normalization, unaffected output
columns and exact graph topology. Larger head and hidden-layer fits were
rejected when they changed working examples or reduced accepted precision.
The conservative accepted update improves accepted correctness on the new
training records from 628 to 630. Old reviewed regression metrics are retained.
Every per-class scalar and geometry-resolved metric on the fixed 9,042-row test
benchmark is preserved. The 9,031-row validation benchmark was used for fitting
strength selection. Fixed test benchmarks were used for release acceptance;
no validation/test row supplied training gradients. Repeated benchmark use does
not support a new blind generalization claim.

The sampled saved-image replay uses 48 deterministic photos, held for 12 frames
each. Final correct labels stay 38/48; expected actions improve from 37 to 38.
It covers existing Left/Right and new positive Down/Thumb Down/Stop Inverted/Like
samples. It does not cover all new photos or deleted negative photos. Exact
per-class counts and limitations are in the current evidence JSON.

The detector, geometry, scalar gates, temporal policy, latest-frame scheduling
and ghost/false-hand rejection are unchanged. No inference pass was added.
ONNX is still 77,797 bytes and FP32 TFLite is still 78,960 bytes. Host inference
comparisons are in `validation/onnx_latency.json` and `validation/latency_comparison.json`.
These measurements exclude camera-to-command and physical-device latency.

| Check | Evidence |
|---|---|
| FP32 conversion on 9,031 validation and 9,042 test rows | `validation/conversion_report.json`; matching top labels and scalar gates |
| Canonical Python/Dart feature and geometry cases | 1,142 cases plus 5,710 known-mass qualifications in `validation/dart_geometry_parity.json` |
| Additional held-out cache geometry | 400 cases in `validation/dart_cache_geometry_parity.json` |
| Native LiteRT fixture inference and Flutter tests | Fresh runtime/tester analysis and tests in delivery `HOST_CHECKS.json` |
| Native Android filter | 42 shared Python/Kotlin policy cases, fresh compile |
| Native USB-NCM | 17 transport groups and independent wire oracle |
| Backend tests | Current results compared with six existing fist/ownership failures; no fully passing suite claimed |

FP32 is the default. INT8 may pass conversion gates, but its manifest blocks
loading until precision-specific geometry/controller and device acceptance.
The Android diagnostic APK is checked against the exact current model assets
and source. iOS compilation, signing, physical camera accuracy and physical
phone/NCM latency remain in `DEVICE_ACCEPTANCE.md`. Ghost tuning is deferred.
