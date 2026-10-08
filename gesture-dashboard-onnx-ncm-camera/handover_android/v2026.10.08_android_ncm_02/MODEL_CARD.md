# Model and runtime snapshot

- Model: TwelveGestureBalancedMLP, ONNX float32, 77,529 bytes.
- SHA-256: `7434fffefd8d4cd3c6e82d8edacfd7c97b811beddea2bff3d506683f4dc0355e`.
- Input: `landmark_features`, float32 [N,76]; the mobile wrapper uses N=1.
- Command outputs: `probabilities` [N,12] in README order.
- Open-set evidence: `known_gesture_mass` [N,1]. `no_gesture` is an application rejection sentinel.
- Inputs originate from MediaPipe's 21 image landmarks. World landmarks and handedness additionally validate live geometry; they are not appended to the 76-feature tensor.
- CPU inference is the reference execution path. No Core ML conversion or quantized substitute is supplied.
- Current model metadata reports offline accepted precision 0.99500 and macro F1 0.99108. These are offline qualification values, not live iPhone/NCM performance.
- Runtime source includes the latest 6 October 2026 geometry changes. ONNX weights were retained.
- Training/evaluation limitations, class mapping, feature names and recorded conversion parity are in gesture_mlp_onnx_metadata.json.
- New-class evaluation includes participant-held-out rows from HaGRID's training partition. Synthetic rotations do not establish unseen real camera angles, distances or lighting performance.
- The complete runtime includes face rejection, low-light processing, detector recovery and stateful landmark/decision logic. Model-only integration is insufficient to reproduce it.
