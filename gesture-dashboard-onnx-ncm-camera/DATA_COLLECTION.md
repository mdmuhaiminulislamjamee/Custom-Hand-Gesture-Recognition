# Collecting images for the next model

Open **05 Data Collection** at http://127.0.0.1:3200.

1. Connect the board camera. Create an ID such as `person-001`; reuse that ID for the same person on later visits.
2. Follow **Collect now**. Directions come first: index finger only, with eight viewing variations including casual placement. Palm and other commands follow, then no-gesture examples (face/ear, other fingers, background and relaxed hands).
3. Choose one of four balanced groups: right hand with fingers toward the camera, right hand with fingers away, left hand with fingers toward, or left hand with fingers away. Then set lighting and distance. Press **Capture image** or press **Space**; a three-second timer gives you time to pose. The live camera and detected hand landmarks are shown side by side.
4. Review the frozen photo and its matching landmark view, then select **Confirm label & save image** or press **Space** again. Choose Retake if needed. The prompt is the human label; a wrong model prediction does not prevent saving.
5. Collect three images in each hand/angle group (12 per prompt). The collector then advances to the next unfinished prompt. You can skip or jump to any prompt. Counts survive server restarts.
6. Saved images for the current prompt appear as thumbnails. Select one or more and use **Delete selected**, or use **Delete last image** in Recent saves. Both ask for confirmation and remove the matching JPEG and JSON label together.

There are 54 prompts and 648 photos for one complete pass per person. Hand-based prompts use the 3 + 3 + 3 + 3 balance above; face/ear and empty-background prompts simply require 12 varied images. This is a starting target, not a guarantee of sufficient training coverage. Vary people, both hands, sessions, backgrounds and conditions. Move naturally between photos; near-identical frames do not replace genuine variation. Keep directions clear while changing the wrist/camera angle. Here “one” means the index-up pose, not an additional command.

The unmirrored board camera retains its calibration: image-right is **Left**, image-left is **Right**, image-up is **Up**. Follow the written instructions; do not reverse labels to match an incorrect prediction.

## Storage and privacy

The default is `D:\Data-Collection`. All gesture folders are created when you create a participant. Photos are organized like:

```text
D:\Data-Collection\
  dataset.json
  participants\
    person-001\
      participant.json
      up\casual\session-20260909...\
        <image-id>.jpg
        <image-id>.json
      open_palm\front\...
      no_gesture\face_ear\...
    person-002\...
```

JPEGs are original board frames without skeletons or overlays. Each matching JSON contains the confirmed label, participant/session/view, hand, finger angle, lighting, distance, timestamp, checksum and frame ID. Available landmarks/features and model output are diagnostics, never automatic labels. The frozen photo and diagnostics come from the same processed frame.

Collect with participants' permission. Use anonymous IDs; images can still contain faces. This collector does not upload images. Back up the entire folder including JSON files. To change storage, set `GESTURE_COLLECTION_DIR` before starting the backend. Missing or unwritable storage produces an error, not a silent save elsewhere.

## Future retraining

Give the entire `Data-Collection` folder for retraining. Saving images does **not** train, change ONNX weights, or invoke Feedback/Safe Learn.

Audit without training:

```powershell
.venv\Scripts\python.exe -m scripts.train_multiview_model --dataset D:\Data-Collection --audit-only
```

The reader supports all eight commands and `no_gesture`, verifies checksums, deduplicates images, and keeps every participant wholly in one train/validation/test split. Audit output lists coverage and cases needing landmark re-extraction or label review. Raw images without usable landmarks are retained: the current 76-feature classifier cannot learn directly from pixels. Such cases require detector evaluation/re-extraction during later training. Backgrounds without a detected hand are valuable detector-negative tests.

Candidate training writes to `artifacts/multiview`, never production. See [MULTIVIEW_TRAINING.md](MULTIVIEW_TRAINING.md) for qualification. More participants may be needed to fill independent splits; never put one person's images in multiple splits to satisfy coverage.

## Recognition changes

- Face-box filtering rejects small hand candidates over detected faces/ears; this is not face identification. Large hands held in front of faces remain eligible.
- Directions require the index itself to be extended and prominent, including a model-supported side-view path.
- Recovered tracks require three consistent observations. Pose rejection no longer resets smoothing every frame. Brief tracking gaps do not immediately re-arm a held command.
- Strong model-agreed index/palm geometry can handle moderate vocabulary uncertainty after a longer hold. Very low vocabulary mass and reviewed-negative vetoes remain rejected.
- Rejected candidates are not drawn as accepted skeletons; raw diagnostics remain available. Class scores are marked as candidates when no command is accepted.

The added model is Google's MediaPipe BlazeFace short-range detector, checksum-pinned by `scripts/download_face_detector.py`. Setup installs it and the release manifest verifies it.

These are runtime improvements, not a retrained model or a promise to recognize every customer pose. Low-resolution/occluded fingers can still be ambiguous. Collect remaining failures and test on unseen participants after retraining.
