# Collecting images for the next model

Open **05 Data Collection** at http://127.0.0.1:3200.

1. Choose **Webcam** or **NCM camera**. Authorize the browser webcam when prompted, or connect the board for NCM. Create an ID such as `person-001`; reuse that ID for the same person on later visits.
2. Follow **Collect now**, or click **Go to no-gesture collection**. Use **Jump to a section** for resting Dorsal, resting Fist, incidental pointing, near-command or unrelated hand shapes, occlusion, two hands, or no-hand scenes.
3. For single-hand prompts, choose one of four balanced groups: right hand with fingers toward the camera, right hand with fingers away, left hand with fingers toward, or left hand with fingers away. Two-hand and no-hand prompts set the hand condition automatically. Set lighting and distance. Press **Capture image** or **Space**; a three-second timer gives you time to pose. The live camera and detected hand landmarks are shown side by side.
4. Review the frozen photo and its matching landmark view, then select **Confirm label & save image** or press **Space** again. Choose Retake if needed. The prompt is the human label; a wrong model prediction does not prevent saving.
5. Collect three images in each hand/angle group (12 per prompt). The collector then advances to the next unfinished prompt. You can skip or jump to any prompt. Counts survive server restarts.
6. Saved images for the current prompt appear as thumbnails. Select one or more and use **Delete selected**, or use **Delete last image** in Recent saves. Both ask for confirmation and remove the matching JPEG and JSON label together.

There are 133 prompts: 72 commands and 61 `no_gesture` cases. A complete optional pass is 1,596 photos. Single-hand prompts use the 3 + 3 + 3 + 3 balance above; two-hand and no-hand prompts need 12 varied images each. Skip sections you cannot complete now and revisit them later. Vary people, both hands, sessions, backgrounds and conditions. Move naturally between photos; near-identical frames do not replace genuine variation.

Collect Fist and Thumbs Down with both hands and genuine camera/wrist-angle
variation. Positive Open Palm examples should point up, left, right, and at
oblique side angles, with the palm surface visible. Downward and diagonal-down
palms belong in reviewed `no_gesture` hard-negative coverage so they cannot
enable object tracking. Collect upward and sideways Fists with both hands.
Capture downward fists, partly bent open hands, and the retired thumb-and-little
sign as `no_gesture` negatives.

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

JPEGs are original selected-camera frames without skeletons or overlays. Each matching JSON contains the confirmed label, source, participant/session/view, prompt group, training use, hand, finger angle, lighting, distance, timestamp, checksum and frame ID. Available landmarks/features and model output are diagnostics, never automatic labels. The frozen photo and diagnostics come from the same processed frame.

Collect with participants' permission. Use anonymous IDs; images can still contain faces. This collector does not upload images. Back up the entire folder including JSON files. To change storage, set `GESTURE_COLLECTION_DIR` before starting the backend. Missing or unwritable storage produces an error, not a silent save elsewhere.

## Future retraining

Give the entire `Data-Collection` folder for retraining. Saving images does **not** train, change ONNX weights, or invoke Feedback/Safe Learn.

Audit without training:

```powershell
.venv\Scripts\python.exe -m scripts.train_multiview_model --dataset D:\Data-Collection --audit-only
```

The reader supports all twelve commands and `no_gesture`, verifies checksums, deduplicates images, and keeps every participant wholly in one train/validation/test split. Audit output separates cases needing landmark review from examples excluded by design. Resting poses that could have the same hand shape as an intentional command, two-hand relationships, occlusion and no-hand scenes remain reviewed raw images for context or detector work; they are not fitted as contradictory labels into the 76-feature single-hand classifier. Only distinct hand-shape cases with valid landmarks enter that classifier's training set.

Candidate training writes to `artifacts/multiview`, never production. See [MULTIVIEW_TRAINING.md](MULTIVIEW_TRAINING.md) for qualification. More participants may be needed to fill independent splits; never put one person's images in multiple splits to satisfy coverage.

## Recognition changes

- Face-box filtering rejects small hand candidates over detected faces/ears; this is not face identification. Large hands held in front of faces remain eligible.
- Directions require the index itself to be extended and prominent, including a model-supported side-view path.
- Recovered tracks require three consistent observations. Pose rejection no longer resets smoothing every frame. Brief tracking gaps do not immediately re-arm a held command.
- Strong model-agreed index/palm geometry can handle moderate vocabulary uncertainty after a longer hold. Very low vocabulary mass and reviewed-negative vetoes remain rejected.
- Rejected candidates are not drawn as accepted skeletons; raw diagnostics remain available. Class scores are marked as candidates when no command is accepted.

The added model is Google's MediaPipe BlazeFace short-range detector, checksum-pinned by `scripts/download_face_detector.py`. Setup installs it and the release manifest verifies it.

These are runtime improvements, not a retrained model or a promise to recognize every customer pose. Low-resolution/occluded fingers can still be ambiguous. Collect remaining failures and test on unseen participants after retraining.
