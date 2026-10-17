# Collecting images for the next model

Open **05 Data Collection** at http://127.0.0.1:3200.

1. Choose **Webcam** or **NCM camera**. Authorize the browser webcam when prompted, or connect the board for NCM. Create an ID such as `person-001`; reuse that ID for the same person on later visits.
2. Follow **Collect now**, or click **Go to no-gesture collection**. Use **Jump to a section** for resting Dorsal, resting Fist, incidental pointing, near-command or unrelated hand shapes, occlusion, two hands, or no-hand scenes.
3. For single-hand prompts, choose the indicated hand and camera angle. Collect Fist across the upper arc from horizontal left through up to horizontal right, with both hands and a 30-degree pitch toward the camera. Collect lower angles as `no_gesture`. Collect the opposite-facing closed hand under the reverse-facing `no_gesture` prompt. Each hand needs six varied images for these one-orientation prompts; other single-hand prompts need three per hand and orientation. Two-hand and no-hand prompts set the hand condition automatically. Set lighting and distance. Press **Capture image** or **Space**; a three-second timer gives you time to pose. The live camera and detected hand landmarks are shown side by side.
4. Review the frozen photo and its matching landmark view, then select **Confirm label & save image** or press **Space** again. Choose Retake if needed. The prompt is the human label; a wrong model prediction does not prevent saving.
5. Collect three images in each hand/angle group (12 per prompt). The collector then advances to the next unfinished prompt. You can skip or jump to any prompt. Counts survive server restarts.
6. Saved images for the current prompt appear as thumbnails. Select one or more and use **Delete selected**, or use **Delete last image** in Recent saves. Both ask for confirmation and remove the matching JPEG and JSON label together.

Use **Upload photo** to import an existing frame without connecting a camera.
Choose its original camera source, participant, session, and prompt first, then
review and confirm its label exactly as for a captured image. Imported images
are converted to a JPEG without overlays before analysis; the saved JPEG and
diagnostics describe that same frame.

There are 135 prompts: 72 commands and 63 `no_gesture` cases. A complete optional pass is 1,620 photos. Skip sections you cannot complete now and revisit them later. Vary people, both hands, sessions, backgrounds and conditions. Move naturally between photos; near-identical frames do not replace genuine variation. Straight touching index/middle fingers can be Peace and are retained as context review, excluded from negative hand-classifier fitting. Positive Fist images marked fingers-away are excluded pending orientation review.

Collect Fist with both hands and genuine camera/wrist-angle variation while
the closed hand stays at or above horizontal. Up-left and up-right are valid;
left-down, right-down and straight-down fists are `no_gesture`, including when
pitched 30 degrees toward the camera. Horizontal left/right are valid boundaries.
Review older downward-positive Fist captures before including them in training. Positive Open Palm examples should point up, left, right, and at
oblique side angles, with the palm surface visible. Downward and diagonal-down
palms belong in reviewed `no_gesture` hard-negative coverage so they cannot
enable object tracking. Collect upright and rotated Fists with both hands.
Capture opposite-facing closed fists, partly bent open hands, and the retired
thumb-and-little sign as `no_gesture` negatives. Opposite-facing fist examples
remain context-only because static 2-D landmarks can be ambiguous.

## Video Lab

Open **06 Video Lab** to record with the selected webcam/NCM camera or upload an
MP4, WebM, MOV, or AVI. Uploads are limited to **five minutes and 256 MB**. Use the
actual deployment camera and retain the full frame with the whole hand visible.
The arm and body may be cropped for every active command. Dorsal is collected only as `no_gesture`. Give the same person the same anonymous ID
across recordings and sessions.

- Valid command clips: about 5–10 seconds, including neutral, presentation,
  hold, and release. Record all twelve commands with both hands.
- Difficult negatives: about 10–20 seconds per scenario, including relaxed
  standing, hands on thighs/desks, folded arms, opposite-facing fists, partial
  shapes, object use, and cropped body landmarks.
- Everyday activity: about 2–5 minutes of ordinary movement. Include natural
  pauses, reaching, walking/turning, typing, multiple hands, and no hands.

Review the original video, add non-overlapping start/end spans, and confirm the
labels. Each positive span represents **one intended action** for evaluation.
Choose a negative subtype for `no_gesture`. Mark ambiguous transitions
`uncertain`; unlabeled/uncertain time is replayed but excluded from scored
exposure. Label based on the recording, never the model's prediction. Record
matched valid and relaxed cases with the same person, setting, and lighting.

**Extract reviewed frames** saves portable JPEG/JSON rows into the existing
participant collection. A one-second sampling interval is a starting point;
review the resulting photos and remove near duplicates. Detection failures are
still saved as reviewed raw images. Context/detector cases remain excluded from
static fitting. Originals and annotation versions stay under `videos/clips/`.
When annotations change, old extracted frames whose labels no longer match
are excluded from fitting; extract the revised spans again.

**Evaluate clip** replays the full hand detection, geometry,
temporal hold, cooldown, release, and action pipeline at recorded timestamps.
It uses an isolated detector and does not send actions to the live dashboard.
Select production or a completed training candidate. Reports include false
actions per hour of labeled negative time, per-command event recall, missed
commands, wrong commands, duplicates, latency, coverage, and an event timeline.
Replay freezes the base ONNX classifier and disables the mutable live-feedback
adapter so a comparison cannot change while the job runs. Reports record the
model identity and adapter policy.
Short exposure is a pilot measurement, not evidence of a zero-error system.

## Model Training and results

Open **07 Model Training**:

1. Run **Audit dataset** after collecting photos or extracting reviewed video
   spans. The audit shows usable features, exclusions, per-class coverage, and
   deterministic participant-isolated training/validation/test assignment.
2. Collect usable training and validation participants before **Train candidate**.
   The worker trains all twelve commands plus internal rejection, retaining
   public data and capping captured-data repeats. Context-only examples do not
   enter the hand classifier. Epoch selection uses validation data.
3. Inspect validation metrics, ONNX parity, and job logs. Incomplete class/person
   coverage is reported as exploratory, even when training completes.
4. Run **Evaluate candidate** to score the frozen public and camera test
   snapshots. Use new independent clips in Video Lab for full-runtime evidence.
5. Download the candidate and JSON report for review. Candidate training and
   evaluation do not promote a model into the running production pipeline.

Jobs run in an isolated Python process with bounded numerical threads, support
cancellation, and retain reports under `artifacts/model_lab/<job-id>/`. Training
dependencies come from `requirements-notebook.txt`. An interrupted job is
reported as failed after restart. The older multi-view script targets a legacy
taxonomy; use Model Training for the twelve-command workflow.

## Hand-only recognition

Dorsal is a hard negative at every angle, distance, body position and background. Use the Dorsal hard-negative prompts for both hands, including against clothing. No positive Dorsal prompts remain. Reviewed legacy Dorsal images keep their original files and provenance; the dataset reader maps their training labels to `no_gesture`. Model Lab also maps public Dorsal rows to rejection before balancing or evaluation.

A relaxed hand that exactly matches a command may qualify because body posture no longer distinguishes intent. Review such frames carefully; do not train identical single-hand features with contradictory positive/negative labels. The existing context-only collection IDs are retained for older data. Dorsal with the body cropped belongs in the no-gesture Dorsal collection.

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
.venv\Scripts\python.exe -m scripts.run_model_lab_job --kind audit --dataset D:\Data-Collection --output artifacts/model_lab/manual-audit
```

The reader supports all twelve commands and `no_gesture`, verifies checksums, deduplicates images, and keeps every participant wholly in one train/validation/test split. Audit output separates cases needing landmark review from examples excluded by design. Resting poses that could have the same hand shape as an intentional command, two-hand relationships, occlusion and no-hand scenes remain reviewed raw images for context or detector work; they are not fitted as contradictory labels into the 76-feature single-hand classifier. Only distinct hand-shape cases with valid landmarks enter that classifier's training set.

Candidate training from Model Training writes to `artifacts/model_lab`, never production. More participants may be needed to fill independent splits; never put one person's images in multiple splits to satisfy coverage. [MULTIVIEW_TRAINING.md](MULTIVIEW_TRAINING.md) describes the legacy directional workflow.

## Recognition changes

- Live inference runs only hand models. Body position, face overlap and clothing do not veto commands.
- Directions require the index itself to be extended and prominent, including a model-supported side-view path.
- Recovery uses enhanced full-frame and recent-hand crops, alternating with global search. Fresh observations must pass normal selection and gesture holds; missing landmarks cannot execute actions. Brief tracking gaps do not immediately re-arm a held command.
- Strong model-agreed index/palm geometry can handle moderate vocabulary uncertainty after a longer hold. Very low vocabulary mass and reviewed-negative vetoes remain rejected.
- Rejected candidates are not drawn as accepted skeletons; raw diagnostics remain available. Class scores are marked as candidates when no command is accepted.

The live detector uses MediaPipe Hand Landmarker only. Historical face/body helpers and assets are retained for reference; Pose is no longer a required runtime asset or setup download.

These are runtime improvements, not a retrained model or a promise to recognize every customer pose. Low-resolution/occluded fingers can still be ambiguous. Collect remaining failures and test on unseen participants after retraining.
