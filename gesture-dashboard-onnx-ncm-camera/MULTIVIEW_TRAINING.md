# Multi-view directional training

For everyday collection, use **05 Data Collection** and follow
[DATA_COLLECTION.md](DATA_COLLECTION.md). The dashboard saves portable JPEG/JSON
records under `D:\Data-Collection`, including all ten commands and hard negatives.
The training command below also accepts `--dataset D:\Data-Collection` directly.
Raw-only cases are retained and reported for later landmark re-extraction.
The CSV capture instructions below remain an alternative for technical users.

This workflow addresses the main v18_20 data limitation: Left, Right, Up, and
Down were created primarily by rotating landmarks from HaGRID `one` examples.
The new workflow keeps the ten-command runtime contract unchanged and adds
reviewed, real-camera observations of the four existing directions.

It does **not** treat four rotated copies of one observation as four new people.
Every participant is assigned to exactly one of train, validation, or untouched
test before any oversampling or augmentation.

## What to collect

Use anonymous participant IDs; do not enter names, email addresses, or customer
account numbers. Obtain the participant's permission before retaining images.
For each direction, the guided tool covers:

- front-facing hand;
- yaw left and right;
- fingertips tilted toward and away from the camera;
- relaxed clockwise and counter-clockwise wrist roll;
- a natural, casually placed pose.

Also vary distance, frame position, left/right hand, background, and normal/low
light across sessions. Do not deliberately blur images or include faces when the
camera can be positioned to avoid them.

The qualification floor is intentionally modest and still requires, per
direction:

| Split | Minimum images | Minimum participants | Minimum viewpoints |
|---|---:|---:|---:|
| Train | 60 | 3 | 6 |
| Validation | 20 | 2 | 4 |
| Untouched test | 20 | 2 | 4 |

At least seven participants are therefore needed, and more are strongly
preferred. The deterministic 70/15/15 assignment can require additional people
before every split reaches its minimum. One person's frames must never be moved
between splits just to make the counts pass.

## Capture from the NCM camera

Install the notebook/training dependencies if they are not already present:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
```

Start the dashboard, connect the board camera, then start a capture session:

```powershell
.\start_ncm_onnx_dashboard.bat
.\.venv\Scripts\python.exe -m scripts.capture_multiview_data `
  --participant operator-001 `
  --source ncm
```

The capture window shows one label/view bucket at a time:

- **Space** saves a reviewed frame when the detected index direction agrees.
- **N** skips the rest of the current bucket.
- **Esc** stops without discarding previously saved work.

Resume the same session by passing the same `--session` value. By default the
tool requests eight images for each of eight viewpoints, or 256 images per
participant across four directions. Change this with `--samples-per-view`.

For a local webcam instead:

```powershell
.\.venv\Scripts\python.exe -m scripts.capture_multiview_data `
  --participant operator-001 `
  --source webcam `
  --webcam-device 0
```

Raw images and `manifest.csv` remain under `data/multiview/` and are ignored by
Git because they can contain personal data. Back them up only to an approved,
access-controlled dataset location.

## Audit, train, and qualify

Check coverage without fitting a model:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_multiview_model --audit-only
```

Once coverage passes, train the candidate:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_multiview_model
```

The trainer:

1. validates every image path, landmark array, feature vector, label, and split;
2. rejects participant leakage;
3. keeps every class at 4,096 rows and replaces at most 25% of each directional
   class with real NCM captures;
4. caps reuse of any captured frame at four training rows and applies only mild
   landmark perturbation to repeated rows;
5. selects the best epoch using public and captured validation data;
6. exports `artifacts/multiview/candidate.onnx`;
7. compares the candidate with the current production model on the old public
   test and the untouched captured-participant test.

For pipeline experimentation before coverage is complete, use
`--allow-incomplete-coverage`. Such a model is explicitly marked exploratory and
cannot pass qualification.

The command never overwrites `models/gesture_mlp_production.onnx`. Even an
offline PASS still requires a live NCM action test across directions, distances,
lighting, backgrounds, partial hands, and no-command scenes before promotion.

## Important limitation

The classifier sees 76 values derived from 2-D hand landmarks, not RGB pixels.
Real images improve it because camera angle and casual placement change the
detected landmark geometry. Lighting, background, motion blur, and very small
hands primarily challenge MediaPipe and must also be measured end to end; a high
classifier score alone does not prove camera reliability.
