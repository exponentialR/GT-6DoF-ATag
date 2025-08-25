# GT-6DoF-ATag

Utilities and scripts to generate ground-truth 6-DoF object poses from RGB using AprilTags and Intel RealSense (D435i).
Designed for training RGB-only pose networks (e.g. ConPose) in pick-and-place/assembly scenarios.

## Contents

* **Calibration**: ChArUco colour camera calibration → `calib_color.yaml` (fx, fy, cx, cy, dist).
* **Board building**: Per-face AprilTag layout from a single frame → `<object_side>.yaml` and a global `boards/tag_registry.yaml`.
* **Shot capture**: Wide “face shots” with tag overlays + meta → `boards/shots/*_raw.png/_ann.png/_meta.json`.
* **Mesh keypoints**: Canonical 3D keypoints per object derived from its mesh → `objects/<object>/keypoints.json`.
* **Annotation**: Click 4 face corners on a shot to compute **T\_board\_object** for that face.
* **Batch annotation**: Drive the annotator across faces/shots automatically.
* **Ground truth**: At runtime, **T\_cam\_object = T\_cam\_board · T\_board\_object**.

---

## Quick start (scripts)

### 0) Generate AprilTags for printing

Contiguous IDs:

```bash
python -m src.gen_april_tags --tag-size-mm 80 --id-start 46 --id-end 49 --pil-text
```

Non-contiguous IDs:

```bash
python -m src.gen_april_tags --tag-size-mm 40 --ids 19,20,21,22,27,28,29,30
```

> Print at the correct physical size. The **black square** edge = `--tag-size-mm`.

---

### 1) Calibrate the colour camera (ChArUco)

Run our interactive calibrator; SPACE to capture a sample, ENTER to solve:

```bash
python charuco_calibrate.py --squares-x 5 --squares-y 3 \
  --square-length-mm 50 --marker-length-mm 37 --dict 7X7_1000 \
  --width 640 --height 480 --min-samples 30 --out calib_color.yaml
```

Outputs `calib_color.yaml` with:

* `camera_matrix` (fx, fy, cx, cy)
* `distortion_coefficients`
* `reproj_rms` (pixel RMS)

---

### 2) Build per-face boards and the tag registry

For each *face* of each object, capture one view with all its tags visible; pick an **origin** tag on that face. The script estimates in-plane offsets/rotation and writes a board YAML + updates the global registry.

```bash
python make_board.py \
  --object_name connection_plate_white_sideA \
  --calib calib_color.yaml \
  --tag_size_mm 80 \
  --out_dir boards \
  --save_shot \
  --z_thresh 0.01         # ensure tags are coplanar (≤ 1 cm off-plane)
# Repeat for other faces (e.g., connection_plate_white_sideB, column_white_sideA, ...)
```

Outputs:

* `boards/<object_face>.yaml` (face layout: origin\_id, tag\_size\_m, 2D offsets + yaw)
* `boards/tag_registry.yaml` (maps tag IDs → object\_face YAML)
* Optional shots in `boards/shots/` for audit (`--save_shot`)

> Different faces can use different tag sizes; each face YAML records its own `tag_size_m`.

---

### 3) Capture “face shots” for later annotation

Guided capture that checks seen tag IDs against the chosen face.

```bash
python -m src.capture_isc_face \
  --calib calib_color.yaml \
  --width 640 --height 480 --fps 30
```

Workflow:

* Enter object name (e.g., `connection_plate_white_sideA`)
* Select face index (usually `[0]` if a single YAML)
* Live preview; **ENTER** to save a shot (raw, annotated, meta JSON)

Outputs (per shot):

```
boards/shots/<object_face>_<face_key>_<YYYYMMDD_HHMMSS>_raw.png
boards/shots/<object_face>_<face_key>_<YYYYMMDD_HHMMSS>_ann.png
boards/shots/<object_face>_<face_key>_<YYYYMMDD_HHMMSS>_meta.json
```

---

### 4) Derive canonical 3D keypoints from meshes (once per object)

We compute the 8 AABB corners in the object’s **mesh frame** and also write a face→corner order that matches your faces.

```bash
python -m src.gen_keypoints_from_obj \
  --obj meshes/connection_plate.obj \
  --object_name connection_plate_white \
  --units_to_m 1.0 \
  --auto_sides \
  --write_object_config
```

Outputs:

* `objects/<object_name>/keypoints.json` (8 corners + face→corner order)
* `objects/<object_name>/object_config.yaml` (units, defaults)

> You can visualise these with `src.view_keypoints_on_obj.py` (we added this utility).

---

### 5) Annotate each face shot → **T\_board\_object** (per face)

Click 4 corners on a saved shot; the tool detects tags to get **T\_cam\_board** and uses your clicks + keypoints to solve **T\_cam\_object**, then derives **T\_board\_object**.

Single shot:

```bash
python -m src.annotate_face_shot --shot boards/shots/<object_face>_..._raw.png
```

Batch across faces (latest shot per face, skips ones already done):

```bash
python -m src.batch_annotate_faces
# Filters:
python -m src.batch_annotate_faces --object-filter connection_plate_white
python -m src.batch_annotate_faces --force  # re-annotate even if YAML exists
```

Outputs (per annotated face):

```
objects/<object_face>/faces/<face_key>_T_board_object.yaml
```

---

## How ground truth is formed

At **data capture** time (later, when you gather training frames):

1. Detect AprilTags in the RGB frame.
2. From a detected tag `i`:

   * Tag solver gives **T\_cam\_tagᵢ** (camera→tagᵢ) using (fx, fy, cx, cy) and that face’s `tag_size_m`.
   * Board YAML gives **T\_board\_tagᵢ** (board→tagᵢ).
   * Then **T\_cam\_board = T\_cam\_tagᵢ · (T\_board\_tagᵢ)⁻¹**.
3. From the annotation step you already have **T\_board\_object** for that face.
4. Final ground truth:

   $$
   \boxed{T_{\text{cam}\to\text{object}} \;=\; T_{\text{cam}\to\text{board}} \cdot T_{\text{board}\to\text{object}}}
   $$
5. Save RGB and **T\_cam\_object** as labels for training.

---

## Block diagram (end-to-end)

```mermaid
flowchart LR
  A[Camera calibration (ChArUco)\n-> calib_color.yaml] --> B
  B[Per-face board build (make_board.py)\n-> boards/face.yaml + boards/tag_registry.yaml] --> C
  C[Capture face shots (capture_isc_face.py)\n-> shots + meta] --> D
  E[Meshes (.obj)] --> F
  F[Canonical 3D keypoints (gen_keypoints_from_obj.py)\n-> keypoints.json] --> D
  D[Annotate shots (annotate_face_shot.py)\n-> T_board_object per face] --> G
  G[Runtime data capture (tag detect + registry lookup)] --> H
  H[Compute T_cam_board from tag + board] --> I
  I[Compose GT pose\nT_cam_object = T_cam_board * T_board_object\n-> save (RGB, pose)] --> J[Train RGB-only pose net]
```
[Calibrate] -> [Boards + registry] -> [Face shots] -> [Annotate -> T_board_object]
         \                                   ^
          \-> [Meshes] -> [Keypoints] -------|
[Runtime] Detect tag -> T_cam_board -> T_cam_object = T_cam_board · T_board_object


---

## File layout (typical)

```
calib_color.yaml
boards/
  tag_registry.yaml
  <object_face>.yaml
  shots/
    <object_face>_<face>_<ts>_raw.png
    <object_face>_<face>_<ts>_ann.png
    <object_face>_<face>_<ts>_meta.json
objects/
  <object_name>/
    keypoints.json
    object_config.yaml
  <object_face>/
    faces/
      <face_key>_T_board_object.yaml
meshes/
  *.obj
src/
  gen_april_tags.py
  charuco_calibrate.py
  make_board.py
  capture_isc_face.py
  gen_keypoints_from_obj.py
  view_keypoints_on_obj.py
  annotate_face_shot.py
  batch_annotate_faces.py
  utils/...
```

---

## Notes & tips

* **Origin tag per face**: choose a tag that’s reliably visible; its choice only fixes the 2D board frame on that face.
* **Tag size matters** at detection: per-face `tag_size_m` is stored in that face’s YAML and used when solving tag pose.
* **Planarity**: keep tags coplanar on a face (use `--z_thresh` in `make_board.py`).
* **Mixed sizes**: fine—just build each face with its true size; the registry keeps them separate.
* **Windows paths**: the code resolves `boards\*.yaml` as well as POSIX paths.
* **RealSense hiccups**: we added warm-up + retry with optional hardware reset to avoid “Frame didn’t arrive” restarts.

---

## What this repo is for

A reproducible pipeline to convert simple AprilTag stickers + a few manual clicks into high-quality 6-DoF ground truth for rigid objects, using only RGB at training time. It bridges physical tags and digital meshes through a board frame so you can train and validate RGB-only pose estimators on your real objects.

---

## Next actions

* [ ] Finish **annotation for all faces** (use `batch_annotate_faces`).
* [ ] Add a small script to **compose and export GT** `(RGB, T_cam_object)` from new capture sessions.
* [ ] (Optional) Add ADD/ADD-S evaluation utilities using your meshes.

