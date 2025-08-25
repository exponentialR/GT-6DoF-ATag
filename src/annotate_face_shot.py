"""
annotate_face_shot.py

Click 4 canonical corners on a saved face shot to compute:
    T_board_object  (pose of object in the board frame for THIS FACE)

Usage:
  python -m src.annotate_face_shot \
    --shot boards/shots/<object>_<face>_<ts>_raw.png

Assumes a sibling JSON meta file (..._meta.json) saved by capture_isc_face.py.
Requires: pupil-apriltags, OpenCV, PyYAML, numpy
"""

from __future__ import annotations
import argparse
import numpy as np
import cv2, yaml
from pathlib import Path
import itertools
from utils.annotation_utils import load_meta, detect_tags, load_board, se3, inv_se3, load_keypoints_fuzzy, choose_face_key, _draw_prompt

def assign_corners_any_order(clicked_xy, face_names, pts3d_dict, K):
    """
    clicked_xy: list of 4 (u,v) in the order the user clicked
    face_names: list of 4 strings (canonical names for that face)
    pts3d_dict: dict name -> [x,y,z] in metres
    K: 3x3 intrinsics

    Returns:
      mapping: {name: (u,v)}  # the best assignment
      ordered2d: (4,2) in the canonical face_names order
      fit: {'rvec','tvec','err'}
    """
    pts2d = np.array(clicked_xy, float).reshape(-1,1,2)
    names = list(face_names)

    best = None
    for perm in itertools.permutations(names, 4):
        X = np.array([pts3d_dict[n] for n in perm], float).reshape(-1,1,3)
        ok, rvec, tvec = cv2.solvePnP(X, pts2d, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        reproj, _ = cv2.projectPoints(X, rvec, tvec, K, None)
        err = float(np.sqrt(np.mean(np.sum((reproj - pts2d)**2, axis=2))))
        # prefer positive depth
        R,_ = cv2.Rodrigues(rvec)
        Xc = (R @ X.reshape(-1,3).T + tvec.reshape(3,1)).T  # (4,3)
        zmean = float(np.mean(Xc[:,2]))

        if best is None or (err < best["err"] and zmean > 0):
            best = {"perm": perm, "rvec": rvec, "tvec": tvec, "err": err}

    if best is None:
        raise RuntimeError("PnP failed for all 24 assignments")

    # Build mapping name -> clicked pixel
    mapping = {name: tuple(clicked_xy[i]) for i,name in enumerate(best["perm"])}
    # Reorder pixels to canonical face_names
    ordered2d = np.array([mapping[n] for n in face_names], float)

    return mapping, ordered2d, {"rvec": best["rvec"], "tvec": best["tvec"], "err": best["err"]}


def collect_four_clicks_any_order(image_bgr) -> list[tuple[float,float]]:
    """Collect 4 clicks in ANY order; supports 'u' to undo."""
    pts = []
    disp = image_bgr.copy()

    def redraw(rms=None):
        vis = disp.copy()
        cv2.putText(vis, "Click any 4 face corners (any order).  ENTER=save  u=undo  q=quit",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
        if rms is not None:
            cv2.putText(vis, f"fit RMS: {rms:.2f}px", (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)
        for i,(u,v) in enumerate(pts):
            cv2.circle(vis, (int(u),int(v)), 4, (0,255,0), -1)
            cv2.putText(vis, str(i+1), (int(u)+6, int(v)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow("Annotate", vis)

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((float(x), float(y)))
            redraw()

    cv2.namedWindow("Annotate", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Annotate", cb)
    redraw()

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q')):  # ESC/q
            raise SystemExit("Aborted.")
        if k == ord('u') and pts:
            pts.pop()
            redraw()
        if k == 13 and len(pts) == 4:   # ENTER
            return pts



def solve_pnp(pts2d: np.ndarray, pts3d: np.ndarray, K: np.ndarray):
    dist = np.zeros((5,1), dtype=float)  # assume rectified (your solver expects pinhole)
    ok, rvec, tvec = cv2.solvePnP(pts3d, pts2d, K, dist, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        raise RuntimeError("solvePnP failed")
    R,_ = cv2.Rodrigues(rvec)
    return se3(R, tvec.reshape(3))

def reproj_err(pts3d, T_cam_obj, K, img, draw=True):
    R = T_cam_obj[:3,:3]; t = T_cam_obj[:3,3].reshape(3,1)
    dist = np.zeros((5,1), float)
    rvec,_ = cv2.Rodrigues(R)
    uv,_ = cv2.projectPoints(pts3d, rvec, t, K, dist)  # (N,1,2)
    uv = uv.reshape(-1,2)
    if draw:
        vis = img.copy()
        for (u,v) in uv:
            cv2.circle(vis, (int(round(u)), int(round(v))), 3, (0,0,255), -1)
        return uv, vis
    return uv, None

def main():
    ap = argparse.ArgumentParser("Annotate a face shot to compute T_board_object")
    ap.add_argument("--shot", required=True, help="Path to *_raw.png")
    ap.add_argument("--family", default="tag36h11")
    args = ap.parse_args()

    shot_raw = Path(args.shot)
    if not shot_raw.exists(): raise SystemExit(f"Shot not found: {shot_raw}")
    if not str(shot_raw).endswith("_raw.png"):
        raise SystemExit("Shot path must end with _raw.png")

    meta_path = Path(str(shot_raw).replace("_raw.png", "_meta.json"))
    if not meta_path.exists(): raise SystemExit(f"Meta json not found: {meta_path}")
    meta = load_meta(meta_path)

    obj_name = meta["object"]
    face_yaml = meta["face_yaml"]
    if not face_yaml:
        raise SystemExit("Meta missing face_yaml (capture in registered mode so face YAML is recorded).")

    # Load image
    img = cv2.imread(str(shot_raw), cv2.IMREAD_COLOR)
    if img is None: raise SystemExit(f"Failed to read image: {shot_raw}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Camera intrinsics
    fx, fy, cx, cy = meta["camera"]["fx"], meta["camera"]["fy"], meta["camera"]["cx"], meta["camera"]["cy"]
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], float)

    # Keypoints & face mapping
    repo_root = Path(__file__).resolve().parents[1]
    pts3d_dict, faces_map, face_key_resolved, kp_path = load_keypoints_fuzzy(obj_name, repo_root)
    if face_key_resolved is None:
        raise SystemExit(f"[!] Keypoints found at {kp_path}, but no face entry matches '{obj_name}'. "
                         f"Add a faces[...] key for this face or use --object_name to a matching one.")
    print(f"[i] Using keypoints: {kp_path}  | face key: {face_key_resolved}")
    face_key = choose_face_key(face_yaml)
    if face_key not in faces_map:
        raise SystemExit(f"Face '{face_key}' not found in keypoints.json faces. Available: {list(faces_map.keys())}")
    names_in_order = faces_map[face_key]
    print(f"[i] Face '{face_key}' click order: {names_in_order}")

    # Board (to convert camera->tag to camera->board)
    board_yaml_path = Path(face_yaml)
    if not board_yaml_path.is_absolute():
        board_yaml_path = (repo_root / board_yaml_path).resolve()
    origin_id, tag_size_m, T_board_tag = load_board(board_yaml_path)

    # Detect tags in this image (get camera->tag)
    dets = detect_tags(gray, fx, fy, cx, cy, tag_size_m, family=args.family)
    if not dets:
        raise SystemExit("No AprilTags detected in shot; cannot get board pose.")
    det_by_id = {int(d.tag_id): d for d in dets}

    # Compute T_cam_board using origin if present, else any seen tag with board mapping
    if origin_id in det_by_id and det_by_id[origin_id].pose_R is not None:
        d = det_by_id[origin_id]
        T_cam_board = se3(d.pose_R.astype(float), d.pose_t.reshape(3).astype(float))
    else:
        # pick any detected tag that exists in the board
        common = [tid for tid in det_by_id.keys() if tid in T_board_tag]
        if not common:
            raise SystemExit("Detected tags do not belong to this face’s board.")
        tid = common[0]
        d = det_by_id[tid]
        T_cam_tag = se3(d.pose_R.astype(float), d.pose_t.reshape(3).astype(float))
        T_board_tag_i = T_board_tag[tid]              # board->tag
        T_tag_board = inv_se3(T_board_tag_i)         # tag->board
        T_cam_board = T_cam_tag @ T_tag_board

    # Ask user to click 4 canonical corners in order
    clicked = collect_four_clicks_any_order(img)

    # Assign names↔pixels by testing all 24 permutations; fit pose
    mapping, ordered2d, fit = assign_corners_any_order(clicked, names_in_order, pts3d_dict, K)
    R, _ = cv2.Rodrigues(fit["rvec"])
    T_cam_obj = se3(R, fit["tvec"].reshape(3))
    rms = float(fit["err"])
    print(f"[i] Reprojection RMS (best assignment): {rms:.2f} px")

    # Solve PnP for camera->object from clicks

    # Reprojection check (all 8 corners)
    # Reprojection check (all keypoints)
    all_pts3d = np.vstack([v for _, v in sorted(pts3d_dict.items())]).astype(np.float32)
    _, vis = reproj_err(all_pts3d, T_cam_obj, K, img, draw=True)

    # RMS on the 4 annotated corners (canonical order)
    pts3d_canon = np.vstack([pts3d_dict[n] for n in names_in_order]).astype(np.float32)
    uv4, _ = reproj_err(pts3d_canon, T_cam_obj, K, img, draw=False)
    err = np.linalg.norm(uv4 - ordered2d, axis=1)
    rms = float(np.sqrt((err ** 2).mean()))

    print(f"[i] Reprojection RMS on 4 points: {rms:.2f} px")

    # Compute board->object and save
    T_board_cam = inv_se3(T_cam_board)
    T_board_obj = T_board_cam @ T_cam_obj

    anno_png = Path(str(shot_raw).replace("_raw.png", "_anno.png"))
    done = img.copy()
    for name in names_in_order:
        u, v = mapping[name]
        cv2.circle(done, (int(u), int(v)), 5, (0, 0, 255), 2)
        cv2.putText(done, name, (int(u) + 6, int(v) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.putText(done, f"RMS={rms:.2f}px", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    cv2.imwrite(str(anno_png), done)

    reproj_png = Path(str(shot_raw).replace("_raw.png", "_reproj.png"))
    cv2.imwrite(str(reproj_png), vis)

    out_dir = repo_root / "objects" / obj_name / "faces"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = out_dir / f"{face_key}_T_board_object.yaml"
    out = {
        "object": obj_name,
        "face_key": face_key,
        "board_yaml": str(board_yaml_path),
        "image": str(shot_raw),
        "rms_px": rms,
        "T_board_object": {"matrix": T_board_obj.tolist()},
        "corner_uv": {name: [float(mapping[name][0]), float(mapping[name][1])] for name in names_in_order},
        "pnp": {
            "rvec": [float(x) for x in fit["rvec"].ravel()],
            "tvec": [float(x) for x in fit["tvec"].ravel()],
        },
        "notes": "Any-order 4-point annotation + tag-based board pose on this image."
    }

    with out_yaml.open("w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f"[i] Wrote {out_yaml}")

    # Show overlay image
    cv2.imshow("Reprojection check (red dots = projected 3D corners)", vis)
    print("[i] Press any key to close.")
    cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
