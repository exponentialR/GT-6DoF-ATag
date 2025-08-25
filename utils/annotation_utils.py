import argparse, json, os, sys, math
from pathlib import Path
import numpy as np
import cv2, yaml
import re, json, glob
from pathlib import Path

try:
    from pupil_apriltags import Detector
except Exception as e:
    raise SystemExit("Please install pupil-apriltags: pip install pupil-apriltags") from e


def _draw_prompt(img, face_key, clicked_pts, rms=None):
    vis = img.copy()
    cv2.putText(vis, f"Face: {face_key}  (click any order: 4 corners)",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    if rms is not None:
        cv2.putText(vis, f"fit RMS: {rms:.2f}px",
                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2)
    for i,(u,v) in enumerate(clicked_pts):
        cv2.circle(vis, (int(u),int(v)), 4, (0,255,0), -1)
        cv2.putText(vis, str(i+1), (int(u)+6, int(v)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(vis, "ENTER=save  u=undo  q=quit",
                (12, img.shape[0]-14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return vis


def _shot_side(object_name: str) -> str | None:
    m = re.search(r'_side([A-D])$', object_name)
    return m.group(1) if m else None

def load_keypoints_fuzzy(object_name: str, repo_root: Path):
    """
    Try to locate keypoints.json for this object, allowing variant names.
    Strategy:
      1) objects/<object_name>/keypoints.json
      2) strip trailing _sideX -> objects/<base>/keypoints.json
      3) scan objects/*/keypoints.json and pick the one that has a face
         whose key endswith _sideX
    Returns: (pts3d_dict, faces_map, chosen_face_key, kp_path)
    Raises FileNotFoundError if nothing suitable is found.
    """
    obj_dir = repo_root / "objects"
    side = _shot_side(object_name)

    # 1) exact
    cand = obj_dir / object_name / "keypoints.json"
    if cand.exists():
        kp = json.loads(cand.read_text())
        pts3d = {k: np.array(v, float) for k, v in kp["points"].items()}
        faces = kp.get("faces", {})
        # prefer exact face key; else fall back by side
        face_key = object_name if object_name in faces else None
        if face_key is None and side:
            for k in faces.keys():
                if k.endswith(f"_side{side}"):
                    face_key = k; break
        return pts3d, faces, face_key, str(cand)

    # 2) strip _sideX and retry
    base = re.sub(r'_side[A-D]$', '', object_name)
    cand = obj_dir / base / "keypoints.json"
    if cand.exists():
        kp = json.loads(cand.read_text())
        pts3d = {k: np.array(v, float) for k, v in kp["points"].items()}
        faces = kp.get("faces", {})
        face_key = object_name if object_name in faces else None
        if face_key is None and side:
            for k in faces.keys():
                if k.endswith(f"_side{side}"):
                    face_key = k; break
        return pts3d, faces, face_key, str(cand)

    # 3) scan all and choose a set that contains the desired side
    if side:
        for kp_path in obj_dir.glob("*/keypoints.json"):
            kp = json.loads(kp_path.read_text())
            faces = kp.get("faces", {})
            for k in faces.keys():
                if k.endswith(f"_side{side}"):
                    pts3d = {n: np.array(v, float) for n, v in kp["points"].items()}
                    return pts3d, faces, k, str(kp_path)

    raise FileNotFoundError(f"No keypoints found for '{object_name}' (side={side}).")


def se3(R, t):
    T = np.eye(4, dtype=float); T[:3,:3] = R; T[:3,3] = t.reshape(3); return T

def inv_se3(T):
    R, t = T[:3,:3], T[:3,3]; Ti = np.eye(4, dtype=float)
    Ti[:3,:3] = R.T; Ti[:3,3] = -R.T @ t; return Ti

def rz(yaw_rad):
    c,s = math.cos(yaw_rad), math.sin(yaw_rad)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]], float)
    return R

def load_meta(meta_path: Path):
    with open(meta_path, "r") as f: m = json.load(f)
    for k in ("object","face_yaml","camera"):
        if k not in m: raise ValueError(f"meta missing '{k}'")
    return m

def load_keypoints(obj_name: str, repo_root: Path):
    kp_path = repo_root / "objects" / obj_name / "keypoints.json"
    with kp_path.open("r") as f: kp = json.load(f)
    pts = {k: np.asarray(v, float).reshape(3) for k,v in kp["points"].items()}
    faces = kp.get("faces", {})
    return pts, faces

def load_board(board_yaml_path: Path):
    y = yaml.safe_load(open(board_yaml_path, "r"))
    origin = int(y["origin_id"])
    tag_size_m = float(y.get("tag_size_m", 0.08))
    # Build T_board_tag for each tag id
    tb = {}
    for e in y.get("tags", []):
        tid = int(e["id"]); cx = float(e["cx"]); cy = float(e["cy"]); yaw_deg = float(e.get("yaw_deg", 0.0))
        Y = math.radians(yaw_deg)
        T = np.eye(4, dtype=float)
        T[:3,:3] = rz(Y)
        T[:3,3] = np.array([cx, cy, 0.0], float)
        tb[tid] = T
    # Origin must be identity
    tb[origin] = np.eye(4, dtype=float)
    return origin, tag_size_m, tb

def detect_tags(image_gray, fx, fy, cx, cy, tag_size_m, family="tag36h11"):
    det = Detector(families=family, nthreads=4, quad_decimate=1.0, refine_edges=True)
    return det.detect(image_gray, estimate_tag_pose=True,
                      camera_params=(fx, fy, cx, cy), tag_size=tag_size_m)

def choose_face_key(face_yaml_path: str) -> str:
    # face key = YAML basename without extension
    base = os.path.splitext(os.path.basename(face_yaml_path))[0]
    return base