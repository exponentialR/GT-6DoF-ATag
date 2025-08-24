"""
capture_isc_face.py

Take one or more wide shots per face for later T_object_from_board annotation.
- Prompts for object name (or pass --object_name)
- Lists faces (YAMLs) for that object from boards/tag_registry.yaml
- Live preview; ENTER to save raw+annotated still + JSON meta
- Keeps running until you press 'q'

Keys in the preview window:
  ENTER  -> save shot
  f      -> switch face
  o      -> switch object
  q or ESC -> quit
"""

import argparse, os, sys, json, time, datetime
import yaml
import numpy as np
import cv2
import pyrealsense2 as rs
from pathlib import Path
from utils.intel_realsense_utils import start_rs, get_frames_with_retry, load_calib, _face_label

try:
    from pupil_apriltags import Detector
except Exception as e:
    raise SystemExit("Please install pupil-apriltags: pip install pupil-apriltags") from e

repo_root = Path(__file__).resolve().parents[1]


def _resolve_yaml_path(p: str) -> str | None:
    if not p:
        return None
    p = p.replace("\\", "/")  # tolerate Windows entries
    cand = Path(p)
    if not cand.is_absolute():
        cand = repo_root / cand
    if not cand.exists():
        alt = repo_root / "boards" / cand.name  # fallback by basename
        if alt.exists():
            cand = alt
    return str(cand)


# --------- helpers ---------
def load_registry(path):
    if not os.path.exists(path):
        print(f"[!] tag registry not found: {path}")
        return {"version": 1, "tags": {}}
    reg = yaml.safe_load(open(path, "r")) or {}
    reg.setdefault("tags", {})
    return reg


def faces_for_object(object_name, registry):
    """Return list of face entries for the object.
       Each entry: dict(yaml, object, tag_ids=set(...))"""
    by_yaml = {}
    for tid, rec in registry["tags"].items():
        try:
            obj = rec.get("object")
            yml_raw = rec.get("yaml")
            if obj is None or yml_raw is None:
                continue
            if obj != object_name:
                continue
            yml = _resolve_yaml_path(yml_raw)
            if not yml:
                print(f"[!] Could not resolve yaml path: {yml_raw}")
                continue
            by_yaml.setdefault(yml, {"object": obj, "tag_ids": set()})
            by_yaml[yml]["tag_ids"].add(int(tid))
        except Exception:
            continue

    faces = []
    for yml, info in by_yaml.items():
        try:
            with open(yml, "r") as f:
                yy = yaml.safe_load(f)
            obj_field = yy.get("object", info["object"])
            tags = yy.get("tags", [])
            if tags and not info["tag_ids"]:
                info["tag_ids"] = set(int(t["id"]) for t in tags if "id" in t)
            faces.append({
                "yaml": yml,
                "object": obj_field,
                "tag_ids": sorted(list(info["tag_ids"]))
            })
        except Exception as e:
            print(f"[!] Could not read {yml}: {e}")
    return faces


def draw_detections(img, dets, colour=(0, 255, 0)):
    vis = img.copy()
    for d in dets:
        pts = d.corners.astype(int)
        cv2.polylines(vis, [pts], True, colour, 2)
        cv2.putText(vis, str(int(d.tag_id)), tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return vis


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# --------- main ---------

def main():
    ap = argparse.ArgumentParser("Capture wide shots per face for later annotation")
    ap.add_argument("--calib", default="calib_color.yaml")
    ap.add_argument("--registry", default=None,
                    help="Path to boards/tag_registry.yaml (defaults to repo_root/boards/tag_registry.yaml)")
    ap.add_argument("--out_dir", default=None,
                    help="Where to save shots; defaults to repo_root/boards/shots")
    ap.add_argument("--object_name", default=None, help="If omitted, will prompt interactively")
    ap.add_argument("--family", default="tag36h11")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--min_expected", type=int, default=1, help="min expected tags to see from the chosen face")

    args = ap.parse_args()

    calib_path = Path(args.calib)
    if not calib_path.exists():
        alt = repo_root / "calib_color.yaml"
        if alt.exists():
            args.calib = str(alt)

    if args.registry is None:
        cand = repo_root / "boards" / "tag_registry.yaml"
        args.registry = str(cand)
    else:
        rp = Path(args.registry)
        if not rp.exists():
            cand = repo_root / args.registry
            if cand.exists():
                args.registry = str(cand)

    if args.out_dir is None:
        args.out_dir = str(repo_root / "boards" / "shots")

    print(f"[i] Using calib: {args.calib}")
    print(f"[i] Using registry: {args.registry}")
    print(f"[i] Shots will be saved to: {args.out_dir}")

    (fx, fy, cx, cy), K = load_calib(args.calib)
    reg = load_registry(args.registry)
    ensure_dir(args.out_dir)

    det = Detector(families=args.family, nthreads=4, quad_decimate=1.0, refine_edges=True)

    # RealSense colour stream

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    start_rs(pipe, cfg, warmup=15)  # short warmup after start

    try:
        object_name = args.object_name
        while True:
            # -------- choose object --------
            if not object_name:
                object_name = input("[?] Enter object name (e.g., connection_plate_white): ").strip()
            faces = faces_for_object(object_name, reg)
            if not faces:
                # List available objects for convenience
                objs = {}
                for tid, rec in reg["tags"].items():
                    obj = rec.get("object")
                    yml = rec.get("yaml")
                    if obj and yml:
                        objs.setdefault(obj, set()).add(yml)
                print(f"[!] No faces found for object '{object_name}' in {args.registry}.")
                if objs:
                    print("    Known objects:")
                    for obj, ymls in objs.items():
                        print(f"      - {obj} ({len(ymls)} face file(s))")
                ch = input("[?] Try a different object name? [y/N]: ").strip().lower()
                if ch == "y":
                    object_name = None
                    continue
                else:
                    print("[i] You can still capture generic shots; they just won’t validate against expected tags.")
                    faces = []  # proceed with free capture

            # Index faces and let user pick
            face_idx = 0
            while True:
                # build a face menu each loop (files might change)
                face_menu = []
                for i, f in enumerate(faces):
                    face_key = os.path.splitext(os.path.basename(f["yaml"]))[0]
                    face_menu.append(f"[{i}] {face_key}  tags={f['tag_ids']}  ({f['yaml']})")
                if face_menu:
                    print("[i] Faces for", object_name)
                    for line in face_menu:
                        print("    ", line)
                    sel = input(
                        f"[?] Select face index [0..{len(faces) - 1}] (or 'o' change object, 'q' quit): ").strip()
                    if sel.lower() == 'q':
                        return
                    if sel.lower() == 'o':
                        object_name = None
                        break
                    try:
                        face_idx = int(sel)
                        if not (0 <= face_idx < len(faces)):
                            print("[!] Out of range.")
                            continue
                    except Exception:
                        print("[!] Enter a valid index, 'o', or 'q'.")
                        continue
                    face = faces[face_idx]
                else:
                    print("[i] Free capture mode (no face validation). Press 'o' to change object or 'q' to quit.")
                    face = None

                # -------- live preview & capture for this face --------
                expected = set(face["tag_ids"]) if face else set()
                face_key = os.path.splitext(os.path.basename(face["yaml"]))[0] if face else "unregistered"
                print("[i] Preview running. ENTER=save, f=switch face, o=switch object, q=quit.")

                while True:
                    frames = get_frames_with_retry(pipe, cfg, max_retries=2, timeout_ms=2000, do_hw_reset=True)
                    c = np.asanyarray(frames.get_color_frame().get_data())
                    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
                    dd = det.detect(g, estimate_tag_pose=False)

                    vis = draw_detections(c, dd)

                    # Line 1: face label
                    label = _face_label(face)
                    cv2.putText(vis, f"Face: {label}", (12, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    det_ids = {int(d.tag_id) for d in dd}
                    if expected:
                        ok = len(expected.intersection(det_ids)) >= max(1, args.min_expected)
                        msg = f"seen={sorted(det_ids)}  exp∩seen={sorted(expected.intersection(det_ids))}"
                        col = (0, 200, 0) if ok else (0, 0, 255)
                    else:
                        ok = True
                        msg = f"seen={sorted(det_ids)}"
                        col = (0, 200, 200)

                    # Line 2: detection status
                    cv2.putText(vis, msg, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

                    # Bottom help
                    cv2.putText(vis, "ENTER=save  f=face  o=object  q=quit",
                                (20, args.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    cv2.imshow("Capture face shot", vis)
                    k = cv2.waitKey(1) & 0xFF

                    if k in (27, ord('q')):  # ESC or q
                        return

                    if k == ord('f'):
                        if faces:
                            face_idx = (face_idx + 1) % len(faces)
                            face = faces[face_idx]
                            expected = set(face["tag_ids"])
                            face_key = os.path.splitext(os.path.basename(face["yaml"]))[0]
                            print(f"[i] Switched to face: {_face_label(face)}  expected={sorted(expected)}")
                        continue

                    # o = change object (leave preview to re-prompt)
                    if k == ord('o'):
                        object_name = None
                        break

                    if k == 13:  # ENTER -> save
                        if expected and not ok:
                            print("[!] Expected face tags not visible. Save anyway? [y/N] ", end="", flush=True)
                            ch = input().strip().lower()
                            if ch != "y":
                                continue

                        ts = timestamp()
                        base = f"{object_name}_{face_key}_{ts}"
                        raw_path = os.path.normpath(os.path.join(args.out_dir, f"{base}_raw.png"))
                        ann_path = os.path.normpath(os.path.join(args.out_dir, f"{base}_ann.png"))
                        meta_path = os.path.normpath(os.path.join(args.out_dir, f"{base}_meta.json"))

                        cv2.imwrite(raw_path, c)
                        cv2.imwrite(ann_path, vis)

                        meta = {
                            "object": object_name,
                            "face_yaml": face["yaml"] if face else None,
                            "expected_tag_ids": sorted(list(expected)),
                            "detected_tag_ids": sorted(list(det_ids)),
                            "image": {
                                "path_raw": raw_path,
                                "path_ann": ann_path,
                                "width": int(args.width),
                                "height": int(args.height),
                            },
                            "camera": {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)},
                            "timestamp": ts,
                        }
                        with open(meta_path, "w") as f:
                            json.dump(meta, f, indent=2)
                        print(f"[+] Saved {raw_path}")
                        print(f"[+] Saved {ann_path}")
                        print(f"[+] Saved {meta_path}")

                        # ask user what to do next
                        nxt = input(
                            "[?] Save another for this face (Enter), switch (f), change object (o), or quit (q)? ").strip().lower()
                        if nxt == 'q':
                            return
                        if nxt == 'o':
                            object_name = None
                            break
                        if nxt == 'f':
                            break
                        # else continue capturing same face

                if object_name is None:
                    # break to outer loop to select object
                    break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
