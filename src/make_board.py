#!/usr/bin/env python3
import argparse, os, sys, yaml, numpy as np, cv2, pyrealsense2 as rs, datetime, tempfile, shutil
from math import atan2, degrees
try:
    from pupil_apriltags import Detector
except Exception as e:
    raise SystemExit("Install pupil-apriltags: pip install pupil-apriltags") from e

def load_calib(path):
    if not os.path.exists(path):
        print(f"[!] {path} not found.")
        print("    Run your ChArUco calibration to generate calib_color.yaml first.")
        sys.exit(1)
    y = yaml.safe_load(open(path))
    cm = y["camera_matrix"]; dc = y["distortion_coefficients"]
    fx, fy, cx, cy = float(cm["fx"]), float(cm["fy"]), float(cm["cx"]), float(cm["cy"])
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], float)
    # Distortion not needed by the apriltag pose solver (expects pinhole), keep for logging/completeness
    D = np.array([[dc["k1"], dc["k2"], dc["p1"], dc["p2"], dc["k3"]]], float)
    return (fx, fy, cx, cy), K, D

def se3(R, t):
    T = np.eye(4, dtype=float)
    T[:3,:3] = R; T[:3,3] = t.reshape(3)
    return T

def inv_se3(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3,3] = -R.T @ t
    return Ti

def load_registry(path):
    if not os.path.exists(path):
        return {"version": 1, "updated": None, "tags": {}}
    with open(path, "r") as f:
        reg = yaml.safe_load(f) or {}
    reg.setdefault("version", 1)
    reg.setdefault("tags", {})
    return reg

def save_registry(path, reg):
    reg["updated"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # atomic write
    fd, tmp = tempfile.mkstemp(prefix=".tag_registry_", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(reg, f, sort_keys=False)
        shutil.move(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser("Interactive board.yaml builder from live AprilTags")
    p.add_argument("--calib", default="calib_color.yaml", help="Path to calib YAML (with fx,fy,cx,cy keys)")
    p.add_argument("--family", default="tag36h11", help="AprilTag family")
    p.add_argument("--tag_size_mm", type=float, default=80.0, help="Black square edge length (mm)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--object_name", type=str, required=True, help="Name of the object, e.g. connection_plate")
    p.add_argument("--out_dir", default="boards", help="Directory for board files")
    p.add_argument("--save_shot", action="store_true", help="Save the captured frame (raw and annotated) for auditing")
    p.add_argument("--shots_dir", default="boards/shots", help="Directory for saved shots if --save_shot is set")
    # planarity controls
    p.add_argument("--z_thresh", type=float, default=0.01, help="Max |z| (m) between selected tags; guards planarity")
    p.add_argument("--allow_nonplanar", action="store_true",
                   help="Warn but still write YAML even if |z| > z_thresh")
    p.add_argument("--registry", default="boards/tag_registry.yaml",
                   help="Path to tag→board registry YAML")
    p.add_argument("--force-registry", action="store_true",
                   help="If a tag already maps to a different YAML, overwrite the mapping")

    return p.parse_args()

def main():
    args = parse_args()

    # Ensure output dirs exist
    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_shot:
        os.makedirs(args.shots_dir, exist_ok=True)

    out_path = os.path.join(args.out_dir, f"{args.object_name}.yaml")
    (fx, fy, cx, cy), K, D = load_calib(args.calib)
    tag_size_m = args.tag_size_mm / 1000.0

    det = Detector(families=args.family, nthreads=4, quad_decimate=1.0, refine_edges=True)

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipe.start(cfg)
    print("[i] Showing live colour stream. Make sure all tags you want are visible in one view.")
    print("[i] Press ENTER to capture this frame for board creation. ESC to quit.")

    frame = None; dets = []
    try:
        while True:
            f = pipe.wait_for_frames()
            c = np.asanyarray(f.get_color_frame().get_data())
            g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
            vis = c.copy()

            # Draw IDs for situational awareness
            dd = det.detect(g, estimate_tag_pose=False)
            for d in dd:
                pts = d.corners.astype(int)
                cv2.polylines(vis, [pts], True, (0,255,0), 2)
                cv2.putText(vis, str(d.tag_id), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            cv2.imshow("Select a view (ENTER to use)", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                return
            if k == 13:  # ENTER
                frame = c.copy()
                dets = det.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                  estimate_tag_pose=True,
                                  camera_params=(fx, fy, cx, cy),
                                  tag_size=tag_size_m)
                if len(dets) == 0:
                    print("[!] No tags detected in captured frame; try again.")
                    continue

                if args.save_shot:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    base = os.path.join(args.shots_dir, f"{args.object_name}_{ts}")
                    raw_path = f"{base}_raw.png"
                    ann_path = f"{base}_ann.png"
                    cv2.imwrite(raw_path, frame)
                    ann = frame.copy()
                    for d in dets:
                        pts = d.corners.astype(int)
                        cv2.polylines(ann, [pts], True, (0, 255, 0), 2)
                        cv2.putText(ann, str(int(d.tag_id)), tuple(pts[0]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imwrite(ann_path, ann)
                    print(f"[i] Saved raw shot to {raw_path}")
                    print(f"[i] Saved shot: {raw_path} and {ann_path}")
                break
    finally:
        pipe.stop(); cv2.destroyAllWindows()

    ids_visible = [int(d.tag_id) for d in dets]
    print(f"[i] Detected IDs in captured frame: {ids_visible}")
    sels = input("[?] Enter comma-separated IDs to include (e.g., 12,37): ").strip()
    try:
        sel_ids = [int(s) for s in sels.replace(" ", "").split(",") if s != ""]
    except Exception:
        print("[!] Could not parse IDs."); return
    sel_ids = [i for i in sel_ids if i in ids_visible]
    if len(sel_ids) < 1:
        print("[!] Need at least one valid ID."); return

    origin = input(f"[?] Choose ORIGIN id from {sel_ids} (default {sel_ids[0]}): ").strip()
    origin_id = int(origin) if origin else sel_ids[0]
    if origin_id not in sel_ids:
        print("[!] Origin must be one of the selected IDs."); return

    # Build SE(3) for each selected tag (camera->tag)
    det_map = {int(d.tag_id): d for d in dets}
    T_cam = {}
    for tid in sel_ids:
        d = det_map.get(tid, None)
        if d is None or d.pose_R is None or d.pose_t is None:
            print(f"[!] Missing pose for id {tid}."); return
        T_cam[tid] = se3(d.pose_R.astype(float), d.pose_t.reshape(3).astype(float))

    # Express all tags in ORIGIN-tag frame
    T_org_cam = inv_se3(T_cam[origin_id])
    entries = []
    nonplanar = False
    for tid in sel_ids:
        T_org_tag = T_org_cam @ T_cam[tid]
        t = T_org_tag[:3,3]
        R = T_org_tag[:3,:3]
        yaw = degrees(atan2(R[1,0], R[0,0]))  # in-plane yaw around +z
        if abs(t[2]) > args.z_thresh:
            print(f"[!] Tag {tid} z offset {t[2]:.3f} m exceeds {args.z_thresh} m.")
            nonplanar = True
        entries.append(dict(id=int(tid), cx=float(t[0]), cy=float(t[1]), yaw_deg=float(yaw)))

    if nonplanar and not args.allow_nonplanar:
        print("[!] Re-capture with a flatter view / re-mount the tags, or pass --allow_nonplanar to proceed.")
        return

    # Sort by id for readability
    entries.sort(key=lambda e: e["id"])

    board = dict(
        object=args.object_name,           # include object name
        family=args.family,
        tag_size_m=tag_size_m,
        origin_id=int(origin_id),
        tags=entries,
        notes="Board frame = origin tag centre; x,y follow origin tag axes; z ≈ 0."
    )
    with open(out_path, "w") as f:
        yaml.safe_dump(board, f, sort_keys=False)
    print(f"[i] Wrote {out_path}")
    print("[i] Example entries:")
    for e in entries:
        print(f"    - {{id: {e['id']}, cx: {e['cx']:.3f}, cy: {e['cy']:.3f}, yaw_deg: {e['yaw_deg']:.1f}}}")

    # ---- Update tag registry ----
    reg = load_registry(args.registry)
    updated, conflicts = 0, []
    for e in entries:
        tid = str(e["id"])
        current = reg["tags"].get(tid)
        if current is None or current.get("yaml") == out_path:
            reg["tags"][tid] = {"object": args.object_name, "yaml": out_path}
            updated += 1
        else:
            if args.force_registry:
                print(f"[!] Overwriting registry for tag {tid}: {current['yaml']} -> {out_path}")
                reg["tags"][tid] = {"object": args.object_name, "yaml": out_path}
                updated += 1
            else:
                conflicts.append((tid, current["yaml"], out_path))

    save_registry(args.registry, reg)
    print(f"[i] Registry updated ({updated} entries) at {args.registry}")
    if conflicts:
        print("[!] Registry conflicts (use --force-registry to overwrite):")
        for tid, oldp, newp in conflicts:
            print(f"    tag {tid}: {oldp}  ->  {newp}")

if __name__ == "__main__":
    main()
