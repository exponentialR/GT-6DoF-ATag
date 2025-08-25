import argparse, json, os
from pathlib import Path
import numpy as np

def load_obj_vertices(path: Path) -> np.ndarray:
    """Parse OBJ and return Nx3 array of vertices (float). Only reads 'v ' lines."""
    vs = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "): continue
            parts = line.strip().split()
            if len(parts) < 4: continue
            try:
                vs.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                pass
    if not vs:
        raise SystemExit(f"[!] No vertices found in {path}")
    return np.asarray(vs, dtype=np.float64)

def corners_from_bounds(vmin, vmax, r=6):
    """Return dict of 8 AABB corners with conventional names; rounded to r decimals."""
    xmin, ymin, zmin = vmin
    xmax, ymax, zmax = vmax
    def R(x): return round(float(x), r)
    P = {
        "xmin_ymin_zmin": [R(xmin), R(ymin), R(zmin)],
        "xmin_ymin_zmax": [R(xmin), R(ymin), R(zmax)],
        "xmin_ymax_zmin": [R(xmin), R(ymax), R(zmin)],
        "xmin_ymax_zmax": [R(xmin), R(ymax), R(zmax)],
        "xmax_ymin_zmin": [R(xmax), R(ymin), R(zmin)],
        "xmax_ymin_zmax": [R(xmax), R(ymin), R(zmax)],
        "xmax_ymax_zmin": [R(xmax), R(ymax), R(zmin)],
        "xmax_ymax_zmax": [R(xmax), R(ymax), R(zmax)],
    }
    return P

def face_corner_names(axis: str, sign: int):
    """Return the 4 corner keys for a given face (±X/±Y/±Z) in a consistent click order:
       top-left → top-right → bottom-right → bottom-left when *looking along +axis*.
    """
    assert axis in ("x","y","z") and sign in (-1, +1)
    if axis == "x":
        side = "xmin" if sign < 0 else "xmax"
        # y+ is 'top', z+ is 'right' when looking along +X
        return [f"{side}_ymax_zmin", f"{side}_ymax_zmax", f"{side}_ymin_zmax", f"{side}_ymin_zmin"]
    if axis == "y":
        side = "ymin" if sign < 0 else "ymax"
        # z+ is 'right', x+ is 'top' when looking along +Y
        return [f"xmax_{side}_zmin", f"xmax_{side}_zmax", f"xmin_{side}_zmax", f"xmin_{side}_zmin"]
    if axis == "z":
        side = "zmin" if sign < 0 else "zmax"
        # x+ is 'right', y+ is 'top' when looking along +Z
        return [f"xmax_ymax_{side}", f"xmin_ymax_{side}", f"xmin_ymin_{side}", f"xmax_ymin_{side}"]

def auto_side_mapping(extents, prefer_thin=True):
    """Pick which axis maps to sideA/B and which to C/D.
       If prefer_thin=True, map the thinnest axis to A/B (plates); else longest."""
    order = np.argsort(extents)   # ascending
    if prefer_thin:
        ab_axis = ["x","y","z"][order[0]]
        cd_axis = ["x","y","z"][order[1]]
    else:
        ab_axis = ["x","y","z"][order[-1]]
        cd_axis = ["x","y","z"][order[-2]]
    return ab_axis, cd_axis

def main():
    ap = argparse.ArgumentParser("Generate canonical keypoints.json (AABB corners) from OBJ")
    ap.add_argument("--obj", required=True, help="Path to OBJ mesh")
    ap.add_argument("--object_name", required=True, help="Logical object name (folder name under objects/)")
    ap.add_argument("--out_dir", default=None, help="Root dir (defaults to repo_root/objects/<object_name>)")
    ap.add_argument("--units_to_m", type=float, default=1.0, help="Scale factor: metres per OBJ unit (Unity=1.0, mm=0.001)")
    ap.add_argument("--round", type=int, default=6, dest="rnd", help="Decimal places to round coordinates")
    ap.add_argument("--sideA", choices=["+X","-X","+Y","-Y","+Z","-Z"], default="+X", help="Face key for sideA")
    ap.add_argument("--sideB", choices=["+X","-X","+Y","-Y","+Z","-Z"], default="-X", help="Face key for sideB")
    ap.add_argument("--sideC", choices=["+X","-X","+Y","-Y","+Z","-Z"], default="+Y", help="Face key for sideC")
    ap.add_argument("--sideD", choices=["+X","-X","+Y","-Y","+Z","-Z"], default="-Y", help="Face key for sideD")
    ap.add_argument("--auto_sides", action="store_true",
                    help="Override sideA..D using geometry (A/B=thinnest axis, C/D=next).")
    ap.add_argument("--write_object_config", action="store_true",
                    help="Also write objects/<obj>/object_config.yaml skeleton with mesh info")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]  # .../GT-6DoF-ATag/
    obj_path = Path(args.obj)
    if not obj_path.is_absolute():
        obj_path = (repo_root / obj_path).resolve()

    V = load_obj_vertices(obj_path)
    vmin = V.min(axis=0) * args.units_to_m
    vmax = V.max(axis=0) * args.units_to_m
    ext = vmax - vmin
    print(f"[i] Bounds (m):")
    print(f"    x: {vmin[0]:.6f} .. {vmax[0]:.6f}   Δ={ext[0]:.6f}")
    print(f"    y: {vmin[1]:.6f} .. {vmax[1]:.6f}   Δ={ext[1]:.6f}")
    print(f"    z: {vmin[2]:.6f} .. {vmax[2]:.6f}   Δ={ext[2]:.6f}")

    points = corners_from_bounds(vmin, vmax, r=args.rnd)

    # Decide side mapping
    def parse_face(s):
        sign = +1 if s[0] == "+" else -1
        axis = s[1].lower()
        return axis, sign
    if args.auto_sides:
        ab_axis, cd_axis = auto_side_mapping(ext, prefer_thin=True)
        sideA, sideB = (f"+{ab_axis.upper()}", f"-{ab_axis.upper()}")
        sideC, sideD = (f"+{cd_axis.upper()}", f"-{cd_axis.upper()}")
        print(f"[i] Auto sides → A/B along thinnest axis '{ab_axis}', C/D along '{cd_axis}'")
    else:
        sideA, sideB, sideC, sideD = args.sideA, args.sideB, args.sideC, args.sideD

    A = face_corner_names(*parse_face(sideA))
    B = face_corner_names(*parse_face(sideB))
    C = face_corner_names(*parse_face(sideC))
    D = face_corner_names(*parse_face(sideD))

    obj_name = args.object_name
    faces = {
        f"{obj_name}_sideA": A,
        f"{obj_name}_sideB": B,
        f"{obj_name}_sideC": C,
        f"{obj_name}_sideD": D,
    }

    # Write keypoints.json
    out_root = Path(args.out_dir) if args.out_dir else (repo_root / "objects" / obj_name)
    out_root.mkdir(parents=True, exist_ok=True)
    kp_path = out_root / "keypoints.json"
    with kp_path.open("w") as f:
        json.dump({"units_to_m": float(args.units_to_m), "points": points, "faces": faces}, f, indent=2)
    print(f"[i] Wrote {kp_path}")

    # Optional object_config.yaml skeleton
    if args.write_object_config:
        import yaml  # std dep
        cfg = {
            "mesh": {
                "path": str(obj_path.relative_to(repo_root) if obj_path.is_relative_to(repo_root) else obj_path),
                "units_to_m": float(args.units_to_m),
                "T_mesh_object": {  # identity by default: mesh frame == object frame
                    "matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
                }
            },
            "notes": "Fill in `T_board_object[face]` after annotation; leave identity if mesh frame is your object frame."
        }
        cfg_path = out_root / "object_config.yaml"
        with cfg_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"[i] Wrote {cfg_path}")

    # Friendly hint
    print("\n[→] Next: run your face annotator, clicking the 4 corners in the order listed for that face.")
    print("    You can edit the faces’ point order later if you prefer a different click order.")
    print("    For plates, `--auto_sides` is handy (A/B on the thinnest dimension).")

if __name__ == "__main__":
    main()
