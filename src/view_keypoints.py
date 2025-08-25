"""
view_keypoints.py
Visualise canonical keypoints and face quads on a mesh.

- Loads objects/<object_name>/keypoints.json
- Optionally loads objects/<object_name>/object_config.yaml for:
    * mesh.path
    * mesh.units_to_m
    * mesh.T_mesh_object (4x4), mapping object->mesh (identity by default)
- Or you can pass --obj explicitly.

Controls:
  Mouse/keys: standard Open3D controls for pan/zoom/orbit
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import open3d as o3d
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

def _load_keypoints(kp_path: Path) -> dict:
    with kp_path.open("r") as f:
        kp = json.load(f)
    kp.setdefault("units_to_m", 1.0)
    if "points" not in kp or not isinstance(kp["points"], dict):
        raise ValueError("keypoints.json missing 'points' dict")
    kp.setdefault("faces", {})
    return kp

def _load_object_cfg(cfg_path: Path) -> dict | None:
    if not cfg_path.exists():
        return None
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg

def _resolve_mesh_path(obj_name: str, args_obj: str | None, cfg: dict | None) -> Path:
    # Priority: CLI --obj > object_config.yaml > meshes/<obj>.obj (best effort)
    if args_obj:
        p = Path(args_obj)
        return (p if p.is_absolute() else (REPO_ROOT / p)).resolve()
    if cfg and "mesh" in cfg and "path" in cfg["mesh"]:
        p = Path(cfg["mesh"]["path"])
        return (p if p.is_absolute() else (REPO_ROOT / p)).resolve()
    # fallback guesses
    cand = REPO_ROOT / "meshes" / f"{obj_name}.obj"
    if cand.exists():
        return cand.resolve()
    raise FileNotFoundError("Could not resolve mesh path. Pass --obj or set meshes/<name>.obj or object_config.yaml: mesh.path")

def _get_units_to_m(kp: dict, cfg: dict | None, cli_scale: float | None) -> float:
    # precedence: CLI override > object_config.yaml > keypoints.json
    if cli_scale and cli_scale > 0:
        return float(cli_scale)
    if cfg and "mesh" in cfg and "units_to_m" in cfg["mesh"]:
        return float(cfg["mesh"]["units_to_m"])
    return float(kp.get("units_to_m", 1.0))

def _get_T_mesh_object(cfg: dict | None) -> np.ndarray:
    # Default: identity (mesh frame == object frame)
    T = np.eye(4, dtype=float)
    if cfg and "mesh" in cfg and "T_mesh_object" in cfg["mesh"]:
        M = cfg["mesh"]["T_mesh_object"].get("matrix")
        if M is not None:
            T = np.array(M, dtype=float).reshape(4,4)
    return T

def _make_sphere(center: np.ndarray, radius: float, colour=(1.0, 0.4, 0.0)) -> o3d.geometry.TriangleMesh:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
    s.paint_uniform_color(colour)
    s.translate(center.reshape(3))
    return s

def _make_quad(points: list[np.ndarray], colour=(0.1, 0.7, 1.0)) -> o3d.geometry.LineSet:
    """Create a closed polyline through 4 points (0-1-2-3-0)."""
    P = np.vstack(points).astype(float)
    lines = [[0,1],[1,2],[2,3],[3,0]]
    ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(P),
                               lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector([colour]*len(lines))
    return ls

def _axis(length=0.05) -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=length, origin=[0,0,0])

def main():
    ap = argparse.ArgumentParser("Visualise keypoints on mesh")
    ap.add_argument("--object_name", required=True, help="e.g. connection_plate_white")
    ap.add_argument("--obj", default=None, help="Path to OBJ (overrides object_config)")
    ap.add_argument("--keypoints", default=None, help="Path to keypoints.json")
    ap.add_argument("--object_config", default=None, help="Path to object_config.yaml")
    ap.add_argument("--units_to_m", type=float, default=None, help="Override scale (m per OBJ unit)")
    ap.add_argument("--point_radius", type=float, default=0.003, help="Sphere radius (m)")
    ap.add_argument("--show_axes", action="store_true", help="Show world axes")
    args = ap.parse_args()

    obj_dir = REPO_ROOT / "objects" / args.object_name
    kp_path = Path(args.keypoints) if args.keypoints else (obj_dir / "keypoints.json")
    cfg_path = Path(args.object_config) if args.object_config else (obj_dir / "object_config.yaml")

    kp = _load_keypoints(kp_path)
    cfg = _load_object_cfg(cfg_path)

    mesh_path = _resolve_mesh_path(args.object_name, args.obj, cfg)
    scale = _get_units_to_m(kp, cfg, args.units_to_m)
    T_mo = _get_T_mesh_object(cfg)  # maps object->mesh

    print(f"[i] Mesh: {mesh_path}")
    print(f"[i] units_to_m: {scale}")
    print(f"[i] T_mesh_object:\n{T_mo}")

    # Load mesh
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    # Apply scale (OBJ units -> metres)
    mesh.scale(scale, center=[0,0,0])

    geoms = [mesh]
    if args.show_axes:
        geoms.append(_axis(length=0.05))

    # Prepare colours per face
    face_colours = {
        "sideA": (0.90, 0.20, 0.20),
        "sideB": (0.20, 0.90, 0.20),
        "sideC": (0.20, 0.55, 0.95),
        "sideD": (0.90, 0.75, 0.20),
    }

    # Build point cloud / spheres
    points_obj = {}
    for name, xyz in kp["points"].items():
        p_obj = np.array(xyz, dtype=float).reshape(3)
        # object -> mesh
        p_h = np.r_[p_obj, 1.0]
        p_mesh = (T_mo @ p_h)[:3]
        points_obj[name] = p_mesh

    # Spheres
    for name, p in points_obj.items():
        geoms.append(_make_sphere(p, radius=args.point_radius, colour=(1.0, 0.5, 0.0)))

    # Face quads (if present)
    faces = kp.get("faces", {})
    for face_key, names in faces.items():
        if not isinstance(names, list) or len(names) < 4:
            continue
        # choose colour by suffix sideX
        colour = (0.1, 0.7, 1.0)
        for k, col in face_colours.items():
            if face_key.endswith(k):
                colour = col; break
        try:
            quad = [_ for _ in (points_obj[names[0]], points_obj[names[1]],
                                points_obj[names[2]], points_obj[names[3]])]
            geoms.append(_make_quad(quad, colour=colour))
        except KeyError:
            print(f"[!] Missing point in face '{face_key}': {names}")

    # Print a legend to the console
    print("\n[i] Canonical points (mesh frame, metres):")
    for name, p in sorted(points_obj.items()):
        print(f"  {name:>16}: [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]")
    if faces:
        print("\n[i] Faces (click order suggestion):")
        for fk, names in faces.items():
            print(f"  {fk}: {names}")

    o3d.visualization.draw_geometries(geoms,
        window_name=f"{args.object_name} – keypoints",
        width=1280, height=800)

if __name__ == "__main__":
    main()
