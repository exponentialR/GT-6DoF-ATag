"""
Batch-launch the face annotator so you don't have to call it manually per shot.

Usage examples:
  python -m src.batch_annotate_faces
  python -m src.batch_annotate_faces --shots-dir boards/shots
  python -m src.batch_annotate_faces --object-filter connection_plate_white
  python -m src.batch_annotate_faces --force

It picks the latest shot per (object, face_key) and runs:
  python -m src.annotate_face_shot --shot <..._raw.png>
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path
from collections import defaultdict

RE_TS = re.compile(r"_(\d{8}_\d{6})_raw\.png$")

def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]

def load_meta_for_shot(raw_png: Path) -> dict | None:
    meta = raw_png.with_name(raw_png.name.replace("_raw.png", "_meta.json"))
    if not meta.exists():
        return None
    try:
        return json.loads(meta.read_text())
    except Exception:
        return None

def face_key_from_meta(meta: dict) -> str | None:
    # face key = basename of face_yaml without extension
    fy = meta.get("face_yaml")
    if not fy:
        return None
    return Path(fy).stem

def shot_timestamp(raw_png: Path, meta: dict | None) -> str:
    # Prefer meta timestamp; fallback to filename pattern
    ts = (meta or {}).get("timestamp")
    if isinstance(ts, str) and len(ts) == 15:  # YYYYMMDD_HHMMSS
        return ts
    m = RE_TS.search(raw_png.name)
    return m.group(1) if m else "00000000_000000"

def build_out_yaml(repo_root: Path, object_name: str, face_key: str) -> Path:
    return repo_root / "objects" / object_name / "faces" / f"{face_key}_T_board_object.yaml"

def main():
    ap = argparse.ArgumentParser("Batch launcher for annotate_face_shot")
    ap.add_argument("--shots-dir", default=None, help="Default: <repo>/boards/shots")
    ap.add_argument("--object-filter", default=None,
                    help="Substring to keep only objects whose name contains this (case-insensitive).")
    ap.add_argument("--force", action="store_true", help="Re-annotate even if output YAML already exists.")
    ap.add_argument("--dry-run", action="store_true", help="List what would run, but don't launch.")
    args = ap.parse_args()

    repo_root = repo_root_from_here()
    shots_dir = Path(args.shots_dir) if args.shots_dir else (repo_root / "boards" / "shots")
    if not shots_dir.exists():
        sys.exit(f"[!] Shots dir not found: {shots_dir}")

    # Collect candidate shots and group by (object, face_key)
    groups: dict[tuple[str,str], list[tuple[str, Path]]] = defaultdict(list)
    for raw_png in sorted(shots_dir.glob("*_raw.png")):
        meta = load_meta_for_shot(raw_png)
        if not meta:
            continue
        obj = meta.get("object")
        face_key = face_key_from_meta(meta)
        if not obj or not face_key:
            continue
        if args.object_filter and args.object_filter.lower() not in obj.lower():
            continue
        ts = shot_timestamp(raw_png, meta)
        groups[(obj, face_key)].append((ts, raw_png))

    if not groups:
        sys.exit("[!] No eligible shots found (are your *_meta.json files present and face-registered?).")

    # Pick latest shot per face
    picks: list[tuple[str,str,Path,str]] = []  # (object, face_key, shot_path, ts)
    for (obj, face_key), items in groups.items():
        ts, shot = sorted(items, key=lambda x: x[0])[-1]  # latest
        picks.append((obj, face_key, shot, ts))

    # Filter out already-annotated faces unless --force
    to_run = []
    for obj, face_key, shot, ts in picks:
        out_yaml = build_out_yaml(repo_root, obj, face_key)
        if out_yaml.exists() and not args.force:
            print(f"[i] Skip (already annotated): {obj} / {face_key}  -> {out_yaml}")
            continue
        to_run.append((obj, face_key, shot, ts, out_yaml))

    if not to_run:
        print("[i] Nothing to do.")
        return

    # Summary
    print("\n[i] Will annotate the following faces (latest shot per face):")
    for obj, face_key, shot, ts, out_yaml in to_run:
        print(f"    - {obj} / {face_key}  ts={ts}  shot={shot.name}  -> {out_yaml}")

    if args.dry_run:
        print("\n[i] Dry-run only; exiting.")
        return

    # Launch annotator per shot
    for obj, face_key, shot, ts, out_yaml in to_run:
        print(f"\n[>] Annotating: {obj} / {face_key}  ({shot})")
        cmd = [sys.executable, "-m", "src.annotate_face_shot", "--shot", str(shot)]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"[!] Annotator exited with code {ret.returncode} for {shot}")
        else:
            if out_yaml.exists():
                print(f"[✓] Wrote {out_yaml}")
            else:
                print(f"[?] Completed, but {out_yaml} not found (check console output).")

if __name__ == "__main__":
    main()
