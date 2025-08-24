import yaml, os
from pathlib import Path

repo = Path(__file__).resolve().parents[1]  # <repo root>
reg_path = repo / "boards" / "tag_registry.yaml"

y = yaml.safe_load(open(reg_path)) or {}
tags = y.get("tags", {})
changed = 0

for tid, rec in list(tags.items()):
    p = rec.get("yaml")
    if not p:
        continue
    # normalise slashes and resolve relative to repo root
    p_norm = p.replace("\\", "/")
    cand = Path(p_norm)
    if not cand.is_absolute():
        cand = repo / cand
    # fallback: look under boards/ by basename
    if not cand.exists():
        alt = repo / "boards" / Path(p_norm).name
        if alt.exists():
            cand = alt
    # write back as repo-relative POSIX path
    try:
        rel = cand.relative_to(repo)
    except ValueError:
        rel = cand  # leave absolute if outside
    rec["yaml"] = str(rel).replace("\\", "/")
    changed += 1

y["tags"] = tags
yaml.safe_dump(y, open(reg_path, "w"), sort_keys=False)
print(f"[i] Normalised {changed} entries in {reg_path}")
