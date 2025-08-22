#!/usr/bin/env python3
import os
import sys
import math
import argparse
import cv2
import numpy as np

# ---------- Optional Pillow (for nicer text + PDF) ----------
USE_PIL_TEXT_DEFAULT = False
try:
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False
    USE_PIL_TEXT_DEFAULT = False

# ---------- Paper presets (mm) ----------
PAPER_MM = {
    "A4": (210.0, 297.0),
    "LETTER": (215.9, 279.4),   # 8.5 x 11 in
    "LEGAL": (215.9, 355.6),    # 8.5 x 14 in
}

# ---------- Defaults / tuning ----------
DEFAULT_OUT_DIR = "apriltags_out"
DEFAULT_PREFIX = "apriltag_36h11"
DEFAULT_DPI = 600
DEFAULT_MARGIN_FRAC = 0.05
DEFAULT_TOP_GAP_FRAC = 0.08
DEFAULT_LABEL_GAP_FRAC = 0.05
LABEL_FONT_SCALE = 1.5
LABEL_THICKNESS = 6

def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))

def px_to_mm(px: int, dpi: int) -> float:
    return px * 25.4 / dpi

def make_tag_bitmap(tag_id: int, side_mm: float, dpi: int) -> np.ndarray:
    side_px = mm_to_px(side_mm, dpi)
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if hasattr(cv2.aruco, "generateImageMarker"):
        tag = cv2.aruco.generateImageMarker(dic, tag_id, side_px, 1)
    else:
        tag = cv2.aruco.drawMarker(dic, tag_id, side_px)
    return tag  # uint8

def estimate_label_size_cv(text: str):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, LABEL_FONT_SCALE, LABEL_THICKNESS)
    return tw, th, base

def draw_label_cv(img, xc, y_top, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, LABEL_FONT_SCALE, LABEL_THICKNESS)
    org = (int(xc - tw/2), int(y_top + th))
    cv2.putText(img, text, org, font, LABEL_FONT_SCALE, (0,0,0), LABEL_THICKNESS, cv2.LINE_AA)

def draw_label_pil(img_rgb, xc, y_top, text):
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=int(36 * LABEL_FONT_SCALE))
    except Exception:
        font = ImageFont.load_default()
    tw = draw.textlength(text, font=font)
    draw.text((int(xc - tw/2), int(y_top)), text, fill=(0,0,0), font=font)
    return np.array(pil_img)

def compute_grid(W, H, margin, tag_px, label_gap_px, use_pil_text):
    """
    Compute the maximum (cols, rows) such that tags + label fit within margins.
    Cell height = top_gap + tag + label_gap + label_height + padding_bottom.
    We use a small padding around each cell derived from tag size.
    """
    # Conservative paddings
    top_gap = int(DEFAULT_TOP_GAP_FRAC * (H - 2*margin) / 3)  # scales with page height
    side_pad = max(8, int(0.02 * tag_px))
    bottom_pad = max(8, int(0.02 * tag_px))

    # Estimate label height using the largest likely text
    sample_text = "AprilTag 36h11 | ID 9999 | 300 mm"
    if use_pil_text and HAVE_PIL:
        # Approximate label height as 36 * LABEL_FONT_SCALE (close enough)
        label_h = int(36 * LABEL_FONT_SCALE * 1.2)
    else:
        _, th, base = estimate_label_size_cv(sample_text)
        label_h = th + base

    cell_w = tag_px + 2*side_pad
    cell_h = top_gap + tag_px + label_gap_px + label_h + bottom_pad

    cols = max(0, (W - 2*margin) // cell_w)
    rows = max(0, (H - 2*margin) // cell_h)
    return cols, rows, top_gap, side_pad, label_h

def make_canvas(paper_w_mm, paper_h_mm, dpi):
    W = mm_to_px(paper_w_mm, dpi)
    H = mm_to_px(paper_h_mm, dpi)
    sheet = np.full((H, W, 3), 255, dtype=np.uint8)
    return sheet, W, H

def layout_sheet(
    paper_w_mm: float,
    paper_h_mm: float,
    dpi: int,
    tag_size_mm: float,
    ids,
    prefix: str,
    out_dir: str,
    use_pil_text: bool,
    margin_frac: float,
    label_gap_frac: float,
):
    os.makedirs(out_dir, exist_ok=True)

    sheet, W, H = make_canvas(paper_w_mm, paper_h_mm, dpi)
    margin = int(margin_frac * min(W, H))

    tag_gray = make_tag_bitmap(ids[0], tag_size_mm, dpi)
    tag_px = tag_gray.shape[0]
    label_gap_px = int(label_gap_frac * ((H - 2*margin) // 3))

    cols, rows, top_gap, side_pad, label_h = compute_grid(
        W, H, margin, tag_px, label_gap_px, use_pil_text
    )
    capacity = cols * rows

    if capacity == 0:
        sys.exit(
            f"Error: A single tag of {tag_size_mm:.1f} mm cannot fit on the selected paper at {dpi} dpi "
            f"with current margins. Reduce tag size or margins."
        )

    if len(ids) > capacity:
        print(
            f"Warning: {len(ids)} tags requested but only {capacity} fit on this page. "
            f"Truncating to {capacity}.",
            file=sys.stderr
        )
        ids = ids[:capacity]

    # Rebuild tag (to avoid resizing artefacts) and place
    tag_rgb_cache = {}
    cell_w = (W - 2*margin) // max(1, cols)
    cell_h = (H - 2*margin) // max(1, rows)

    y_tag_offset = top_gap

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(ids):
                break
            tag_id = ids[idx]
            if tag_id not in tag_rgb_cache:
                gray = make_tag_bitmap(tag_id, tag_size_mm, dpi)
                tag_rgb_cache[tag_id] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            tag_rgb = tag_rgb_cache[tag_id]

            x0 = margin + c * cell_w
            y0 = margin + r * cell_h
            x_tag = x0 + (cell_w - tag_px) // 2
            y_tag = y0 + y_tag_offset

            # Bounds safety
            y1 = y_tag + tag_px
            x1 = x_tag + tag_px
            if y1 > H or x1 > W:
                idx += 1
                continue

            sheet[y_tag:y1, x_tag:x1] = tag_rgb

            # Label
            label_ascii = f"AprilTag 36h11 | ID {tag_id} | {int(tag_size_mm)} mm"
            x_center = x0 + cell_w // 2
            y_label = y_tag + tag_px + label_gap_px

            if use_pil_text and HAVE_PIL:
                label_unicode = f"AprilTag 36h11 • ID {tag_id} • {int(tag_size_mm)} mm"
                sheet[:] = draw_label_pil(sheet, x_center, y_label, label_unicode)
            else:
                draw_label_cv(sheet, x_center, y_label, label_ascii)

            idx += 1

    # Filenames
    paper_name = f"{int(round(paper_w_mm))}x{int(round(paper_h_mm))}mm"
    png_path = os.path.join(
        out_dir, f"{prefix}_{int(tag_size_mm)}mm_{paper_name}_{dpi}dpi.png"
    )
    cv2.imwrite(png_path, sheet)

    if HAVE_PIL:
        im = Image.fromarray(sheet)
        pdf_path = os.path.join(out_dir, f"{prefix}_{int(tag_size_mm)}mm_{paper_name}.pdf")
        im.save(pdf_path, "PDF", resolution=dpi)
        print("Saved:", png_path, "and", pdf_path)
    else:
        print("Saved:", png_path, "(install Pillow to also produce PDF)")

def parse_paper(paper: str, custom_mm: str | None, orientation: str):
    if custom_mm:
        try:
            w_mm, h_mm = [float(x) for x in custom_mm.lower().replace("mm","").split("x")]
        except Exception:
            sys.exit("Bad --paper-mm format. Use like: --paper-mm 210x297")
    else:
        key = paper.upper()
        if key not in PAPER_MM:
            sys.exit(f"Unknown paper '{paper}'. Choose from: {', '.join(PAPER_MM.keys())} or use --paper-mm WxH.")
        w_mm, h_mm = PAPER_MM[key]

    if orientation.lower() == "landscape":
        w_mm, h_mm = max(w_mm, h_mm), min(w_mm, h_mm)
    else:
        # portrait
        w_mm, h_mm = min(w_mm, h_mm), max(w_mm, h_mm)

    return w_mm, h_mm

def build_id_list(start_id: int, end_id: int):
    if end_id < start_id:
        sys.exit("--id-end must be >= --id-start.")
    return list(range(start_id, end_id + 1))

def main():
    ap = argparse.ArgumentParser(
        description="Generate printable AprilTag 36h11 sheets with auto grid and capacity checks."
    )
    ap.add_argument("--tag-size-mm", type=float, required=True,
                    help="Side length of tag in millimetres (e.g., 60)")
    ap.add_argument("--paper", type=str, default="A4",
                    help="Paper preset: A4, LETTER, LEGAL (ignored if --paper-mm is provided)")
    ap.add_argument("--paper-mm", type=str, default=None,
                    help="Custom paper size WxH in millimetres (e.g., 210x297). Overrides --paper.")
    ap.add_argument("--orientation", type=str, default="portrait", choices=["portrait","landscape"],
                    help="Paper orientation (default: portrait)")
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"Rendering DPI (default: {DEFAULT_DPI})")
    ap.add_argument("--id-start", type=int, required=True, help="First AprilTag ID (inclusive)")
    ap.add_argument("--id-end", type=int, required=True, help="Last AprilTag ID (inclusive)")
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--prefix", type=str, default=DEFAULT_PREFIX, help=f"Filename prefix (default: {DEFAULT_PREFIX})")
    ap.add_argument("--pil-text", action="store_true",
                    help="Use Pillow for nicer text (needs Pillow installed). Also writes PDF if Pillow available.")
    ap.add_argument("--margin-frac", type=float, default=DEFAULT_MARGIN_FRAC,
                    help=f"Page margin as fraction of min(width,height). Default {DEFAULT_MARGIN_FRAC:.2f}")
    ap.add_argument("--label-gap-frac", type=float, default=DEFAULT_LABEL_GAP_FRAC,
                    help=f"Fraction of cell height between tag and label. Default {DEFAULT_LABEL_GAP_FRAC:.2f}")

    args = ap.parse_args()

    use_pil_text = bool(args.pil_text) and HAVE_PIL
    if args.pil_text and not HAVE_PIL:
        print("Note: Pillow not found; falling back to OpenCV text. Install Pillow for nicer text/PDF.",
              file=sys.stderr)

    paper_w_mm, paper_h_mm = parse_paper(args.paper, args.paper_mm, args.orientation)
    ids = build_id_list(args.id_start, args.id_end)

    layout_sheet(
        paper_w_mm=paper_w_mm,
        paper_h_mm=paper_h_mm,
        dpi=args.dpi,
        tag_size_mm=args.tag_size_mm,
        ids=ids,
        prefix=args.prefix,
        out_dir=args.out_dir,
        use_pil_text=use_pil_text,
        margin_frac=args.margin_frac,
        label_gap_frac=args.label_gap_frac,
    )

    print(f"Done. Open '{args.out_dir}/' and print at 100% (Actual size).")

if __name__ == "__main__":
    main()
