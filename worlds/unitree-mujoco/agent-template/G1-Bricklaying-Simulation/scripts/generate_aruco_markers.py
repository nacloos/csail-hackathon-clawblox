"""Generate ArUco markers for table localization.

Produces 8 individual PNGs and a combined A4 sheet, ready to print at 300 DPI.
Output goes to out_figures/aruco/.

Usage:
    conda activate g1-analysis
    python scripts/generate_aruco_markers.py
"""

import os
import numpy as np
import cv2
from PIL import Image

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
ARUCO_DICT  = cv2.aruco.DICT_5X5_50
NUM_MARKERS = 8          # IDs 0..7
MARKER_CM   = 7.0 * 7.0 / 6.7        # for some reason the print is scaled down a little?
DPI         = 300
BORDER_BITS = 1          # white quiet-zone width in marker bit units

# Letter sheet dimensions at 300 DPI: 8.5 × 11 in
LETTER_W_PX = int(round(8.5 * DPI))   # 2550
LETTER_H_PX = int(round(11.0 * DPI))  # 3300

# 2×2 grid → 4 markers per page, 2 pages for 8 markers
GRID_COLS         = 2
GRID_ROWS         = 2
MARKERS_PER_PAGE  = GRID_COLS * GRID_ROWS

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out_figures", "aruco")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def marker_px_size(marker_cm: float, border_bits: int, dpi: int) -> tuple[int, int]:
    """
    Return the pixel size of the central marker and the total (including border) of ArUco.
    """
    px_per_cm = dpi / 2.54
    px_marker = ((marker_cm * px_per_cm) // 5) * 5
    px_total = px_marker / 5 * (5 + 2 * border_bits)
    return int(px_marker), int(px_total)


def add_id_label(img: np.ndarray, marker_id: int, font_scale: float = 1.0, thickness: int = 2) -> np.ndarray:
    """Return a copy of img with 'ID: {marker_id}' drawn below the image."""
    buffer_h = 20
    label_h   = int(40 * font_scale) + buffer_h
    canvas    = np.full((img.shape[0] + label_h, img.shape[1]), 255, dtype=np.uint8)
    canvas[:img.shape[0], :] = img
    text      = f"ID: {marker_id}"
    font      = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (canvas.shape[1] - tw) // 2
    y = img.shape[0] + th + buffer_h
    cv2.putText(canvas, text, (x, y), font, font_scale, 0, thickness, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    body_px, total_px = marker_px_size(MARKER_CM, BORDER_BITS, DPI)

    print(f"Marker body: {body_px} px  |  total (with border): {total_px} px  |  DPI: {DPI}")
    print(f"Physical size: {MARKER_CM} cm body, ~{total_px / DPI * 2.54:.1f} cm total")
    print(f"Output dir: {os.path.abspath(OUT_DIR)}\n")

    marker_imgs = []
    for marker_id in range(NUM_MARKERS):
        img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, total_px,
                                            borderBits=BORDER_BITS)
        path = os.path.join(OUT_DIR, f"aruco_marker_{marker_id:02d}.png")
        Image.fromarray(img).save(path, dpi=(DPI, DPI))
        print(f"  Wrote {path}")
        marker_imgs.append(img)

    # -----------------------------------------------------------------------
    # 2-page PDF, each page Letter (8.5×11 in), 2×2 grid of markers.
    # -----------------------------------------------------------------------
    cell_w = LETTER_W_PX // GRID_COLS
    cell_h = LETTER_H_PX // GRID_ROWS

    pages = []
    num_pages = NUM_MARKERS // MARKERS_PER_PAGE
    for page_idx in range(num_pages):
        sheet = np.full((LETTER_H_PX, LETTER_W_PX), 255, dtype=np.uint8)
        for local_idx in range(MARKERS_PER_PAGE):
            global_idx = page_idx * MARKERS_PER_PAGE + local_idx
            row = local_idx // GRID_COLS
            col = local_idx % GRID_COLS

            img = marker_imgs[global_idx]
            labeled = add_id_label(img, global_idx)

            y0 = row * cell_h
            x0 = col * cell_w
            block_h, block_w = labeled.shape
            y1 = y0 + max(0, (cell_h - block_h) // 2)
            x1 = x0 + max(0, (cell_w - block_w) // 2)
            h = min(block_h, LETTER_H_PX - y1)
            w = min(block_w, LETTER_W_PX - x1)
            sheet[y1:y1 + h, x1:x1 + w] = labeled[:h, :w]

        pages.append(Image.fromarray(sheet))

    pdf_path = os.path.join(OUT_DIR, "aruco_markers.pdf")
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=DPI)
    print(f"\n  Wrote {pdf_path}  (2 pages, Letter, 2×2 grid)")

    # -----------------------------------------------------------------------
    # Placement instructions
    # -----------------------------------------------------------------------
    print("""
=============================================================
Print aruco_markers.pdf at 100% / actual size (NOT "fit to page").
2 pages, 4 markers each (2×2 grid), Letter paper.

After placing, measure each marker's center position and yaw
in the table frame (tape measure from table origin corner).
Record as MARKER_POSITIONS_TABLE = {id: (x_m, y_m, yaw_rad), ...}
in src/bricklaying/perception/aruco_localizer.py (next step).
=============================================================
""")


if __name__ == "__main__":
    main()
