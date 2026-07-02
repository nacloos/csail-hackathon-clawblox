"""
Generates synthetic lighting condition variations on a set of images.
"""
from pathlib import Path
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------
# Lighting transforms
# ---------------------------------------------------------------------

def adjust_exposure(img: np.ndarray, factor: float) -> np.ndarray:
    """Multiply image intensity (simulate exposure change)."""
    img = img.astype(np.float32) * factor
    return np.clip(img, 0, 255).astype(np.uint8)


def adjust_color_temperature(img: np.ndarray, r_scale: float, g_scale: float, b_scale: float) -> np.ndarray:
    """Scale RGB channels to simulate warm/cold lighting."""
    img = img.astype(np.float32)
    img[..., 0] *= r_scale  # R
    img[..., 1] *= g_scale  # G
    img[..., 2] *= b_scale  # B
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------

def generate_variations(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        p for p in input_dir.iterdir()
        if "rgb" in str(p) and p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ])

    if not image_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    for img_path in image_paths:
        img = np.array(Image.open(img_path).convert("RGB"))

        variations = {
            "normal": img,
            "dark": adjust_exposure(img, 0.6),
            "bright": adjust_exposure(img, 1.4),
            "cold": adjust_color_temperature(img, r_scale=0.75, g_scale=1.0, b_scale=1.25),
            "warm": adjust_color_temperature(img, r_scale=1.25, g_scale=1.0, b_scale=0.75),
        }

        for name, variant in variations.items():
            out_name = f"{img_path.stem}_{name}{img_path.suffix}"
            out_path = output_dir / out_name
            Image.fromarray(variant).save(out_path)

    print(f"Generated lighting variations for {len(image_paths)} images.")
    print(f"Saved to: {output_dir}")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path, help="Directory containing original images")
    parser.add_argument("output_dir", type=Path, help="Directory to save lighting variations")
    args = parser.parse_args()

    generate_variations(args.input_dir, args.output_dir)
