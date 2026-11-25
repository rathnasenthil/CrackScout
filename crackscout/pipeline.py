"""Load one image, run a registered segmentation method, and save outputs."""

from pathlib import Path

import cv2

from crackscout.io import load_image
from crackscout.preprocess import denoise, to_grayscale
from crackscout.segment import segment
from crackscout.visualize import save_comparison


def run_baseline(
    input_path: str | Path,
    output_dir: str | Path = "outputs",
    method: str = "otsu",
) -> dict[str, Path]:
    """Load one image, segment it, and write the mask and comparison figure.

    Files are named from the input stem and the method, so otsu, adaptive,
    and blackhat outputs do not replace one another. Returns paths to the
    saved mask and comparison image.
    """
    source = Path(input_path)
    image = load_image(source)
    gray = denoise(to_grayscale(image))
    mask = segment(gray, method=method)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    mask_path = destination / f"{source.stem}_{method}_mask.png"
    comparison_path = destination / f"{source.stem}_{method}_comparison.png"

    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"Failed to write mask: {mask_path}")
    save_comparison(image, mask, comparison_path, method=method)

    return {"mask": mask_path, "comparison": comparison_path}
