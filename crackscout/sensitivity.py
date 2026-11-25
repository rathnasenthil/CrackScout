"""Fixed black-hat parameter grid.

The grid is a controlled list, not a search. Nothing here ranks a setting
or treats mask statistics as crack-detection accuracy.
"""

from pathlib import Path

import cv2
import numpy as np

from crackscout.io import load_image
from crackscout.preprocess import denoise, to_grayscale
from crackscout.segment import segment
from crackscout.visualize import save_comparison

# Current black-hat defaults are (15, 25). The other pairs change one
# parameter at a time. This list is not a tuning procedure.
BLACKHAT_SENSITIVITY_CONFIGS: tuple[tuple[int, int], ...] = (
    (15, 25),
    (9, 25),
    (21, 25),
    (15, 15),
    (15, 35),
)

LIMITATIONS = """\
Limitations of these measurements:
- White-pixel percentage counts every foreground pixel. Cracks, shadows, stains, and joints are not distinguished.
- Connected components use 8-connectivity. Regions that touch, including along a diagonal, merge into one component. Breaks in a region increase the count. The count is not a count of cracks.
- These figures describe the binary mask only. They do not measure detection accuracy, and they do not show whether a shadow was removed.
- The five (kernel_size, threshold) pairs are fixed. This script does not rank them or select a parameter.
"""


def describe_mask(mask: np.ndarray) -> dict[str, int | float]:
    """Count white pixels and 8-connected foreground components.

    The mask must already be a single-channel image of 0 and 255.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("Expected a single-channel uint8 mask")
    if np.any((mask != 0) & (mask != 255)):
        raise ValueError("Mask must contain only 0 and 255")
    total = int(mask.size)
    if total == 0:
        raise ValueError("Mask has no pixels")
    white = int(np.count_nonzero(mask == 255))
    label_count, _ = cv2.connectedComponents(mask, connectivity=8)
    return {
        "white_pixels": white,
        "total_pixels": total,
        "white_percent": 100.0 * white / total,
        "connected_components": int(label_count - 1),
    }


def config_slug(kernel_size: int, threshold: int) -> str:
    """Return the filename fragment for one black-hat configuration."""
    return f"blackhat_k{kernel_size}_t{threshold}"


def planned_outputs(
    stem: str, output_dir: str | Path
) -> list[dict[str, object]]:
    """Return the files this experiment would write, without creating them."""
    destination = Path(output_dir)
    planned: list[dict[str, object]] = []
    for kernel_size, threshold in BLACKHAT_SENSITIVITY_CONFIGS:
        slug = config_slug(kernel_size, threshold)
        planned.append(
            {
                "kernel_size": kernel_size,
                "threshold": threshold,
                "mask": destination / f"{stem}_{slug}_mask.png",
                "comparison": destination / f"{stem}_{slug}_comparison.png",
            }
        )
    return planned


def summary_path(stem: str, output_dir: str | Path) -> Path:
    """Return the path of the text report for this input stem."""
    return Path(output_dir) / f"{stem}_blackhat_sensitivity.txt"


def format_report(input_path: str | Path, rows: list[dict[str, object]]) -> str:
    """Format the measurement table and its limitations."""
    lines = [
        "Black-hat parameter sensitivity",
        f"Input: {input_path}",
        "Preprocessing: grayscale conversion and denoise(), shared with run_baseline.py.",
        "Segmentation: segment(method='blackhat') at the kernel size and threshold below.",
        "White percent is 100 * (pixels equal to 255) / (all pixels).",
        "Connected components are 8-connected foreground regions. Background is excluded.",
        "",
        f"{'kernel':>6}  {'threshold':>9}  {'white_px':>10}  {'white_%':>10}  {'components':>11}",
    ]
    for row in rows:
        lines.append(
            f"{row['kernel_size']:>6}  {row['threshold']:>9}  "
            f"{row['white_pixels']:>10}  {row['white_percent']:>10.4f}  "
            f"{row['connected_components']:>11}"
        )
        lines.append(f"         mask: {row['mask']}")
        lines.append(f"         comparison: {row['comparison']}")
    lines.extend(["", LIMITATIONS.rstrip(), ""])
    return "\n".join(lines)


def _refuse_existing(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    listed = "\n".join(f"  {path}" for path in existing)
    raise FileExistsError(
        "Refusing to overwrite existing output:\n"
        f"{listed}\n"
        "Choose another --output-dir. This experiment does not replace existing files."
    )


def run_blackhat_sensitivity(
    input_path: str | Path,
    output_dir: str | Path = "outputs",
) -> list[dict[str, object]]:
    """Run the fixed black-hat grid and write masks, figures, and a report.

    Raises FileExistsError if any planned file is already present.
    """
    source = Path(input_path)
    image = load_image(source)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    planned = planned_outputs(source.stem, destination)
    report_path = summary_path(source.stem, destination)
    reserved = [report_path]
    for item in planned:
        reserved.append(item["mask"])
        reserved.append(item["comparison"])
    _refuse_existing(reserved)

    gray = denoise(to_grayscale(image))
    rows: list[dict[str, object]] = []
    for item in planned:
        kernel_size = int(item["kernel_size"])
        threshold = int(item["threshold"])
        mask = segment(
            gray,
            method="blackhat",
            kernel_size=kernel_size,
            threshold=threshold,
        )
        stats = describe_mask(mask)
        mask_path = item["mask"]
        comparison_path = item["comparison"]
        if not cv2.imwrite(str(mask_path), mask):
            raise OSError(f"Failed to write mask: {mask_path}")
        label = f"blackhat, kernel {kernel_size}, threshold {threshold}"
        save_comparison(image, mask, comparison_path, method=label)
        rows.append(
            {
                "kernel_size": kernel_size,
                "threshold": threshold,
                "white_pixels": stats["white_pixels"],
                "total_pixels": stats["total_pixels"],
                "white_percent": stats["white_percent"],
                "connected_components": stats["connected_components"],
                "mask": mask_path,
                "comparison": comparison_path,
            }
        )

    report_path.write_text(format_report(source, rows), encoding="utf-8")
    return rows
