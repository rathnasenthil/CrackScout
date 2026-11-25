"""Save a side-by-side view of the input and the baseline mask."""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_comparison(
    image: np.ndarray,
    mask: np.ndarray,
    path: str | Path,
    method: str,
) -> None:
    """Write a two-panel figure: input image and binary mask.

    The figure labels this as an unevaluated baseline.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image shape {image.shape[:2]} does not match mask shape {mask.shape[:2]}"
        )

    if image.ndim == 2:
        display = image
        cmap = "gray"
    elif image.ndim == 3 and image.shape[2] == 3:
        display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cmap = None
    else:
        raise ValueError(f"Expected a grayscale or BGR image, got shape {image.shape}")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(display, cmap=cmap)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Mask ({method}, unevaluated)")
    axes[1].axis("off")

    figure.suptitle(
        "Classical OpenCV segmentation. This figure is not an accuracy evaluation."
    )
    figure.tight_layout()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)
