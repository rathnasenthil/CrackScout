"""Shared preparation steps used before segmentation."""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to a single-channel uint8 grayscale image."""
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(
            f"Expected a grayscale or BGR image, got shape {image.shape}"
        )
    if gray.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 image, got dtype {gray.dtype}")
    return gray


def denoise(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Blur fine concrete texture before thresholding.

    kernel_size must be a positive odd integer. The default 5x5 kernel is a
    light blur so thin dark lines are less likely to disappear.
    """
    if gray.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {gray.shape}")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be a positive odd integer, got {kernel_size}"
        )
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
