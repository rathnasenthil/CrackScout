"""Segmentation methods.

Add a later method by writing a function that accepts a grayscale image and
returns a uint8 mask (0 background, 255 foreground), then register it in
SEGMENTERS. The baseline script selects a method by name.
"""

from collections.abc import Callable

import cv2
import numpy as np

Segmenter = Callable[..., np.ndarray]


def _require_gray_uint8(gray: np.ndarray) -> np.ndarray:
    if gray.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {gray.shape}")
    if gray.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 grayscale image, got dtype {gray.dtype}")
    return gray


def _kernel_size(size: int, name: str) -> int:
    if size < 1:
        raise ValueError(f"{name} must be >= 1, got {size}")
    return size


def _require_odd_size(size: int, name: str, minimum: int) -> int:
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError(f"{name} must be an integer, got {size!r}")
    if size < minimum or size % 2 == 0:
        raise ValueError(
            f"{name} must be an odd integer greater than or equal to {minimum}, got {size}"
        )
    return size


def _require_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return float(value)


def _require_unit_threshold(value: int, name: str = "threshold") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < 0 or value > 255:
        raise ValueError(f"{name} must be from 0 through 255, got {value}")
    return value


def _require_binary_mask(mask: np.ndarray) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("Segmentation must return a single-channel uint8 mask")
    if np.any((mask != 0) & (mask != 255)):
        raise ValueError("Segmentation mask must contain only 0 and 255")
    return mask


def segment_otsu(
    gray: np.ndarray,
    *,
    open_kernel: int = 3,
    close_kernel: int = 5,
) -> np.ndarray:
    """Threshold dark pixels with Otsu, then clean the mask.

    Assumes the structure of interest is darker than the surrounding concrete.
    Opening removes small bright specks. Closing bridges short gaps.
    The result is an unevaluated binary mask, not a verified crack map.
    """
    _require_gray_uint8(gray)
    open_size = _kernel_size(open_kernel, "open_kernel")
    close_size = _kernel_size(close_kernel, "close_kernel")

    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    open_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_size, open_size)
    )
    close_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_element)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_element)


_ADAPTIVE_METHODS = {
    "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    "mean": cv2.ADAPTIVE_THRESH_MEAN_C,
}


def segment_adaptive(
    gray: np.ndarray,
    *,
    block_size: int = 35,
    C: float = 5,
    adaptive_method: str = "gaussian",
) -> np.ndarray:
    """Mark pixels darker than a local mean or Gaussian-weighted mean.

    THRESH_BINARY_INV turns those darker pixels into 255. block_size is the
    odd neighborhood width and must be at least 3. C is subtracted from the
    local estimate, so a positive C keeps pixels that are darker than their
    neighborhood. The mask is an unevaluated candidate, not a crack score.
    """
    _require_gray_uint8(gray)
    block = _require_odd_size(block_size, "block_size", minimum=3)
    constant = _require_number(C, "C")
    if not isinstance(adaptive_method, str):
        raise ValueError(
            f"adaptive_method must be a string, got {adaptive_method!r}"
        )
    try:
        adaptive_flag = _ADAPTIVE_METHODS[adaptive_method]
    except KeyError:
        known = ", ".join(sorted(_ADAPTIVE_METHODS))
        raise ValueError(
            f"adaptive_method must be one of: {known}. Got {adaptive_method!r}."
        ) from None
    return cv2.adaptiveThreshold(
        gray,
        255,
        adaptive_flag,
        cv2.THRESH_BINARY_INV,
        block,
        constant,
    )


def segment_blackhat(
    gray: np.ndarray,
    *,
    kernel_size: int = 15,
    threshold: int = 25,
) -> np.ndarray:
    """Highlight dark features that are smaller than the structuring element.

    Black-hat is the closing of the image minus the image. threshold is the
    cutoff on that response, from 0 through 255. kernel_size is the odd
    ellipse width and must be at least 3. The mask is an unevaluated
    candidate, not a crack score.
    """
    _require_gray_uint8(gray)
    size = _require_odd_size(kernel_size, "kernel_size", minimum=3)
    cutoff = _require_unit_threshold(threshold)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    response = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, element)
    _, mask = cv2.threshold(response, cutoff, 255, cv2.THRESH_BINARY)
    return mask


SEGMENTERS: dict[str, Segmenter] = {
    "otsu": segment_otsu,
    "adaptive": segment_adaptive,
    "blackhat": segment_blackhat,
}


def available_methods() -> list[str]:
    """Return registered segmentation method names."""
    return sorted(SEGMENTERS)


def segment(gray: np.ndarray, method: str = "otsu", **kwargs: object) -> np.ndarray:
    """Run one registered segmentation method on a grayscale image."""
    try:
        segmenter = SEGMENTERS[method]
    except KeyError:
        known = ", ".join(available_methods()) or "(none)"
        raise ValueError(
            f"Unknown segmentation method '{method}'. Available: {known}"
        ) from None
    return _require_binary_mask(segmenter(gray, **kwargs))
