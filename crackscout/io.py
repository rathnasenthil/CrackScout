"""Load concrete images from disk."""

from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Load a color image as a BGR uint8 array.

    Raises:
        FileNotFoundError: the path is missing or is not a file.
        ValueError: the path exists but OpenCV cannot decode it.
    """
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {image_path}\n"
            "Add a concrete-crack photo you have rights to use under samples\\, "
            "then pass that file to --input. Example:\n"
            "  python run_baseline.py --input samples\\your_crack_image.jpg"
        )

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}. "
            "The file may be empty, truncated, or not a supported format "
            "(jpg, jpeg, png, bmp, tif, tiff)."
        )
    return image
