"""Crop a YOLO-seg mask to its bounding box.

The ECUSTFD GrabCut pipeline (in MATLAB) operates on the mask cropped
to the Faster R-CNN food bounding box. We replicate exactly that: a
boolean ``True`` pixel inside the bbox plays the role of the
``GC_PR_FGD == 3`` label.

Why a separate folder "segmentation_runtime":
- The end-to-end pipeline needs a uniform "Detection with mask" API
  regardless of where the mask came from (YOLO-seg inference, pre-
  computed SAM masks, val labels). This folder contains the
  YOLO-seg-specific loader and the mask-cropping helper.
- Pure Python with no Ultralytics dependency in ``crop_to_bbox.py``
  keeps it unit-testable.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def crop_mask_to_bbox(
    mask: np.ndarray,
    bbox: Sequence[float],
) -> np.ndarray:
    """Crop a full-image mask to the bounding box and binarize it.

    Args:
        mask: full-image boolean or uint8 mask of shape ``(H, W)``.
            ``0`` / ``False`` is background, anything else is foreground.
        bbox: ``[x1, y1, x2, y2]`` in pixels. Coordinates outside the
            image are clipped.

    Returns:
        Boolean mask of shape ``(bbox_h, bbox_w)``. Returned dtype
        is ``bool`` (so subsequent shape formulas can use boolean
        indexing directly).

    Raises:
        ValueError: if the bbox is degenerate (zero or negative
            width/height) or fully outside the image.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D (H, W), got shape {mask.shape}")

    img_h, img_w = mask.shape
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Degenerate bbox {tuple(bbox)}: x2 <= x1 or y2 <= y1"
        )

    # Clip to image bounds.
    x1c = max(0, int(round(x1)))
    y1c = max(0, int(round(y1)))
    x2c = min(img_w, int(round(x2)))
    y2c = min(img_h, int(round(y2)))

    if x2c <= x1c or y2c <= y1c:
        raise ValueError(
            f"Bbox {tuple(bbox)} lies entirely outside image of shape "
            f"{mask.shape} after clipping."
        )

    crop = mask[y1c:y2c, x1c:x2c]
    return crop.astype(bool)


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert any 2-D numeric mask to boolean.

    Any non-zero value is treated as foreground. This mimics the
    behaviour of the C++ label ``GC_PR_FGD == 3`` (single integer
    label) for any YOLO-seg output that uses 0/1/255 byte values.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D (H, W), got shape {mask.shape}")
    return mask != 0


__all__ = ["crop_mask_to_bbox", "binarize_mask"]
