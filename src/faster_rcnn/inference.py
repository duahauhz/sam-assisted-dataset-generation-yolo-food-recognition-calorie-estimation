# -*- coding: utf-8 -*-
"""Faster R-CNN inference + GrabCut segmentation.

Architecture mirrors the YOLO26-seg pipeline
(``src/segmentation_runtime/yolo_infer.py``):

1. Load ``fasterrcnn_mobilenet_v3_large_320_fpn`` from a trained ``.pt``
   checkpoint (torchvision format, exported via ``train_faster_rcnn.py``).
2. Run forward pass on an image; return ``Detection`` dicts with
   ``class_name``, ``conf``, ``bbox`` (xyxy in original image coords),
   and ``mask`` (None — masks come from GrabCut, see step 3).
3. For each food bbox (non-coin), run GrabCut (OpenCV) to obtain a
   foreground mask inside the bbox.  The mask is cropped to the bbox
   and returned as a boolean ``(H, W)`` array where ``True`` means
   foreground (``GC_PR_FGD`` equivalent).

The output detection dicts are **identical in schema** to the
``src/segmentation_runtime`` API, so the rest of the pipeline
(coin calibration, view pairing, volume formulas, beta correction,
calorie estimation, metrics) needs **zero changes**.

Volume formulas
---------------
All five shape-specific formulas are verbatim ports from the original
MATLAB/C++ baseline (``ECUSTFD/faster_rcnn/grabcut_mex.cpp``).
They are reused from ``src.volume_models`` — no duplication here.

Mask contract
-------------
``mask`` values returned here are boolean ``(H, W)`` arrays where
``True`` corresponds to C++ label ``GC_PR_FGD == 3``.
The ``src.volume_models`` functions expect this format exactly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F_vision

from src.constants import FOOD_CLASSES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Class names in the order used by torchvision COCO-trained Faster R-CNN.
# The model was trained with ``n_classes=21`` (19 foods + coin + background).
# Index 0 is reserved for the "__background__" class by torchvision's convention.
# Indices 1..20 map to the 20 classes from build_dataset.py::_CLASS_NAMES.
_FOOD_LIST: List[str] = [
    "apple", "banana", "bread", "bun", "coin",
    "doughnut", "egg", "fried_dough_twist", "grape",
    "lemon", "litchi", "mango", "mooncake",
    "orange", "peach", "pear", "plum", "qiwi",
    "sachima", "tomato",
]
# torchvision index → class name
# index 0 = background (torchvision built-in)
# indices 1..20 = 20 trained foreground classes (19 foods + coin)
_IDX_TO_NAME: List[str] = ["__background__"] + _FOOD_LIST
# Verify total length is 21 (0=background + 20 trained classes)
assert len(_IDX_TO_NAME) == 21, f"Expected 21 classes, got {len(_IDX_TO_NAME)}"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _normalize_device(device: Union[str, int, None]) -> str:
    """Normalize input device to a string representation."""
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, int):
        return f"cuda:{device}"
    if device.isdigit():
        return f"cuda:{device}"
    return device


def load_faster_rcnn(weights_path: str | Path, device: str | int | None = None) -> torch.nn.Module:
    """Load a Faster R-CNN model and restore weights from a ``.pt`` checkpoint.

    Args:
        weights_path: Path to a trained Faster R-CNN checkpoint
            (created by ``train_faster_rcnn.py``).
        device: Torch device, e.g. ``"cuda:0"``, ``"cpu"``, or ``None``
            (auto-selects CUDA if available).

    Returns:
        A Faster R-CNN model in ``model.eval()`` state.
    """
    n_classes = 21  # 19 foods + coin + background (torchvision convention)
    device_str = _normalize_device(device)

    # Build architecture (same as training).
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

    # Restore weights.
    # Source: ``train_faster_rcnn.py`` saves {"epoch", "model_state", "optimizer_state", "loss"}
    # (see lines 440-444). We check for both formats for robustness.
    # NOTE: weights_only=False is required because the checkpoint contains
    # the optimizer state which uses legacy torch.storage.UntypedStorage
    # (tagged with id 0). torch>=2.6 defaults weights_only=True which
    # refuses this format. The checkpoint is local and trusted.
    ckpt = torch.load(str(weights_path), map_location=device_str, weights_only=False)
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt  # plain state_dict
    model.load_state_dict(state, strict=False)
    model.to(device_str)
    model.eval()

    _log.info("Loaded Faster R-CNN from %s (device=%s)", weights_path, device_str)
    return model


# ---------------------------------------------------------------------------
# GrabCut
# ---------------------------------------------------------------------------

#: Number of GrabCut iterations per call.
_GRABCUT_ITERS = 10


def _grabcut_crop(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    iters: int = _GRABCUT_ITERS,
) -> np.ndarray:
    """Run GrabCut inside a bounding box; return a boolean foreground mask.

    This is a pure-Python port of ``ECUSTFD/faster_rcnn/grabcut_mex.cpp``
    lines 120-135.  The C++ source uses OpenCV's ``grabCut`` with
    ``GC_INIT_WITH_RECT`` on the full image (not cropped first) with the
    rect in absolute image coordinates; we replicate that by passing the
    full image and the full-image rect.

    The foreground mask is extracted as ``mask == cv2.GC_PR_FGD`` (value 3)
    after GrabCut, then cropped to the bbox region.

    Args:
        image: Full BGR image as ``np.ndarray`` (H, W, 3).
        bbox: ``(x1, y1, x2, y2)`` in image pixel coordinates.
        iters: Number of GrabCut iterations.

    Returns:
        Boolean ``(bbox_h, bbox_w)`` array where ``True`` = foreground
        (equivalent to C++ ``GC_PR_FGD == 3``).
    """
    x1, y1, x2, y2 = bbox
    # OpenCV Rect: (x, y, width, height)
    rect = (int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1)))

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Initialise with the bbox rect (GC_INIT_WITH_RECT).
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_RECT)

    # Crop mask to the bbox region.
    crop = mask[int(round(y1)):int(round(y2)), int(round(x1)):int(round(x2))]

    # GC_PR_FGD == 3 in OpenCV.  Return boolean: True = foreground.
    return (crop == cv2.GC_PR_FGD) | (crop == cv2.GC_FGD)


def apply_grabcut_to_detections(
    detections: List[dict],
    image: np.ndarray,
) -> List[dict]:
    """Add GrabCut masks to detection dicts for food classes (non-coin).

    ``coin`` detections are skipped — the coin is used only for scale
    calibration and does not appear in the volume formula.

    Args:
        detections: List of detection dicts as returned by
            ``predict_one`` (each has ``class_name``, ``conf``, ``bbox``,
            ``mask=None``).
        image: Full BGR image as ``np.ndarray`` (H, W, 3).

    Returns:
        Same list with ``mask`` populated for food-class detections.
        Coin detections retain ``mask=None``.
    """
    out: List[dict] = []
    for d in detections:
        d = dict(d)  # shallow copy
        if d["class_name"] == "coin":
            d["mask"] = None
        else:
            d["mask"] = _grabcut_crop(image, d["bbox"])
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _log_transform_image(image_path: str | Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Load an image from disk, convert to BGR (OpenCV format), return with original size."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise IOError(f"cv2.imread returned None for: {image_path}")
    orig_h, orig_w = image.shape[:2]
    return image, (orig_w, orig_h)


def _torch_to_detections(
    torch_outputs: list[dict],
    orig_size: tuple[int, int],
) -> List[dict]:
    """Convert torchvision Faster R-CNN output to our standard detection dicts.

    Bboxes are returned in the original image coordinate space
    (not the normalised [0,1] or transformed space) so they can be
    used directly for GrabCut and for coordinate comparison with the
    YOLO-seg pipeline.
    """
    detections: List[dict] = []
    for output in torch_outputs:
        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        orig_w, orig_h = orig_size

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            cls_idx = int(label)
            if cls_idx >= len(_IDX_TO_NAME):
                continue
            class_name = _IDX_TO_NAME[cls_idx]
            if class_name == "__background__":
                continue

            detections.append({
                "class_name": class_name,
                "conf": float(score),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "mask": None,
            })
    return detections


@torch.no_grad()
def predict_one(
    model: torch.nn.Module,
    image_path: str | Path,
    conf: float = 0.25,
    iou_threshold: float = 0.3,
    device: str | int | None = None,
) -> List[dict]:
    """Run Faster R-CNN inference + GrabCut on a single image.

    Args:
        model: Loaded Faster R-CNN model (from ``load_faster_rcnn``).
        image_path: Path to the image file.
        conf: Minimum confidence threshold.  Detections below this are dropped.
        iou_threshold: NMS IoU threshold (passed to torchvision).
        device: Torch device override.  If ``None`` the device the model
            was loaded on is used.

    Returns:
        List of detection dicts with keys:
        - ``class_name``: str
        - ``conf``: float
        - ``bbox``: ``(x1, y1, x2, y2)`` in original image coordinates
        - ``mask``: boolean ``(H, W)`` array (foreground pixels) or ``None``
          for coin detections.
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = _normalize_device(device)

    # Load image and convert to tensor.
    image = cv2.imread(str(image_path))
    if image is None:
        raise IOError(f"cv2.imread returned None for: {image_path}")
    orig_h, orig_w = image.shape[:2]
    orig_size = (orig_w, orig_h)

    # Convert BGR → RGB and make CHW tensor.
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = F_vision.to_tensor(rgb).to(device)

    # Forward pass.
    outputs = model([tensor])

    # Apply confidence filter and NMS.
    output = outputs[0]
    keep = torch.ops.torchvision.nms(output["boxes"], output["scores"], iou_threshold)
    output = {
        "boxes": output["boxes"][keep],
        "labels": output["labels"][keep],
        "scores": output["scores"][keep],
    }

    # Convert to detection dicts.
    detections = _torch_to_detections([output], orig_size)

    # Confidence filter.
    detections = [d for d in detections if d["conf"] >= conf]

    # Apply GrabCut for food classes.
    detections = apply_grabcut_to_detections(detections, image)

    _log.info(
        "predict_one(%s, conf=%.2f): %d dets, classes=%s",
        Path(image_path).name, conf, len(detections),
        {d["class_name"]: sum(1 for x in detections if x["class_name"] == d["class_name"])
         for d in detections}
    )
    return detections


@torch.no_grad()
def predict_many(
    model: torch.nn.Module,
    image_paths: Iterable[str | Path],
    conf: float = 0.25,
    iou_threshold: float = 0.3,
    device: str | int | None = None,
) -> dict[str, List[dict]]:
    """Run inference on multiple images.

    Args:
        model: Loaded Faster R-CNN model.
        image_paths: Iterable of image file paths.
        conf: Confidence threshold.
        iou_threshold: NMS IoU threshold.
        device: Torch device override.

    Returns:
        Dict mapping ``str(image_path)`` → list of detection dicts.
    """
    out: dict[str, List[dict]] = {}
    for p in image_paths:
        out[str(p)] = predict_one(model, p, conf=conf, iou_threshold=iou_threshold, device=device)
    return out


# ---------------------------------------------------------------------------
# Module init
# ---------------------------------------------------------------------------

_log = logging.getLogger(__name__)

__all__ = [
    "load_faster_rcnn",
    "predict_one",
    "predict_many",
    "apply_grabcut_to_detections",
]


# Backward-compatible alias
load_model = load_faster_rcnn
