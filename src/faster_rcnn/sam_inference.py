# -*- coding: utf-8 -*-
"""SAM1-based segmentation for Faster R-CNN detections.

This module is the SAM-backed alternative to GrabCut
(``src/faster_rcnn/inference.py``). It uses the same Faster R-CNN bbox
detector, but replaces the OpenCV GrabCut segmentation with **SAM1
(vit_b) box-prompted** segmentation.

Architecture (mirrors ``src/faster_rcnn/inference.py``):
1. Faster R-CNN detects ``bbox`` + ``class`` (torchvision).
2. SAM1 ``set_image`` is called once per image.
3. For each food detection, SAM1 ``predict(box=det_bbox)`` returns a
   boolean mask shaped ``(H, W)``.
4. The mask is cropped to the bbox region (same shape as GrabCut output).
5. Coin detections get ``mask=None`` (no SAM needed).

The output detection dicts are **identical in schema** to the GrabCut
output (``class_name``, ``conf``, ``bbox``, ``mask``), so the downstream
pipeline (coin calibration, view pairing, volume formulas, beta
correction, calorie estimation, metrics) needs **zero changes**.

References
----------
- SAM1 model: ``models/sam/sam_vit_b_01ec64.pth`` (vit_b, ~375 MB).
- Package: ``segment_anything`` (Meta AI).
- Existing SAM1 box-prompt script (uses GT bboxes):
  ``src/data_prep_SAM1/sam_masks_full/segment_sam1_box.py`` (lines 144-165).
  Runtime uses *detected* bboxes, which is the principled choice for
  evaluation.
- Pre-computed SAM masks (GT-bbox-prompted, 2978 ``.npy`` files, mean
  mIoU 0.743):
  ``data/processed/sam_masks_full/masks/<stem>.npy``.
  These are NOT directly usable here because the bbox source differs;
  per-image SAM at runtime is the canonical option.

Pipeline cost (per image, RTX-class GPU)
----------------------------------------
- Faster R-CNN forward:  ~30 ms.
- SAM image encoding:    ~150 ms (vit_b).
- SAM mask per detection: ~30 ms.
- Total: ~250 ms per image.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

# Reuse the Faster R-CNN tensor -> detection dict conversion from inference.py.
# The contract is the same: torchvision Faster R-CNN output -> our dict schema.
from .inference import _torch_to_detections, _normalize_device

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default paths (mirror src/data_prep_SAM1/sam_masks_full/segment_sam1_box.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("E:/AI_Research/dlt8").resolve()
DEFAULT_SAM_CHECKPOINT = PROJECT_ROOT / "models" / "sam" / "sam_vit_b_01ec64.pth"
DEFAULT_SAM_MODEL_TYPE = "vit_b"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_sam(
    checkpoint_path: Optional[Union[str, Path]] = None,
    model_type: str = DEFAULT_SAM_MODEL_TYPE,
    device: Optional[Union[str, int]] = None,
) -> "SamPredictor":
    """Load SAM1 from disk and return a ``SamPredictor``.

    Args:
        checkpoint_path: Path to ``sam_vit_b_01ec64.pth``. Defaults to
            ``E:/AI_Research/dlt8/models/sam/sam_vit_b_01ec64.pth``.
        model_type: SAM backbone. Default ``"vit_b"`` (matches the
            checkpoint shipped in the project).
        device: Torch device. If ``None``, auto-selects CUDA.

    Returns:
        A ``SamPredictor`` ready for ``set_image`` + ``predict``.

    Raises:
        FileNotFoundError: if the checkpoint does not exist.
        ImportError: if the ``segment_anything`` package is not installed.
    """
    # Imported lazily so this module can be imported even when the user
    # only uses GrabCut (segment_anything is an optional heavy dependency).
    from segment_anything import sam_model_registry, SamPredictor

    if checkpoint_path is None:
        checkpoint_path = DEFAULT_SAM_CHECKPOINT
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {checkpoint_path}. "
            f"Download from notebook 01 Cell 0 (URL: "
            f"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)."
        )

    device = _normalize_device(device)

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    sam.eval()

    predictor = SamPredictor(sam)
    _log.info(
        "Loaded SAM1 %s from %s on device=%s",
        model_type, checkpoint_path, device,
    )
    return predictor


# ---------------------------------------------------------------------------
# SAM segmentation (single bbox)
# ---------------------------------------------------------------------------
def _sam_segment_crop(
    predictor: "SamPredictor",
    image_rgb: np.ndarray,
    bbox: Tuple[float, float, float, float],
    *,
    multimask_output: bool = False,
) -> np.ndarray:
    """Run SAM with bbox prompt on the full image; return mask cropped to bbox.

    Pattern mirrors ``segment_sam1_box.py`` (data_prep_SAM1) which uses
    ``multimask_output=False`` (single mask per box, default convention).
    When ``multimask_output=True``, SAM returns 3 masks and we pick the
    one with the highest IoU-predictor score.

    Args:
        predictor: ``SamPredictor`` with ``set_image`` already called.
        image_rgb: Full image as ``np.ndarray`` ``(H, W, 3)``, RGB.
        bbox: ``(x1, y1, x2, y2)`` in image pixel coordinates (0-indexed,
            matches Faster R-CNN output).
        multimask_output: see above.

    Returns:
        Boolean ``(bbox_h, bbox_w)`` array where ``True`` = foreground.
    """
    H, W = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox

    # Clip to image bounds (matches bbox handling in segment_sam1_box.py:148-156).
    x1c = max(0, int(round(x1)))
    y1c = max(0, int(round(y1)))
    x2c = min(W, int(round(x2)))
    y2c = min(H, int(round(y2)))

    if x2c <= x1c or y2c <= y1c:
        # Degenerate bbox after clipping — return empty mask.
        return np.zeros((max(1, y2c - y1c), max(1, x2c - x1c)), dtype=bool)

    box = np.array([x1c, y1c, x2c, y2c], dtype=np.float32)

    with torch.no_grad():
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=multimask_output,
        )

    if multimask_output:
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(bool)
    else:
        mask = masks[0].astype(bool)

    # Crop to bbox region (same contract as GrabCut output).
    return mask[y1c:y2c, x1c:x2c]


def apply_sam_to_detections(
    detections: List[dict],
    predictor: "SamPredictor",
    image_rgb: np.ndarray,
    *,
    multimask_output: bool = False,
) -> List[dict]:
    """Add SAM masks to detection dicts for food classes (non-coin).

    ``coin`` detections are skipped — the coin is used only for scale
    calibration and does not appear in the volume formula.

    Args:
        detections: List of detection dicts as returned by
            ``_torch_to_detections`` (each has ``class_name``, ``conf``,
            ``bbox``, ``mask=None``).
        predictor: ``SamPredictor`` with ``set_image`` already called.
        image_rgb: Full RGB image (H, W, 3) — same one passed to
            ``set_image``.
        multimask_output: SAM ``multimask_output`` flag.

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
            d["mask"] = _sam_segment_crop(
                predictor, image_rgb, d["bbox"],
                multimask_output=multimask_output,
            )
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_one_with_sam(
    frcnn_model: torch.nn.Module,
    sam_predictor: "SamPredictor",
    image_path: Union[str, Path],
    conf: float = 0.25,
    iou_threshold: float = 0.3,
    device: Optional[Union[str, int]] = None,
    frcnn_device: Optional[Union[str, int]] = None,
    multimask_output: bool = False,
) -> List[dict]:
    """Run Faster R-CNN bbox detection + SAM1 box-prompt segmentation on one image.

    Returns the SAME detection dict schema as
    ``src.faster_rcnn.inference.predict_one`` so the downstream pipeline
    (coin calibration, view pairing, volume formulas, beta correction,
    calorie estimation, metrics) is unchanged.

    Args:
        frcnn_model: Loaded Faster R-CNN model (from ``load_faster_rcnn``).
        sam_predictor: Loaded SAM (from ``load_sam``).
        image_path: Path to the image file.
        conf: Minimum confidence threshold for detections.
        iou_threshold: NMS IoU threshold for Faster R-CNN.
        frcnn_device: Torch device override (default: from model).
        multimask_output: SAM ``multimask_output`` flag (default False,
            matches existing SAM1 box-prompt convention).

    Returns:
        List of detection dicts with keys:
        - ``class_name``: str
        - ``conf``: float
        - ``bbox``: ``(x1, y1, x2, y2)`` in original image coordinates
        - ``mask``: boolean ``(H, W)`` array (foreground pixels) or ``None``
          for coin detections.
    """
    from torchvision.transforms import functional as F_vision

    if frcnn_device is None:
        frcnn_device = device
    if frcnn_device is None:
        frcnn_device = next(frcnn_model.parameters()).device
    else:
        frcnn_device = _normalize_device(frcnn_device)

    # Load image.
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise IOError(f"cv2.imread returned None for: {image_path}")
    orig_h, orig_w = image_bgr.shape[:2]
    orig_size = (orig_w, orig_h)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Faster R-CNN forward.
    tensor = F_vision.to_tensor(image_rgb).to(frcnn_device)
    outputs = frcnn_model([tensor])

    # Apply NMS.
    output = outputs[0]
    keep = torch.ops.torchvision.nms(output["boxes"], output["scores"], iou_threshold)
    output = {
        "boxes": output["boxes"][keep],
        "labels": output["labels"][keep],
        "scores": output["scores"][keep],
    }

    # Convert to detection dicts (mask=None initially).
    detections = _torch_to_detections([output], orig_size)
    detections = [d for d in detections if d["conf"] >= conf]

    # SAM set_image once per image.
    sam_predictor.set_image(image_rgb)

    # Apply SAM to each food detection.
    detections = apply_sam_to_detections(
        detections, sam_predictor, image_rgb,
        multimask_output=multimask_output,
    )

    _log.info(
        "predict_one_with_sam(%s, conf=%.2f): %d dets, classes=%s",
        Path(image_path).name, conf, len(detections),
        {d["class_name"]: sum(1 for x in detections if x["class_name"] == d["class_name"])
         for d in detections}
    )
    return detections


@torch.no_grad()
def predict_many_with_sam(
    frcnn_model: torch.nn.Module,
    sam_predictor: "SamPredictor",
    image_paths: Iterable[Union[str, Path]],
    conf: float = 0.25,
    iou_threshold: float = 0.3,
    frcnn_device: Optional[Union[str, int]] = None,
    multimask_output: bool = False,
) -> dict:
    """Run inference on multiple images.

    Returns:
        Dict mapping ``str(image_path)`` → list of detection dicts.
    """
    out: dict = {}
    for p in image_paths:
        out[str(p)] = predict_one_with_sam(
            frcnn_model, sam_predictor, p,
            conf=conf, iou_threshold=iou_threshold,
            frcnn_device=frcnn_device,
            multimask_output=multimask_output,
        )
    return out


__all__ = [
    "load_sam",
    "predict_one_with_sam",
    "predict_many_with_sam",
    "apply_sam_to_detections",
    "DEFAULT_SAM_CHECKPOINT",
    "DEFAULT_SAM_MODEL_TYPE",
]
