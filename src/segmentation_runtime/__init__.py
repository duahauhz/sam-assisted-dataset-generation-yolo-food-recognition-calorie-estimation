"""Segmentation runtime — YOLO-seg inference + mask cropping.

This folder is the bridge between the trained YOLO26-seg checkpoint
(``runs/yolo_seg/ecustfd_yolo26seg-2/weights/best.pt``) and the
geometry pipeline (volume formulas, calorie). Concretely:

- ``yolo_infer.py`` — load the model and run inference on images,
  returning a uniform list of detection dicts.
- ``crop_to_bbox.py`` — crop full-image masks to the detection bbox,
  the same way the original paper cropped GrabCut masks to the
  Faster R-CNN box.

The rest of the pipeline consumes only the dict API; no Ultralytics
imports leak outside this folder.
"""

from .crop_to_bbox import binarize_mask, crop_mask_to_bbox
from .yolo_infer import (
    _CLASS_NAMES,
    load_yolo_seg,
    predict_many,
    predict_one,
)

__all__ = [
    "load_yolo_seg",
    "predict_one",
    "predict_many",
    "crop_mask_to_bbox",
    "binarize_mask",
    "_CLASS_NAMES",
]
