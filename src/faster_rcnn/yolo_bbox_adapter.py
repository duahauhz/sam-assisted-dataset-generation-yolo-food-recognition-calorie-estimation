# -*- coding: utf-8 -*-
"""YOLO26 Bbox-only Adapter for Two-Stage Evaluation (with GrabCut or SAM1).

This module enables evaluating YOLO26 models trained in bbox-only mode
(e.g., ``models/ecustfd_yolo26seg_bbox_best.pt``) with downstream segmentation
backends (GrabCut / SAM1) and the unified calorie/volume estimation pipeline.

Contract:
- Input: Image path or numpy array.
- Output: List of Detection dicts:
    {
        "class_name": str,
        "conf": float,
        "bbox": [x1, y1, x2, y2],  # xyxy in original pixel coords
        "mask": None                # Filled later by GrabCut or SAM
    }
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

_log = logging.getLogger(__name__)


class YOLOBboxDetector:
    """Wrapper around Ultralytics YOLO to output standardized detection dicts."""

    def __init__(self, model_path: Union[str, Path], device: Optional[str] = None):
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model weights not found at: {self.model_path}")

        _log.info("[YOLO Adapter] Loading model from %s...", self.model_path)
        self.model = YOLO(str(self.model_path))
        self.device = device

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        conf: float = 0.25,
        iou: float = 0.50,
        imgsz: int = 480,
    ) -> List[Dict[str, Any]]:
        """Run YOLO bbox inference and return standardized detection dicts."""
        if isinstance(image, (str, Path)):
            img_path = str(image)
        else:
            img_path = image

        results = self.model.predict(
            source=img_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        if not results:
            return detections

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        names = r.names  # dict of {class_id: class_name}

        for box, score, cid in zip(xyxy, confs, cls_ids):
            cname = names.get(cid, str(cid))
            detections.append({
                "class_name": cname,
                "conf": float(score),
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "mask": None,
            })

        return detections
