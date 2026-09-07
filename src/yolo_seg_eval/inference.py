# -*- coding: utf-8 -*-
"""YOLO26-seg inference wrapper for evaluation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.segmentation_runtime.crop_to_bbox import crop_mask_to_bbox 
from src.segmentation_runtime.yolo_infer import _results_to_detections, _CLASS_NAMES 

_log = logging.getLogger(__name__)

# Helper function để chuẩn hóa device
def _normalize_device(device: Union[str, int, None]) -> str: 
    """Normalize input device for PyTorch and Ultralytics."""
    if device is None: # Nếu device là None
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if isinstance(device, int): # Nếu device là int
        return f"cuda:{device}"
    if isinstance(device, str) and device.isdigit():
        return f"cuda:{device}"
    return str(device)

# Load model từ file weights
def load_yolo_seg(weights_path: Union[str, Path], device: Optional[Union[str, int]] = None) -> YOLO:
    """Load a YOLO-seg model from disk."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO-seg model weights not found at: {weights_path}")

    model = YOLO(str(weights_path))
    _log.info("Loaded YOLO-seg model from %s", weights_path)
    return model

# Chạy inference trên một ảnh
def predict_one(
    model: YOLO,# Mô hình YOLO đã được load
    image_path: Union[str, Path],# Đường dẫn đến ảnh
    conf: float = 0.25,# Ngưỡng tin cậy
    iou_threshold: float = 0.50,# Ngưỡng IoU
    imgsz: int = 480,# Kích thước ảnh đầu vào mô hình
    device: Optional[Union[str, int]] = None,# Device để chạy inference
) -> List[Dict[str, Any]]:
    """Run YOLO-seg prediction on a single image and return detection dicts."""
    dev = _normalize_device(device) # Chuẩn hóa device
    p = Path(image_path) # Chuyển đổi đường dẫn sang Path
    if not p.exists(): # Nếu không tìm thấy ảnh
        raise FileNotFoundError(f"Image not found: {p}") # Raise lỗi

    image_bgr = cv2.imread(str(p)) # Đọc ảnh
    if image_bgr is None: # Nếu đọc ảnh thất bại
        raise IOError(f"cv2.imread returned None for: {p}") # Raise lỗi
    orig_h, orig_w = image_bgr.shape[:2] # Lấy chiều cao và chiều rộng ảnh

    # DEBUG: Log conf threshold being used
    _log.debug(f"[INFERENCE] predict_one: {p.name}, conf={conf:.4f}, iou={iou_threshold:.2f}")
    
    results = model.predict(
        source=str(p), # Nguồn ảnh
        conf=conf, # Ngưỡng tin cậy
        iou=iou_threshold, # Ngưỡng IoU
        imgsz=imgsz, # Kích thước ảnh đầu vào mô hình
        device=dev, # Device để chạy inference
        verbose=False, # Tắt chế độ verbose
    )

    if not results or results[0].masks is None or results[0].boxes is None: # Nếu không tìm thấy object nào
        _log.debug(f"[INFERENCE] {p.name}: NO DETECTIONS (masks={results[0].masks is not None if results else False}, boxes={results[0].boxes is not None if results else False})")
        return [] # Trả về list rỗng

    r = results[0]
    _log.debug(f"[INFERENCE] {p.name}: YOLO returned {len(r.boxes)} boxes before processing")
    detections = _results_to_detections(
        r, # Kết quả dự đoán từ YOLO
        class_names=_CLASS_NAMES, # Danh sách tên các lớp
        orig_w=orig_w, # Chiều rộng ảnh gốc
        orig_h=orig_h, # Chiều cao ảnh gốc
        imgsz=imgsz, # Kích thước ảnh đầu vào mô hình
    )

    # Chuẩn hóa tên các lớp
    for d in detections:
        if d.get("class_name") == "fired_dough_twist": # Nếu tên lớp là "fired_dough_twist"
            d["class_name"] = "fried_dough_twist" # Sửa thành "fried_dough_twist"
        elif d.get("class_name") == "kiwi": # Nếu tên lớp là "kiwi"
            d["class_name"] = "qiwi" # Sửa thành "qiwi"

    # In log thông tin dự đoán
    _log.info(
        "[RESULT] predict_one(%s, conf=%.2f): %d dets, classes=%s",
        p.name, conf, len(detections),
        {d["class_name"]: sum(1 for x in detections if x["class_name"] == d["class_name"])
         for d in detections}
    )
    return detections


# Export các hàm cần thiết
__all__ = [
    "load_yolo_seg",
    "predict_one",
    "_normalize_device",
]
