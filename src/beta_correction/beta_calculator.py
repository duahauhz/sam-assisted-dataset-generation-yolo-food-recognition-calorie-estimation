"""Beta correction module for per-class volume calibration.

This module implements the paper-faithful beta calibration:
    beta_k = sum(V_real_k) / sum(V_est_raw_k)
calculated on the training set for each food class k, matching
`ECUSTFD/faster_rcnn/xls_results_analysis.m` (lines 96-97).

At test time:
    V_calibrated = beta_k * V_est_raw
"""

from __future__ import annotations

import json # Đọc và ghi file json
import math # Thư viện toán học
from pathlib import Path # Thư viện xử lý đường dẫn file
from typing import Dict, List, Sequence, Tuple # Xử lý kiểu dữ liệu



from src.constants import FOOD_CLASSES # Hằng số


# Tính toán beta per class
def compute_class_betas(
    sample_records: Sequence[Dict[str, float | str]],
) -> Dict[str, float]:
    """Compute per-class beta factors from train set sample predictions.

    Args:
        sample_records: List of dicts, each containing:
            - `class_name`: str
            - `v_tilde_cm3`: float (raw estimated volume)
            - `v_real_cm3`: float (ground truth real volume)

    Returns:
        Dict mapping `class_name` -> `beta_k`. If `sum(v_tilde_cm3)` is 0 or
        invalid, returns 1.0.
    """
    class_real: Dict[str, float] = {} # Real volume per class
    class_est: Dict[str, float] = {} # Estimated volume per class

    for rec in sample_records: # lặp qua từng mẫu
        cls_name = str(rec["class_name"]) # Lấy tên class
        v_est = float(rec.get("v_tilde_cm3", 0.0)) # Lấy thể tích ước tính
        v_real = float(rec.get("v_real_cm3", 0.0)) # Lấy thể tích thực tế

        # Skip invalid, zero, or NaN predictions/ground-truths (matches MATLAB line 84 & 124)
        if math.isnan(v_est) or math.isnan(v_real) or v_est <= 0.0 or v_real <= 0.0: # Bỏ qua các giá trị không hợp lệ, bằng 0 hoặc NaN
            continue

        class_real[cls_name] = class_real.get(cls_name, 0.0) + v_real # Cộng dồn thể tích thực tế
        class_est[cls_name] = class_est.get(cls_name, 0.0) + v_est # Cộng dồn thể tích ước tính

    betas: Dict[str, float] = {} # Beta per class
    for cls_name in FOOD_CLASSES: # Lặp qua từng class
        real_sum = class_real.get(cls_name, 0.0) # Lấy tổng thể tích thực tế
        est_sum = class_est.get(cls_name, 0.0) # Lấy tổng thể tích ước tính
        if est_sum > 0.0 and real_sum > 0.0: # Nếu tổng thể tích ước tính và thực tế lớn hơn 0
            betas[cls_name] = float(real_sum / est_sum) # Tính beta
        else:
            betas[cls_name] = 1.0 # Nếu không thỏa mãn thì beta = 1.0

    return betas


# Lưu beta vào file json
def save_betas(betas: Dict[str, float], output_path: Path) -> None:
    """Save computed betas to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True) # Tạo thư mục nếu chưa có
    with output_path.open("w", encoding="utf-8") as f: # Mở file để ghi
        json.dump(betas, f, indent=2) # Lưu beta vào file json


# Load beta từ file json
def load_betas(input_path: Path) -> Dict[str, float]:
    """Load betas from a JSON file."""
    if not input_path.exists(): # Nếu file không tồn tại
        return {} # Trả về dict rỗng
    with input_path.open("r", encoding="utf-8") as f: # Mở file để đọc
        return json.load(f) # Trả về beta


# Hiệu chỉnh thể tích
def calibrate_volume(v_raw: float, class_name: str, betas: Dict[str, float]) -> float:
    """Apply beta calibration factor to raw volume: V_calibrated = beta_k * V_raw."""
    beta = betas.get(class_name, 1.0) # Lấy beta theo class
    return v_raw * beta # Trả về thể tích đã hiệu chỉnh
