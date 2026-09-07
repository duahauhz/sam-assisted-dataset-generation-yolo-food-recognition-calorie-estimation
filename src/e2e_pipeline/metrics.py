"""Mean Error (ME) metrics, paper-faithful.

Paper Eq. in §4 reports signed mean error per class:

    ME_volume(k) = (1 / n_k) * Σ_i (v_i − V_i) / V_i
    ME_mass(k)   = (1 / n_k) * Σ_i (m_i − M_i) / M_i

These are *signed* means (not absolute), and the final number is
multiplied by 100 to report as a percentage.

The runtime MATLAB code uses the same definition but multiplies by
100 at the end. We mirror that.

Filtering:
- The paper drops "misidentified" images from ME. We replicate by
  excluding samples whose pair was filtered out by the confidence
  threshold.
- A sample is counted only if both ``v_pred`` and ``v_real`` are
  finite and ``v_real > 0``.
"""

from __future__ import annotations

import math # import math
from dataclasses import dataclass, field # dataclass and field
from typing import Dict, List, Sequence # type hint

#
@dataclass(frozen=True)
class PerClassResult: # Kết quả cho mỗi class
    """Mean error summary for one class."""

    class_name: str # Tên class
    n_samples: int # Số lượng mẫu
    me_volume_pct: float | None # sai số trung bình thể tích
    me_mass_pct: float | None # sai số trung bình khối lượng
    abs_me_volume_pct: float | None  # sai số trung bình tuyệt đối thể tích


@dataclass(frozen=True)
class MetricReport: # Báo cáo tổng hợp
    """Aggregate metric report across all classes."""

    per_class: List[PerClassResult] # Kết quả cho mỗi class
    overall: Dict[str, float] # Kết quả tổng hợp


def _safe_mean(values: List[float]) -> float | None: # Hàm tính trung bình
    if not values:# Nếu không có giá trị
        return None # Trả về None
    return sum(values) / len(values) # Tính trung bình

# Hàm tính toán me per class cho từng class
def compute_me_per_class(
    class_names: Sequence[str], # Tên class
    v_pred: Sequence[float], # Thể tích dự đoán
    v_real: Sequence[float], # Thể tích thực tế
    m_pred: Sequence[float] | None = None, # Khối lượng dự đoán
    m_real: Sequence[float] | None = None, # Khối lượng thực tế
) -> List[PerClassResult]:
    """Compute the paper's Mean Error per class.

    Args:
        class_names: e.g. ``["apple", "apple", "banana", ...]``.
        v_pred: predicted volumes (cm^3).
        v_real: reference volumes (cm^3).
        m_pred: optional predicted masses (g).
        m_real: optional reference masses (g).

    Returns:
        Per-class ``PerClassResult`` (only classes with at least one
        valid sample appear).
    """
    have_mass = (# Kiểm tra có khối lượng không
        m_pred is not None# Khối lượng dự đoán không rỗng
        and m_real is not None# Khối lượng thực tế không rỗng
        and len(m_pred) == len(class_names)# Khối lượng dự đoán có cùng số lượng với class names
        and len(m_real) == len(class_names)# Khối lượng thực tế có cùng số lượng với class names
    )

    by_cls: Dict[str, Dict[str, List[float]]] = {} # Dictionary để lưu trữ kết quả theo class
    for i, cls in enumerate(class_names): # Lặp qua từng class
        cls = str(cls)# Chuyển class sang string
        v_p = float(v_pred[i])# Chuyển thể tích dự đoán sang float
        v_r = float(v_real[i])# Chuyển thể tích thực tế sang float
        if not (math.isfinite(v_p) and math.isfinite(v_r) and v_r > 0):# Kiểm tra thể tích dự đoán và thực tế có hợp lệ không
            continue # Bỏ qua nếu không hợp lệ
        bucket = by_cls.setdefault(# Lấy kết quả theo class
            cls, {"me_vol": [], "abs_me_vol": [], "me_mass": []}
        )
        rel = (v_p - v_r) / v_r # Tính sai số trung bình
        bucket["me_vol"].append(rel) # Thêm sai số trung bình vào dictionary
        bucket["abs_me_vol"].append(abs(rel)) # Thêm sai số trung bình tuyệt đối vào dictionary
        if have_mass: # Nếu có khối lượng
            m_p = float(m_pred[i])  # type: ignore[index]
            m_r = float(m_real[i])  # type: ignore[index]
            if math.isfinite(m_p) and math.isfinite(m_r) and m_r > 0: # Kiểm tra khối lượng dự đoán và thực tế có hợp lệ không
                bucket["me_mass"].append((m_p - m_r) / m_r) # Thêm sai số trung bình khối lượng vào dictionary

    out: List[PerClassResult] = [] # Danh sách kết quả cho mỗi class
    for cls in sorted(by_cls): # Sắp xếp kết quả theo class
        b = by_cls[cls] # Lấy kết quả theo class
        out.append(# Thêm kết quả vào danh sách
            PerClassResult(
                class_name=cls,# Tên class
                n_samples=len(b["me_vol"]), # Số lượng mẫu
                me_volume_pct=(# Sai số trung bình thể tích
                    100.0 * _safe_mean(b["me_vol"]) if b["me_vol"] else None
                ),
                me_mass_pct=(# Sai số trung bình khối lượng
                    100.0 * _safe_mean(b["me_mass"]) if b["me_mass"] else None
                ),
                abs_me_volume_pct=(# Sai số trung bình tuyệt đối thể tích
                    100.0 * _safe_mean(b["abs_me_vol"]) if b["abs_me_vol"] else None
                ),
            )
        )
    return out

# Tổng hợp kết quả
def aggregate(report: List[PerClassResult]) -> Dict[str, float]:
    """Aggregate per-class results into a single overall number.

    Both ME and abs-ME are averaged as the simple mean of the per-class
    numbers (each class counts equally, regardless of sample count).
    This matches the "average per class" rows in the paper's tables.
    """
    if not report: # Nếu không có kết quả
        return {
            "mean_me_volume_pct": float("nan"),# Sai số trung bình thể tích
            "mean_abs_me_volume_pct": float("nan"),# Sai số trung bình tuyệt đối thể tích
            "mean_me_mass_pct": float("nan"),# Sai số trung bình khối lượng
            "n_classes": 0,# Số lượng class
        }

    me_v = [r.me_volume_pct for r in report if r.me_volume_pct is not None]# Sai số trung bình thể tích
    abs_me_v = [r.abs_me_volume_pct for r in report if r.abs_me_volume_pct is not None]# Sai số trung bình tuyệt đối thể tích
    me_m = [r.me_mass_pct for r in report if r.me_mass_pct is not None]# Sai số trung bình khối lượng

    return {
        "mean_me_volume_pct": sum(me_v) / len(me_v) if me_v else float("nan"),# Trung bình sai số trung bình thể tích 
        "mean_abs_me_volume_pct": sum(abs_me_v) / len(abs_me_v) if abs_me_v else float("nan"),# Trung bình sai số trung bình tuyệt đối thể tích
        "mean_me_mass_pct": sum(me_m) / len(me_m) if me_m else float("nan"),# Trung bình sai số trung bình khối lượng
        "n_classes": len(report),# Số lượng class
    }


__all__ = [
    "PerClassResult",
    "MetricReport",
    "compute_me_per_class",
    "aggregate",
]
