"""Runtime MATLAB baseline calorie estimation.

The runtime MATLAB code (``ECUSTFD/faster_rcnn/faster_rcnn_rec.m``)
does not apply the paper's beta correction. Instead it uses the
``q_k`` factor from ``food_info.xls`` directly:

    C = q_k * v_tilde       (kcal)

where ``q_k`` is the per-class "kcal per cm^3" factor that
pre-multiplies density and energy:

    q_k = rho_k * c_k       (paper-consistent definition)

This module exists to allow a direct head-to-head comparison with the
runtime MATLAB baseline (no beta correction). For the paper-faithful
estimate, use ``paper_formula.py``.
"""

from __future__ import annotations

from dataclasses import dataclass # Xử lý dataclass
from typing import Dict, Optional # Xử lý kiểu dữ liệu

from .food_info_xls import get_q_factor # Lấy q factor


@dataclass(frozen=True) # Tạo dataclass
class CalorieResultRuntime:
    """Result of one runtime-style calorie estimation."""

    class_name: str # Tên lớp
    v_tilde_cm3: float # Thể tích
    q_kcal_per_cm3: float # Q factor
    kcal: float # Calo

# Ước tính calo
def estimate_calorie_runtime(
    class_name: str, # Tên lớp
    v_tilde_cm3: float, # Thể tích
    food_info: Dict[str, Dict[str, float | str]], # Thông tin thực phẩm
) -> CalorieResultRuntime:# Trả về CalorieResultRuntime
    """Estimate kcal using the runtime MATLAB formula.

    Args:
        class_name: e.g. ``"apple"``.
        v_tilde_cm3: raw geometric volume (cm^3).
        food_info: parsed ``food_info.xls`` (from ``parse_food_info``).

    Returns:
        ``CalorieResultRuntime`` with the q factor and kcal.

    Raises:
        KeyError: if the class is missing from ``food_info`` or its
            ``kcal_per_cm3`` value is ``None``.
    """
    q = get_q_factor(food_info, class_name) # Lấy q factor
    if q is None: # Kiểm tra q factor có tồn tại không
        raise KeyError(
            f"Missing kcal_per_cm3 for class '{class_name}' in food_info."
        )

    kcal = float(q) * float(v_tilde_cm3) # Tính calo
    return CalorieResultRuntime(
        class_name=class_name,
        v_tilde_cm3=float(v_tilde_cm3),
        q_kcal_per_cm3=float(q),
        kcal=kcal,
    )


__all__ = ["CalorieResultRuntime", "estimate_calorie_runtime"]
