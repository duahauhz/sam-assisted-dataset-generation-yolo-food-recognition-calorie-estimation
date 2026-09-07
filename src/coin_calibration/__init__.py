"""Coin-based per-view scale estimation (cm/pixel).

A 1-Yuan coin (diameter 25 mm = 2.5 cm) appears in every ECUSTFD image.
The detection bounding box of the coin gives the pixel diameter; the
ratio 2.5 cm / pixel-diameter is the cm/pixel scale. This module implements
that conversion for a single view (top OR side) and validates the
result.

Why a separate folder:
- The formula is one-liner but is the *only* source of physical units
  in the entire pipeline. Any bug here propagates cubed into volume.
- ``alpha_T`` and ``alpha_S`` are computed independently, so each call
  is single-view.

Coin-diameter convention (matches MATLAB baseline):
-----------------------------------------------
The MATLAB baseline (``faster_rcnn/faster_rcnn_rec.m`` line 130) computes:

    top_pixel = 2.5 / ((y2 + y1 - x2 - x1) / 2)

i.e. it uses ``(y_sum - x_sum) / 2`` rather than ``(W + H) / 2``.
That expression is only correct when the bbox is roughly square
(``(y2 - y1) ≈ (x2 - x1)``); otherwise it silently biases the
estimate. YOLO/Faster R-CNN coin detections on ECUSTFD are nearly
square, so the two conventions agree to within 1-2 %% in practice.

We adopt ``(W + H) / 2`` here (the natural "mean diameter" of a
rectangular bbox) because:
  (a) it is rotation-invariant: a bbox of a rotated coin gives the
      same answer for any orientation;
  (b) it reduces to ``2.5 / W`` when W = H, which matches the
      square-bbox limit that the MATLAB baseline effectively assumes;
  (c) when the coin detection is degenerate (very long or very tall)
      the MATLAB formula would silently blow up; ``(W + H) / 2`` does
      not.

If you want strict MATLAB bit-for-bit parity, swap ``alpha = 2.5 /
((w + h) / 2.0)`` for ``alpha = 2.5 / abs(((y1 + y2) - (x1 + x2)) /
2.0)`` below.
"""

from __future__ import annotations

from dataclasses import dataclass # Xử lý dataclass
from typing import Iterable, List, Sequence # Xử lý kiểu dữ liệu

from src.constants import COIN_DIAMETER_MM, COIN_DIAMETER_CM # Hằng số


@dataclass(frozen=True) # Tạo dataclass
class CoinScale:
    """Result of coin-calibration for one view.

    ``alpha_cm_per_pixel`` is the cm/pixel scale used by the volume
    formulas. ``coin_bbox`` is the original bbox (xyxy) that produced
    the estimate, kept for debugging/visualisation.
    """

    alpha_cm_per_pixel: float # Tỷ lệ đồng xu
    coin_bbox: tuple[float, float, float, float] # Bounding box của coin

# Tính toán chiều rộng và chiều cao của bounding box
def _bbox_width_height(bbox: Sequence[float]) -> tuple[float, float]:
    if len(bbox) != 4: # Kiểm tra bounding box có 4 phần tử không
        raise ValueError(
            f"bbox must have 4 elements [x1, y1, x2, y2], got {len(bbox)}"
        )
    x1, y1, x2, y2 = bbox # Tách bounding box
    return (x2 - x1), (y2 - y1) # Tính toán chiều rộng và chiều cao

# Tính toán tỷ lệ đồng xu từ bounding box của coin
def compute_coin_scale(coin_bbox: Sequence[float]) -> CoinScale:
    """Compute the cm/pixel scale from a single coin bounding box.

    Implements the runtime MATLAB baseline formula
    (``faster_rcnn_rec.m`` line 130, where the coin diameter is 2.5 cm
    — see ``ECUSTFD/paper/1705.07632v3.pdf`` §2.2). The paper itself
    only specifies the diameter, not the formula.

        alpha = 2.5 cm / mean(W_box, H_box)

    See the module docstring for the convention choice (we use
    ``mean(W, H)`` rather than the literal MATLAB expression).

    Args:
        coin_bbox: ``[x1, y1, x2, y2]`` in *original image* coordinates.
            Should be in pixels; any absolute scale cancels out.

    Returns:
        ``CoinScale`` with the alpha and the bbox preserved.

    Raises:
        ValueError: if the bbox is degenerate (zero or negative
            width/height).
    """
    w, h = _bbox_width_height(coin_bbox) # Tính toán chiều rộng và chiều cao
    if w <= 0 or h <= 0: # Kiểm tra chiều rộng và chiều cao
        raise ValueError(
            f"Degenerate coin bbox {tuple(coin_bbox)}: "
            f"width={w}, height={h}. Cannot compute alpha."
        )
    # mean(W, H) — see module docstring for the convention choice.
    alpha = COIN_DIAMETER_CM / ((w + h) / 2.0) # Tính toán tỷ lệ đồng xu
    return CoinScale(alpha_cm_per_pixel=alpha, coin_bbox=tuple(coin_bbox)) # Trả về CoinScale

# Tính toán tỷ lệ đồng xu cho nhiều view
def compute_coin_scales(
    coin_bboxes_per_view: dict[str, Sequence[float] | None],
) -> dict[str, CoinScale | None]:
    """Compute scales for many views in one call.

    Args:
        coin_bboxes_per_view: mapping ``{"top": bbox_xyxy_or_None,
            "side": bbox_xyxy_or_None, ...}``. ``None`` means the view
            has no detected coin (alpha is left as ``None``).

    Returns:
        Mapping with the same keys; each value is either a ``CoinScale``
        or ``None``.
    """
    out: dict[str, CoinScale | None] = {}# Tạo dict để lưu kết quả
    for view, bbox in coin_bboxes_per_view.items(): # Lặp qua từng view
        if bbox is None: # Nếu view không có coin
            out[view] = None
        else:
            out[view] = compute_coin_scale(bbox) # Tính toán tỷ lệ đồng xu
    return out

# Lọc coin từ danh sách các detection
def filter_coin_detections(
    detections: Iterable[dict],
    coin_class_name: str = "coin",
) -> List[dict]:
    """Return only the coin detections from a list of detection dicts.

    Each detection dict must have at least ``"class_name"``. Filters
    whose value differs from ``coin_class_name``.

    The YOLO-seg wrapper in ``src/segmentation_runtime`` produces a
    superset of all detections; this helper is provided so the
    pipeline orchestrator can split coins from foods without coupling
    to the segmentation wrapper's internal naming.
    """
    return [d for d in detections if d.get("class_name") == coin_class_name] # Lọc coin


# Chọn coin có độ tin cậy cao nhất
def select_highest_conf(detections: Sequence[dict]) -> dict | None:
    """Pick the detection with the highest confidence.

    Returns ``None`` if the list is empty. Ties are broken by the
    detection order (stable).
    """
    if not detections: # Kiểm tra danh sách rỗng
        return None # Trả về None
    return max(detections, key=lambda d: float(d.get("conf", 0.0))) # Chọn coin có độ tin cậy cao nhất
