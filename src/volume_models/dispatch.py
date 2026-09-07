"""Dispatch + shared inputs for the per-shape volume formulas."""

from __future__ import annotations

from dataclasses import dataclass # Xử lý dataclass
from typing import Literal # Xử lý literal

import numpy as np # Thư viện numpy

from .shape_models import (# Models volume
    volume_column, # Model column
    volume_ellipsoid, # Model ellipsoid
    volume_grape, # Model grape
    volume_torus, # Model torus
    volume_unknown, # Model unknown
)

ShapeName = Literal["ellipsoid", "column", "unknown", "grape", "torus"] # Literal types

_SHAPE_TO_FN = {
    "ellipsoid": volume_ellipsoid,
    "column": volume_column,
    "unknown": volume_unknown,
    "grape": volume_grape,
    "torus": volume_torus,
}


@dataclass(frozen=True)
class ShapeInputs:
    """Container for the input pair (top mask, side mask) plus per-view scales.

    ``alpha_t`` and ``alpha_s`` are derived independently from the coin
    bounding box detected in each view (``alpha = 2.5 cm / mean(W, H)``).
    Units are cm/pixel.
    """

    top_mask: np.ndarray  # (H_T, W_T) bool
    side_mask: np.ndarray  # (H_S, W_S) bool
    alpha_t: float # scale top
    alpha_s: float # scale side

# Tính toán thể tích
def compute_volume(shape: str, inputs: ShapeInputs) -> float:
    """Dispatch to the correct shape formula.

    Raises ``ValueError`` for unknown shape names so that a typo in
    ``SHAPE_MODELS`` is caught immediately rather than silently
    returning 0.
    """
    fn = _SHAPE_TO_FN.get(shape) # Lấy model
    if fn is None: # Kiểm tra shape
        raise ValueError( # Raise error
            f"Unknown shape '{shape}'. "
            f"Expected one of {sorted(_SHAPE_TO_FN)}."
        )

    if inputs.top_mask.dtype != bool: # Kiểm tra mask top
        top_mask = inputs.top_mask.astype(bool) # Ép kiểu mask top
    else:
        top_mask = inputs.top_mask # Gán mask top
    if inputs.side_mask.dtype != bool: # Kiểm tra mask side
        side_mask = inputs.side_mask.astype(bool) # Ép kiểu mask side
    else:
        side_mask = inputs.side_mask # Gán mask side

    if shape == "ellipsoid": # Nếu shape là ellipsoid
        return float(fn(side_mask, inputs.alpha_s)) # Tính toán thể tích
    return float(fn(top_mask, inputs.alpha_t, side_mask, inputs.alpha_s)) # Tính toán thể tích
