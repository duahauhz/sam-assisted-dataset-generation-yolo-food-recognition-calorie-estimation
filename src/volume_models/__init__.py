"""Volume estimation from top + side binary masks.

Pure geometry — no model weights, no I/O. Inputs are boolean numpy arrays
shaped (H, W) for top and side mask crops, plus the per-view scale
alpha (cm/pixel) derived from the coin bounding box. Output is the
raw estimated volume in cm^3.

The five formulas are ported verbatim from the original MATLAB/C++
implementation in ``ECUSTFD/faster_rcnn/grabcut_mex.cpp`` (function
``CalVolume``, lines 181-473). The cross-view top-scale correction
``alpha_T' = min(alpha_T, alpha_S * L_S_max / H_A)`` is also preserved
exactly as in the C++ source — it is not described in the dataset paper
but it is essential for reproducing the runtime numbers.

See ``README.md`` in this folder for the equations and per-shape notes.
"""

from .shape_models import (
    volume_ellipsoid,
    volume_column,
    volume_unknown,
    volume_grape,
    volume_torus,
)
from .dispatch import compute_volume, ShapeInputs

__all__ = [
    "volume_ellipsoid",
    "volume_column",
    "volume_unknown",
    "volume_grape",
    "volume_torus",
    "compute_volume",
    "ShapeInputs",
]
