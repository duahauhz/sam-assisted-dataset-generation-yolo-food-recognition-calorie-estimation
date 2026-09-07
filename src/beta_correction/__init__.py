"""Beta correction module for per-class volume calibration.

Matches paper's offline analysis script `ECUSTFD/faster_rcnn/xls_results_analysis.m` (lines 96-97).
"""

from .beta_calculator import (
    calibrate_volume,
    compute_class_betas,
    load_betas,
    save_betas,
)

__all__ = [
    "compute_class_betas",
    "calibrate_volume",
    "save_betas",
    "load_betas",
]
