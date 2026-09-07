"""Calorie estimation — runtime MATLAB baseline + paper-faithful reference.

This folder implements two calorie estimators, both of which follow
the canonical ECUSTFD pipeline:

  (1) ``runtime_formula.py``  — reads ``food_info.xls`` (the on-disk
      q-factor table). This is what ``faster_rcnn_rec.m`` does at
      runtime.

  (2) ``paper_faithful.py``   — recomputes q from paper Table 1
      (``q = rho * energy``). For paper-reproduction work this is the
      authoritative source; for runtime parity with the shipped
      xls use (1).

Both compute the same canonical formula:

    C = q_k * V_tilde       (kcal)

where ``V_tilde`` is the raw geometric volume (cm^3) from the
shape-specific models in ``src.volume_models`` and ``q_k`` is the
per-class factor. The paper itself only states (§3.4 page 4):
"After getting volume, food's calorie is obtained by searching
related tables" — it does not give an explicit formula.

The paper does NOT contain a beta correction. Beta correction
appears only in the offline analysis script
``xls_results_analysis.m`` (lines 96-97, 134-136) and is **not** a
runtime formula. It is implemented in ``src.beta_correction``.

Folder structure:
- ``food_info_xls.py``     — parser for ``ECUSTFD/faster_rcnn/food_info.xls``
- ``runtime_formula.py``   — ``estimate_calorie_runtime`` (xls-driven).
- ``paper_faithful.py``    — ``paper_q_kcal_per_cm3`` /
                              ``paper_calorie_kcal`` (Table-1-driven).
"""

from __future__ import annotations

from .food_info_xls import EXPECTED_HEADER, get_q_factor, parse_food_info
from .runtime_formula import CalorieResultRuntime, estimate_calorie_runtime
from .paper_faithful import (
    PAPER_DENSITY_G_CM3,
    PAPER_ENERGY_PER_G,
    paper_q_kcal_per_cm3,
    paper_calorie_kcal,
)
# Backward-compatibility alias for ground truth loader (lazy-loaded to avoid circular imports)
def __getattr__(name: str):
    if name == "load_ground_truth":
        from src.faster_rcnn.eval_pipeline import load_ground_truth
        return load_ground_truth
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # runtime (xls-driven)
    "CalorieResultRuntime",
    "estimate_calorie_runtime",
    # paper-faithful (Table-1-driven)
    "PAPER_DENSITY_G_CM3",
    "PAPER_ENERGY_PER_G",
    "paper_q_kcal_per_cm3",
    "paper_calorie_kcal",
    # xls parser
    "parse_food_info",
    "get_q_factor",
    "EXPECTED_HEADER",
    "load_ground_truth",
]