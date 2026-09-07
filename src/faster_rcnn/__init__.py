# -*- coding: utf-8 -*-
"""Make this directory a Python package."""
import sys
from pathlib import Path

# Ensure project root (e.g. dlt8) and src/ are in sys.path for `from src...` imports
_SRC_DIR = Path(__file__).resolve().parent.parent
_ROOT_DIR = _SRC_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from .faster_rcnn_guard import (
    LOGS_DIR,
    MODEL_DIR,
    PREDICTIONS_DIR,
    ROOT,
    make_run_dir,
    make_timestamp,
    project_root,
    setup_logger,
)
from .inference import (
    apply_grabcut_to_detections,
    load_faster_rcnn,
    predict_many,
    predict_one,
)
from .sam_inference import (
    apply_sam_to_detections,
    load_sam,
    predict_many_with_sam,
    predict_one_with_sam,
)
from .eval_pipeline import main as run_eval

__all__ = [
    "LOGS_DIR",
    "MODEL_DIR",
    "PREDICTIONS_DIR",
    "ROOT",
    "apply_grabcut_to_detections",
    "apply_sam_to_detections",
    "load_faster_rcnn",
    "load_sam",
    "make_run_dir",
    "make_timestamp",
    "predict_many",
    "predict_many_with_sam",
    "predict_one",
    "predict_one_with_sam",
    "project_root",
    "run_eval",
    "setup_logger",
]
