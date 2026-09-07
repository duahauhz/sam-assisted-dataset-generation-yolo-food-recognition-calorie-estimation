# -*- coding: utf-8 -*-
"""Project logger used by Faster R-CNN scripts/notebooks.

Mirrors the pattern used in the YOLO26-seg pipeline (see log headers in
``outputs/predictions/04_run_*/``).  Both the build-dataset step and
the training step write to per-run files under ``outputs/logs/``.

Typical usage:

    from src.faster_rcnn.faster_rcnn_guard import setup_logger, project_root

    logger = setup_logger(__name__, timestamp="20260804-221051")
    logger.info("hello")
    print(project_root())
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Direct PyTorch cache to E: drive to prevent disk full error on C: drive
os.environ.setdefault("TORCH_HOME", "E:/AI_Research/.cache/torch")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def project_root() -> Path:
    """Return the project root regardless of the current working directory."""
    return Path("E:/AI_Research/dlt8").resolve()


ROOT = project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

LOGS_DIR = ROOT / "outputs" / "logs"
PREDICTIONS_DIR = ROOT / "outputs" / "predictions"
MODEL_DIR = ROOT / "models"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def _force_utf8() -> None:
    """Best-effort UTF-8 on Windows consoles (cp1252 default)."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def make_timestamp() -> str:
    """Return a filesystem-safe timestamp like ``20260804-221051``."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def setup_logger(
    name: str,
    *,
    timestamp: Optional[str] = None,
    log_stem: str = "faster_rcnn",
) -> logging.Logger:
    """Return a logger that mirrors file + console output.

    Args:
        name: ``__name__`` of the caller.
        timestamp: explicit timestamp string, generated if ``None``.
        log_stem: file prefix (e.g. ``"05_train"`` or ``"05_build"``).

    Output:
        ``<ROOT>/outputs/logs/<log_stem>_<timestamp>.log``
    """
    _force_utf8()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = timestamp or make_timestamp()
    log_path = LOGS_DIR / f"{log_stem}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers on re-init (Jupyter re-runs cell).
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"LOG_PATH = {log_path}")
    return logger


def make_run_dir(run_stem: str, timestamp: Optional[str] = None) -> Path:
    """Create and return a timestamped run dir under ``outputs/predictions/``."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts = timestamp or make_timestamp()
    run_dir = PREDICTIONS_DIR / f"{run_stem}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
