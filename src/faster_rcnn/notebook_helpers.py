# -*- coding: utf-8 -*-
"""Helpers used by 05_faster_rcnn_new_train.ipynb.

This module contains every reusable piece of logic that the notebook
needs (deps check, dataset summary, run manifest).
The notebook's job is reduced to:

    from faster_rcnn import notebook_helpers as nh
    nh.setup_logging()
    nh.check_dependencies()
    nh.run_build_dataset()
    nh.print_dataset_summary()
    nh.run_train()
    nh.print_output_manifest()
    nh.print_done()

Every ``DEFAULT_*`` (paths, epochs, batch, ... ) lives in the underlying
CLI scripts (``build_dataset.py``, ``train_faster_rcnn.py``); this
helpers module does NOT hardcode any training hyperparameter -- it only
reads what those scripts already wrote to disk.

Note on history
---------------
Before Aug 2026 this notebook trained BOTH ``old`` (raw VOC XML, paper
2017 baseline) and ``new`` (patched XML, SAM-pipeline bboxes).  As of
2026-08-08 we keep ONLY ``new`` -- ``old`` has been removed because the
project no longer compares against the 2017-paper raw bboxes.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from faster_rcnn.faster_rcnn_guard import (
    LOGS_DIR, MODEL_DIR, ROOT, make_run_dir, make_timestamp, setup_logger,
)


# ---------------------------------------------------------------------------
# Shared globals
# ---------------------------------------------------------------------------

PRED_DIR = ROOT / "outputs/predictions"
DATA_ROOT = ROOT / "data/processed/faster_rcnn_seg"
RUN_DIR_GLOBAL = None  # set by setup_logging()

# Single variant this notebook trains.  Kept as a module constant so that
# downstream code can reference it without scattering string literals.
DEFAULT_VARIANT = "new"


# ---------------------------------------------------------------------------
# Cell 1: logging
# ---------------------------------------------------------------------------

def setup_logging() -> Path:
    """Create the per-run log file + run dir.

    Returns the run_dir so the notebook can reference it if needed.
    """
    global RUN_DIR_GLOBAL
    ts = make_timestamp()
    logger = setup_logger(
        "05a_faster_rcnn_train", timestamp=ts, log_stem="05a_faster_rcnn_train",
    )
    logger.info("=" * 70)
    logger.info(f"=== 05a -- Faster R-CNN ({DEFAULT_VARIANT}) -- Train one model ===")
    logger.info("=" * 70)
    logger.info(f"PROJECT_ROOT = {ROOT}")
    logger.info(f"LOG_PATH     = {LOGS_DIR / f'05a_faster_rcnn_train_{ts}.log'}")
    logger.info(f"Timestamp    = {ts}")
    run_dir = make_run_dir("05a_faster_rcnn_train", timestamp=ts)
    logger.info(f"RUN_DIR      = {run_dir}")
    RUN_DIR_GLOBAL = run_dir
    return run_dir


# ---------------------------------------------------------------------------
# Cell 2: dependency check
# ---------------------------------------------------------------------------

def check_dependencies() -> bool:
    """Return True iff torch + torchvision + pycocotools are importable."""
    log = logging.getLogger()
    log.info("Sanity checks:")
    log.info(f"  python : {sys.version.split()[0]}")
    log.info(f"  cwd    : {Path.cwd()}")
    status = {}
    for name, modname in [
        ("torch", "torch"), ("torchvision", "torchvision"), ("pycocotools", "pycocotools"),
    ]:
        try:
            m = __import__(modname)
            status[name] = (True, getattr(m, "__version__", "?"))
        except ImportError as exc:
            status[name] = (False, str(exc))
    for name, (ok, ver) in status.items():
        log.info(f"  {name:<12} : {'OK' if ok else 'FAIL'} ({ver})")
    if not all(ok for ok, _ in status.values()):
        log.error("Missing dep -- run: pip install torch torchvision pycocotools")
        return False
    log.info(f"  torch CUDA      : {status['torch'][1]} -> cuda={__import__('torch').cuda.is_available()}")
    return True


# ---------------------------------------------------------------------------
# Subprocess live runner
# ---------------------------------------------------------------------------

def _run_cmd_live(cmd: List[str]) -> None:
    """Run a command as a subprocess, streaming stdout live to sys.stdout."""
    import os
    log = logging.getLogger()
    log.info("Running: " + " ".join(cmd))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    if proc.stdout:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

    proc.wait()
    if proc.returncode != 0:
        log.error(f"Command exit code = {proc.returncode}")
        raise SystemExit(proc.returncode)


# ---------------------------------------------------------------------------
# Cell 3: launch build_dataset.py
# ---------------------------------------------------------------------------

def run_build_dataset(max_samples: int | None = None) -> None:
    """Launch ``build_dataset.py --variant new`` as a subprocess."""
    log = logging.getLogger()
    cmd = [
        sys.executable, "-u",
        str(ROOT / "src/faster_rcnn/build_dataset.py"),
        "--variant", DEFAULT_VARIANT,
    ]
    if max_samples is not None:
        cmd += ["--max-samples", str(max_samples)]
    _run_cmd_live(cmd)
    log.info(f"build_dataset.py ({DEFAULT_VARIANT}) finished successfully.")


# ---------------------------------------------------------------------------
# Cell 4: dataset summary
# ---------------------------------------------------------------------------

def _safe_count(p: Path) -> Dict:
    """Return image/annotation counts from a COCO JSON, or zeros if missing."""
    if not p.exists():
        return {"images": 0, "annotations": 0, "missing": True}
    j = json.loads(p.read_text(encoding="utf-8"))
    return {"images": len(j["images"]), "annotations": len(j["annotations"]), "missing": False}


def collect_dataset_summary() -> Dict[str, Dict[str, Dict]]:
    """Return ``{split: {images, annotations, missing}}`` for the single variant."""
    base = DATA_ROOT / DEFAULT_VARIANT
    return {
        "train": _safe_count(base / "annotations/train.json"),
        "test":  _safe_count(base / "annotations/test.json"),
    }


def _first_sample_bbox() -> Dict | None:
    """Return one bbox sample from the train split, or None if empty."""
    p = DATA_ROOT / DEFAULT_VARIANT / "annotations/train.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text(encoding="utf-8"))
    if not j["annotations"]:
        return None
    a = j["annotations"][0]
    img = next(i for i in j["images"] if i["id"] == a["image_id"])
    return {"file_name": img["file_name"], "category_id": a["category_id"], "bbox": a["bbox"]}


def print_dataset_summary() -> None:
    """Log image/annotation counts for the variant + one sample bbox."""
    log = logging.getLogger()
    summary = collect_dataset_summary()
    for split in ("train", "test"):
        s = summary[split]
        suffix = "  (file missing!)" if s['missing'] else ""
        log.info(f"[{split:<5}] images: {s['images']}, objects: {s['annotations']}{suffix}")
    sample = _first_sample_bbox()
    if sample:
        log.info(f"[sample] file={sample['file_name']}  cat={sample['category_id']}  bbox={sample['bbox']}")


# ---------------------------------------------------------------------------
# Cell 5: launch train
# ---------------------------------------------------------------------------

def run_train() -> None:
    """Launch ``train_faster_rcnn.py --data-variant new``.

    All hyperparameters are hardcoded in ``train_faster_rcnn.py`` as
    ``DEFAULT_*`` constants; this launcher passes only the variant.
    """
    log = logging.getLogger()
    cmd = [
        sys.executable, "-u", str(ROOT / "src/faster_rcnn/train_faster_rcnn.py"),
        "--data-variant", DEFAULT_VARIANT,
    ]
    _run_cmd_live(cmd)
    log.info(f"train ({DEFAULT_VARIANT}) finished.")


# ---------------------------------------------------------------------------
# Cell 6: output manifest
# ---------------------------------------------------------------------------

def print_output_manifest() -> None:
    """Log the paths of best.pt / history.json / summary.txt for the variant run."""
    log = logging.getLogger()
    log.info("=" * 70)
    log.info("OUTPUT ARTIFACTS")
    log.info("=" * 70)
    runs = sorted(PRED_DIR.glob(f"05_train_{DEFAULT_VARIANT}_*"),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        log.warning(f"  [{DEFAULT_VARIANT}] no run found")
        return
    run = runs[-1]
    log.info(f"  run_dir  : {run}")
    log.info(f"  summary  : {run / 'summary.txt'}")
    log.info(f"  history  : {run / 'history.json'}")
    log.info(f"  ckpt     : {run / 'best.pt'}")
    log.info(f"  best.pt copied to: "
             f"{MODEL_DIR / f'faster_rcnn_{DEFAULT_VARIANT}_best.pt'}")
    log.info("")
    log.info("Combined run summary: " + str(RUN_DIR_GLOBAL))


# ---------------------------------------------------------------------------
# Cell 7: cleanup / done
# ---------------------------------------------------------------------------

def print_done() -> None:
    log = logging.getLogger()
    log.info("=" * 70)
    log.info("✅ 05a_faster_rcnn_train.ipynb -- DONE")
    log.info("Next: run 06_faster_rcnn_eval.ipynb (set MODEL_VARIANT='new') "
             "to evaluate mAP / ME_volume / ME_mass.")
    log.info("=" * 70)
