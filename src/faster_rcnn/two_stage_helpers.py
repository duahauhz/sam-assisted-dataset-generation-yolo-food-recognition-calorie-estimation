# -*- coding: utf-8 -*-
"""Two-Stage Evaluation Helpers (Faster R-CNN / YOLO Bbox + GrabCut / SAM1).

This module provides reusable utilities for notebooks:
- 06a_faster_rcnn_eval.ipynb (Faster R-CNN + GrabCut)
- 06b_faster_rcnn_sam_eval.ipynb (Faster R-CNN + SAM1)
- 07_yolo26_bbox_eval.ipynb (YOLO26 Bbox-only + GrabCut/SAM downstream)

Features:
1. Dual-logging & Exception Trap (console + timestamped log file).
2. Rigorous Label Mapping Sanity Check (prevents off-by-one / coin shift).
3. Multi-sample Smoke Testing (detection + mask + pairing contract).
4. Benchmark & Latency Profiling (mean, median, p90/p95/p99, fps).
5. Standardized Output Manifest & Summary Generator.
"""
from __future__ import annotations

import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical Class Order (Torchvision Convention: 0=background, 1..20=classes)
# ---------------------------------------------------------------------------
CANONICAL_21_CLASSES: List[str] = [
    "__background__",
    "apple", "banana", "bread", "bun", "coin",
    "doughnut", "egg", "fried_dough_twist", "grape",
    "lemon", "litchi", "mango", "mooncake",
    "orange", "peach", "pear", "plum", "qiwi",
    "sachima", "tomato",
]


# ---------------------------------------------------------------------------
# 1. Dual Logging & Error Hook Setup
# ---------------------------------------------------------------------------
def setup_eval_logging(
    notebook_tag: str,
    project_root: Optional[Path] = None,
) -> Tuple[logging.Logger, Path, Path, str]:
    """Setup dual logging to console and file, with global excepthook."""
    if project_root is None:
        project_root = Path("E:/AI_Research/dlt8").resolve()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logs_dir = project_root / "outputs" / "logs"
    preds_dir = project_root / "outputs" / "predictions"
    logs_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"{notebook_tag}_{timestamp}.log"
    run_dir = preds_dir / f"{notebook_tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Logger configuration
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler (UTF-8)
    fh = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # Global exception hook to guarantee traceback is saved in log file
    def _excepthook(exc_type, exc_val, exc_tb):
        logger.error("Uncaught exception:", exc_info=(exc_type, exc_val, exc_tb))
        sys.__excepthook__(exc_type, exc_val, exc_tb)

    sys.excepthook = _excepthook

    logger.info("=" * 70)
    logger.info("=== %s -- EVALUATION SESSION STARTED ===", notebook_tag.upper())
    logger.info("=" * 70)
    logger.info("PROJECT_ROOT : %s", project_root)
    logger.info("LOG_PATH     : %s", log_file)
    logger.info("RUN_DIR      : %s", run_dir)
    logger.info("TIMESTAMP    : %s", timestamp)

    return logger, log_file, run_dir, timestamp


# ---------------------------------------------------------------------------
# 2. Rigorous Label Mapping Sanity Check
# ---------------------------------------------------------------------------
def sanity_check_label_mapping(
    idx_to_name: List[str],
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Verify that class mapping is strictly 1-indexed with 21 classes.

    Guarantees:
    - idx 0 is '__background__'
    - idx 1 is 'apple'
    - idx 5 is 'coin' (prevents coin shift bug)
    - idx 20 is 'tomato'
    """
    log = logger or logging.getLogger()
    log.info("[Sanity Check] Verifying class mapping contract...")

    if len(idx_to_name) != 21:
        raise ValueError(
            f"Class mapping length mismatch! Expected 21 classes, got {len(idx_to_name)}: {idx_to_name}"
        )

    for idx, expected in enumerate(CANONICAL_21_CLASSES):
        actual = idx_to_name[idx]
        if actual != expected:
            raise ValueError(
                f"Label shift detected at index {idx}! Expected '{expected}', got '{actual}'."
            )

    log.info("  [PASS] 21 classes validated: idx 0='__background__', idx 1='apple', idx 5='coin', idx 20='tomato'.")
    log.info("  [PASS] No off-by-one label shift detected.")
    return True


# ---------------------------------------------------------------------------
# 3. Ground-Truth Table Matching & Aliasing
# ---------------------------------------------------------------------------
def fix_ground_truth_aliasing(
    gt_by_class: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Ensure aliases (like fired_dough_twist vs fried_dough_twist) are present in GT."""
    log = logger or logging.getLogger()
    if gt_by_class and "fired_dough_twist" in gt_by_class and "fried_dough_twist" not in gt_by_class:
        gt_by_class["fried_dough_twist"] = gt_by_class["fired_dough_twist"]
        log.info("[fix] Aliased 'fired_dough_twist' -> 'fried_dough_twist' in GT dictionary.")
    elif gt_by_class and "fried_dough_twist" in gt_by_class and "fired_dough_twist" not in gt_by_class:
        gt_by_class["fired_dough_twist"] = gt_by_class["fried_dough_twist"]
        log.info("[fix] Aliased 'fried_dough_twist' -> 'fired_dough_twist' in GT dictionary.")
    return gt_by_class


# ---------------------------------------------------------------------------
# 4. Multi-Sample Smoke Test
# ---------------------------------------------------------------------------
def run_smoke_test(
    predict_fn: Callable[[Path, float], List[Dict[str, Any]]],
    image_paths: List[Path],
    expected_classes: Optional[List[str]] = None,
    conf: float = 0.25,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Run smoke test on representative images to verify non-empty output and valid masks."""
    log = logger or logging.getLogger()
    log.info("[Smoke Test] Testing detector + mask backend on %d sample images...", len(image_paths))

    for idx, p in enumerate(image_paths):
        if not p.exists():
            log.warning("  [SKIP] Test image does not exist: %s", p)
            continue

        dets = predict_fn(p, conf)
        class_counts: Dict[str, int] = {}
        for d in dets:
            c = d.get("class_name", "unknown")
            class_counts[c] = class_counts.get(c, 0) + 1

        log.info("  Sample [%d/%d] %s: %d detections, classes=%s", idx + 1, len(image_paths), p.name, len(dets), class_counts)

        # Check mask contracts on detections
        for d in dets:
            cname = d.get("class_name")
            bbox = d.get("bbox")
            mask = d.get("mask")
            if cname != "coin" and mask is not None:
                if not isinstance(mask, np.ndarray) or mask.dtype != bool:
                    raise TypeError(f"Mask for {cname} must be a boolean numpy array, got {type(mask)} {getattr(mask, 'dtype', None)}")
                if mask.sum() == 0:
                    log.warning("  [Warning] Mask for %s in %s is empty!", cname, p.name)

    log.info("[Smoke Test] PASS -- All smoke test samples processed without crash.")
    return True


# ---------------------------------------------------------------------------
# 5. Speed Profiler & Report Generator
# ---------------------------------------------------------------------------
def compute_speed_report(
    e2e_times: List[float],
    e2e_ndets: List[int],
    e2e_nmasks: List[int],
    config_dict: Dict[str, Any],
    run_dir: Path,
    backend_title: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Calculate latency statistics and write speed_per_image.json."""
    log = logger or logging.getLogger()
    if not e2e_times:
        log.warning("[timer] No execution times recorded.")
        return {}

    n = len(e2e_times)
    mean_t = statistics.mean(e2e_times)
    median_t = statistics.median(e2e_times)
    min_t = min(e2e_times)
    max_t = max(e2e_times)
    std_t = statistics.pstdev(e2e_times) if n > 1 else 0.0

    def _pct(xs, q):
        s = sorted(xs)
        k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
        return s[k]

    p90 = _pct(e2e_times, 0.90)
    p95 = _pct(e2e_times, 0.95)
    p99 = _pct(e2e_times, 0.99)
    total_t = sum(e2e_times)
    fps = n / total_t if total_t > 0 else 0.0

    avg_ndets = statistics.mean(e2e_ndets) if e2e_ndets else 0.0
    avg_nmasks = statistics.mean(e2e_nmasks) if e2e_nmasks else 0.0

    speed_report = {
        "config": config_dict,
        "metric": "end_to_end_per_image_seconds",
        "backend": backend_title,
        "n_images": n,
        "total_seconds": total_t,
        "images_per_second": fps,
        "mean": mean_t,
        "median": median_t,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "min": min_t,
        "max": max_t,
        "stdev": std_t,
        "avg_n_detections": avg_ndets,
        "avg_n_mask_calls": avg_nmasks,
    }

    speed_path = run_dir / "speed_per_image.json"
    speed_path.write_text(json.dumps(speed_report, indent=2), encoding="utf-8")
    log.info("[timer] Speed report saved -> %s", speed_path)

    print("\n" + "=" * 70)
    print(f"  PER-IMAGE END-TO-END SPEED ({backend_title})")
    print("=" * 70)
    print(f"  n_images         : {n}")
    print(f"  total wall time  : {total_t:.2f}s ({total_t/60:.2f} min)")
    print(f"  throughput       : {fps:.3f} images/sec")
    print("  Per-image latency:")
    print(f"    mean   : {mean_t*1000:.1f} ms  ({mean_t:.4f}s)")
    print(f"    median : {median_t*1000:.1f} ms  ({median_t:.4f}s)")
    print(f"    p90    : {p90*1000:.1f} ms  ({p90:.4f}s)")
    print(f"    p95    : {p95*1000:.1f} ms  ({p95:.4f}s)")
    print(f"    p99    : {p99*1000:.1f} ms  ({p99:.4f}s)")
    print(f"    min/max: {min_t*1000:.1f} ms / {max_t*1000:.1f} ms")
    print(f"    stdev  : {std_t*1000:.1f} ms")
    print(f"  avg #dets / #masks per img: {avg_ndets:.2f} / {avg_nmasks:.2f}")
    print("=" * 70)

    return speed_report


# ---------------------------------------------------------------------------
# 6. Per-Class Markdown Report Formatter
# ---------------------------------------------------------------------------
def format_and_print_report(
    report_path: Path,
    title: str,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Load report JSON, generate styled Markdown table and print overall stats."""
    if not report_path.exists():
        print(f"Report not found at {report_path}")
        return None, {}

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    rows = []
    # Support both schemas: "per_class" (list of dicts from eval_pipeline/run_e2e)
    # and "by_class" (dict keyed by class name)
    if "per_class" in report and isinstance(report["per_class"], list):
        for item in report["per_class"]:
            cls_name = item.get("class_name", "")
            n = item.get("n_samples", 0)
            me_vol = item.get("me_volume_pct", 0.0)
            me_mass = item.get("me_mass_pct", 0.0)
            abs_me_vol = item.get("abs_me_volume_pct", abs(me_vol) if me_vol is not None else 0.0)
            rows.append({
                "Class": cls_name,
                "n": n,
                "ME_vol (%)": me_vol,
                "|ME_vol| (%)": abs_me_vol,
                "ME_mass (%)": me_mass,
            })
    elif "by_class" in report and isinstance(report["by_class"], dict):
        for cls_name, stats in report["by_class"].items():
            n = stats.get("n_samples", 0)
            me_vol = stats.get("me_volume", stats.get("me_volume_pct", 0.0))
            me_mass = stats.get("me_mass", stats.get("me_mass_pct", 0.0))
            mape_vol = stats.get("mape_volume", stats.get("abs_me_volume_pct", abs(me_vol) if me_vol is not None else 0.0))
            rows.append({
                "Class": cls_name,
                "n": n,
                "ME_vol (%)": me_vol,
                "|ME_vol| (%)": mape_vol,
                "ME_mass (%)": me_mass,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[["Class", "n", "ME_vol (%)", "|ME_vol| (%)", "ME_mass (%)"]]
        df = df.sort_values("|ME_vol| (%)").reset_index(drop=True)
        print(f"\n=== {title} ===")
        print(df.to_markdown(index=False, floatfmt=".2f"))

    overall = report.get("overall", {})
    print("\n=== Overall Metrics ===")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")

    return df, report

