# -*- coding: utf-8 -*-
"""Compute vision metrics (Precision / Recall / mAP@50 / mAP@50-95) on the ECUSTFD
test split for any detector whose bbox predictions we can obtain.

This script fills the gap in the Bảng 1 (vision metrics table) of the paper:

- ``04_e2e_paper_faithful_beta.ipynb`` / YOLO26-seg
- ``06a_faster_rcnn_eval.ipynb``       / Faster R-CNN + GrabCut
- ``06b_faster_rcnn_sam_eval.ipynb``   / Faster R-CNN + SAM1
- ``07a_yolo_bbox_eval.ipynb``         / YOLO26 bbox-only + GrabCut
- ``07b_yolo_bbox_sam_eval.ipynb``     / YOLO26 bbox-only + SAM1

The five end-to-end notebooks (04, 06a, 06b, 07a, 07b) all run inference on
the **test** split (1733 stems — see ``summary.txt`` in each output dir) but
only store ME_volume / ME_mass / speed in their reports; they do NOT save
bbox predictions to disk and do NOT compute P/R/mAP. This script

1. Loads each detector (Faster R-CNN / YOLO-seg / YOLO-bbox) once.
2. Re-runs inference on the **test** image set, collecting bbox
   predictions in COCO ``results`` format: ``[image_id, category_id,
   bbox=[x,y,w,h], score]``.
3. Loads ground-truth bbox annotations from
   ``data/processed/faster_rcnn_seg/new/annotations/test.json``
   (COCO format).
4. Computes COCO metrics with ``pycocotools.cocoeval.COCOeval``:
   ``AP@[0.50:0.95]``, ``AP@50``, ``AP@75``, ``AR@1``, ``AR@10``,
   ``AR@100``, plus the standard 12-point precision/recall grid.

The script NEVER re-trains — it only re-uses the existing weights files:
``models/faster_rcnn_new_best.pt``,
``models/ecustfd_yolo26seg_best.pt``,
``models/ecustfd_yolo26seg_bbox_best.pt``.

Usage
-----

::

    # Run all five notebook presets (recommended):
    python -m src.faster_rcnn.eval_vision --run-all

    # Run just one preset (e.g. YOLO26-seg from notebook 04):
    python -m src.faster_rcnn.eval_vision --preset 04_yolo26_seg

    # Run a custom detector / weights:
    python -m src.faster_rcnn.eval_vision \\
        --detector faster_rcnn \\
        --weights  models/faster_rcnn_new_best.pt

    # Smoke test on first 10 images (fast):
    python -m src.faster_rcnn.eval_vision --preset 06a_faster_rcnn --max-images 10

By default the script **excludes** the ``coin`` class from mAP (matches the
food-only convention used elsewhere in the repo). Pass ``--include-coin`` to
include it.

Notes on coordinate spaces
--------------------------
- **Faster R-CNN** outputs ``boxes`` in original-image pixel space
  (``x1, y1, x2, y2``). Convert ``xyxy → xywh`` directly.
- **YOLO-bbox** (``ultralytics``) outputs ``boxes.xyxy`` in **original-image
  pixel space** — same as Faster R-CNN.  ``xyxy → xywh`` directly.
- **YOLO-seg** outputs ``boxes.xyxy`` in **original-image pixel space**
  AND returns ``masks`` in **letterboxed mask canvas** space. We only need
  bbox predictions, so we use the original-space bbox (same as bbox-only).
  See ``src.segmentation_runtime.yolo_infer._results_to_detections`` for
  the full coordinate machinery (we do not need it here for bbox-only
  mAP).

Confidence threshold is fixed at 0.0 — we want the full P/R curve.
``COCOeval`` handles score thresholding internally at the IoU sweep.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# pycocotools
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "pycocotools is required for this script. "
        "Install via `pip install pycocotools`."
    ) from _exc

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_GT_JSON = (
    PROJECT_ROOT
    / "data" / "processed" / "faster_rcnn_seg" / "new" / "annotations" / "test.json"
)
TEST_SPLIT_TXT = PROJECT_ROOT / "data" / "raw" / "ECUSTFD" / "ImageSets" / "Main" / "test.txt"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "ECUSTFD" / "JPEGImages"
WEIGHTS_DIR = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log = logging.getLogger("eval_vision")


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Class name handling
# ---------------------------------------------------------------------------

# Faster R-CNN class names (index 0 = __background__).
_FASTER_RCNN_CLASSES: List[str] = [
    "apple", "banana", "bread", "bun", "coin",
    "doughnut", "egg", "fried_dough_twist", "grape",
    "lemon", "litchi", "mango", "mooncake",
    "orange", "peach", "pear", "plum", "qiwi",
    "sachima", "tomato",
]
assert len(_FASTER_RCNN_CLASSES) == 20

# YOLO26-seg / YOLO26-bbox class names (Ultralytics, 0-indexed).
# NOTE: in the trained checkpoint the class is spelled ``fired_dough_twist``
# (one 'r'). Both YOLO adapters emit ``fired_dough_twist`` as the class
# name (we normalise to ``fried_dough_twist`` for the COCO categories to
# match ``test.json``).
_YOLO_CLASSES: List[str] = [
    "apple", "banana", "bread", "bun", "coin",
    "doughnut", "egg", "fired_dough_twist", "grape",
    "lemon", "litchi", "mango", "mooncake",
    "orange", "peach", "pear", "plum", "qiwi",
    "sachima", "tomato",
]


def _normalise_yolo_class(name: str) -> str:
    """Map ``fired_dough_twist`` (YOLO spelling) → ``fried_dough_twist``
    (Faster R-CNN / test.json spelling)."""
    return "fried_dough_twist" if name == "fired_dough_twist" else name


# ---------------------------------------------------------------------------
# Detection container
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    image_id: int
    category_id: int
    bbox_xywh: Tuple[float, float, float, float]
    score: float
    class_name: str  # for diagnostics only; COCO uses category_id


# ---------------------------------------------------------------------------
# Detector runners
# ---------------------------------------------------------------------------

class DetectorRunner:
    """Base class — subclasses implement :meth:`detect_all`."""

    name: str = "base"

    def __init__(self, device: str = "0") -> None:
        self.device = device
        self.model = None

    def load(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def detect_all(
        self, image_paths: List[Path], stem_to_image_id: Dict[str, int]
    ) -> List[Detection]:
        raise NotImplementedError

    def unload(self) -> None:
        self.model = None
        with contextlib.suppress(Exception):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class FasterRCNNRunner(DetectorRunner):
    """Wrapper around ``src.faster_rcnn.inference`` (torchvision)."""

    name = "faster_rcnn"

    def __init__(self, weights_path: Path, device: str = "0") -> None:
        super().__init__(device=device)
        self.weights_path = weights_path
        # Category id mapping: torchvision idx → COCO category_id.
        # COCO test.json uses 1-indexed category_ids; torchvision uses
        # 0=background + 1..20 for the 20 classes.  After dropping
        # background, torchvision idx 1 → COCO category_id 1, etc.
        self._name_to_cat: Dict[str, int] = {}

    def load(self) -> None:
        from src.faster_rcnn.inference import load_faster_rcnn

        self.model = load_faster_rcnn(self.weights_path, device=self.device)
        # Build name → category_id from COCO GT.
        gt = COCO(str(TEST_GT_JSON))
        self._name_to_cat = {c["name"]: c["id"] for c in gt.loadCats(gt.getCatIds())}

    def detect_all(
        self, image_paths: List[Path], stem_to_image_id: Dict[str, int]
    ) -> List[Detection]:
        from src.faster_rcnn.inference import predict_one

        detections: List[Detection] = []
        n = len(image_paths)
        t0 = time.time()
        for i, p in enumerate(image_paths):
            # We use conf=0.0 so COCOeval can sweep the full P/R curve.
            dets = predict_one(self.model, p, conf=0.0, device=self.device)
            stem = p.stem  # e.g. apple015S(1)
            image_id = stem_to_image_id.get(stem)
            if image_id is None:
                _log.warning("Image stem %s not in test.json — skipped.", stem)
                continue
            for d in dets:
                cat_name = d["class_name"]
                if cat_name not in self._name_to_cat:
                    continue
                x1, y1, x2, y2 = d["bbox"]
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                detections.append(
                    Detection(
                        image_id=image_id,
                        category_id=self._name_to_cat[cat_name],
                        bbox_xywh=(float(x1), float(y1), float(w), float(h)),
                        score=float(d["conf"]),
                        class_name=cat_name,
                    )
                )
            if (i + 1) % 200 == 0 or (i + 1) == n:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                _log.info(
                    "  [%s] %d / %d  (%.1f img/s, elapsed %.1fs)",
                    self.name, i + 1, n, rate, elapsed,
                )
        return detections


class YOLOBboxRunner(DetectorRunner):
    """Wrapper around ``src.faster_rcnn.yolo_bbox_adapter.YOLOBboxDetector``."""

    name = "yolo_bbox"

    def __init__(self, weights_path: Path, device: str = "0") -> None:
        super().__init__(device=device)
        self.weights_path = weights_path
        self._name_to_cat: Dict[str, int] = {}

    def load(self) -> None:
        from src.faster_rcnn.yolo_bbox_adapter import YOLOBboxDetector

        # Device string for Ultralytics: '0' = first GPU, 'cpu' = CPU.
        dev = self.device if str(self.device).isdigit() else "cpu"
        self.model = YOLOBboxDetector(self.weights_path, device=dev)
        gt = COCO(str(TEST_GT_JSON))
        self._name_to_cat = {c["name"]: c["id"] for c in gt.loadCats(gt.getCatIds())}

    def detect_all(
        self, image_paths: List[Path], stem_to_image_id: Dict[str, int]
    ) -> List[Detection]:
        detections: List[Detection] = []
        n = len(image_paths)
        t0 = time.time()
        for i, p in enumerate(image_paths):
            dets = self.model.predict(
                p, conf=0.0, iou=0.5, imgsz=480,
            )
            stem = p.stem
            image_id = stem_to_image_id.get(stem)
            if image_id is None:
                continue
            for d in dets:
                cat_name = _normalise_yolo_class(d["class_name"])
                if cat_name not in self._name_to_cat:
                    continue
                x1, y1, x2, y2 = d["bbox"]
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                detections.append(
                    Detection(
                        image_id=image_id,
                        category_id=self._name_to_cat[cat_name],
                        bbox_xywh=(float(x1), float(y1), float(w), float(h)),
                        score=float(d["conf"]),
                        class_name=cat_name,
                    )
                )
            if (i + 1) % 200 == 0 or (i + 1) == n:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                _log.info(
                    "  [%s] %d / %d  (%.1f img/s, elapsed %.1fs)",
                    self.name, i + 1, n, rate, elapsed,
                )
        return detections


class YOLOSegRunner(YOLOBboxRunner):
    """YOLO-seg model — bbox branch only. Reuses YOLOBboxAdapter since
    ``ultralytics.YOLO.predict`` returns both boxes and masks; we just
    read ``result.boxes.xyxy``."""

    name = "yolo_seg"

    def __init__(self, weights_path: Path, device: str = "0") -> None:
        super().__init__(weights_path=weights_path, device=device)

    def load(self) -> None:
        # We bypass the YOLOBboxDetector wrapper and use Ultralytics directly
        # because we need access to raw ``result.boxes`` (xyxy in original
        # pixel coords is already what Ultralytics returns; the letterbox
        # coordinate issue only affects masks, not boxes).
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as _exc:  # pragma: no cover
            raise ImportError(
                "ultralytics is required for YOLO-seg. "
                "Install via `pip install ultralytics`."
            ) from _exc
        dev = self.device if str(self.device).isdigit() else "cpu"
        self.model = YOLO(str(self.weights_path))
        self._yolo_device = dev
        gt = COCO(str(TEST_GT_JSON))
        self._name_to_cat = {c["name"]: c["id"] for c in gt.loadCats(gt.getCatIds())}

    def detect_all(
        self, image_paths: List[Path], stem_to_image_id: Dict[str, int]
    ) -> List[Detection]:
        detections: List[Detection] = []
        n = len(image_paths)
        t0 = time.time()
        for i, p in enumerate(image_paths):
            results = self.model.predict(
                source=str(p),
                conf=0.0,
                iou=0.5,
                imgsz=480,
                device=self._yolo_device,
                verbose=False,
            )
            if not results:
                continue
            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            stem = p.stem
            image_id = stem_to_image_id.get(stem)
            if image_id is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            names = r.names
            for box, score, cid in zip(xyxy, confs, cls_ids):
                cat_name = _normalise_yolo_class(names.get(cid, str(cid)))
                if cat_name not in self._name_to_cat:
                    continue
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                detections.append(
                    Detection(
                        image_id=image_id,
                        category_id=self._name_to_cat[cat_name],
                        bbox_xywh=(float(x1), float(y1), float(w), float(h)),
                        score=float(score),
                        class_name=cat_name,
                    )
                )
            if (i + 1) % 200 == 0 or (i + 1) == n:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                _log.info(
                    "  [%s] %d / %d  (%.1f img/s, elapsed %.1fs)",
                    self.name, i + 1, n, rate, elapsed,
                )
        return detections


# ---------------------------------------------------------------------------
# COCO eval
# ---------------------------------------------------------------------------

def _detections_to_coco_json(detections: List[Detection]) -> List[dict]:
    """Convert list of Detection dataclasses to COCO results JSON list."""
    return [
        {
            "image_id": d.image_id,
            "category_id": d.category_id,
            "bbox": [d.bbox_xywh[0], d.bbox_xywh[1], d.bbox_xywh[2], d.bbox_xywh[3]],
            "score": d.score,
        }
        for d in detections
    ]


def evaluate_coco(
    detections: List[Detection],
    out_dir: Path,
    include_coin: bool = False,
    cat_ids: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Run COCOeval on a list of detections, returning a metrics dict.

    Args:
        detections: list of :class:`Detection` dataclasses.
        out_dir: directory to write ``predictions.json`` and ``metrics.json``.
        include_coin: if False, exclude ``coin`` (category name) from mAP.
        cat_ids: explicit category-id filter (overrides ``include_coin``).

    Returns:
        Dict with keys: ``precision``, ``recall``, ``mAP_50``, ``mAP_50_95``,
        ``mAP_75``, ``AR_100``, and the raw ``COCOeval.stats`` array (12 vals).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter detections by category_id if requested.
    if cat_ids is not None:
        cat_id_set = set(cat_ids)
        detections = [d for d in detections if d.category_id in cat_id_set]
    elif not include_coin:
        gt_coco = COCO(str(TEST_GT_JSON))
        all_cats = gt_coco.loadCats(gt_coco.getCatIds())
        coin_id = next((c["id"] for c in all_cats if c["name"] == "coin"), None)
        if coin_id is not None:
            n_before = len(detections)
            detections = [d for d in detections if d.category_id != coin_id]
            _log.info(
                "Excluded coin class (cat_id=%d): %d → %d detections",
                coin_id, n_before, len(detections),
            )

    # Write predictions JSON.
    pred_json = _detections_to_coco_json(detections)
    pred_path = out_dir / "predictions.json"
    with open(pred_path, "w") as f:
        json.dump(pred_json, f)
    _log.info("Wrote %d predictions to %s", len(pred_json), pred_path)

    if len(pred_json) == 0:
        _log.warning("No predictions to evaluate — skipping COCOeval.")
        return {"precision": 0.0, "recall": 0.0, "mAP_50": 0.0, "mAP_50_95": 0.0, "mAP_75": 0.0, "AR_100": 0.0, "stats": [0.0] * 12}

    # Run COCOeval.
    coco_gt = COCO(str(TEST_GT_JSON))
    coco_dt = coco_gt.loadRes(str(pred_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    # Capture the printed summary to a string by redirecting stdout.
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coco_eval.summarize()
    summary_text = buf.getvalue()
    _log.info("COCOeval summary:\n%s", summary_text)

    stats = coco_eval.stats  # 12 values: AP @[.5:.95], AP50, AP75, AP_small/medium/large, AR1/10/100, AR_small/medium/large
    # COCOeval.stats indices:
    # 0  = AP @ IoU=0.50:0.95 | area=all      | maxDets=100  (= mAP@50-95)
    # 1  = AP @ IoU=0.50      | area=all      | maxDets=100  (= mAP@50)
    # 2  = AP @ IoU=0.75      | area=all      | maxDets=100  (= mAP@75)
    # 3  = AP @ IoU=0.50:0.95 | area=small    | maxDets=100
    # 4  = AP @ IoU=0.50:0.95 | area=medium   | maxDets=100
    # 5  = AP @ IoU=0.50:0.95 | area=large    | maxDets=100
    # 6  = AR @ IoU=0.50:0.95 | area=all      | maxDets=1
    # 7  = AR @ IoU=0.50:0.95 | area=all      | maxDets=10
    # 8  = AR @ IoU=0.50:0.95 | area=all      | maxDets=100 (= recall approx)
    # 9-11 = AR small/medium/large
    metrics = {
        "mAP_50_95": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR_1": float(stats[6]),
        "AR_10": float(stats[7]),
        "AR_100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
    }
    # Aliases matching the paper's column names.
    metrics_alias = {
        "Precision (%)": metrics["mAP_50"],  # best-effort alias — COCO AP@50 is closest to "precision at IoU=0.5"
        "Recall (%)": metrics["AR_100"],
        "mAP@50 (%)": metrics["mAP_50"],
        "mAP@50-95 (%)": metrics["mAP_50_95"],
        # raw
        "precision": metrics["mAP_50"],
        "recall": metrics["AR_100"],
        "mAP_50": metrics["mAP_50"],
        "mAP_50_95": metrics["mAP_50_95"],
        "mAP_75": metrics["mAP_75"],
        "stats": [float(x) for x in stats],
    }
    metrics.update(metrics_alias)

    # Write metrics JSON.
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    _log.info("Wrote metrics to %s", metrics_path)

    # Also write the COCOeval summary text.
    summary_path = out_dir / "coco_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_test_stem_set() -> List[str]:
    if not TEST_SPLIT_TXT.exists():
        raise FileNotFoundError(f"Test split file missing: {TEST_SPLIT_TXT}")
    stems = [line.strip() for line in TEST_SPLIT_TXT.read_text().splitlines() if line.strip()]
    return stems


def _build_stem_to_image_id(gt_coco: COCO) -> Dict[str, int]:
    """Build a ``{stem: image_id}`` map from the COCO GT.

    The test.json ``images`` entries carry a ``stem`` field (e.g.
    ``apple015S(1)``) plus ``file_name`` (e.g. ``apple015S(1).JPG``).
    """
    out: Dict[str, int] = {}
    for img in gt_coco.dataset["images"]:
        stem = img.get("stem")
        if stem:
            out[stem] = int(img["id"])
        else:
            # Fallback: strip extension from file_name.
            stem = Path(img["file_name"]).stem
            out[stem] = int(img["id"])
    return out


def _resolve_image_paths(stems: List[str]) -> List[Path]:
    """Map each stem to its image file under IMAGES_DIR."""
    out: List[Path] = []
    missing: List[str] = []
    for stem in stems:
        cand = IMAGES_DIR / f"{stem}.JPG"
        if cand.exists():
            out.append(cand)
        else:
            # Try lowercase extension.
            cand2 = IMAGES_DIR / f"{stem}.jpg"
            if cand2.exists():
                out.append(cand2)
            else:
                missing.append(stem)
    if missing:
        _log.warning(
            "Missing %d / %d test image files (showing first 5): %s",
            len(missing), len(stems), missing[:5],
        )
    return out


def _build_runner(detector: str, weights: Path, device: str) -> DetectorRunner:
    if detector == "faster_rcnn":
        return FasterRCNNRunner(weights, device=device)
    if detector == "yolo_bbox":
        return YOLOBboxRunner(weights, device=device)
    if detector == "yolo_seg":
        return YOLOSegRunner(weights, device=device)
    raise ValueError(f"Unknown detector: {detector!r}")


# ---------------------------------------------------------------------------
# Preset detector configs (matches the 5 end-to-end notebooks)
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, str]] = {
    "04_yolo26_seg": {
        "detector": "yolo_seg",
        "weights": "ecustfd_yolo26seg_best.pt",
    },
    "06a_faster_rcnn": {
        "detector": "faster_rcnn",
        "weights": "faster_rcnn_new_best.pt",
    },
    "06b_faster_rcnn_sam": {
        "detector": "faster_rcnn",
        "weights": "faster_rcnn_new_best.pt",
    },
    "07a_yolo_bbox_grabcut": {
        "detector": "yolo_bbox",
        "weights": "ecustfd_yolo26seg_bbox_best.pt",
    },
    "07b_yolo_bbox_sam": {
        "detector": "yolo_bbox",
        "weights": "ecustfd_yolo26seg_bbox_best.pt",
    },
}


def run_one(
    preset_name: str,
    out_root: Path,
    device: str = "0",
    include_coin: bool = False,
    max_images: Optional[int] = None,
) -> Dict[str, float]:
    """Run vision metrics for one preset; return metrics dict."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset {preset_name!r}. Known: {list(PRESETS)}")
    cfg = PRESETS[preset_name]
    weights_path = WEIGHTS_DIR / cfg["weights"]
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    runner = _build_runner(cfg["detector"], weights_path, device=device)
    runner.load()
    try:
        stems = _load_test_stem_set()
        if max_images is not None:
            stems = stems[:max_images]
        gt_coco = COCO(str(TEST_GT_JSON))
        stem_to_image_id = _build_stem_to_image_id(gt_coco)
        image_paths = _resolve_image_paths(stems)
        _log.info(
            "[%s] %d test images, weights=%s",
            preset_name, len(image_paths), weights_path,
        )
        detections = runner.detect_all(image_paths, stem_to_image_id)
        out_dir = out_root / preset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        # Persist raw detections (dataclass → list of dicts) for the run.
        with open(out_dir / "detections_raw.json", "w") as f:
            json.dump(
                [
                    {
                        "image_id": d.image_id,
                        "category_id": d.category_id,
                        "bbox": list(d.bbox_xywh),
                        "score": d.score,
                        "class_name": d.class_name,
                    }
                    for d in detections
                ],
                f,
            )
        return evaluate_coco(detections, out_dir, include_coin=include_coin)
    finally:
        runner.unload()


def run_all(
    out_root: Path,
    device: str = "0",
    include_coin: bool = False,
    max_images: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Run vision metrics for all 5 presets; return nested metrics dict."""
    out_root.mkdir(parents=True, exist_ok=True)
    all_metrics: Dict[str, Dict[str, float]] = {}
    for preset_name in PRESETS:
        _log.info("=" * 70)
        _log.info("Running preset %s", preset_name)
        _log.info("=" * 70)
        all_metrics[preset_name] = run_one(
            preset_name, out_root, device=device,
            include_coin=include_coin, max_images=max_images,
        )
    # Write summary table.
    summary_path = out_root / "summary_all.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    _log.info("Wrote summary table to %s", summary_path)

    # Pretty-print the Bảng 1 columns.
    cols = ["mAP_50_95", "mAP_50", "mAP_75", "AR_100"]
    print("\n" + "=" * 90)
    print("VISION METRICS SUMMARY (test split, food-only, IoU sweep [0.5..0.95])")
    print("=" * 90)
    print(f"{'Preset':<28} | {'mAP@50-95':>10} | {'mAP@50':>10} | {'mAP@75':>10} | {'AR@100':>10}")
    print("-" * 90)
    for preset_name, m in all_metrics.items():
        print(
            f"{preset_name:<28} | "
            f"{m.get('mAP_50_95', 0)*100:>9.2f}% | "
            f"{m.get('mAP_50', 0)*100:>9.2f}% | "
            f"{m.get('mAP_75', 0)*100:>9.2f}% | "
            f"{m.get('AR_100', 0)*100:>9.2f}%"
        )
    print("=" * 90)
    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute Precision / Recall / mAP@50 / mAP@50-95 on ECUSTFD test split."
    )
    parser.add_argument(
        "--detector",
        choices=["faster_rcnn", "yolo_bbox", "yolo_seg"],
        help="Detector type. Ignored if --preset or --run-all is given.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Path to weights file. Ignored if --preset or --run-all is given.",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS),
        help="Run one of the 5 notebook presets (auto-fills detector+weights).",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all 5 presets and write a summary table.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "predictions" / "vision_metrics",
        help="Root output directory.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device id (e.g. '0', 'cpu').",
    )
    parser.add_argument(
        "--include-coin",
        action="store_true",
        help="Include the 'coin' class in mAP (default: exclude).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit to the first N test images (smoke test).",
    )
    args = parser.parse_args(argv)

    _setup_logging()

    if args.run_all:
        run_all(args.out_dir, device=args.device, include_coin=args.include_coin, max_images=args.max_images)
        return 0

    if args.preset:
        run_one(args.preset, args.out_dir, device=args.device, include_coin=args.include_coin, max_images=args.max_images)
        return 0

    if args.detector and args.weights:
        # Manual mode — create a temp preset.
        preset_name = f"manual_{args.detector}_{args.weights.stem}"
        PRESETS[preset_name] = {"detector": args.detector, "weights": args.weights.name}
        run_one(preset_name, args.out_dir, device=args.device, include_coin=args.include_coin, max_images=args.max_images)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())