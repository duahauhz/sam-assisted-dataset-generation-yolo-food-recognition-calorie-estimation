# -*- coding: utf-8 -*-
"""Faster R-CNN evaluation pipeline — E2E inference + GrabCut + volume + beta + metrics.

This module is the Faster R-CNN equivalent of ``src/e2e_pipeline/run_e2e.py``.
It uses:

    Faster R-CNN inference (from ``src.faster_rcnn.inference``)
        -> GrabCut mask generation
    Coin calibration (from ``src.coin_calibration``)
    View pairing (from ``src.view_pairing``)
    Volume estimation (from ``src.volume_models``)
    Beta correction (from ``src.beta_correction``)
    Calorie estimation (from ``src.calorie_estimation``)
    Metrics (from ``src.e2e_pipeline.metrics``)

Key differences from the YOLO26-seg pipeline
--------------------------------------------
1. Masks come from GrabCut (OpenCV) applied to Faster R-CNN bboxes,
   not from YOLO's built-in segmentation.
2. ``src.faster_rcnn.inference`` reuses the exact same detection dict
   schema (``class_name``, ``conf``, ``bbox``, ``mask``) so the
   downstream modules need zero changes.

Mask contract
-------------
``mask`` values are boolean ``(H, W)`` arrays where ``True`` = foreground
(equivalent to C++ label ``GC_PR_FGD == 3``).  This is identical to
the contract in ``src/segmentation_runtime`` and is expected by all
``src.volume_models`` functions.
"""

from __future__ import annotations

import argparse # Xử lý đối số dòng lệnh
import csv # Đọc và ghi file csv
import json # Đọc và ghi file json
import logging # Ghi log
from dataclasses import asdict, dataclass # Xử lý dataclass
from pathlib import Path # Xử lý đường dẫn file
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union # Xử lý kiểu dữ liệu

import numpy as np # Thư viện xử lý mảng

from src.calorie_estimation import (
    estimate_calorie_runtime, # Ước tính calo
    parse_food_info, # Phân tích thông tin thực phẩm
)
from src.coin_calibration import (
    compute_coin_scale,# Tính toán tỷ lệ đồng xu
    filter_coin_detections,# Lọc coin
    select_highest_conf, # Chọn coin có độ tin cậy cao nhất
)
from src.constants import DENSITY_G_CM3, FOOD_CLASSES, SHAPE_MODELS # Hằng số
from src.e2e_pipeline.dataset_split import (
    get_paper_50_50_split, # Lấy 50 50 split
    group_top_side, # Nhóm top và side
    load_split, # Load split
    make_pairs, # Tạo pairs
    parse_filename, # Phân tích filename
    resolve_image_paths, # Giải quyết đường dẫn ảnh
)
from src.e2e_pipeline.metrics import aggregate, compute_me_per_class # Tổng hợp và tính toán me per class

from src.beta_correction import (
    calibrate_volume, # Hiệu chỉnh thể tích
    compute_class_betas, # Tính toán beta per class
    save_betas, # Lưu beta
)
from src.segmentation_runtime import crop_mask_to_bbox # Cắt mask theo bbox
from src.volume_models import ShapeInputs, compute_volume # Tính toán thể tích

# Faster R-CNN inference module (our GrabCut wrapper).
from src.faster_rcnn.inference import load_faster_rcnn, predict_one as frcnn_predict_one # Load model và predict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_DATA_ROOT = Path("E:/AI_Research/dlt8/data/raw/ECUSTFD")
DEFAULT_IMAGES_DIR = DEFAULT_DATA_ROOT / "JPEGImages"
DEFAULT_IMAGESETS_DIR = DEFAULT_DATA_ROOT / "ImageSets" / "Main"
DEFAULT_MODEL_OLD = Path("E:/AI_Research/dlt8/models/faster_rcnn_old_best.pt")
DEFAULT_MODEL_NEW = Path("E:/AI_Research/dlt8/models/faster_rcnn_new_best.pt")
DEFAULT_FOOD_INFO = Path("E:/AI_Research/dlt8/ECUSTFD/faster_rcnn/food_info.xls")
DEFAULT_OUTPUT_DIR = Path("E:/AI_Research/dlt8/outputs")
DEFAULT_SAM_CHECKPOINT = Path("E:/AI_Research/dlt8/models/sam/sam_vit_b_01ec64.pth")


# ---------------------------------------------------------------------------
# Sample result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    pair_id: str
    class_name: str
    item_id: str  # e.g. "apple001"
    top_path: str
    side_path: str
    top_conf: float
    side_conf: float
    v_tilde_cm3: float
    mass_g: float
    kcal: float
    alpha_t: float
    alpha_s: float
    # GT columns (added 2026-08-10 to expose detector-mismatch diagnostics
    # directly in CSV; empty/NaN when ground truth was not loaded).
    gt_class: str = ""          # ground-truth class name (e.g. "doughnut")
    gt_volume_cm3: float = float("nan")
    gt_mass_g: float = float("nan")


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def _load_ground_truth(
    density_xls: Path,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Load per-item real volume (cm^3) and real mass (g) from density.xls."""
    import xlrd  # requires xlrd<2.0

    wb = xlrd.open_workbook(str(density_xls))
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for sheet_name in wb.sheet_names():
        if sheet_name == "mix":
            continue
        sh = wb.sheet_by_name(sheet_name)
        items: Dict[str, Tuple[float, float]] = {}
        for r in range(1, sh.nrows):
            item_id = str(sh.cell_value(r, 0))
            v = float(sh.cell_value(r, 2))
            m = float(sh.cell_value(r, 3))
            items[item_id] = (v, m)
        if items:
            out[sheet_name] = items

    # Fix typo in density.xls: sheet is 'fired_dough_twist' instead of 'fried_dough_twist'
    if "fired_dough_twist" in out and "fried_dough_twist" not in out:
        out["fried_dough_twist"] = out["fired_dough_twist"]

    return out


# Public alias
load_ground_truth = _load_ground_truth



# ---------------------------------------------------------------------------
# Item ID extraction & coin/mask helpers
# ---------------------------------------------------------------------------

_FALLBACK_ALPHA_CM_PER_PX = 0.108  # ECUSTFD 1-yuan coin median (~2.5cm / 23px)


def _safe_compute_coin_scale(bbox: Sequence[float]) -> float:
    """Compute coin scale in cm/pixel; returns fallback if bbox is invalid."""
    try:
        scale_obj = compute_coin_scale(bbox)
        if scale_obj is not None and hasattr(scale_obj, "alpha_cm_per_pixel"):
            return float(scale_obj.alpha_cm_per_pixel)
    except Exception as e:
        log = logging.getLogger("faster_rcnn_eval_pipeline")
        log.debug("Coin scale calculation error on bbox %s: %s; using fallback", bbox, e)
    return _FALLBACK_ALPHA_CM_PER_PX


def _crop_or_pass_through(
    mask: Optional[np.ndarray],
    bbox: Sequence[float],
) -> Optional[np.ndarray]:
    """If mask.shape already matches bbox size, return mask as-is.
    Otherwise call crop_mask_to_bbox (full-image mask case).
    Handles both Numpy bool mask and None gracefully.
    """
    if mask is None:
        return None
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = bbox
    bbox_h = int(round(y2)) - int(round(y1))
    bbox_w = int(round(x2)) - int(round(x1))
    if h == bbox_h and w == bbox_w:
        # Already cropped by _grabcut_crop or _sam_segment_crop -- pass through.
        return mask.astype(bool) if mask.dtype != bool else mask
    return crop_mask_to_bbox(mask, bbox)


def _extract_item_id(image_path: Path) -> Optional[str]:
    parsed = parse_filename(image_path.name)
    if parsed is None:
        return None
    stem, _view, _idx = parsed
    return stem


# ---------------------------------------------------------------------------
# Process one (top, side) pair
# ---------------------------------------------------------------------------

def _process_one_pair(
    top_dets: List[dict],
    side_dets: List[dict],
    conf_threshold: float,
    food_info: dict,
    item_id: str = "",
) -> List[SampleResult]:
    """Process a single (top, side) image pair.

    Returns one ``SampleResult`` per successfully paired class.
    """
    log = logging.getLogger("faster_rcnn_eval_pipeline")

    # 1. Coin calibration (independent per view, with cross-fill / fallback).
    top_coins = filter_coin_detections(top_dets, coin_class_name="coin")
    side_coins = filter_coin_detections(side_dets, coin_class_name="coin")
    best_top_coin = select_highest_conf(top_coins)
    best_side_coin = select_highest_conf(side_coins)

    alpha_t: Optional[float] = None
    alpha_s: Optional[float] = None

    if best_top_coin is not None:
        alpha_t = _safe_compute_coin_scale(best_top_coin["bbox"])
    if best_side_coin is not None:
        alpha_s = _safe_compute_coin_scale(best_side_coin["bbox"])

    # Policy:
    # - If BOTH coins found -> use each view's own alpha (paper-faithful)
    # - If ONE coin found -> cross-fill alpha from available view
    # - If NEITHER coin found -> fallback to default ECUSTFD coin alpha (0.108 cm/px)
    if alpha_t is not None and alpha_s is None:
        alpha_s = alpha_t
        log.debug("%s: side coin missing, cross-filling alpha_t=%.4f", item_id, alpha_t)
    elif alpha_s is not None and alpha_t is None:
        alpha_t = alpha_s
        log.debug("%s: top coin missing, cross-filling alpha_s=%.4f", item_id, alpha_s)
    elif alpha_t is None and alpha_s is None:
        log.debug("%s: no coin in either view, using fallback alpha=%.4f", item_id, _FALLBACK_ALPHA_CM_PER_PX)
        alpha_t = _FALLBACK_ALPHA_CM_PER_PX
        alpha_s = _FALLBACK_ALPHA_CM_PER_PX

    # 2. Import view pairing types locally to avoid import cycle.
    from src.view_pairing import Detection, pair_and_filter_cross_view_soft

    def _to_detection(d: dict) -> Detection:
        return Detection(
            class_name=d["class_name"],
            conf=float(d["conf"]),
            bbox=tuple(d["bbox"]),
            mask=d["mask"],
        )

    filtered_pairs = pair_and_filter_cross_view_soft(
        [_to_detection(d) for d in top_dets],
        [_to_detection(d) for d in side_dets],
        primary_conf=conf_threshold,
        top_1_only=True,
    )

    # 3. Per pair: compute volume from GrabCut mask, compute calorie.
    out: List[SampleResult] = []
    for p in filtered_pairs:
        cls = p.class_name
        shape = SHAPE_MODELS.get(cls)
        if shape is None:
            log.warning("No shape model for class '%s'", cls)
            continue

        # Crop mask to bbox safely (supports both cropped and full-image masks).
        try:
            top_mask = _crop_or_pass_through(p.top.mask, p.top.bbox)
            side_mask = _crop_or_pass_through(p.side.mask, p.side.bbox)
        except (ValueError, AttributeError) as e:
            log.debug("Skipping %s/%s: mask crop failed: %s", item_id, cls, e)
            continue

        if top_mask is None or side_mask is None or top_mask.size == 0 or side_mask.size == 0:
            log.debug("Skipping %s/%s: empty mask", item_id, cls)
            continue

        try:
            v_tilde = compute_volume(
                shape,
                ShapeInputs(
                    top_mask=top_mask,
                    side_mask=side_mask,
                    alpha_t=alpha_t,
                    alpha_s=alpha_s,
                ),
            )
        except Exception as e:
            log.warning("Volume computation failed for %s/%s (%s): %s", item_id, cls, shape, e)
            continue

        if not np.isfinite(v_tilde) or v_tilde <= 0:
            log.debug("Skipping %s/%s: invalid volume %.4f", item_id, cls, v_tilde)
            continue

        try:
            r = estimate_calorie_runtime(cls, v_tilde, food_info)
            kcal = float(r.kcal)
            mass = float(v_tilde) * float(DENSITY_G_CM3.get(cls, 0.0))
        except (KeyError, ValueError) as e:
            log.warning("Calorie estimation failed for %s/%s: %s", item_id, cls, e)
            kcal = 0.0
            mass = 0.0

        out.append(
            SampleResult(
                pair_id=f"{item_id}_{cls}",
                class_name=cls,
                item_id=item_id,
                top_path="",
                side_path="",
                top_conf=p.top.conf,
                side_conf=p.side.conf,
                v_tilde_cm3=float(v_tilde),
                mass_g=mass,
                kcal=kcal,
                alpha_t=alpha_t,
                alpha_s=alpha_s,
            )
        )

    if not out:
        # Missed-detection penalty (paper protocol): every pair in the split
        # is evaluated; a pair with no gate-passing prediction contributes a
        # zero-volume sample, i.e. a 100% volume error, to the per-class MAE.
        # Coverage stays honest: these rows keep empty paths/confidences, so
        # they are excluded from n_pairs_with_samples.
        fallback_cls = ""
        if item_id:
            for fc in FOOD_CLASSES:
                if item_id.startswith(fc):
                    fallback_cls = fc
                    break
        if not fallback_cls:
            all_dets = [d for d in top_dets + side_dets if d.get("class_name") != "coin"]
            if all_dets:
                fallback_cls = max(all_dets, key=lambda d: float(d.get("conf", 0.0)))["class_name"]
            else:
                fallback_cls = FOOD_CLASSES[0]

        out.append(
            SampleResult(
                pair_id=f"{item_id}_{fallback_cls}",
                class_name=fallback_cls,
                item_id=item_id,
                top_path="",
                side_path="",
                top_conf=0.0,
                side_conf=0.0,
                v_tilde_cm3=0.0,
                mass_g=0.0,
                kcal=0.0,
                alpha_t=alpha_t or _FALLBACK_ALPHA_CM_PER_PX,
                alpha_s=alpha_s or _FALLBACK_ALPHA_CM_PER_PX,
            )
        )

    return out


# ---------------------------------------------------------------------------
# Match sample to ground truth
# ---------------------------------------------------------------------------

def _match_sample_to_ground_truth(
    class_name: str,
    item_id: str | None,
    gt_by_class: Dict[str, Dict[str, Tuple[float, float]]],
) -> Tuple[float, float] | None:
    if item_id is None:
        return None
    class_gt = gt_by_class.get(class_name)
    if class_gt is None:
        return None
    return class_gt.get(item_id)


# ---------------------------------------------------------------------------
# Evaluate on a set of pairs (with optional beta calibration)
# ---------------------------------------------------------------------------

def _evaluate_samples_set(
    model,
    pairs: List[Tuple[Path, Path]],
    groups: List[dict],
    conf_threshold: float,
    food_info: dict,
    gt_by_class: Dict[str, Dict[str, Tuple[float, float]]] | None,
    betas: Dict[str, float] | None = None,
    infer_fn=None,
    mask_backend: str = "grabcut",
    sam_predictor=None,
) -> Tuple[List[SampleResult], List[float], List[float], List[float], List[float]]:
    """Evaluate all (top, side) pairs using the given inference function.

    Args:
        model: Detection model (Faster R-CNN or YOLO).
        pairs: List of (top_path, side_path) tuples.
        groups: Groups dict from dataset_split.
        conf_threshold: Confidence threshold for pairing.
        food_info: Parsed food_info.xls.
        gt_by_class: Ground truth from density.xls.
        betas: Optional beta calibration dict.
        infer_fn: Inference function with signature ``(model, path, conf) -> List[dict]``.
            Used only when ``mask_backend == "grabcut"`` (default).
        mask_backend: ``"grabcut"`` (default) or ``"sam"``. Switches the
            segmentation backend; Faster R-CNN bbox detection is unchanged.
        sam_predictor: Pre-loaded ``SamPredictor`` (from ``load_sam``),
            required when ``mask_backend == "sam"``.
    """
    log = logging.getLogger("faster_rcnn_eval_pipeline")
    if infer_fn is None:
        import src.faster_rcnn.inference as _inf
        infer_fn = _inf.predict_one

    if mask_backend == "sam" and sam_predictor is None:
        raise ValueError(
            "mask_backend='sam' requires sam_predictor to be provided. "
            "Call load_sam() first and pass the predictor here."
        )

    # Reference-counted cache to prevent CPU RAM exhaustion (see
    # CPU_RAM_EXHAUSTION_ROOT_CAUSE.md). Each image is released from cache
    # as soon as no upcoming pair needs it, keeping peak memory bounded
    # to the working set (a few dozen images) rather than all images.
    from collections import Counter
    image_refcount: Counter = Counter()
    for top, side in pairs:
        image_refcount[str(top)] += 1
        image_refcount[str(side)] += 1

    cache: Dict[str, List[dict]] = {}

    def _get_dets(p: Path) -> List[dict]:
        p_str = str(p)
        if p_str not in cache:
            infer_floor_conf = min(float(conf_threshold), 0.01)
            if mask_backend == "grabcut":
                cache[p_str] = infer_fn(model, p, conf=infer_floor_conf)
            elif mask_backend == "sam":
                import src.faster_rcnn.sam_inference as _frcnn_sam
                cache[p_str] = _frcnn_sam.predict_one_with_sam(
                    model, sam_predictor, p, conf=infer_floor_conf,
                )
            else:
                raise ValueError(
                    f"Unknown mask_backend={mask_backend!r}; "
                    f"expected 'grabcut' or 'sam'."
                )
        return cache[p_str]

    def _should_prefetch(pair_idx: int) -> tuple:
        """Prefetch next pair's images if they're about to be released (reduces I/O latency)."""
        if pair_idx + 1 >= len(pairs):
            return None, None
        return pairs[pair_idx + 1]

    samples: List[SampleResult] = []
    for pair_idx, (top, side) in enumerate(pairs):
        top_dets = _get_dets(top)
        side_dets = _get_dets(side)

        # Prefetch optimization: Load next pair's images early if not already cached
        # (overlaps disk I/O with current pair processing, ~10-15% speedup on HDD/slow SSD)
        next_top, next_side = _should_prefetch(pair_idx)
        if next_top and str(next_top) not in cache:
            _ = _get_dets(next_top)
        if next_side and str(next_side) not in cache and str(next_side) != str(next_top):
            _ = _get_dets(next_side)

        # Release cache entries once no remaining pair needs them.
        for _p in (top, side):
            _p_str = str(_p)
            image_refcount[_p_str] -= 1
            if image_refcount[_p_str] <= 0:
                cache.pop(_p_str, None)

        item_id = _extract_item_id(top) or _extract_item_id(side) or ""
        ps = _process_one_pair(
            top_dets, side_dets, conf_threshold, food_info, item_id=item_id
        )
        for s in ps:
            is_penalty = (s.top_conf == 0.0 and s.side_conf == 0.0 and s.v_tilde_cm3 == 0.0)
            if not is_penalty:
                s.top_path = str(top)
                s.side_path = str(side)
                # Apply beta calibration if available.
                if betas and s.class_name in betas:
                    s.v_tilde_cm3 = calibrate_volume(s.v_tilde_cm3, s.class_name, betas)
                    s.mass_g = s.v_tilde_cm3 * float(DENSITY_G_CM3.get(s.class_name, 0.0))
            # Penalty rows keep empty paths (honest coverage) but still get
            # GT matched, so their 100% error lands in the right class bucket.
            # Look up ground-truth volume/mass for this sample so the CSV row
            # exposes the detector-vs-GT mismatch directly (2026-08-10 fix).
            gt = _match_sample_to_ground_truth(s.class_name, s.item_id, gt_by_class or {})
            if gt is not None:
                s.gt_volume_cm3, s.gt_mass_g = gt
                s.gt_class = s.class_name
            else:
                # Search the entire gt_by_class for an item_id match even when
                # the predicted class is wrong -- this is what we need to flag
                # detector-collapse bugs (e.g. predicted "bun", real "doughnut").
                for _real_cls, _items in (gt_by_class or {}).items():
                    if s.item_id in _items:
                        s.gt_volume_cm3, s.gt_mass_g = _items[s.item_id]
                        s.gt_class = _real_cls
                        break
        samples.extend(ps)

    # Separate predicted vs ground-truth for metric computation.
    v_pred = [s.v_tilde_cm3 for s in samples]
    v_real: List[float] = []
    m_pred: List[float] = []
    m_real: List[float] = []
    for s in samples:
        lookup_cls = s.gt_class if s.gt_class else s.class_name
        gt = _match_sample_to_ground_truth(lookup_cls, s.item_id, gt_by_class or {})
        if gt is None:
            v_real.append(float("nan"))
            m_real.append(float("nan"))
        else:
            v_real.append(gt[0])
            m_real.append(gt[1])
        m_pred.append(s.mass_g)

    detector_collapsed = False
    if len(samples) >= 50:
        unique_cls = {s.class_name for s in samples if s.class_name}
        if len(unique_cls) <= 1:
            detector_collapsed = True
            log.warning(
                "[COLLAPSE] Detector produced %d samples with only %d unique "
                "class label(s) (%s). Almost certainly detector collapse at "
                "conf=%.2f. Check CSV's class_name vs gt_class columns.",
                len(samples), len(unique_cls), sorted(unique_cls), conf_threshold,
            )

    return samples, v_pred, v_real, m_pred, m_real


# ---------------------------------------------------------------------------
# Run one (split, conf, apply_beta) configuration
# ---------------------------------------------------------------------------

def _run_one_config(
    *,
    model_weights: Path,
    model_variant: str,  # "old" or "new"
    split: str,
    images_dir: Path,
    imagesets_dir: Path,
    conf_threshold: float,
    food_info: dict,
    gt_by_class: Dict[str, Dict[str, Tuple[float, float]]] | None,
    out_dir: Path,
    apply_beta: bool = False,
    device: str | int | None = None,
    mask_backend: str = "grabcut",
    sam_predictor=None,
    infer_fn=None,
    iou_threshold: float = 0.5,
    **kwargs,
) -> Dict[str, Path]:
    """Run the full Faster R-CNN evaluation pipeline for one configuration.

    Args:
        mask_backend: ``"grabcut"`` (default) or ``"sam"``. Selects the
            segmentation backend used for mask generation. Faster R-CNN
            bbox detection is unchanged.
        sam_predictor: Pre-loaded ``SamPredictor`` (required when
            ``mask_backend == "sam"``). Passed through to both the
            train (beta fit) and test (evaluation) calls.
    """
    log = logging.getLogger("faster_rcnn_eval_pipeline")
    log.info(
        "=== config: variant=%s, split=%s, conf=%.2f, apply_beta=%s, mask_backend=%s ===",
        model_variant, split, conf_threshold, apply_beta, mask_backend,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model.
    log.info("Loading Faster R-CNN model: %s", model_weights)
    model = load_faster_rcnn(model_weights, device=device)
    log.info("Model loaded on device=%s", next(model.parameters()).device)

    betas: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Beta calibration: fit on train split, apply to test.
    # ------------------------------------------------------------------
    if apply_beta:
        log.info("Beta calibration requested: fitting on train split (paper 50/50)")
        unique_map = {
            p.name.lower(): p
            for p in (list(images_dir.glob("*.JPG")) + list(images_dir.glob("*.jpg")))
        }
        all_image_paths = list(unique_map.values())
        train_paths, _ = get_paper_50_50_split(all_image_paths)
        tr_groups = group_top_side(train_paths)
        tr_pairs = make_pairs(tr_groups)
        log.info("  Train split: %d stems, %d pairs", len(train_paths), len(tr_pairs))

        tr_samples, tr_vpred, tr_vreal, _, _ = _evaluate_samples_set(
            model, tr_pairs, tr_groups, conf_threshold, food_info, gt_by_class,
            betas=None,
            infer_fn=infer_fn,
            mask_backend=mask_backend,
            sam_predictor=sam_predictor,
        )
        log.info("  Train samples processed: %d", len(tr_samples))

        tr_records = []
        for i, s in enumerate(tr_samples):
            tr_records.append({
                "class_name": s.class_name,
                "v_tilde_cm3": tr_vpred[i],
                "v_real_cm3": tr_vreal[i],
            })

        betas = compute_class_betas(tr_records)
        beta_json_path = out_dir / f"betas_train_conf{int(conf_threshold*100)}.json"
        save_betas(betas, beta_json_path)
        log.info("  Beta calibration done: %d classes -> %s", len(betas), beta_json_path.name)

    # ------------------------------------------------------------------
    # Load split and group pairs.
    # ------------------------------------------------------------------
    split_file = imagesets_dir / f"{split}.txt"
    stems = load_split(split_file)
    paths = resolve_image_paths(stems, images_dir)
    log.info("  Split '%s': %d/%d images resolved", split, len(paths), len(stems))

    groups = group_top_side(paths)
    pairs = make_pairs(groups)
    max_pairs = kwargs.get("max_pairs")
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
        log.info("  [Testing] Sliced pairs to max_pairs=%d", len(pairs))
    log.info("  (top, side) pairs: %d", len(pairs))

    # ------------------------------------------------------------------
    # Evaluate on test split.
    # ------------------------------------------------------------------
    samples, v_pred, v_real, m_pred, m_real = _evaluate_samples_set(
        model, pairs, groups, conf_threshold, food_info, gt_by_class,
        betas=betas if apply_beta else None,
        infer_fn=infer_fn,
        mask_backend=mask_backend,
        sam_predictor=sam_predictor,
    )
    log.info("  Samples after pairing: %d", len(samples))

    # Count unique (top, side) image pairs that produced at least one sample.
    # Penalty rows (missed detections) keep empty paths, so they are excluded
    # here -- this is the honest Coverage numerator.
    unique_pairs_with_samples = len(
        {(s.top_path, s.side_path) for s in samples if s.top_path and s.side_path}
    )
    log.info("  Unique pairs with samples: %d / %d", unique_pairs_with_samples, len(pairs))

    # Determine detector-collapse flag for evaluated samples (heuristic: >=50 samples, <=1 class)
    detector_collapsed: Optional[bool] = None
    if len(samples) >= 50:
        unique_cls = {s.class_name for s in samples if s.class_name}
        detector_collapsed = len(unique_cls) <= 1

    # ------------------------------------------------------------------
    # Compute per-class ME.
    # ------------------------------------------------------------------
    class_names = [s.class_name for s in samples]
    per_class = compute_me_per_class(
        class_names, v_pred, v_real, m_pred=m_pred, m_real=m_real
    )
    overall = aggregate(per_class)

    log.info(
        "  Overall ME_vol=%.4f%%  |ME_vol|=%.4f%%  ME_mass=%.4f%%  (n_classes=%d)",
        overall["mean_me_volume_pct"], overall["mean_abs_me_volume_pct"],
        overall["mean_me_mass_pct"], overall["n_classes"]
    )

    # ------------------------------------------------------------------
    # Write outputs.
    # ------------------------------------------------------------------
    tag_suffix = "_beta" if apply_beta else ""
    tag = f"{split}_conf{int(conf_threshold*100)}{tag_suffix}"

    csv_path = out_dir / f"samples_{tag}.csv"
    json_path = out_dir / f"report_{tag}.json"

    fieldnames = [
        "pair_id", "class_name", "item_id", "top_path", "side_path",
        "top_conf", "side_conf", "v_tilde_cm3", "mass_g", "kcal",
        "alpha_t", "alpha_s",
        # gt_* columns added 2026-08-10 to expose detector-collapse diagnostics
        # directly in CSV. The 99.86% bun-collapse we saw on 20260810-001221
        # was invisible until these columns were added. gt_class is the *true*
        # label (item_id lookup across all gt_by_class entries, even when the
        # predicted class is wrong); class_name is the *predicted* label.
        "gt_class", "gt_volume_cm3", "gt_mass_g",
    ]  # noqa: E501
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in samples:
            w.writerow(asdict(s))
    log.info("  Wrote: %s", csv_path)

    report = {
        "config": {
            "model_variant": model_variant,
            "model_weights": str(model_weights),
            "split": split,
            "conf_threshold": conf_threshold,
            "apply_beta": apply_beta,
            "mask_backend": mask_backend,
            "n_pairs": len(pairs),
            "n_samples": len(samples),
            "n_pairs_with_samples": unique_pairs_with_samples,
            "detector": f"faster_rcnn_{mask_backend}",
            # 2026-08-12: expose detector-collapse flag so downstream
            # consumers can surface it in benchmark tables without
            # re-running the collapse detector. ``None`` is fine here
            # if there are fewer than 50 samples (heuristic N/A).
            "detector_collapsed": detector_collapsed,
        },
        "betas": betas if apply_beta else {},
        "per_class": [
            {
                "class_name": r.class_name,
                "n_samples": r.n_samples,
                "me_volume_pct": r.me_volume_pct,
                "me_mass_pct": r.me_mass_pct,
                "abs_me_volume_pct": r.abs_me_volume_pct,
            }
            for r in per_class
        ],
        "overall": overall,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("  Wrote: %s", json_path)

    return {"csv": csv_path, "json": json_path}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Faster R-CNN E2E evaluation pipeline. "
            "Choose segmentation backend via --mask-backend (grabcut or sam)."
        ),
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_MODEL_OLD),
        help="Path to Faster R-CNN .pt checkpoint. Default: faster_rcnn_old_best.pt",
    )
    parser.add_argument(
        "--variant",
        default="old",
        choices=["old", "new"],
        help="Model variant label ('old' or 'new') used in output filenames.",
    )
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR))
    parser.add_argument("--imagesets-dir", default=str(DEFAULT_IMAGESETS_DIR))
    parser.add_argument("--food-info", default=str(DEFAULT_FOOD_INFO))
    parser.add_argument("--density-xls", default=str(DEFAULT_DATA_ROOT / "density.xls"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--split",
        action="append",
        choices=["val", "test"],
        help="Repeatable: evaluate on val and/or test split.",
    )
    parser.add_argument(
        "--conf",
        action="append",
        type=float,
        help="Repeatable: confidence threshold (e.g. 0.5 0.8).",
    )
    parser.add_argument(
        "--apply-beta",
        action="store_true",
        help="Compute beta calibration on train split and apply to evaluation.",
    )
    parser.add_argument(
        "--mask-backend",
        default="grabcut",
        choices=["grabcut", "sam"],
        help="Segmentation backend: 'grabcut' (OpenCV) or 'sam' (SAM1 vit_b).",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default=str(DEFAULT_SAM_CHECKPOINT),
        help="Path to SAM1 checkpoint (only used when --mask-backend=sam).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (e.g. '0' for GPU 0, 'cpu'). Default: auto.",
    )
    args = parser.parse_args()

    splits = args.split or ["test"]
    confs = args.conf or [0.8]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    food_info = parse_food_info(args.food_info)
    try:
        gt_by_class = _load_ground_truth(Path(args.density_xls))
    except Exception as e:
        print(f"Failed to load density.xls: {e}. Continuing without GT.")
        gt_by_class = None

    # Load SAM once if requested.
    sam_predictor = None
    if args.mask_backend == "sam":
        from src.faster_rcnn.sam_inference import load_sam
        sam_predictor = load_sam(
            checkpoint_path=args.sam_checkpoint,
            model_type="vit_b",
            device=args.device,
        )

    for split in splits:
        for conf in confs:
            print(
                f"\n=== variant={args.variant} split={split} conf={conf} "
                f"apply_beta={args.apply_beta} mask_backend={args.mask_backend} ==="
            )
            _run_one_config(
                model_weights=Path(args.weights),
                model_variant=args.variant,
                split=split,
                images_dir=Path(args.images_dir),
                imagesets_dir=Path(args.imagesets_dir),
                conf_threshold=conf,
                food_info=food_info,
                gt_by_class=gt_by_class,
                out_dir=out_dir / "predictions",
                apply_beta=args.apply_beta,
                device=args.device,
                mask_backend=args.mask_backend,
                sam_predictor=sam_predictor,
            )


if __name__ == "__main__":
    main()
