# -*- coding: utf-8 -*-
"""YOLO26-seg evaluation pipeline — E2E inference + relaxed coin-gate + volume + beta + metrics.

This module is the YOLO26-seg counterpart to ``src/faster_rcnn/eval_pipeline.py``
and ``src/yolo_bbox/eval_pipeline.py``. It maintains 100% parity in benchmark logic,
relaxed coin calibration, view pairing, beta calibration, and output schemas.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.calorie_estimation import (
    estimate_calorie_runtime,
    parse_food_info,
)
from src.coin_calibration import (
    compute_coin_scale,
    filter_coin_detections,
    select_highest_conf,
)
from src.constants import DENSITY_G_CM3, FOOD_CLASSES, SHAPE_MODELS
from src.e2e_pipeline.dataset_split import (
    get_paper_50_50_split,
    group_top_side,
    load_split,
    make_pairs,
    parse_filename,
    resolve_image_paths,
)
from src.e2e_pipeline.metrics import aggregate, compute_me_per_class

from src.beta_correction import (
    calibrate_volume,
    compute_class_betas,
    save_betas,
)
from src.segmentation_runtime.crop_to_bbox import crop_mask_to_bbox
from src.volume_models import ShapeInputs, compute_volume

from src.yolo_seg_eval.inference import load_yolo_seg, predict_one as yolo_seg_predict_one

_FALLBACK_ALPHA_CM_PER_PX = 0.108  # ECUSTFD 1-yuan coin median (~2.5cm / 23px)


@dataclass
class SampleResult:
    pair_id: str
    class_name: str
    item_id: str
    top_path: str
    side_path: str
    top_conf: float
    side_conf: float
    v_tilde_cm3: float
    mass_g: float
    kcal: float
    alpha_t: float
    alpha_s: float
    gt_class: str = ""
    gt_volume_cm3: float = float("nan")
    gt_mass_g: float = float("nan")


def _safe_compute_coin_scale(bbox: Sequence[float]) -> float:
    """Compute coin scale in cm/pixel; returns fallback if bbox is invalid."""
    try:
        scale_obj = compute_coin_scale(bbox)
        if scale_obj is not None and hasattr(scale_obj, "alpha_cm_per_pixel"):
            return float(scale_obj.alpha_cm_per_pixel)
    except Exception as e:
        log = logging.getLogger("yolo_seg_eval_pipeline")
        log.debug("Coin scale calculation error on bbox %s: %s; using fallback", bbox, e)
    return _FALLBACK_ALPHA_CM_PER_PX


def _crop_or_pass_through(
    mask: Optional[np.ndarray],
    bbox: Sequence[float],
) -> Optional[np.ndarray]:
    """Pass through if mask shape matches bbox, else crop."""
    if mask is None:
        return None
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = bbox
    bbox_h = int(round(y2)) - int(round(y1))
    bbox_w = int(round(x2)) - int(round(x1))
    if h == bbox_h and w == bbox_w:
        return mask.astype(bool) if mask.dtype != bool else mask
    return crop_mask_to_bbox(mask, bbox)


def _extract_item_id(image_path: Path) -> Optional[str]:
    parsed = parse_filename(image_path.name)
    if parsed is None:
        return None
    stem, _view, _idx = parsed
    return stem


def _process_one_pair(
    top_dets: List[dict],
    side_dets: List[dict],
    conf_threshold: float,
    food_info: dict,
    item_id: str = "",
) -> List[SampleResult]:
    """Process a single (top, side) image pair with relaxed coin calibration."""
    log = logging.getLogger("yolo_seg_eval_pipeline")

    # 1. Coin calibration (cross-fill / fallback)
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

    # 2. View pairing
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

    # 3. Compute volume & calorie
    out: List[SampleResult] = []
    for p in filtered_pairs:
        cls = p.class_name
        shape = SHAPE_MODELS.get(cls)
        if shape is None:
            log.warning("No shape model for class '%s'", cls)
            continue

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


def _evaluate_samples_set(
    model,
    pairs: List[Tuple[Path, Path]],
    groups: List[dict],
    conf_threshold: float,
    food_info: dict,
    gt_by_class: Dict[str, Dict[str, Tuple[float, float]]] | None,
    betas: Dict[str, float] | None = None,
    infer_fn=None,
    iou_threshold: float = 0.50,
    imgsz: int = 480,
    device: Optional[Union[str, int]] = None,
) -> Tuple[List[SampleResult], List[float], List[float], List[float], List[float]]:
    """Evaluate all pairs using YOLO26-seg model."""
    log = logging.getLogger("yolo_seg_eval_pipeline")

    if infer_fn is None:
        import src.yolo_seg_eval.inference as _yolo_seg_inf
        infer_fn = _yolo_seg_inf.predict_one

    # Reference-counted cache: an image is only kept in RAM while at least
    # one *upcoming* pair still needs it. This bounds peak memory to the
    # working set (a handful of images) instead of every image in the split
    # (previously ~39k images cached simultaneously -> CPU RAM OOM at
    # low/zero conf thresholds, where each image yields hundreds of
    # detections/masks). See CPU_RAM_EXHAUSTION_ROOT_CAUSE.md.
    from collections import Counter
    image_refcount: Counter = Counter()
    for top, side in pairs:
        image_refcount[str(top)] += 1
        image_refcount[str(side)] += 1

    cache: Dict[str, List[dict]] = {}

    def _get_dets(p: Path) -> List[dict]:
        # Include conf_threshold in cache key to avoid using wrong threshold
        cache_key = f"{p}__conf{conf_threshold:.4f}"
        if cache_key not in cache:
            log.debug(f"[CACHE MISS] {p.name}, calling infer with conf={conf_threshold:.2f}")
            infer_floor_conf = min(float(conf_threshold), 0.01)
            cache[cache_key] = infer_fn(
                model, p, conf=infer_floor_conf, iou_threshold=iou_threshold,
                imgsz=imgsz, device=device,
            )
            log.debug(f"[CACHE STORE] {p.name}: {len(cache[cache_key])} detections")
        else:
            log.debug(f"[CACHE HIT] {p.name}: returning cached {len(cache[cache_key])} detections")
        return cache[cache_key]

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
        if next_top:
            _next_t_key = f"{next_top}__conf{conf_threshold:.4f}"
            if _next_t_key not in cache:
                _ = _get_dets(next_top)
        if next_side and str(next_side) != str(next_top):
            _next_s_key = f"{next_side}__conf{conf_threshold:.4f}"
            if _next_s_key not in cache:
                _ = _get_dets(next_side)

        # Release cache entries once no remaining pair needs them.
        for _p in (top, side):
            _p_str = str(_p)
            image_refcount[_p_str] -= 1
            if image_refcount[_p_str] <= 0:
                _p_key = f"{_p}__conf{conf_threshold:.4f}"
                cache.pop(_p_key, None)

        item_id = _extract_item_id(top) or _extract_item_id(side) or ""
        ps = _process_one_pair(top_dets, side_dets, conf_threshold, food_info, item_id=item_id)
        for s in ps:
            is_penalty = (s.top_conf == 0.0 and s.side_conf == 0.0 and s.v_tilde_cm3 == 0.0)
            if not is_penalty:
                s.top_path = str(top)
                s.side_path = str(side)
                if betas and s.class_name in betas:
                    s.v_tilde_cm3 = calibrate_volume(s.v_tilde_cm3, s.class_name, betas)
                    s.mass_g = s.v_tilde_cm3 * float(DENSITY_G_CM3.get(s.class_name, 0.0))
            # Penalty rows keep empty paths (honest coverage) but still get
            # GT matched, so their 100% error lands in the right class bucket.

            gt = _match_sample_to_ground_truth(s.class_name, s.item_id, gt_by_class or {})
            if gt is not None:
                s.gt_volume_cm3, s.gt_mass_g = gt
                s.gt_class = s.class_name
            else:
                for _real_cls, _items in (gt_by_class or {}).items():
                    if s.item_id in _items:
                        s.gt_volume_cm3, s.gt_mass_g = _items[s.item_id]
                        s.gt_class = _real_cls
                        break
        samples.extend(ps)

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

    return samples, v_pred, v_real, m_pred, m_real


def _run_one_config(
    *,
    model_weights: Path,
    model_variant: str = "yolo26_seg",
    split: str = "test",
    images_dir: Path,
    imagesets_dir: Path,
    conf_threshold: float = 0.80,
    food_info: dict,
    gt_by_class: Dict[str, Dict[str, Tuple[float, float]]] | None,
    out_dir: Path,
    apply_beta: bool = True,
    device: str | int | None = None,
    infer_fn=None,
    iou_threshold: float = 0.50,
    imgsz: int = 480,
    max_pairs: Optional[int] = None,
    **kwargs,
) -> Dict[str, Path]:
    """Run the full YOLO26-seg E2E pipeline for one configuration."""
    log = logging.getLogger("yolo_seg_eval_pipeline")
    log.info(
        "=== config: variant=%s, split=%s, conf=%.2f, apply_beta=%s, max_pairs=%s ===",
        model_variant, split, conf_threshold, apply_beta, max_pairs,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading YOLO-seg model: %s", model_weights)
    model = load_yolo_seg(model_weights, device=device)

    betas: Dict[str, float] = {}

    # Beta calibration on train 50/50 split
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
        if max_pairs is not None:
            tr_pairs = tr_pairs[:max_pairs]
            log.info("  [Testing] Sliced train pairs to max_pairs=%d", len(tr_pairs))
        log.info("  Train split: %d stems, %d pairs", len(train_paths), len(tr_pairs))

        tr_samples, tr_vpred, tr_vreal, _, _ = _evaluate_samples_set(
            model, tr_pairs, tr_groups, conf_threshold, food_info, gt_by_class,
            betas=None,
            infer_fn=infer_fn,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
            device=device,
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

    # Load split and group pairs
    split_file = imagesets_dir / f"{split}.txt"
    stems = load_split(split_file)
    paths = resolve_image_paths(stems, images_dir)
    log.info("  Split '%s': %d/%d images resolved", split, len(paths), len(stems))

    groups = group_top_side(paths)
    pairs = make_pairs(groups)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
        log.info("  [Testing] Sliced test pairs to max_pairs=%d", len(pairs))
    log.info("  (top, side) pairs: %d", len(pairs))

    # Evaluate on test split
    samples, v_pred, v_real, m_pred, m_real = _evaluate_samples_set(
        model, pairs, groups, conf_threshold, food_info, gt_by_class,
        betas=betas if apply_beta else None,
        infer_fn=infer_fn,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        device=device,
    )
    log.info("  Samples after pairing: %d", len(samples))

    # Count unique (top, side) image pairs that produced at least one sample.
    # Previously this counted unique item_id strings, which undercounts:
    # one item spans many (top, side) view pairs.
    unique_pairs_with_samples = len(
        {(s.top_path, s.side_path) for s in samples if s.top_path and s.side_path}
    )
    log.info("  Unique pairs with samples: %d / %d", unique_pairs_with_samples, len(pairs))

    detector_collapsed: Optional[bool] = None
    if len(samples) >= 50:
        unique_cls = {s.class_name for s in samples if s.class_name}
        detector_collapsed = len(unique_cls) <= 1

    # Compute metrics
    class_names = [s.class_name for s in samples]
    per_class = compute_me_per_class(class_names, v_pred, v_real, m_pred=m_pred, m_real=m_real)
    overall = aggregate(per_class)

    log.info(
        "  Overall ME_vol=%.4f%%  |ME_vol|=%.4f%%  ME_mass=%.4f%%  (n_classes=%d)",
        overall["mean_me_volume_pct"], overall["mean_abs_me_volume_pct"],
        overall["mean_me_mass_pct"], overall["n_classes"]
    )

    # Output files
    tag_suffix = "_beta" if apply_beta else ""
    tag = f"{split}_conf{int(conf_threshold*100)}{tag_suffix}"
    csv_path = out_dir / f"samples_{tag}.csv"
    json_path = out_dir / f"report_{tag}.json"

    fieldnames = [
        "pair_id", "class_name", "item_id", "top_path", "side_path",
        "top_conf", "side_conf", "v_tilde_cm3", "mass_g", "kcal",
        "alpha_t", "alpha_s",
        "gt_class", "gt_volume_cm3", "gt_mass_g",
    ]
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
            "n_pairs": len(pairs),
            "n_samples": len(samples),
            "n_pairs_with_samples": unique_pairs_with_samples,
            "detector": model_variant,
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
