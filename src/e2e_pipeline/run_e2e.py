"""End-to-end pipeline orchestrator.

This module glues together:

    YOLO-seg inference
        -> segmentation_runtime.predict_one
        -> segmentation_runtime.crop_mask_to_bbox
    Coin calibration
        -> coin_calibration.compute_coin_scale
    View pairing
        -> view_pairing.pair_top_side
    Volume estimation
        -> volume_models.compute_volume
    Calorie estimation
        -> calorie_estimation.estimate_calorie_runtime

The single calorie formula is the runtime MATLAB baseline:

    C = q_k * V_tilde       (kcal)

There is no beta correction. ``--conf`` can be repeated to evaluate
multiple confidence thresholds. The runner writes one CSV and one
JSON per configuration; ``render_report.py`` aggregates them into a
single Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from src.segmentation_runtime import (
    crop_mask_to_bbox,
    load_yolo_seg,
    predict_one,
)
from src.view_pairing import (
    Detection,
    filter_pairs_by_confidence,
    pair_top_side,
)
from src.volume_models import ShapeInputs, compute_volume

from .dataset_split import (
    group_top_side,
    load_split,
    make_pairs,
    parse_filename,
    resolve_image_paths,
)
from .metrics import aggregate, compute_me_per_class

from src.beta_correction import (
    calibrate_volume,
    compute_class_betas,
    save_betas,
)
from src.e2e_pipeline.dataset_split import (
    get_paper_50_50_split,
    group_top_side,
    load_split,
    make_pairs,
    parse_filename,
    resolve_image_paths,
)
from src.e2e_pipeline.metrics import aggregate, compute_me_per_class

DEFAULT_DATA_ROOT = Path("E:/AI_Research/dlt8/data/raw/ECUSTFD")
DEFAULT_IMAGES_DIR = DEFAULT_DATA_ROOT / "JPEGImages"
DEFAULT_IMAGESETS_DIR = DEFAULT_DATA_ROOT / "ImageSets" / "Main"
def _resolve_default_weights() -> Path:
    candidates = [
        Path("E:/AI_Research/dlt8/models/ecustfd_yolo26seg_trainval-3_best.pt"),
        Path("E:/AI_Research/dlt8/runs/yolo_seg/ecustfd_yolo26seg_trainval-3/weights/best.pt"),
        Path("E:/AI_Research/dlt8/models/ecustfd_yolo26seg_trainval_best.pt"),
        Path("E:/AI_Research/dlt8/models/ecustfd_yolo26seg-2_best.pt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


DEFAULT_WEIGHTS = _resolve_default_weights()

DEFAULT_FOOD_INFO = Path("E:/AI_Research/dlt8/ECUSTFD/faster_rcnn/food_info.xls")
DEFAULT_OUTPUT_DIR = Path("E:/AI_Research/dlt8/outputs")


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


def _detection_from_dict(d: dict) -> Detection:
    return Detection(
        class_name=d["class_name"],
        conf=float(d["conf"]),
        bbox=tuple(d["bbox"]),
        mask=d["mask"],
    )


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
    # 1. Coin calibration (independent per view).
    top_coins = filter_coin_detections(top_dets, coin_class_name="coin")
    side_coins = filter_coin_detections(side_dets, coin_class_name="coin")
    best_top_coin = select_highest_conf(top_coins)
    best_side_coin = select_highest_conf(side_coins)
    if best_top_coin is None or best_side_coin is None:
        return []
    alpha_t_obj = compute_coin_scale(best_top_coin["bbox"])
    alpha_s_obj = compute_coin_scale(best_side_coin["bbox"])
    alpha_t = alpha_t_obj.alpha_cm_per_pixel
    alpha_s = alpha_s_obj.alpha_cm_per_pixel

    # 2. Pair top + side.
    pairs = pair_top_side(
        [_detection_from_dict(d) for d in top_dets],
        [_detection_from_dict(d) for d in side_dets],
    )
    pairs = filter_pairs_by_confidence(pairs.pairs, conf_threshold=conf_threshold)

    # 3. Per pair: crop mask, compute volume, compute calorie.
    out: List[SampleResult] = []
    for p in pairs:
        cls = p.class_name
        shape = SHAPE_MODELS.get(cls)
        if shape is None:
            continue
        try:
            top_mask = crop_mask_to_bbox(p.top.mask, p.top.bbox)
            side_mask = crop_mask_to_bbox(p.side.mask, p.side.bbox)
            v_tilde = compute_volume(
                shape,
                ShapeInputs(
                    top_mask=top_mask,
                    side_mask=side_mask,
                    alpha_t=alpha_t,
                    alpha_s=alpha_s,
                ),
            )
        except Exception:
            continue

        if not np.isfinite(v_tilde) or v_tilde <= 0:
            continue

        try:
            r = estimate_calorie_runtime(cls, v_tilde, food_info)
            kcal = float(r.kcal)
            mass = float(v_tilde) * float(DENSITY_G_CM3.get(cls, 0.0))
        except KeyError:
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
    return out


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
            item_id = str(sh.cell_value(r, 0))  # e.g. "apple001"
            v = float(sh.cell_value(r, 2))
            m = float(sh.cell_value(r, 3))
            items[item_id] = (v, m)
        if items:
            out[sheet_name] = items
    return out


def _extract_item_id(image_path: Path) -> Optional[str]:
    parsed = parse_filename(image_path.name)
    if parsed is None:
        return None
    stem, _view, _idx = parsed
    return stem


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
    groups: List[Dict[str, List[Path]]],
    conf_threshold: float,
    food_info: dict,
    gt_by_class: Dict[str, Dict[str, Tuple[float, float]]] | None,
    betas: Dict[str, float] | None = None,
) -> Tuple[List[SampleResult], List[float], List[float], List[float], List[float]]:
    cache: Dict[str, List[dict]] = {}
    for g in groups:
        for p in g.get("top", []) + g.get("side", []):
            if str(p) not in cache:
                cache[str(p)] = predict_one(
                    model, p, conf=conf_threshold, imgsz=480
                )

    samples: List[SampleResult] = []
    for top, side in pairs:
        top_dets = cache.get(str(top), [])
        side_dets = cache.get(str(side), [])
        item_id = _extract_item_id(top) or _extract_item_id(side) or ""
        ps = _process_one_pair(
            top_dets, side_dets, conf_threshold, food_info, item_id=item_id
        )
        for s in ps:
            s.top_path = str(top)
            s.side_path = str(side)
            # Apply beta calibration if requested
            if betas and s.class_name in betas:
                s.v_tilde_cm3 = calibrate_volume(s.v_tilde_cm3, s.class_name, betas)
                s.mass_g = s.v_tilde_cm3 * float(DENSITY_G_CM3.get(s.class_name, 0.0))
        samples.extend(ps)

    v_pred = [s.v_tilde_cm3 for s in samples]
    v_real: List[float] = []
    m_pred: List[float] = []
    m_real: List[float] = []
    for s in samples:
        gt = _match_sample_to_ground_truth(s.class_name, s.item_id, gt_by_class or {})
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
    weights: Path,
    split: str,
    images_dir: Path,
    imagesets_dir: Path,
    conf_threshold: float,
    food_info: dict,
    gt_by_class: Dict[str, Dict[str, Tuple[float, float]]] | None,
    out_dir: Path,
    apply_beta: bool = False,
    device: str | int | None = None,
) -> Dict[str, Path]:
    """Run the pipeline for one (split, conf) configuration."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_yolo_seg(weights, device=device)

    betas: Dict[str, float] = {}

    if apply_beta:
        print(f"  [{split}@{conf_threshold}] --apply-beta enabled. Computing train set betas via 50/50 paper split...")
        unique_map = {p.name.lower(): p for p in (list(images_dir.glob("*.JPG")) + list(images_dir.glob("*.jpg")))}
        all_image_paths = list(unique_map.values())
        train_paths, _ = get_paper_50_50_split(all_image_paths)


        tr_groups = group_top_side(train_paths)
        tr_pairs = make_pairs(tr_groups)
        tr_samples, tr_vpred, tr_vreal, _, _ = _evaluate_samples_set(
            model, tr_pairs, tr_groups, conf_threshold, food_info, gt_by_class, betas=None
        )

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
        print(f"  [{split}@{conf_threshold}] Computed betas for {len(betas)} classes -> {beta_json_path.name}")


    # 1. Load split stems.
    split_file = imagesets_dir / f"{split}.txt"
    stems = load_split(split_file)
    paths = resolve_image_paths(stems, images_dir)
    print(f"  [{split}@{conf_threshold}] resolved {len(paths)} of {len(stems)} images")

    # 2. Group into top/side pairs.
    groups = group_top_side(paths)
    pairs = make_pairs(groups)
    print(f"  [{split}@{conf_threshold}] {len(pairs)} (top, side) pairs")

    # 3. Evaluate samples
    samples, v_pred, v_real, m_pred, m_real = _evaluate_samples_set(
        model, pairs, groups, conf_threshold, food_info, gt_by_class, betas=betas if apply_beta else None
    )
    print(f"  [{split}@{conf_threshold}] {len(samples)} classes detected after pairing")

    # 4. Compute ME per class.
    class_names = [s.class_name for s in samples]
    per_class = compute_me_per_class(
        class_names, v_pred, v_real, m_pred=m_pred, m_real=m_real
    )
    overall = aggregate(per_class)

    # 5. Write CSV + JSON.
    tag_suffix = "_beta" if apply_beta else ""
    tag = f"{split}_conf{int(conf_threshold*100)}{tag_suffix}"
    if apply_beta:
        csv_path = out_dir / "samples_test_paper_conf80_beta.csv"
        json_path = out_dir / "report_test_paper_conf80_beta.json"
    else:
        csv_path = out_dir / f"samples_{tag}.csv"
        json_path = out_dir / f"report_{tag}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(asdict(samples[0]).keys()) if samples else [
                "pair_id", "class_name", "top_path", "side_path",
                "top_conf", "side_conf", "v_tilde_cm3",
                "mass_g", "kcal", "alpha_t", "alpha_s",
            ],
        )
        w.writeheader()
        for s in samples:
            w.writerow(asdict(s))

    report = {
        "config": {
            "split": split,
            "conf_threshold": conf_threshold,
            "apply_beta": apply_beta,
            "n_pairs": len(pairs),
            "n_samples": len(samples),
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

    return {"csv": csv_path, "json": json_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end calorie pipeline.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
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
        help="Compute beta calibration factor on train split and apply to test predictions.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Forwarded to Ultralytics (e.g. '0' for GPU 0, 'cpu').",
    )
    args = parser.parse_args()

    splits = args.split or ["val"]
    confs = args.conf or [0.5]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    food_info = parse_food_info(args.food_info)
    try:
        gt_by_class = _load_ground_truth(Path(args.density_xls))
    except Exception as e:
        print(f"Failed to load density.xls: {e}. Continuing without GT.")
        gt_by_class = None

    for split in splits:
        for conf in confs:
            print(f"\n=== split={split} conf={conf} apply_beta={args.apply_beta} ===")
            _run_one_config(
                weights=Path(args.weights),
                split=split,
                images_dir=Path(args.images_dir),
                imagesets_dir=Path(args.imagesets_dir),
                conf_threshold=conf,
                food_info=food_info,
                gt_by_class=gt_by_class,
                out_dir=out_dir / "predictions",
                apply_beta=args.apply_beta,
                device=args.device,
            )


if __name__ == "__main__":
    main()

