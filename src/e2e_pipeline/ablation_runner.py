"""Ablation Evaluation Script for Comparative Models (YOLOv8-seg, YOLO11-seg, etc.).

Runs end-to-end evaluation with Beta Calibration and generates comparison reports.
"""

from __future__ import annotations

import argparse, json, shutil, sys
from pathlib import Path
import torch
from ultralytics import YOLO

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_ROOT, OUTPUT_ROOT, RAW_DATA_ROOT

from src.calorie_estimation import parse_food_info
from src.e2e_pipeline.run_e2e import _load_ground_truth, _run_one_config


def get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate Ablation Models E2E")
    ap.add_argument(
        "--weights",
        default="models/ecustfd_yolov8seg_trainval_best.pt",
        help="Path to model weights checkpoint",
    )
    ap.add_argument(
        "--model-name",
        default="yolov8n-seg",
        help="Display name for model in ablation report",
    )
    ap.add_argument("--conf", type=float, default=0.8, help="Confidence threshold")
    ap.add_argument("--split", default="test", help="Dataset split (default: test)")
    return ap


def run_ablation(
    weights_path: str | Path,
    model_name: str,
    conf: float = 0.8,
    split: str = "test",
) -> dict:
    weights_path = Path(weights_path)
    print(f"=== Running Ablation Evaluation for {model_name} ===")
    print(f"Weights: {weights_path}")
    print(f"Confidence: {conf} | Split: {split}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Copy weights to models/ablation/ for clean organization
    ablation_weights_dir = Path("models/ablation")
    ablation_weights_dir.mkdir(parents=True, exist_ok=True)
    dst_weights = ablation_weights_dir / f"{model_name}_best.pt"
    if weights_path.resolve() != dst_weights.resolve() and weights_path.exists():
        shutil.copy2(weights_path, dst_weights)
        print(f"Saved weights copy -> {dst_weights}")

    # 1. Run E2E pipeline with beta calibration
    ablation_pred_dir = OUTPUT_ROOT / "predictions" / "ablation"
    ablation_pred_dir.mkdir(parents=True, exist_ok=True)

    food_info_path = RAW_DATA_ROOT / "food_info.xls"
    if not food_info_path.exists():
        food_info_path = Path("ECUSTFD/faster_rcnn/food_info.xls")
    food_info = parse_food_info(food_info_path)

    density_xls = RAW_DATA_ROOT / "density.xls"
    gt_by_class = _load_ground_truth(density_xls)

    images_dir = RAW_DATA_ROOT / "JPEGImages"
    imagesets_dir = RAW_DATA_ROOT / "ImageSets" / "Main"

    print(f"\n[1/3] Running E2E evaluation for {model_name}...")
    res = _run_one_config(
        weights=dst_weights,

        split=split,
        conf_threshold=conf,
        images_dir=images_dir,
        imagesets_dir=imagesets_dir,
        food_info=food_info,
        gt_by_class=gt_by_class,
        out_dir=ablation_pred_dir,
        device="0" if torch.cuda.is_available() else "cpu",
        apply_beta=True,
    )

    # 2. Measure Speed & Latency
    print(f"\n[2/3] Measuring Speed & Latency...")
    model = YOLO(str(dst_weights))
    data_yaml = Path("data/processed/yolo_ecustfd_seg/ecustfd-seg.yaml")
    metrics = model.val(data=str(data_yaml), split="val", plots=False, workers=0, verbose=False)

    speed_dict = metrics.speed
    prep_ms = speed_dict.get("preprocess", 0.0)
    inf_ms = speed_dict.get("inference", 0.0)
    post_ms = speed_dict.get("postprocess", 0.0)
    tot_ms = prep_ms + inf_ms + post_ms
    fps = 1000.0 / tot_ms if tot_ms > 0 else 0.0

    ablation_summary = {
        "model_name": model_name,
        "weights": str(dst_weights),
        "split": split,
        "conf": conf,
        "fps": round(fps, 1),
        "total_latency_ms": round(tot_ms, 2),
        "mask_mAP50": round(float(metrics.seg.map50), 4),
        "box_mAP50": round(float(metrics.box.map50), 4),
        "outputs": {k: str(v) for k, v in res.items()},
    }

    # Save ablation summary JSON
    summary_json = OUTPUT_ROOT / "reports" / "ablation" / f"summary_{model_name}.json"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(ablation_summary, indent=2), encoding="utf-8")

    print(f"\n[3/3] Completed Ablation Evaluation for {model_name}!")
    print(f"Summary JSON: {summary_json}")
    print(f"Throughput: {ablation_summary['fps']} FPS ({ablation_summary['total_latency_ms']} ms/image)")
    print(f"Mask mAP50: {ablation_summary['mask_mAP50'] * 100:.1f}%")

    return ablation_summary


def main() -> int:
    args = get_parser().parse_args()
    run_ablation(
        weights_path=args.weights,
        model_name=args.model_name,
        conf=args.conf,
        split=args.split,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
