# -*- coding: utf-8 -*-
"""Train YOLO26-seg on ECUSTFD dataset.

Prerequisites:
    1. Run convert_to_yolo_seg.py first to produce data/processed/yolo_ecustfd_seg/
    2. Install ultralytics:  pip install ultralytics

Usage:
    # Full training (Ultralytics default hyperparameters)
    python src/yolo_seg/train/train_yolo26_seg.py

    # Override ONLY the two hyperparameters we justify on hardware grounds
    python src/yolo_seg/train/train_yolo26_seg.py --imgsz 480 --batch 4

Hyperparameter policy:
    We deliberately pass the MINIMUM number of overrides to Ultralytics.
    The two overrides below are the only ones we can justify empirically
    (they were chosen from VRAM profiling on RTX 4050 Laptop 6.4 GB):

      * --imgsz 480   — 640 OOMs at any batch on this GPU; 480 fits.
      * --batch 4     — even at imgsz=480, batch>=8 OOMs once Seg head
                        + AMP activates the mask tensor pipeline.

    Every other hyperparameter (epochs, patience, optimizer, lr0, momentum,
    warmup, all augmentation hsv_*/translate/scale/mosaic/mixup/...) is left
    to Ultralytics's defaults.  Ultralytics auto-saves a complete
    `args.yaml` snapshot inside the run folder, so the actual values used
    are auditable afterwards.
"""

import argparse
import shutil
from pathlib import Path

ROOT      = Path("E:/AI_Research/dlt8")
OUT_ROOT  = ROOT / "data/processed/yolo_ecustfd_seg"
YAML_PATH = OUT_ROOT / "ecustfd-seg.yaml"
RUNS_DIR  = ROOT / "runs/yolo_seg"   # Ultralytics writes here
MODEL_DIR = ROOT / "models"          # save best .pt here after training

# ─── Hyperparameters we override (justified on hardware grounds) ─────────────
# Each has a one-line reason; everything else is left to Ultralytics defaults.
DEFAULT_MODEL  = "yolo26n-seg.pt"   # architecture choice
DEFAULT_IMGSZ  = 480                # VRAM: 640 OOMs on RTX 4050 6.4 GB
DEFAULT_BATCH  = 4                  # VRAM: batch>=8 OOMs with Seg head + AMP
DEFAULT_WORKERS = 0                 # Windows: workers>0 spawns pickling subprocesses
                                   # (we saw 13+ python.exe processes during the
                                   # killed run). On Windows this is a stability
                                   # hazard, not a performance win. Ultralytics
                                   # default is 8.
DEFAULT_DEVICE  = "0"               # GPU0. Set "cpu" if no CUDA.


def get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    # ── Configuration (paths / architecture) ──────────────────────────────────
    ap.add_argument("--model",  default=DEFAULT_MODEL,
                    help=f"Pretrained YOLO-seg model (default: {DEFAULT_MODEL})")
    ap.add_argument("--data",   default=str(YAML_PATH),
                    help=f"Dataset YAML (default: {YAML_PATH})")
    ap.add_argument("--project", default=str(RUNS_DIR),
                    help=f"Ultralytics project dir (default: {RUNS_DIR})")
    ap.add_argument("--name",   default="ecustfd_yolo26seg",
                    help="Experiment name inside project/ (default: ecustfd_yolo26seg)")
    ap.add_argument("--resume", default=None,
                    help="Path to a .pt checkpoint to resume from")
    # ── Only the hyperparameters we justify overriding ───────────────────────
    ap.add_argument("--imgsz",    type=int,    default=DEFAULT_IMGSZ,
                    help="Input image size (must be multiple of 32)")
    ap.add_argument("--batch",    type=int,    default=DEFAULT_BATCH,
                    help="Batch size")
    ap.add_argument("--workers",  type=int,    default=DEFAULT_WORKERS,
                    help="DataLoader workers (0=safe on Windows)")
    ap.add_argument("--device",                  default=DEFAULT_DEVICE,
                    help="Device: '0'=GPU0, 'cpu', 'mps'")
    return ap


def main() -> None:
    args = get_parser().parse_args()

    # ── Sanity checks ─────────────────────────────────────────────────────────
    if not args.resume and not Path(args.data).exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {args.data}\n"
            "Run convert_to_yolo_seg.py first!"
        )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Import ultralytics ─────────────────────────────────────────────────────
    from ultralytics import YOLO

    # ── Load or resume model ──────────────────────────────────────────────────
    if args.resume:
        print(f"[resume] Loading from {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"[train]  Loading pretrained: {args.model}")
        model = YOLO(args.model)

    # ── Launch training ───────────────────────────────────────────────────────
    # NOTE: only pass keys we deliberately set.  Everything else (epochs,
    # patience, optimizer, lr0, momentum, warmup, all augmentation params,
    # amp, close_mosaic, ...) is left to Ultralytics defaults.
    results = model.train(
        data     = args.data,
        imgsz    = args.imgsz,
        batch    = args.batch,
        workers  = args.workers,
        device   = args.device,
        project  = args.project,
        name     = args.name,
        verbose  = True,
    )

    # ── Copy best checkpoint to models/ ──────────────────────────────────────
    best_ckpt = Path(results.save_dir) / "weights" / "best.pt"
    if best_ckpt.exists():
        dst = MODEL_DIR / f"{args.name}_best.pt"
        shutil.copy2(best_ckpt, dst)
        print(f"\n✅ Best checkpoint saved: {dst}")

    last_ckpt = Path(results.save_dir) / "weights" / "last.pt"
    if last_ckpt.exists():
        dst_last = MODEL_DIR / f"{args.name}_last.pt"
        shutil.copy2(last_ckpt, dst_last)
        print(f"✅ Last checkpoint saved: {dst_last}")

    print("\n[done] Training complete.")
    print(f"Results: {results.save_dir}")


if __name__ == "__main__":
    main()
