# -*- coding: utf-8 -*-
"""Train Faster R-CNN (torchvision) on a built ECUSTFD dataset.

Mirrors the YOLO26 training script
(``src/yolo_seg/train/train_yolo26_seg.py``) in structure:

* Only ``--imgsz`` is meaningful in name; for Faster R-CNN we
  actually expose ``--data-variant`` (old | new), ``--epochs``,
  ``--batch``, ``--workers``, ``--device``.
* The checkpoint ``best.pt`` is copied to ``models/`` at the end.
* A per-run log file is written under ``outputs/logs/``.

Why COCO JSON?
--------------
The dataset produced by ``build_dataset.py`` is in COCO format.  We
load it with ``torchvision.datasets.CocoDetection`` and feed it into
``torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn``
(or whatever backbone is chosen).

Run (called from notebooks 05/06):

    python src/faster_rcnn/train_faster_rcnn.py \\
        --data-variant old --epochs 10 --batch 4 --workers 0 --device 0
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

# Make ``src.faster_rcnn`` and ``src.X`` importable when called as a script.
# Both paths are required:
#   - ROOT_DIR             -> so "from src.faster_rcnn import ..." (after
#                             __init__.py runs, "from src.X import ..." inside
#                             inference.py / eval_pipeline.py works because
#                             ROOT_DIR is in sys.path)
#   - ROOT_DIR / "src"     -> so "from faster_rcnn import ..." (relative
#                             sub-package) is resolvable
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from faster_rcnn.faster_rcnn_guard import (  # noqa: E402
    LOGS_DIR,
    MODEL_DIR,
    PREDICTIONS_DIR,
    ROOT,
    make_run_dir,
    make_timestamp,
    setup_logger,
)


# ---------------------------------------------------------------------------
# Defaults (justified on hardware grounds — see YOLO26 script for the
# equivalent reasoning on RTX 4050 6.4 GB)
# ---------------------------------------------------------------------------

DEFAULT_DATA_VARIANT = "old"   # or "new"
DEFAULT_EPOCHS  = 100          # match yolo26seg (which trained 100 epochs)
DEFAULT_BATCH   = 4            # VRAM: Faster R-CNN ~2 GB per batch step
DEFAULT_WORKERS = 0            # Windows: workers>0 spawns pickling subprocesses
DEFAULT_DEVICE  = "0"          # GPU0. Set "cpu" if no CUDA.
DEFAULT_BACKBONE = "fasterrcnn_mobilenet_v3_large_320_fpn"
# Resize policy: mirror yolo26seg's --imgsz 480 (see train_yolo26_seg.py
# DEFAULT_IMGSZ=480). We use the SHORT side = 480 and keep aspect ratio.
# yolo26 resize to a fixed 480x480 square; torchvision's resize on a
# variable-sized detection image should keep aspect ratio (otherwise we
# distort bbox aspect ratios). We follow torchvision's reference pattern.
DEFAULT_IMGSZ   = 480          # short side; matches yolo26seg's --imgsz


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-variant", choices=["old", "new"],
                    default=DEFAULT_DATA_VARIANT,
                    help="Which bbox variant (old=raw XML / new=patched XML)")
    ap.add_argument("--epochs",   type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch",    type=int, default=DEFAULT_BATCH)
    ap.add_argument("--workers",  type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--device",   default=DEFAULT_DEVICE)
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE,
                    help="torchvision detection model name")
    ap.add_argument("--lr",       type=float, default=0.005,
                    help="Initial learning rate (SGD)")
    ap.add_argument("--imgsz",    type=int, default=DEFAULT_IMGSZ,
                    help="Short-side resize (matches yolo26seg's --imgsz)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Limit train/val to first N samples (smoke test)")
    return ap


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def build_datasets(data_root: Path, max_samples: Optional[int], imgsz: int):
    """Build the torchvision CocoDetection train + val datasets.

    Faster R-CNN does not need a separate val split during training; we
    use the trainval split for training and the test split for eval
    later (in ``06_faster_rcnn_eval.ipynb``).  To stay close to the
    paper-faithful 1245/1733 split, we just instantiate two CocoDetection
    objects — one over the train folder and one over the test folder.

    Resize policy (mirrors yolo26seg's --imgsz):
        yolo26seg resizes input to a fixed 480×480 square.  Faster R-CNN
        accepts variable-size tensors (no requirement to be square), so we
        resize so the SHORT side = imgsz and keep aspect ratio — the same
        pattern torchvision's reference detection training uses.  This
        keeps bbox aspect ratios intact and matches yolo26seg's intent
        (downscale images to fit a 6.4 GB GPU).
    """
    import torch  # local import keeps --help fast
    from torchvision.datasets import CocoDetection
    from torchvision.transforms import functional as F

    class _CocoWrap(CocoDetection):
        """Apply PIL → Tensor, short-side resize, and clip boxes.

        We intentionally keep the transform minimal here: Faster R-CNN
        does not need aggressive augmentation (the dataset is small),
        and torchvision's reference training loop already does
        ToTensor + Normalize inside the model.
        """

        def __init__(self, img_dir, ann_file, imgsz, transforms=None):
            super().__init__(str(img_dir), str(ann_file))
            self._imgsz = int(imgsz)
            self._xfm = transforms

        def __getitem__(self, idx):
            from PIL import Image
            img_id = self.ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = Path(self.root) / img_info["file_name"]
            with Image.open(img_path) as im:
                img = im.convert("RGB")
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)
            # Short-side resize before ToTensor so we can also scale boxes.
            W, H = img.size  # PIL: (width, height)
            scale = self._imgsz / float(min(H, W))
            new_W = int(round(W * scale))
            new_H = int(round(H * scale))
            img = img.resize((new_W, new_H))
            img = F.to_tensor(img)
            # COCO target uses absolute [x, y, w, h]; convert to [x1, y1, x2, y2]
            boxes, labels = [], []
            for obj in target:
                x, y, w, h = obj["bbox"]
                x1, y1, x2, y2 = x * scale, y * scale, (x + w) * scale, (y + h) * scale
                # Clip to resized image dims
                x1 = max(0.0, min(x1, new_W)); x2 = max(0.0, min(x2, new_W))
                y1 = max(0.0, min(y1, new_H)); y2 = max(0.0, min(y2, new_H))
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(obj["category_id"])
            target_out = {
                "boxes":  torch.as_tensor(boxes,  dtype=torch.float32).reshape(-1, 4),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([idx]),
            }
            if self._xfm is not None:
                img, target_out = self._xfm(img, target_out)
            return img, target_out

    train_img = data_root / "images" / "train"
    train_ann = data_root / "annotations" / "train.json"
    test_img  = data_root / "images" / "test"
    test_ann  = data_root / "annotations" / "test.json"

    train_ds = _CocoWrap(train_img, train_ann, imgsz)
    test_ds  = _CocoWrap(test_img,  test_ann, imgsz)

    if max_samples is not None:
        train_ds.ids = train_ds.ids[:max_samples]
        test_ds.ids  = test_ds.ids[:max_samples]

    return train_ds, test_ds


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def train(args, logger) -> dict:
    """Train one model and return a summary dict."""
    import torch
    import torch.utils.data
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_320_fpn,
        fasterrcnn_resnet50_fpn,
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        FasterRCNN_ResNet50_FPN_Weights,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    data_root = ROOT / "data/processed/faster_rcnn_seg" / args.data_variant
    if not (data_root / "annotations/train.json").exists():
        raise FileNotFoundError(
            f"Dataset not built for variant '{args.data_variant}': {data_root}\n"
            f"Run: python src/faster_rcnn/build_dataset.py --variant {args.data_variant}"
        )

    logger.info(f"Data variant root: {data_root}")

    train_ds, test_ds = build_datasets(data_root, args.max_samples, args.imgsz)
    logger.info(f"Train dataset: {len(train_ds)} samples")
    logger.info(f"Test  dataset: {len(test_ds)}  samples")

    # DataLoader
    def _collate(batch):
        return tuple(zip(*batch))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, collate_fn=_collate,
    )

    device = torch.device("cuda" if args.device != "cpu" else "cpu")
    logger.info(f"Device: {device}")

    # Build model
    n_classes = 21   # 19 foods + coin + 1 background (added by torchvision)
    logger.info(f"Loading backbone: {args.backbone}")
    if args.backbone == "fasterrcnn_mobilenet_v3_large_320_fpn":
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
    elif args.backbone == "fasterrcnn_resnet50_fpn":
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

# ---------------------------------------------------------------------------
# TeeStream Logger (captures stdout/stderr/tqdm into the log file)
# ---------------------------------------------------------------------------

class TeeStream:
    """Tee stdout/stderr to both console and log file."""
    def __init__(self, log_path: Path):
        self.console = sys.stdout
        self.file = open(log_path, "a", encoding="utf-8", buffering=1)

    def write(self, data: str):
        try:
            self.console.write(data)
            self.console.flush()
        except Exception:
            pass
        if not self.file.closed:
            try:
                self.file.write(data)
                self.file.flush()
            except Exception:
                pass

    def flush(self):
        try:
            self.console.flush()
        except Exception:
            pass
        if not self.file.closed:
            try:
                self.file.flush()
            except Exception:
                pass

    def isatty(self):
        return getattr(self.console, "isatty", lambda: False)()


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def train(args, logger) -> dict:
    """Train one model and return a summary dict."""
    import torch
    import torch.utils.data
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_320_fpn,
        fasterrcnn_resnet50_fpn,
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        FasterRCNN_ResNet50_FPN_Weights,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from tqdm import tqdm

    data_root = ROOT / "data/processed/faster_rcnn_seg" / args.data_variant
    if not (data_root / "annotations/train.json").exists():
        raise FileNotFoundError(
            f"Dataset not built for variant '{args.data_variant}': {data_root}\n"
            f"Run: python src/faster_rcnn/build_dataset.py --variant {args.data_variant}"
        )

    logger.info(f"Data variant root: {data_root}")

    train_ds, test_ds = build_datasets(data_root, args.max_samples, args.imgsz)
    logger.info(f"Train dataset: {len(train_ds)} samples")
    logger.info(f"Test  dataset: {len(test_ds)}  samples")

    # DataLoader
    def _collate(batch):
        return tuple(zip(*batch))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, collate_fn=_collate,
    )

    device = torch.device("cuda" if args.device != "cpu" else "cpu")
    logger.info(f"Device: {device}")

    # Build model
    n_classes = 21   # 19 foods + coin + 1 background (added by torchvision)
    logger.info(f"Loading backbone: {args.backbone}")
    if args.backbone == "fasterrcnn_mobilenet_v3_large_320_fpn":
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
    elif args.backbone == "fasterrcnn_resnet50_fpn":
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Train loop
    logger.info(f"Training for {args.epochs} epochs")
    logger.info("-" * 88)
    logger.info(
        f"  {'Epoch':>8}  {'GPU_mem':>8}  {'loss_cls':>10}  {'loss_box':>10}  "
        f"{'rpn_cls':>10}  {'rpn_box':>10}  {'total_loss':>10}  {'instances':>10}  {'imgsz':>8}"
    )
    logger.info("-" * 88)

    history = []
    best_loss = float("inf")
    best_ckpt = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        sum_cls = 0.0
        sum_box = 0.0
        sum_rpn_cls = 0.0
        sum_rpn_box = 0.0
        n_batches = 0
        t0 = time.time()

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"  {epoch:>3}/{args.epochs}",
            file=sys.stdout,
            leave=True,
            dynamic_ncols=True,
        )

        for i, (imgs, targets) in pbar:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(v for v in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            c_cls = loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
            c_box = loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()
            c_rpn_cls = loss_dict.get("loss_objectness", torch.tensor(0.0)).item()
            c_rpn_box = loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item()
            c_total = loss.item()

            sum_cls += c_cls
            sum_box += c_box
            sum_rpn_cls += c_rpn_cls
            sum_rpn_box += c_rpn_box
            epoch_loss += c_total
            n_batches += 1
            n_inst = sum(len(t["labels"]) for t in targets)

            nb = n_batches
            avg_cls = sum_cls / nb
            avg_box = sum_box / nb
            avg_rpn_cls = sum_rpn_cls / nb
            avg_rpn_box = sum_rpn_box / nb
            avg_total = epoch_loss / nb
            gpu_mem = f"{torch.cuda.max_memory_allocated() / (1024**3):.3f}G" if device.type == "cuda" else "0.000G"

            pbar.set_postfix_str(
                f"{gpu_mem:>8}  {avg_cls:>10.4f}  {avg_box:>10.4f}  {avg_rpn_cls:>10.4f}  {avg_rpn_box:>10.4f}  {avg_total:>10.4f}  {n_inst:>10d}  {args.imgsz:>8d}"
            )

        dt = time.time() - t0
        nb = max(n_batches, 1)
        avg_cls = sum_cls / nb
        avg_box = sum_box / nb
        avg_rpn_cls = sum_rpn_cls / nb
        avg_rpn_box = sum_rpn_box / nb
        avg_total = epoch_loss / nb

        gpu_mem = f"{torch.cuda.max_memory_allocated() / (1024**3):.3f}G" if device.type == "cuda" else "0.000G"

        history.append({
            "epoch": epoch,
            "loss": avg_total,
            "loss_cls": avg_cls,
            "loss_box": avg_box,
            "rpn_cls": avg_rpn_cls,
            "rpn_box": avg_rpn_box,
            "gpu_mem": gpu_mem,
            "wall_s": dt,
        })

        if avg_total < best_loss:
            best_loss = avg_total
            best_ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": avg_total,
            }

    logger.info("-" * 88)

    # Save best checkpoint to a per-run dir
    run_dir = make_run_dir(f"05_train_{args.data_variant}",
                           timestamp=make_timestamp())
    ckpt_path = run_dir / "best.pt"
    torch.save(best_ckpt, ckpt_path)
    logger.info(f"Saved best checkpoint: {ckpt_path}")

    # Copy best.pt → models/  (mirrors YOLO26 train_yolo26_seg.py)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dst = MODEL_DIR / f"faster_rcnn_{args.data_variant}_best.pt"
    shutil.copy2(ckpt_path, dst)
    logger.info(f"Best checkpoint copied to: {dst}")

    # Save training history JSON
    history_path = run_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    logger.info(f"History saved: {history_path}")

    # Save summary
    summary_path = run_dir / "summary.txt"
    lines = [
        "=" * 88,
        f"05 -- TRAIN FASTER R-CNN ({args.data_variant}) -- SUMMARY",
        "=" * 88,
        f"PROJECT_ROOT    : {ROOT}",
        f"Data variant    : {args.data_variant}",
        f"Backbone        : {args.backbone}",
        f"Epochs          : {args.epochs}",
        f"Batch / Workers : {args.batch} / {args.workers}",
        f"Imgsz (short)  : {args.imgsz}",
        f"Device          : {args.device}",
        f"Train samples   : {len(train_ds)}",
        f"Best epoch loss : {best_loss:.4f}",
        f"Best checkpoint : {dst}",
        f"Run dir         : {run_dir}",
        "",
        "Per-epoch loss:",
        f"  {'Epoch':>8}  {'GPU_mem':>8}  {'loss_cls':>10}  {'loss_box':>10}  {'rpn_cls':>10}  {'rpn_box':>10}  {'total_loss':>10}  {'time':>7}",
    ]
    for h in history:
        ep_str = f"{h['epoch']}/{args.epochs}"
        lines.append(
            f"  {ep_str:>8}  {h.get('gpu_mem', '0G'):>8}  {h.get('loss_cls', 0.0):>10.4f}  "
            f"{h.get('loss_box', 0.0):>10.4f}  {h.get('rpn_cls', 0.0):>10.4f}  "
            f"{h.get('rpn_box', 0.0):>10.4f}  {h['loss']:>10.4f}  {h['wall_s']:>6.1f}s"
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Summary written: {summary_path}")

    return {
        "data_variant": args.data_variant,
        "best_loss": best_loss,
        "history": history,
        "ckpt_path": str(dst),
        "run_dir": str(run_dir),
    }


class MultiTeeStream:
    """Tee stdout/stderr to console and multiple log files."""
    def __init__(self, log_paths: List[Path]):
        self.console = sys.stdout
        self.files = [open(p, "a", encoding="utf-8", buffering=1) for p in log_paths]

    def write(self, data: str):
        try:
            self.console.write(data)
            self.console.flush()
        except Exception:
            pass
        for f in self.files:
            if not f.closed:
                try:
                    f.write(data)
                    f.flush()
                except Exception:
                    pass

    def flush(self):
        try:
            self.console.flush()
        except Exception:
            pass
        for f in self.files:
            if not f.closed:
                try:
                    f.flush()
                except Exception:
                    pass

    def close(self):
        for f in self.files:
            if not f.closed:
                try:
                    f.close()
                except Exception:
                    pass

    def isatty(self):
        return getattr(self.console, "isatty", lambda: False)()


def main() -> None:
    args = get_parser().parse_args()
    ts = make_timestamp()
    
    log_path1 = LOGS_DIR / f"05_train_{args.data_variant}_{ts}.log"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_logs_dir = ROOT / "data/processed/faster_rcnn_seg/logs"
    dataset_logs_dir.mkdir(parents=True, exist_ok=True)
    log_path2 = dataset_logs_dir / f"train_run_{args.data_variant}_{ts}.log"

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    tee = MultiTeeStream([log_path1, log_path2])
    sys.stdout = tee
    sys.stderr = tee

    try:
        logger = logging.getLogger("faster_rcnn_train")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if logger.handlers:
            for h in list(logger.handlers):
                logger.removeHandler(h)

        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        logger.info("=" * 88)
        logger.info(f"[run ] Faster R-CNN training (variant={args.data_variant})")
        logger.info(f"[cwd ] {ROOT}")
        logger.info(f"[log1] {log_path1}")
        logger.info(f"[log2] {log_path2}")
        logger.info(f"[ts  ] {ts}")
        logger.info(f"[hyp ] backbone={args.backbone} imgsz={args.imgsz} batch={args.batch} workers={args.workers} device='{args.device}' lr={args.lr}")
        logger.info("-" * 88)
        logger.info("[ok  ] model loaded.\n")

        train(args, logger)
        logger.info("Done.")
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        tee.close()


if __name__ == "__main__":
    main()
