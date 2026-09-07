# YOLO26-seg training pipeline for ECUSTFD

YOLO26-seg (Ultralytics) trained on ECUSTFD, replaces the original
MATLAB Faster R-CNN + GrabCut pipeline.

## Folder layout

```
yolo_seg/
├── README.md                              ← this file
├── requirements.txt                       ← ultralytics, opencv, numpy, ...
├── dataset/                               ← Step 1: build YOLO-seg dataset
│   ├── convert_to_yolo_seg.py             ← SAM .npy masks → YOLO polygons
│   ├── verify_yolo_dataset.py             ← integrity checks
│   └── __init__.py
└── train/                                 ← Step 2: train + export
    ├── train_yolo26_seg.py                ← training entry point
    └── __init__.py
```

## Pipeline (no beta)

This folder is the **replacement** for the MATLAB Faster R-CNN +
GrabCut pipeline. It does NOT contain any beta-correction code; the
only calorie formula downstream is the runtime MATLAB baseline
``C = q_k * V_tilde``.

## Setup

```bash
pip install -r src/yolo_seg/requirements.txt
```

## Workflow

### 1. Build dataset (one-time)

```bash
python src/yolo_seg/dataset/convert_to_yolo_seg.py
```

**Default split (paper-faithful, no data leakage):**

The converter reads the official ECUSTFD split files:
- `data/raw/ECUSTFD/ImageSets/Main/trainval.txt` → **train**
  (1245 stems × 2 views = 2490 images)
- `data/raw/ECUSTFD/ImageSets/Main/test.txt`     → **val/test**
  (1733 stems × 2 views = 3466 images, held out for evaluation)

This matches the original ECUSTFD paper protocol (train Faster R-CNN
on trainval, evaluate on test). Using this split ensures no test
image leaks into the training set.

⚠️ **Do not** use the legacy hash split (`--no-use-imagesets-split`)
for paper evaluation — it leaks test images into the train set,
which will be flagged immediately by reviewers.

Input:
- `data/processed/sam_masks_full/masks/<stem>.npy` (one per
  ``<class><item><S|T>(N).JPG`` filename).
- `data/raw/ECUSTFD/JPEGImages/<stem>.JPG`.

Output:
```
E:\AI_Research\dlt8\data\processed\yolo_ecustfd_seg\
├── ecustfd-seg.yaml      ← dataset config (20 classes)
├── images/{train,val}/   ← JPGs (split by ECUSTFD trainval.txt / test.txt)
└── labels/{train,val}/   ← polygon .txt files
```

**If you have an existing `data/processed/yolo_ecustfd_seg/` from the
old hash split**, delete it before re-running:

```bash
rm -rf data/processed/yolo_ecustfd_seg
python src/yolo_seg/dataset/convert_to_yolo_seg.py
```

After the rebuild, expect:
- `images/train/`: ~2490 images (from trainval.txt)
- `images/val/`:   ~3466 images (from test.txt)

### 2. Verify

```bash
python src/yolo_seg/dataset/verify_yolo_dataset.py
```

Checks: matching files, label format, coordinate range, class IDs,
Ultralytics YAML validity.

### 3. Train

```bash
python src/yolo_seg/train/train_yolo26_seg.py
```

Defaults: `yolo26n-seg.pt`, 100 epochs, imgsz 480, batch 4,
GPU device 0. Override via CLI flags.

### 4. Output

- Checkpoints: `runs/yolo_seg/ecustfd_yolo26seg/weights/{best,last}.pt`
- Best copy: `models/ecustfd_yolo26seg_best.pt`

## Class set (20 classes)

The YOLO-seg class set **matches ECUSTFD exactly** (19 foods + coin):

| Idx | Class            | Shape      | Provenance       |
|----:|------------------|------------|------------------|
|  0  | apple            | ellipsoid  | C++ port         |
|  1  | banana           | unknown    | C++ port         |
|  2  | bread            | column     | C++ port         |
|  3  | bun              | unknown    | C++ port         |
|  4  | coin             | (n/a)      | coin, not food   |
|  5  | doughnut         | torus      | C++ port         |
|  6  | egg              | ellipsoid  | C++ port         |
|  7  | fired_dough_twist| unknown    | C++ port         |
|  8  | grape            | grape      | C++ port (special) |
|  9  | lemon            | ellipsoid  | C++ port         |
| 10  | litchi           | unknown    | inferred         |
| 11  | mango            | unknown    | C++ port         |
| 12  | mooncake         | unknown    | inferred         |
| 13  | orange           | ellipsoid  | C++ port         |
| 14  | peach            | ellipsoid  | inferred         |
| 15  | pear             | unknown    | C++ port         |
| 16  | plum             | ellipsoid  | C++ port         |
| 17  | qiwi             | ellipsoid  | C++ port         |
| 18  | sachima          | column     | C++ port         |
| 19  | tomato           | ellipsoid  | C++ port         |

Notes:
- The class "kiwi" that appeared in earlier versions of
  `convert_to_yolo_seg.py` and `yolo_infer.py` was a label alias
  for **qiwi fruit**. It has been removed; only `qiwi` (the ECUSTFD
  name) remains. Any pre-existing `ecustfd-seg.yaml` with the 21-class
  schema must be **regenerated** by deleting
  `data/processed/yolo_ecustfd_seg/` and re-running
  `convert_to_yolo_seg.py`.
- "C++ port" means the shape formula comes from
  `ECUSTFD/faster_rcnn/grabcut_mex.cpp` lines 9-28 (the authors'
  hard-coded class→shape table).
- "inferred" means the class is in ECUSTFD but not in the C++
  hard-coded table; the shape was inferred by reasoning about the
  food geometry. See `src/constants.py::SHAPE_MODELS` docstring.

## Volume formulas (no beta)

For each detection, the pipeline:

1. **Coin calibration** — `alpha = 2.5 cm / mean(W_box, H_box)` for
   each view (top, side). See `src/coin_calibration/`.
2. **Crop masks** to the food bounding box.
3. **Compute volume** using the per-shape formula:
   - `ellipsoid`: ``V = (PI/4) * alpha_S^3 * sum(L_i^2)``
   - `column`:   ``V = (sA * alpha_T'^2) * (mean_h * alpha_S)``
   - `unknown`:  ``V = sA * alpha_T'^2 * sB * alpha_S / LB_MAX^2``
   - `torus`:    ``V = (PI^1.5/4) * hB^2 * alpha_T' * alpha_S^2 * (sqrt(sA+sAE) + sqrt(sAE))``
   - `grape`:    ``V = 0.81 * sA * alpha_T'^2 * sB * alpha_S / LB_MAX^2``

   (verbatim ports of `ECUSTFD/faster_rcnn/grabcut_mex.cpp`).
4. **Compute calorie** — `C = q_k * V_tilde` (single runtime formula,
   from `src/calorie_estimation/`).

There is **no beta correction** anywhere in this pipeline.

## Class notes

- `coin` dominates (~99% of images) — may bias mAP; we keep it
  because the coin is needed for cm/pixel calibration.
- Hard classes (poor accuracy): banana, grape, mooncake (per
  `src/constants.py::HARD_CLASSES`).
- Backbone freezing via `--freeze N` improves convergence on
  small-instance classes.

## Models

| Suffix       | Params |  mAPmask | When to use              |
|--------------|-------:|---------:|--------------------------|
| yolo26n-seg  | 9.1M   | 33.9     | Default — fastest iter.  |
| yolo26s-seg  | 10.4M  | 40.0     | More capacity            |
| yolo26m-seg  | 23.6M  | 44.1     | Best accuracy vs speed   |
| yolo26l-seg  | 28.0M  | 45.5     | Higher-end GPU           |
| yolo26x-seg  | 62.8M  | 47.0     | Max accuracy             |

*mAPmask on COCO val2017. Numbers from Ultralytics 8.4.x.*

## Status — initial real training run (2026-07-29)

This section is the **lab notebook** for the first real training run
on RTX 4050 Laptop 6.4 GB VRAM. Earlier runs under `runs/yolo_seg/`
(`ecustfd_yolo26seg`, `ecustfd_yolo26seg-2`, `ecustfd_yolo26seg-2_val`,
`_logs`) were **test runs only** and were thrown away before this run.

### Configuration (this run)

| Parameter        | Value                          | Reason                                  |
|------------------|--------------------------------|-----------------------------------------|
| Pretrained       | `yolo26n-seg.pt`               | Smallest, fits 6.4 GB VRAM              |
| Epochs           | 100                            | Default                                 |
| Image size       | 480                            | Reduced from 640 (VRAM-limited)         |
| Batch size       | 4                              | Largest that fits in 6.4 GB at 480px    |
| Device           | `0` (CUDA)                     | Single RTX 4050 Laptop                  |
| AMP              | True                           | Halves activation memory                |
| Patience         | 20                             | Early stop if no mAP improvement        |
| close_mosaic     | 10                             | Disable mosaic in last 10 epochs        |
| Freeze backbone  | 0 (none)                       | Dataset small; full fine-tune           |
| Workers          | 4                              | Dataset loader parallelism              |
| Splits           | ECUSTFD `trainval.txt` / `test.txt` | **No data leakage** (paper-faithful)|

### Pre-run cleanup (this run)

```bash
# 1. Throw away all test runs
rm -rf runs/yolo_seg/ecustfd_yolo26seg runs/yolo_seg/ecustfd_yolo26seg-2 \
       runs/yolo_seg/ecustfd_yolo26seg-2_val runs/yolo_seg/_logs

# 2. Throw away the leaked dataset (old hash split)
rm -rf data/processed/yolo_ecustfd_seg

# 3. Rebuild dataset with paper-faithful trainval/test split
python src/yolo_seg/dataset/convert_to_yolo_seg.py
# Expected: images/train ~2490, images/val ~3466 (ECUSTFD official)
```

### What to monitor during training

While the trainer is running, the following metrics are good
indicators of health:

| Metric              | Healthy range      | If out of range              |
|---------------------|--------------------|------------------------------|
| `train/seg_loss`    | monotonic decrease | Plateaus → reduce LR         |
| `val/seg_loss`      | tracks train       | Diverges → overfitting       |
| `metrics/mAP50(M)`  | increases to ~0.6+ | < 0.3 → check labels         |
| `metrics/mAP50-95(M)` | increases to ~0.4+ | < 0.2 → check labels       |
| GPU mem             | < 6.0 GB           | > 6.2 GB → OOM imminent      |
| Epoch time          | ~30-60 s           | > 5 min → check disk I/O     |

### Expected artifacts after a clean run

```
runs/yolo_seg/ecustfd_yolo26seg_trainval/
├── weights/
│   ├── best.pt              ← use this for evaluation
│   └── last.pt              ← resume from this if training stops
├── results.csv              ← per-epoch metric log
├── results.png              ← training curves
├── confusion_matrix.png
└── args.yaml                ← exact hyperparameters used
```

Then copy best to `models/`:

```bash
cp runs/yolo_seg/ecustfd_yolo26seg_trainval/weights/best.pt \
   models/ecustfd_yolo26seg_trainval_best.pt
```

### Anti-OOM checklist (6.4 GB VRAM)

- **batch ≤ 4** at imgsz 480 (we tested this; batch=8 OOMs)
- **imgsz = 480** (640 OOMs even at batch=2)
- **AMP enabled** (`--amp`, default True; saves ~40% memory)
- **close_mosaic = 10** (mosaic aug allocates 4-image buffers)
- **workers = 4** (more workers don't help; data is on local SSD)
- If you still OOM: `--cache disk` (offline SSD, not RAM)
- If you still OOM: `--freeze 5` (freeze first 5 backbone layers)

### Failed runs / lessons learned (older test runs)

- `ecustfd_yolo26seg` (first test): ran on **old hash-split** dataset
  → leak. Discarded. Lesson: use ECUSTFD official splits.
- `ecustfd_yolo26seg-2`: same leak problem + slightly different
  hyperparameters. Discarded.
- `ecustfd_yolo26seg-2_val`: validation-only run. Discarded.

None of these `.pt` files should be used for evaluation; the
real one is `ecustfd_yolo26seg_trainval/weights/best.pt` after this
initial real run completes.