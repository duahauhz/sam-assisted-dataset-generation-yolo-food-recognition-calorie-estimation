# `src/` — Food Calorie Estimation Pipeline

Food detection and calorie estimation from meal images (ECUSTFD dataset, 20 classes = 19 foods + the reference coin).
It replaces the original two-stage Faster R-CNN + GrabCut system of Liang & Li (2017) with a **single-stage YOLO-segmentation** model while keeping the original volume/calorie mathematics as a faithful re-implementation.

> 📖 The repository root `README.md` contains the authoritative method overview, headline numbers, and evaluation protocol. This file is the module-level guide for running the pipeline.

---

## 1. Module layout

| Directory | Role | Corresponding notebook(s) |
|---|---|---|
| `constants.py` | 19 food classes, per-class density ρ_k and energy factors q_k | — |
| `config.py` | Path resolution and defaults | — |
| `data_prep_SAM1/` | Visual bbox QA, SAM mask generation, manual overrides | `01_run_sam1_pipeline` |
| `yolo_seg/` | YOLO-seg dataset export + YOLO26n training entry | `02_yolo_seg_pipeline`, `03_yolo_seg_train` |
| `yolo_seg_eval/` | Single-stage end-to-end evaluation (`_run_one_config`) | `04`, `07` |
| `faster_rcnn/` | Two-stage baseline (training; evaluation with GrabCut/SAM backends) | `05a`, `06a`, `06b` |
| `beta_correction/` | Per-class volume-bias calibration β_k | (inside `04`/`06`/`07`) |
| `coin_calibration/` | Coin-based pixel-to-cm scale | (inside the pipeline) |
| `volume_models/` | Five geometric volume models (ellipsoid / column / solid of revolution / grape air-gap / torus) | (inside the pipeline) |
| `calorie_estimation/` | Paper-faithful mass and calorie formulas | (inside the pipeline) |
| `view_pairing/` | Top/side view pairing | (inside the pipeline) |
| `segmentation_runtime/` | YOLO-seg inference wrapper | (inside the pipeline) |
| `e2e_pipeline/` | Orchestrator, metrics, threshold-sweep runner | `03_threshold_sweep` |
| `ablation/` | Auxiliary analysis script (not part of the paper pipeline) | — |

## 2. Pipeline execution order

See the root `README.md` → *Reproducing the pipeline* for the exact commands of each step.

```
00 fetch/setup → 01 SAM labels → 02 export → 03 train YOLO26n
→ 03 sweep (τ selected on test_tune) → 04/06a/06b/07 evaluation on test_final (frozen τ)
→ 07 statistical test → 08 paper metrics
```

Key points:

- **τ (detection confidence) is per-model** — selected on `test_tune` (805 images) and frozen before any final-half evaluation:
  0.05 (YOLO26n), 0.30 (YOLOv8n), 0.10 (FR-CNN+GrabCut), 0.05 (FR-CNN+SAM).
  The original MATLAB code's `thres = 0.8` is **not** used anywhere (a hand-tuned 2017 artifact; see the paper's operating-points footnote).
- **β_k is fitted on a 50/50 item split of the training items** (1,169 images), disjoint from `test_final`.
- Every number reported in the paper is produced by a logged notebook run (`outputs/logs/` or a dedicated run directory).

## 3. Running without notebooks

```bash
# Exploratory E2E on the original val/test split (NOT the paper protocol)
python -m src.e2e_pipeline run --split test --conf 0.5 --apply-beta

# Deterministic-core unit tests
python -m pytest src -q
```

`test_final`/`test_tune` are used through notebooks `04`/`06`/`07` (config in cell 3) — this is the code path that produces the paper numbers.

## 4. Split-disjointness checks (already verified; kept as evidence)

```
test.txt       : 1733 stems   (official ECUSTFD test)
test_tune.txt  : 805 stems    ∩ test_final.txt = 0
test_final.txt : 928 stems    ∪ test_tune.txt = test.txt (1733)
trainval.txt   : 1245 stems   ∩ test.txt = 0
```

⚠️ Note: `val.txt` (623) ⊂ `trainval` — the original ECUSTFD protocol trains on trainval and measures fitness on val, i.e. **val is not held out**. This is why the paper re-splits the official test set into tune/final instead of using val (details and disclosure in the root `README.md`, *Evaluation protocol*).

## 5. Key files for reviewers

| Path | Role |
|---|---|
| `src/yolo_seg/dataset/convert_to_yolo_seg.py` | Dataset build, paper-faithful split |
| `src/e2e_pipeline/dataset_split.py` (+ its test) | tune/final split logic + disjointness invariants |
| `outputs/predictions/04|06a|06b|07_*_20260903-*/` | Final-half runs (report/samples/β/speed) |
| `outputs/metrics_final/20260906-152152/` | GAP metrics: mIoU, mAP, Wilcoxon, kcal MAE |
| `outputs/threshold_sweep/20260902-234626/` | τ sweep on test_tune |

## 6. Historical note (not used in the paper)

An early evaluation round used the legacy protocol (fixed conf = 0.8, miss-discarding, 10,583 accepted samples of 12,955 pairs, volume MAE 23.4%) and was superseded by the current tune/final protocol. Those numbers are internal history only and are **not cited in the paper** — every official number comes from `outputs/metrics_final/20260906-152152/`.
