# Committed outputs inventory

This document lists exactly which evaluation artifacts are committed to the public
repository and what each one is the evidence for. Everything else under `outputs/`
is excluded by `.gitignore` (raw intermediates, superseded runs, training logs).

## `outputs/metrics_final/` — paper headline metrics

| Run dir | Status | Purpose |
|:--|:--|:--|
| `20260906-152152/` | **committed** | final GAP metrics run (GAP5 predicted-class fix): `gap1_mask_quality_*` (mIoU/mDice), `gap2_mask_map50.json` (mask/box mAP), `gap3_wilcoxon.json` (item-level tests), `gap5_kcal.json` (calorie MAE), `summary_all_gaps.json` |
| `20260904-142641/` | superseded | earlier GAP5 variant (GT-class kcal); ignored by `.gitignore` |

`gap3_wilcoxon.json` reports **item-level** two-sided Wilcoxon signed-rank tests
(N = 41 items, per-item mean absolute relative volume error) as the reportable
test; pair-level rows are labeled `pair_level_reference_only` because the 6,398
pairs are clustered within items (pseudo-replication).

## `outputs/predictions/` — final-half E2E runs (928 images, τ frozen)

| Run dir | Pipeline | τ | Files |
|:--|:--|:--:|:--|
| `04_e2e_paper_faithful_beta_20260903-090343/` | YOLO26n-Seg | 0.05 | `report_*.json`, `samples_*.csv`, `betas_train_*.json`, `speed_per_image.json`, `summary.txt` |
| `06a_faster_rcnn_eval_20260903-131303/` | FR-CNN + GrabCut | 0.10 | same five files |
| `06b_faster_rcnn_sam_eval_20260903-100830/` | FR-CNN + SAM | 0.05 | same five files |
| `07_e2e_yolov8_paper_faithful_20260903-090734/` | YOLOv8n-Seg | 0.30 | same five files |
| `05_train_new_20260815-032324/` | excluded | — | 146 MB intermediate (ignored) |

Each `samples_*.csv` has 6,398 data rows (one per top–side pair, penalized
protocol: missed pairs carry a 100% volume error with their ground-truth class).
Volume MAE and coverage use all 6,398 pairs; `gap5_kcal.json` scores the accepted
subset (N = 6,334–6,389 per model, same protocol as paper Table II).

`speed_per_image.json` measures end-to-end per-image latency over the 2,088
images processed in each session (final-half evaluation + β-fit pass), so the
throughput column of the paper's Table II is a session-level figure, not a
928-image-only figure; the same 2,088-image set is used for all four pipelines.

## `outputs/threshold_sweep/`

| Run dir | Status | Purpose |
|:--|:--|:--|
| `20260902-234626/` | **committed** | tuning-half sweep behind the frozen τ values (per-conf `report_*.json` + `samples_*.csv` under each model dir) |
| `20260901-082552/`, `20260902-195613/` | superseded | earlier sweeps; ignored by `.gitignore` |

## `outputs/logs/` — matching console logs

Committed for the four final runs (`04/06a/06b/07_*.log`, ~250 KB each).
The 9.5 MB `05_train_new_*` training log is excluded.

## Model checkpoints (`models/`, all committed)

- `ecustfd_yolo26seg_best.pt` / `ecustfd_yolov8seg_best.pt` — YOLO single-stage
  models (with `*_last.pt` alongside; run weights under `runs/**/weights/` are
  excluded — `models/` holds the single canonical copy of each).
- `faster_rcnn_new_best.pt` — two-stage detector (MobileNetV3 FPN).
- Stock pretrained bases (`yolo26n*.pt`, `yolov8n-seg.pt`) — **excluded**
  (Ultralytics re-downloads them automatically on the first training run).
- `sam/sam_vit_b_01ec64.pth` — **excluded** (upstream download).

## Not committed anywhere in the repo

- `data/` — original ECUSTFD images/annotations and all processed derivatives.
- `TMP/` — private working material, audits, and the paper sources themselves.
- `releases/ecustfd-seg-release/` SAM-mask payload (2.8 GB) and `SHA256SUMS` —
  published via Zenodo with checksums; the repo keeps the lightweight text
  artifacts (splits, patched XMLs, audit tables, README, CITATION.cff, LICENSE)
  so the structure is reviewable in-place. Zenodo record:
  https://doi.org/10.5281/zenodo.22664532 (resolves once published).
