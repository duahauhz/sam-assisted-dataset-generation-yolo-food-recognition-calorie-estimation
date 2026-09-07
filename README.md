# SAM-Assisted Dataset Generation and Single-Stage Food Calorie Estimation on ECUSTFD

Official implementation of **"SAM-Assisted Dataset Generation for Food Recognition and Calorie Estimation Using Single-Stage YOLO Segmentation"**.

We upgrade the ECUSTFD food-calorie benchmark [1] with a semi-automatically generated **instance-segmentation layer**: a frozen SAM ViT-B, box-prompted with the original annotations, produces 6,062 polygon masks over all 2,978 images. Every mask was audited visually against per-instance proxy-IoU diagnostics; 2,940/2,978 images (98.7%) needed no manual correction, while 37 boxes were corrected in the VOC XMLs and one mask was replaced by a manual override. On this layer we build a **single-stage** calorie-estimation system in which one YOLO segmentation model replaces the original two-stage Faster R-CNN [2] + GrabCut [3] pipeline, while the original volume/calorie mathematics is kept as a faithful re-implementation.

**Headline results** (final half of the official test set: 928 images, 6,398 top–side pairs, penalized protocol where every missed pair counts as a 100% volume error; frozen per-model operating points; RTX 4050 Laptop):

| Pipeline | end-to-end mIoU | Volume MAE % | Calorie MAE (kcal) | Coverage % | Throughput (img/s) |
|:--|:--:|:--:|:--:|:--:|:--:|
| Faster R-CNN + GrabCut [1]–[3] | 0.925 | 27.06 | 65.01 | 99.86 | 0.46 |
| Faster R-CNN + SAM [4] | 0.975\* | 24.87 | 63.57 | 99.86 | 2.20 |
| YOLOv8n-Seg (single-stage) | 0.942 | **24.54** | **40.29** | 99.52 | **51.6** |
| YOLO26n-Seg (single-stage) | 0.940 | 25.38 | 41.60 | 99.31 | 41.6 |

\* Reference masks are the SAM-generated labels of the annotation layer, so the online-SAM pipeline is structurally favored in IoU; read jointly with the MAE columns. Mask mAP@50 of the learned models against the generated labels: 0.9715 (YOLOv8n), 0.9408 (YOLO26n). The contribution is the **single-stage design + dataset layer** rather than a specific detector version: at the item level (N = 41 items), accuracy differences between all four pipelines are **not statistically significant** (two-sided Wilcoxon signed-rank on per-item mean absolute relative volume error, p = 0.45–1.00), while the single-stage throughput advantage is 19–113× over the two-stage baselines (90–113× vs. GrabCut). Volume MAE and coverage are computed over all 6,398 pairs under the penalized protocol; calorie MAE over the scored subset (N = 6,334–6,389 accepted samples, paper Table II protocol); throughput is measured end-to-end over the 2,088 images processed in each evaluation session (final-half + β-fit images), n per model.

The derived annotation layer itself is published as a separate package — see [Annotation release](#annotation-release) (labels-only, no images).

---

## Contents

- [Method overview](#method-overview)
- [Repository layout](#repository-layout)
- [Environment](#environment)
- [Data: obtaining ECUSTFD](#data-obtaining-ecustfd)
- [Reproducing the pipeline](#reproducing-the-pipeline)
- [Evaluation protocol](#evaluation-protocol)
- [Reproducing the paper numbers](#reproducing-the-paper-numbers)
- [Test suite](#test-suite)
- [Annotation release](#annotation-release)
- [Model checkpoints](#model-checkpoints)
- [Committed-outputs inventory](docs/COMMITTED_OUTPUTS.md)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Method overview

Four pipelines share the same volume/calorie mathematics; only the vision stage differs:

1. **`01`–`02` Label generation (offline).** Every original bounding box prompts a frozen SAM ViT-B; masks are reviewed visually against per-instance proxy-IoU diagnostics, 37 boxes are corrected in the VOC XMLs, and one mask is replaced by a manual override. Output: YOLO-seg label set (20 classes = 19 foods + reference coin) with full provenance flags (`raw_gt_box_prompt` / `patched_gt_box_prompt` / `manual_mask_override`).
2. **Detection + segmentation (online).** Either a two-stage baseline (Faster R-CNN + GrabCut / SAM) or a single-stage YOLO-segmentation model (YOLO26n / YOLOv8n, both trained on the generated labels).
3. **Geometry (inherited from [1]).** Coin-calibrated pixel scale α; per-class dispatch to five geometric volume models (ellipsoid / column / solid of revolution / grape air-gap / torus), integrating the two viewpoint masks row-wise.
4. **Calibration + energy.** Per-class volume-bias correction β_k fitted on a 50/50 item split of the training items (1,169 images), then mass (ρ_k) and calories (q_k · Ṽ) with the predicted class's parameters, as in the original detected-class loop.

## Repository layout

```
├── README.md                  # this file
├── LICENSE                    # code license (MIT)
├── requirements.txt           # Python dependencies
├── src/                       # all code + pipeline notebooks (see src/README.md)
│   ├── constants.py           #   19 food classes, density ρ_k, energy factors q_k
│   ├── config.py              #   path resolution & defaults
│   ├── data_prep_SAM1/        #   01: bbox verification, SAM mask generation, overrides
│   ├── yolo_seg/              #   02: YOLO-seg dataset export + training entry
│   ├── faster_rcnn/           #   05a/06a/06b: two-stage baseline (train; eval with GrabCut/SAM masks)
│   ├── yolo_seg_eval/         #   04/07: single-stage E2E evaluation pipeline
│   ├── beta_correction/       #   per-class β_k calibration
│   ├── coin_calibration/      #   coin-based pixel scale
│   ├── volume_models/         #   3D geometric volume algorithms
│   ├── calorie_estimation/    #   mass & calorie formulas (paper-faithful)
│   ├── view_pairing/          #   top/side view pairing
│   ├── segmentation_runtime/  #   YOLO-seg inference wrapper
│   ├── e2e_pipeline/          #   orchestrator, metrics, threshold-sweep runner
│   └── 00–08_*.ipynb          #   numbered pipeline notebooks (see below)
├── models/                    # trained checkpoints (see Model checkpoints)
├── runs/                      # Ultralytics training evidence (curves, confusion matrices, results.csv)
├── outputs/                   # evaluation artifacts committed as paper evidence
│   ├── metrics_final/         #   GAP analyses: mask quality, mAP, Wilcoxon, kcal MAE
│   ├── predictions/           #   final-half E2E run dirs (reports/samples/speed/summary)
│   ├── threshold_sweep/       #   tuning-half threshold sweep (conf grid per model)
│   └── logs/                  #   matching console logs for the committed runs
├── releases/ecustfd-seg-release/   # annotation layer (text artifacts committed; 2.8 GB SAM masks via Zenodo)
├── logs/                      # environment/data-fetch logs
├── docs/                      # repo-facing notes: committed-outputs inventory
└── data/                      # NOT committed (see .gitignore) — raw & processed data
```

### Pipeline notebooks (`src/`, in execution order)

| Notebook | Role | Paper artifact |
|:--|:--|:--|
| `00_fetch_data` / `00_setup_env` | download ECUSTFD, environment report | — |
| `01_run_sam1_pipeline` | SAM label generation + QA | annotation layer |
| `02_yolo_seg_pipeline` | dataset export to YOLO-seg format | `data/processed/yolo_ecustfd_seg` |
| `03_yolo_seg_train` | train YOLO26n-Seg | `models/ecustfd_yolo26seg_*.pt` |
| `03_threshold_sweep` | per-model confidence sweep on **tuning half** | `outputs/threshold_sweep/…` |
| `04_e2e_paper_faithful_beta` | YOLO26n E2E on **final half** (τ=0.05, β) | `outputs/predictions/04_…/` |
| `05a_faster_rcnn_train` | train Faster R-CNN (MobileNetV3 FPN) | `models/faster_rcnn_new_best.pt` |
| `05b_yolo_seg_v8n_train` | train YOLOv8n-Seg | `models/ecustfd_yolov8seg_*.pt` |
| `06a_faster_rcnn_eval` | FR-CNN + GrabCut E2E (τ=0.10) | `outputs/predictions/06a_…/` |
| `06b_faster_rcnn_sam_eval` | FR-CNN + SAM E2E (τ=0.05) | `outputs/predictions/06b_…/` |
| `07_e2e_yolov8_paper_faithful` | YOLOv8n E2E (τ=0.30) | `outputs/predictions/07_…/` |
| `07_statistical_test` | Wilcoxon signed-rank tests | `outputs/metrics_final/` |
| `08_paper_metrics_final` | headline GAP metrics (mIoU/mAP/Wilcoxon/kcal) | `outputs/metrics_final/<run>/` |

Every notebook is self-contained (config in cell 3) and logs concurrently to console and `outputs/logs/` (or a run dir). All numbers reported in the paper are produced by these notebooks; every committed metrics file traces to exactly one logged run. Note: two notebook numbers are intentionally duplicated (`03` train + sweep, `07` E2E + statistics) — the table's order is the execution order.

## Environment

```bash
pip install -r requirements.txt          # torch, ultralytics, opencv, scipy, jupyter, …
```

Tested environment: Windows 11 + Git Bash, Python 3.14.4, PyTorch 2.13.0 (CUDA 13.0), Ultralytics 8.4.87; hardware used for all reported results: NVIDIA RTX 4050 Laptop GPU (6 GB). The CPU-only path also runs, but the two-stage baselines become impractically slow (GrabCut ≈ 0.46 img/s is already a GPU measurement). Two additional one-time downloads:

- **SAM ViT-B** checkpoint `sam_vit_b_01ec64.pth` → `models/sam/` (from [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)).
- **ECUSTFD images** (see below).

## Data: obtaining ECUSTFD

This repository does **not** redistribute the ECUSTFD photographs (the original authors ask that their paper be cited; their repository is the distribution channel). One-time setup:

```bash
git clone https://github.com/Liang-yc/ECUSTFD-resized-.git
# place JPEGImages/ + Annotations/ + ImageSets/ + density.xls under data/raw/ECUSTFD/
```

Notebook `00_fetch_data.ipynb` automates this and writes an inventory log. The derived **labels/masks** need not be regenerated — they are available pre-built in the annotation release and by running `01`–`02`.

## Reproducing the pipeline

Minimal path from raw ECUSTFD to paper numbers (each step = one notebook; `00_fetch_data` / `00_setup_env` are the prerequisite environment/data steps — run once):

```bash
# 0) one-time: download ECUSTFD + environment report
jupyter nbconvert --execute src/00_fetch_data.ipynb
# 1) labels: SAM masks + QA + YOLO-seg export
jupyter nbconvert --execute src/01_run_sam1_pipeline.ipynb
jupyter nbconvert --execute src/02_yolo_seg_pipeline.ipynb

# 2) training
jupyter nbconvert --execute src/03_yolo_seg_train.ipynb        # YOLO26n-Seg
jupyter nbconvert --execute src/05a_faster_rcnn_train.ipynb   # Faster R-CNN
jupyter nbconvert --execute src/05b_yolo_seg_v8n_train.ipynb # YOLOv8n-Seg

# 3) operating points: confidence sweep on the tuning half (β disabled)
jupyter nbconvert --execute src/03_threshold_sweep.ipynb

# 4) final-half evaluation with frozen τ and β — one notebook per pipeline
jupyter nbconvert --execute src/04_e2e_paper_faithful_beta.ipynb      # YOLO26n,  τ=0.05
jupyter nbconvert --execute src/06a_faster_rcnn_eval.ipynb            # + GrabCut, τ=0.10
jupyter nbconvert --execute src/06b_faster_rcnn_sam_eval.ipynb        # + SAM,     τ=0.05
jupyter nbconvert --execute src/07_e2e_yolov8_paper_faithful.ipynb    # YOLOv8n,   τ=0.30

# 5) statistics + headline metrics
jupyter nbconvert --execute src/07_statistical_test.ipynb
jupyter nbconvert --execute src/08_paper_metrics_final.ipynb
```

For exploratory runs without notebooks, the underlying CLI is:

```bash
python -m src.e2e_pipeline run --split test --conf 0.05 --apply-beta   # exploratory val/test splits
```

**Important:** the frozen operating points above are the ones used for every reported number. Re-running the sweep (step 3) may select a different confidence value on your hardware; to reproduce the paper exactly, keep the notebook configs as committed.

## Evaluation protocol

The official ECUSTFD split has no held-out test: its `val.txt` is part of `trainval` (validation fitness was computed on images seen in training), and the original MATLAB code hard-codes `thres=0.8` — a hand-tuned 2017 artifact. We therefore:

- **Split the official test set** (1,733 images) into two disjoint halves: `test_tune` (805 images, 6,557 pairs) for threshold selection and `test_final` (928 images, 6,398 pairs) for reporting — lists in `releases/ecustfd-seg-release/splits/` (local) and `data/raw/ECUSTFD/ImageSets/Main/`.
- **Tune, then freeze** per-model confidence thresholds on the tuning half (volume MAE minimized subject to ≥85% coverage, β disabled; ≤0.3 pp differences treated as noise, broken toward higher coverage): τ = 0.05 (YOLO26n), 0.30 (YOLOv8n), 0.10 (FR-CNN+GrabCut), 0.05 (FR-CNN+SAM). The 0.8 default is **not** used anywhere. Because the penalized protocol charges a 100% error for every miss, the sweep naturally favors low τ (high coverage); the residual cost is a small number of low-confidence false-positive detections (e.g., coin false positives on background objects), which is why coverage is always reported alongside the penalized MAE — see the threshold-sweep report for the full coverage/MAE curve per model.
- **Penalized protocol**: an unaccepted pair is not discarded — it contributes a zero-volume sample carrying its ground-truth reference (a 100% volume error in the correct class). Report fields `n_pairs` / `n_samples` / `n_pairs_with_samples` make this auditable (6,398 → 6,354–6,389 accepted depending on model).
- **Item-level statistics**: the 6,398 pairs cluster into 41 items (median 81 pairs/item), so all significance tests are Wilcoxon signed-rank at the item level (N = 41); pair-level counts are pseudo-replicated and never used for inference.
- **Known limitation (disclosed in the paper):** the YOLO `best.pt` checkpoints were selected by validation fitness on the official test split (which contains the final half) — a mild checkpoint-selection leakage (~1 pp mAP) favoring the YOLO pipelines; the Faster R-CNN baseline is unaffected.

## Reproducing the paper numbers

The committed artifacts double as evidence. Each final-half run directory contains `report_*.json` (overall + per-class metrics), `samples_*.csv` (6,398 rows, one per pair), `betas_train_*.json` (fitted β_k), `speed_per_image.json` (n = 2,088 images), and `summary.txt`; the matching console log lives in `outputs/logs/`:

| Artifact | Path |
|:--|:--|
| YOLO26n final run | `outputs/predictions/04_e2e_paper_faithful_beta_20260903-090343/` |
| FR-CNN + GrabCut final run | `outputs/predictions/06a_faster_rcnn_eval_20260903-131303/` |
| FR-CNN + SAM final run | `outputs/predictions/06b_faster_rcnn_sam_eval_20260903-100830/` |
| YOLOv8n final run | `outputs/predictions/07_e2e_yolov8_paper_faithful_20260903-090734/` |
| Threshold sweep (tuning half) | `outputs/threshold_sweep/20260902-234626/` |
| GAP metrics: mIoU / mAP / Wilcoxon / kcal MAE | `outputs/metrics_final/20260906-152152/` (source of every headline number) |

Mapping from paper Table II to files: mIoU ← `gap1_mask_quality_summary.json` (`mean_iou_micro`); Volume MAE ← run `report_*.json` → `overall.mean_abs_me_volume_pct` (penalized: misses carry 100% error); Calorie MAE ← `gap5_kcal.json` → `mae_kcal`; Coverage ← `report_*.json` → `n_pairs_with_samples / n_pairs`; Throughput ← `speed_per_image.json` → `images_per_second`.

## Test suite

```bash
python -m pytest src -q
```

52 unit tests (all passing) cover the deterministic core: geometric volume models, coin calibration, calorie formulas (paper-faithful cases), view pairing, dataset split invariants (tune/final disjoint, union = official test), and metrics. Inference-level smoke tests skip gracefully when weights/GPU are absent.

## Annotation release

The derived annotation layer is packaged for publication at `releases/ecustfd-seg-release/` (labels-only, ~2.9 GB with SAM masks):

- `yolo_ecustfd_seg/` — YOLO-seg labels (train 1,245 / val 1,733 = official trainval/test images) + per-image SAM mask `.npy` files with provenance, + dataset YAML.
- `patched_xml/` — the 37 corrected VOC XMLs (drop-in replacements).
- `audit/` — per-instance IoU index, per-image stats, per-class counts.
- `splits/` — official lists + `test_tune` / `test_final`.
- `SHA256SUMS`, `LICENSE` (CC BY 4.0), `CITATION.cff`, and a README with image-obtaining instructions.

It contains **no ECUSTFD images**; users clone the original repository and overlay this layer by matching file names. See the release README for the full mapping notes (known naming quirks of the original release are documented there). Zenodo DOI pending publication.

## Model checkpoints

Trained weights used for every reported number are included in this repository:

| File | Size | SHA-256 |
|:--|:--:|:--|
| `models/ecustfd_yolo26seg_best.pt` (YOLO26n-Seg) | 6.2 MB | `7466e87fda45e0f2441f75e51d9f5db1c40cc861ed5d86701297067f29dcf833` |
| `models/ecustfd_yolov8seg_best.pt` (YOLOv8n-Seg) | 6.5 MB | `519a037041daff60f53ae63b9e972e9cd3f7b149d7dd77c2b5b991f68a9e1aba` |
| `models/faster_rcnn_new_best.pt` (FR-CNN, MobileNetV3 FPN) | 145 MB | `fe9de4f4659c0bc7fb7a8480ea962f1c7adbad8fc793773c88de02bcabc4dbee` |

Also present: `*_last.pt` YOLO checkpoints and the Ultralytics training evidence under `runs/` (curves, confusion matrices, `results.csv`; run weights are not duplicated — use the `models/` files above). The stock pretrained bases (`yolo26n-seg.pt`, `yolov8n-seg.pt`, …) are not committed; Ultralytics downloads them automatically on the first training run. SAM `sam_vit_b_01ec64.pth` (375 MB) is likewise a one-time upstream download — see `.gitignore`.

## Citation

If you use this code or the annotation layer, please cite both this work and the original ECUSTFD paper:

```bibtex
@article{thiswork2026,
  title   = {SAM-Assisted Dataset Generation for Food Recognition and Calorie
             Estimation Using Single-Stage YOLO Segmentation},
  author  = {(authors per final published version)},
  year    = {2026},
  note    = {Preprint; see releases/ecustfd-seg-release for the annotation layer}
}

@article{liang2017computer,
  title   = {Computer Vision-based Food Calorie Estimation: Dataset, Method, and Experiment},
  author  = {Liang, Yanchao and Li, Jianhua},
  journal = {arXiv preprint arXiv:1705.07632},
  year    = {2017}
}
```

## License

- **Code** (this repository, `src/`, evaluation scripts, notebooks): **MIT** — see `LICENSE`.
- **Annotation layer** (`releases/ecustfd-seg-release/`): **CC BY 4.0** — see its `LICENSE` file.
- **ECUSTFD images**: property of the original authors; this repository does not redistribute them.
- Third-party components retain their upstream licenses: Ultralytics (AGPL-3.0), PyTorch (BSD-style), Segment Anything (Apache 2.0). Downstream users of the training code that links Ultralytics must comply with AGPL-3.0.

## Acknowledgements

ECUSTFD is the work of Yanchao Liang and Jianhua Li [1]; its images remain the property of the original authors. We use Meta's Segment Anything Model [4], Ultralytics YOLOv8 [6] and YOLO26 [8], and torchvision's Faster R-CNN [2] with a MobileNetV3 [19] backbone. The geometric volume models, coin calibration, and energy factors are re-implemented from the original MATLAB/C++ release.

## References

(Numbering matches the paper's reference list.)

[1] Y. Liang and J. Li, "Computer vision-based food calorie estimation: dataset, method, and experiment," arXiv preprint arXiv:1705.07632, 2017.
[2] S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards real-time object detection with region proposal networks," in *Adv. Neural Inf. Process. Syst. (NIPS)*, 2015.
[3] C. Rother, V. Kolmogorov, and A. Blake, "'GrabCut': Interactive foreground extraction using iterated graph cuts," *ACM Trans. Graphics*, vol. 23, no. 3, pp. 309–314, 2004.
[4] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, and R. Girshick, "Segment Anything," arXiv preprint arXiv:2304.02643, 2023.
[5] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, real-time object detection," in *Proc. IEEE CVPR*, 2016.
[6] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," Ultralytics, 2023. Software, ver. 8.4.87.
[7] R. Sapkota and M. Karkee, "Ultralytics YOLO evolution: An overview of YOLO26, YOLO11, YOLOv8, and YOLOv5 object detectors for computer vision and pattern recognition," arXiv preprint arXiv:2510.09653, 2025.
[8] G. Jocher, J. Qiu, M. Liu, S. Lyu, F. C. Akyon, and M. E. Kalfaoglu, "Ultralytics YOLO26: Unified real-time end-to-end vision models," arXiv preprint arXiv:2606.03748, 2026.
[9] A. Myers, N. Johnston, V. Rathod, A. Korattikara, A. Gorban, N. Silberman, S. Guadarrama, G. Papandreou, J. Huang, and K. Murphy, "Im2Calories: Towards an automated mobile vision food diary," in *Proc. IEEE ICCV*, 2015.
[10] P. Pouladzadeh, P. Kuhad, S. V. B. Peddi, A. Yassine, and S. Shirmohammadi, "Mobile cloud based food calorie measurement," in *Proc. IEEE ICMEW*, 2014.
[11] J. He, Z. Shao, J. Wright, D. Kerr, C. Boushey, and F. Zhu, "Multi-task image-based dietary assessment for food recognition and portion size estimation," in *Proc. IEEE MIPR*, 2020.
[12] J. He, R. Mao, Z. Shao, J. L. Wright, D. A. Kerr, C. J. Boushey, and F. Zhu, "An end-to-end food image analysis system," in *Proc. IS&T Electronic Imaging*, 2021.
[13] Z. Shao, Y. Han, J. He, R. Mao, J. Wright, D. Kerr, C. Boushey, and F. Zhu, "An integrated system for mobile image-based dietary assessment," in *Proc. 3rd Workshop on AIxFood (AI&Food '21), ACM Multimedia (MM)*, 2021.
[14] J. Dehais, M. Anthimopoulos, S. Shevchik, and S. G. Mougiakakou, "Two-view 3D reconstruction for food volume estimation," *IEEE Trans. Multimedia*, vol. 19, no. 5, pp. 1090–1099, 2017.
[15] J. Gao, W. Tan, L. Ma, Y. Wang, and W. Tang, "MUSEFood: Multi-sensor-based food volume estimation on smartphones," in *Proc. IEEE SmartWorld/SCALCOM/UIC/ATC/CBDCom/IOP/SCI*, 2019.
[16] A. AlMughrabi, U. Haroon, R. Marques, and P. Radeva, "VolETA: One- and few-shot food volume estimation," arXiv preprint arXiv:2407.01717, 2024.
[17] A. AlMughrabi, U. Haroon, R. Marques, and P. Radeva, "VolTex: Food volume estimation using text-guided segmentation and neural surface reconstruction," in *Proc. IEEE/CVF CVPRW*, 2025.
[18] K. He, G. Gkioxari, P. Dollár, and R. Girshick, "Mask R-CNN," in *Proc. IEEE ICCV*, 2017.
[19] A. Howard, M. Sandler, G. Chu, L.-C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, Q. V. Le, and H. Adam, "Searching for MobileNetV3," in *Proc. IEEE/CVF ICCV*, 2019.
[20] F. Mumuni and A. Mumuni, "Segment Anything Model for automated image data annotation: empirical studies using text prompts from Grounding DINO," arXiv preprint arXiv:2406.19057, 2024.
