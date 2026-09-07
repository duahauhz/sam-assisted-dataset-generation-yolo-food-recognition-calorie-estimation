# Beta Correction Module

This module implements the per-class multiplicative volume calibration factor $\beta_k$:

$$V_{\text{calibrated}} = \beta_k \times V_{\text{raw}}$$

where $\beta_k$ is computed on the **Train set** for each food class $k$ matching the official paper implementation in `ECUSTFD/faster_rcnn/xls_results_analysis.m` (lines 96-97 & 134-136):

$$\beta_k = \frac{\sum V_{\text{real (train)}}}{\sum V_{\text{raw\_est (train)}}}$$

## Usage
1. Compute $\beta_k$ on train set predictions using `compute_class_betas(train_sample_records)`.
2. Save to JSON (`outputs/predictions/betas_train_conf80.json`).
3. Apply `calibrate_volume(v_raw, class_name, betas)` when evaluating test set samples.
