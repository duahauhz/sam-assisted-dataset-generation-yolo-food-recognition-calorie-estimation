# Confidence Threshold Sweep Report

- **Session Timestamp**: `20260902-234626`
- **Evaluation Split**: `test_tune`
- **Optimization Method**: Constrained Optimization ($\min 	ext{MAE}$ s.t. $	ext{Coverage} \ge 85.0\%$)
- **Beta Calibration during Sweep**: `False` (Unbiased baseline sweep)

## 1. Selected Optimal Thresholds ($th^*$)

| Model Architecture   |   Optimal Conf ($th^*) | Macro MAE (%)   | Coverage (%)   | Bias (%)   | Pairs Detected   |   Total Samples |
|:---------------------|-----------------------:|:----------------|:---------------|:-----------|:-----------------|----------------:|
| YOLOv8n-Seg          |                   0.95 | 19.20%          | 94.7%          | 11.25%     | 6211/6557        |            6211 |
| YOLO26n-Seg          |                   0.05 | 20.27%          | 100.0%         | 12.66%     | 6557/6557        |            6557 |
| Faster-RCNN+GrabCut  |                   0.1  | 19.03%          | 100.0%         | -12.33%    | 6557/6557        |            6557 |
| Faster-RCNN+SAM      |                   0.05 | 19.24%          | 100.0%         | -14.94%    | 6557/6557        |            6557 |

## 2. Downstream Protocol (STEP 4 & STEP 5)

1. **Freeze**: Do NOT modify or tune $th^*$ on the Test set.
2. **Test Split Evaluation**: Run End-to-End volume and calorie estimation on the **Test Split** (1,733 images) using frozen $th^*$.
3. **Beta Calibration ($eta$)**: Apply 50/50 train-fit $eta$ scaling in Step 5 to obtain final publication-ready accuracy.
4. **Instance Segmentation mAP50**: Compute Mask mAP50 on the Test set.
5. **Latency & Throughput (FPS)**: Benchmark End-to-End runtime per image.
