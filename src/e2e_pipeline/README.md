# `src/e2e_pipeline/` — pipeline end-to-end + evaluation

> ⚠️ **Vai trò hiện tại**: module này phục vụ **chạy thăm dò** (split val/test gốc).
> Số liệu paper sinh từ notebook `04/06a/06b/07` (dùng `src/yolo_seg_eval/eval_pipeline.py`
> với split `test_final`/`test_tune` + τ freeze từng model + β_k). Xem `README.md` gốc repo.

## Phương pháp

Folder này **ghép nối** tất cả các layer đã xây (segmentation → coin →
pairing → volume → calorie) thành một pipeline chạy được, đồng thời
cung cấp metric evaluator.

### Công thức calorie runtime

```
C = q_k * V_tilde       (kcal)
```

Công thức này là runtime MATLAB baseline (xem
`ECUSTFD/faster_rcnn/faster_rcnn_rec.m` line 156). `q_k` đọc từ
`food_info.xls`. (Paper dùng biến thể paper-faithful đầy đủ —
predicted-class q_k + β_k, xem `src/calorie_estimation/paper_faithful.py`.)

### Cách chạy

```bash
# Dump YOLO-seg metrics (đã có sẵn từ training)
python -m src.e2e_pipeline.evaluate_seg

# Chạy end-to-end trên split gốc (thăm dò — KHÔNG phải protocol paper):
python -m src.e2e_pipeline run --split val --conf 0.5
python -m src.e2e_pipeline run --split test --conf 0.5 --apply-beta
```

Mỗi cấu hình tạo ra 2 file:
- `outputs/predictions/samples_<split>_conf<int>.csv` — per-sample.
- `outputs/predictions/report_<split>_conf<int>.json` — per-class ME
  + overall.

Tổng hợp thành 1 markdown:

```bash
python -m src.e2e_pipeline.render_report
```

### Per-class evaluation

```bash
python -m src.e2e_pipeline.compare_protocols \
    --samples-csv outputs/predictions/samples_test_conf5_beta.csv
```

### Unit test

```bash
python -m src.e2e_pipeline.test_dataset_split
python -m src.e2e_pipeline.test_metrics
```

## Các module trong folder

| File | Trách nhiệm |
|---|---|
| `dataset_split.py` | Parse filename `xxxS(1).JPG / T(1).JPG`, đọc `test.txt`/`train.txt`/`val.txt`/`test_tune.txt`/`test_final.txt`, ghép ảnh top+side theo stem. |
| `metrics.py` | Tính **Mean Error (ME)** theo class (paper §4) — cả volume và mass, có dấu, ×100 ra %. |
| `run_e2e.py` | Orchestrator chính: YOLO-seg → pair → volume → calorie → ME. |
| `ablation_runner.py` | Chạy sweep conf cho 1 model (dùng bởi `03_threshold_sweep.ipynb`). |
| `compare_protocols.py` | Per-class Mean Error dựa trên CSV output của `run_e2e.py`. |
| `evaluate_seg.py` | Re-export metric segmentation (Ultralytics) ra JSON + Markdown. |
| `render_report.py` | Tổng hợp JSON output của `run_e2e.py` thành 1 Markdown. |
| `test_dataset_split.py` | Unit test pairing helpers (+ bất biến tune/final disjoint). |
| `test_metrics.py` | Unit test ME formula. |

## Định nghĩa metric (paper-faithful)

```
ME_volume(k) = (1 / n_k) * Σ_i (v_pred_i − v_real_i) / v_real_i   (×100 để ra %)
ME_mass(k)   = (1 / n_k) * Σ_i (m_pred_i − m_real_i) / m_real_i
```

- **Có dấu** (paper §4, không phải absolute). MAE/|ME| tổng hợp lấy từ report JSON.
- Protocol paper (penalized): cặp không được accept **không bị loại** — mang sai số
  100% volume kèm class ground-truth; xem `n_pairs`/`n_samples`/`n_pairs_with_samples`
  trong report JSON.
- Ground truth đọc từ `data/raw/ECUSTFD/density.xls`.
- Conf: mỗi model dùng τ riêng được freeze từ sweep trên `test_tune`
  (0.05/0.30/0.10/0.05) — **không** dùng 0.8 mặc định của MATLAB gốc.

## Lưu ý về đơn vị `density.xls`

File ghi nhãn cột `volume(mm^3)` nhưng giá trị thực tế là cm³ (xem
`TMP/conet_baseline_analysis.md` §2.7 — tài liệu nội bộ). Code xử lý
như cm³ — nếu thực sự là mm³ thì phải chia 1000.

## Artifacts sinh ra

```
outputs/predictions/
  samples_<split>_conf<int>[_beta].csv
  report_<split>_conf<int>[_beta].json
outputs/threshold_sweep/<ts>/    # từ 03_threshold_sweep.ipynb
outputs/reports/
  seg_metrics.json / seg_metrics.md
```

## Tham chiếu

- Paper: `ECUSTFD/paper/1705.07632v3.pdf` §4 (Mean Error).
- Splits: `data/raw/ECUSTFD/ImageSets/Main/{test,train,val,test_tune,test_final}.txt`.
- Ground truth: `data/raw/ECUSTFD/density.xls`.
- Runtime MATLAB baseline: `ECUSTFD/faster_rcnn/faster_rcnn_rec.m` line 156.
