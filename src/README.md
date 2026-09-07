# `src/` — Food Calorie Estimation Pipeline

Pipeline phát hiện và ước lượng calo từ ảnh thực phẩm (ECUSTFD dataset, 20 lớp = 19 thực phẩm + đồng xu chuẩn).
Thay Faster R-CNN + GrabCut của hệ gốc (Liang & Li, 2017) bằng **single-stage YOLO-segmentation**, giữ nguyên phần toán thể tích/calo gốc (re-implementation trung thành).

> 📖 **Tổng quan phương pháp, số liệu chính, protocol đánh giá**: xem `README.md` ở gốc repo (bản chuẩn, tiếng Anh). File này là hướng dẫn chạy ở cấp module.

---

## 1. Cấu trúc module

| Thư mục | Vai trò | Notebook tương ứng |
|---|---|---|
| `constants.py` | 19 lớp thực phẩm, hệ số khối lượng riêng ρ_k, năng lượng q_k | — |
| `config.py` | Giải quyết đường dẫn & default | — |
| `data_prep_SAM1/` | Kiểm tra bbox trực quan, sinh mask SAM, override thủ công | `01_run_sam1_pipeline` |
| `yolo_seg/` | Export dataset YOLO-seg + train YOLO26n | `02`, `03_yolo_seg_train` |
| `yolo_seg_eval/` | Đánh giá E2E single-stage (`_run_one_config`) | `04`, `07` |
| `faster_rcnn/` | Baseline hai giai đoạn (train + eval, GrabCut/SAM backend) | `05a`, `06a`, `06b` |
| `beta_correction/` | Hiệu chỉnh β_k theo lớp | (trong 04/06/07) |
| `coin_calibration/` | Quy đổi px → cm từ đồng xu | (trong pipeline) |
| `volume_models/` | 5 mô hình thể tích hình học (ellipsoid/column/revolution/grape/torus) | (trong pipeline) |
| `calorie_estimation/` | Công thức khối lượng & calo (paper-faithful) | (trong pipeline) |
| `view_pairing/` | Ghép cặp ảnh nhìn từ trên / từ cạnh | (trong pipeline) |
| `segmentation_runtime/` | Wrapper inference YOLO-seg | (trong pipeline) |
| `e2e_pipeline/` | Orchestrator, metrics, threshold sweep | `03_threshold_sweep` |
| `legacy_src/` | Module thử nghiệm giai đoạn đầu (không dùng cho paper) | — |

## 2. Quy trình chạy (đúng thứ tự notebook)

Xem chi tiết lệnh từng bước ở `README.md` gốc → mục *Reproducing the pipeline*.

```
00 fetch/setup → 01 SAM labels → 02 export → 03 train YOLO26n
→ 03 sweep (chọn τ trên test_tune) → 04/06a/06b/07 eval trên test_final (τ đã freeze)
→ 07 statistical test → 08 paper metrics
```

Điểm cần nhớ:

- **τ (conf) mỗi model khác nhau** — được chọn trên `test_tune` (805 ảnh) rồi freeze:
  0.05 (YOLO26n), 0.30 (YOLOv8n), 0.10 (FR-CNN+GrabCut), 0.05 (FR-CNN+SAM).
  **Không dùng default 0.8** của MATLAB gốc (artifact 2017, xem paper §IV footnote).
- **β_k được fit trên 50/50 item split của tập train** (1,169 ảnh), tách rời khỏi `test_final`.
- Mọi số liệu trong paper sinh từ notebook có log (`outputs/logs/` hoặc run dir riêng).

## 3. Chạy nhanh không qua notebook

```bash
# E2E trên split val/test gốc (thăm dò, không phải protocol paper)
python -m src.e2e_pipeline run --split test --conf 0.5 --apply-beta

# Unit test deterministic core
python -m pytest src -q
```

Split `test_final`/`test_tune` được dùng qua notebook 04/06/07 (config cell 3) — đây là đường sinh số liệu paper.

## 4. Kiểm chứng không leak (đã chạy, giữ làm bằng chứng)

```
test.txt      : 1733 stems   (test gốc ECUSTFD)
test_tune.txt : 805 stems    ∩ test_final = 0
test_final.txt: 928 stems    ∪ test_tune  = test.txt (1733)
trainval.txt  : 1245 stems   ∩ test = 0
```

⚠️ Lưu ý: `val.txt` (623) ⊂ `trainval` — protocol gốc ECUSTFD train trên trainval và đo fitness trên val, tức **val không phải held-out**. Đó là lý do paper tự tách test thành tune/final thay vì dùng val (chi tiết + disclosure trong README gốc, mục *Evaluation protocol*).

## 5. Files quan trọng (reviewer kiểm chứng)

| Đường dẫn | Vai trò |
|---|---|
| `src/yolo_seg/dataset/convert_to_yolo_seg.py` | Build dataset, paper-faithful split |
| `src/e2e_pipeline/dataset_split.py` (+ test) | Logic chia tune/final + bất biến disjoint |
| `outputs/predictions/04|06a|06b|07_*_20260903-*/` | Run cuối trên test_final (report/samples/β/speed) |
| `outputs/metrics_final/20260906-152152/` | GAP metrics: mIoU, mAP, Wilcoxon, kcal MAE |
| `outputs/threshold_sweep/20260902-234626/` | Sweep chọn τ trên test_tune |

## 6. Ghi chú lịch sử (không dùng cho paper)

Các kết quả giai đoạn đầu (protocol cũ conf=0.8, MAPE ~23% trên 10,583 mẫu gồm cả train items) đã bị thay bằng protocol tune/final hiện tại. Số liệu cũ chỉ còn giá trị nội bộ, **không trích dẫn vào paper** — mọi số chính thức lấy từ `outputs/metrics_final/20260906-152152/`.
