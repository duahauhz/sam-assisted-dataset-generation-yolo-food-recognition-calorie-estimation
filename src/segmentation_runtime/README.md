# `src/segmentation_runtime/` — YOLO-seg inference + mask cropping

## Phương pháp

Folder này chứa **cầu nối duy nhất** giữa checkpoint YOLO26-seg đã
train và phần geometry pipeline. Hai module:

- `yolo_infer.py` — bọc Ultralytics: load weights, chạy `model.predict`,
  chuẩn hoá output thành list các dict `{"class_name", "conf", "bbox",
  "mask"}`.
- `crop_to_bbox.py` — cắt mask full-image về bbox detection (đúng cách
  mà code MATLAB `grabcut_mex.cpp` cắt GrabCut mask về bbox của
  Faster R-CNN), ra boolean mask đóng vai `GC_PR_FGD == 3`.

## Lý do tách thành folder riêng

- Phần này là **inference** (YOLO), không phải training. Reviewer paper
  chỉ cần đọc folder này + `volume_models/` + `coin_calibration/` +
  `calorie_estimation/` là đủ hiểu end-to-end, không bị nhiễu bởi
  code training YOLO (`src/yolo_seg/`).
- `crop_to_bbox.py` thuần numpy, dễ unit-test. YOLO integration test
  chỉ chạy nếu weights có sẵn.
- Đảm bảo **chỉ folder này** import `ultralytics` — phần còn lại của
  pipeline không phụ thuộc model framework.

## Cách chạy

```bash
# Unit test (không cần weights)
python -m src.segmentation_runtime.test_segmentation

# Smoke test (cần best.pt + 1 ảnh ECUSTFD)
# Đã include trong test_segmentation, tự skip nếu thiếu weights.
```

## API

```python
from src.segmentation_runtime import (
    load_yolo_seg, predict_one, crop_mask_to_bbox, binarize_mask,
)

model = load_yolo_seg(
    r"runs/yolo_seg/ecustfd_yolo26seg-2/weights/best.pt",
    device="0",    # hoặc "cpu"
)

dets = predict_one(model, "path/to/top.jpg", conf=0.5, imgsz=480)
for d in dets:
    print(d["class_name"], d["conf"], d["bbox"])
    cropped = crop_mask_to_bbox(d["mask"], d["bbox"])
    # cropped.shape == (bbox_h, bbox_w), dtype=bool
    # → đưa vào src.volume_models.compute_volume(...)
```

## Lưu ý

- Class set mặc định là **20 class** (19 food + 1 coin), khớp với
  ECUSTFD gốc (`src/constants.py::FOOD_CLASSES`). Không có class
  `kiwi` nào thêm vào — `qiwi` (kiwi fruit) đã có trong ECUSTFD từ
  đầu. Tra cứu tên đúng qua `FOOD_CLASSES` và `CLASS_TO_IDX`.
- Imgsz mặc định = 480 (đúng với checkpoint `yolo26n-seg` đã train).
- Confidence threshold mặc định = 0.25 (Ultralytics default). Khi report
  ME, threshold 0.5 và 0.8 sẽ được dùng để so sánh — xem
  `src/e2e_pipeline/`.

## Tham chiếu

- YOLO26-seg checkpoint: `runs/yolo_seg/ecustfd_yolo26seg-2/weights/best.pt`.
- Training code (không nằm trong folder này): `src/yolo_seg/`.
- Paper gốc: `ECUSTFD/paper/1705.07632v3.pdf` §3.3 (GrabCut).
