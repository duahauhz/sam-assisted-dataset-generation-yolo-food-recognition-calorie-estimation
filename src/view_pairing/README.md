# `src/view_pairing/` — ghép cặp top-view ↔ side-view theo class

## Phương pháp

Paper ECUSTFD §3.6 giả định: top-view và side-view chụp **cùng một** phần
thức ăn, với cùng đồng xu calibration. Ghép cặp được thực hiện theo
chính sách "pick the highest confidence per class per view":

- Với mỗi class xuất hiện ở **cả hai view** → tạo cặp `(top, side)`.
- Class chỉ xuất hiện ở một view → đánh dấu `missing`, KHÔNG tính ME
  (paper §4: "all misidentified test images are discarded").
- Đồng xu (`coin`) **không bao giờ** được ghép cặp — nó chỉ dùng để
  hiệu chỉnh alpha.

## Lý do tách thành folder riêng

- Là cầu nối duy nhất giữa output của 2 lần chạy YOLO-seg (top + side)
  và các công thức volume. Tách riêng để dễ audit chính sách ghép.
- Không phụ thuộc numpy/Ultralytics — dễ unit-test với detection giả lập.
- Hỗ trợ cả 2 ngưỡng conf (0.5 và 0.8) để so sánh với paper.

## Cách chạy

```bash
python -m src.view_pairing.test_pairing
```

## API

```python
from src.view_pairing import (
    Detection,
    pair_top_side,
    filter_pairs_by_confidence,
    pairs_to_dicts,
)

top_dets = [
    Detection(class_name="apple", conf=0.92, bbox=(10, 20, 100, 200), mask=top_mask),
    Detection(class_name="coin",  conf=0.99, bbox=(300, 10, 400, 110)),
]
side_dets = [
    Detection(class_name="apple", conf=0.85, bbox=(0, 50, 90, 250), mask=side_mask),
]

result = pair_top_side(top_dets, side_dets)
# result.pairs  -> list of PairedDetection (only "apple" here)
# result.missing -> list of classes that appeared in only one view

# Lọc theo ngưỡng conf (0.5 cho YOLO, 0.8 cho paper)
kept = filter_pairs_by_confidence(result.pairs, conf_threshold=0.5)

# Xuất CSV-friendly
rows = pairs_to_dicts(kept)
```

## Lưu ý

- Nếu một top-view phát hiện 2 quả apple (2 bbox), chỉ lấy bbox có
  conf cao nhất. Đây là chính sách paper §3.6 và code MATLAB
  `faster_rcnn_rec.m` (giả định ≤1 phần thức ăn / class / view).
- Hàm `filter_pairs_by_confidence` dùng `min(top_conf, side_conf)` —
  yêu cầu **cả hai** view vượt ngưỡng. Đúng nguyên tắc paper.

## Tham chiếu

- Paper: `ECUSTFD/paper/1705.07632v3.pdf` §3.6, §4.
- Code MATLAB: `ECUSTFD/faster_rcnn/faster_rcnn_rec.m` lines 100-126.
- Ghi chú tiếng Việt: `TMP/conet_baseline_analysis.md` Section 2.4.
