# Manual mask overrides — workflow & toolchain

Ghi lại workflow thực tế đã patch mask cho `grape001T(5)`.
Đọc file này khi muốn thêm ảnh override mới vào bảng `OVERRIDES` trong
`apply_mask_overrides.py`.

---

## 1. Khi nào cần override?

Khi SAM1 (`segment_sam1_box.py`) trả mask bám background hoặc lệch hẳn
so với chùm nho — thường vì `bbox_voc` (từ GT label) quá rộng hoặc trải
qua vùng nền tối giống màu nho. Kiểm tra nhanh:

```python
import numpy as np
d = np.load("data/processed/sam_masks_full/masks/<stem>.npy",
            allow_pickle=True).item()
for o in d["objects"]:
    print(o["class"], "area:", o["area"], "bbox_voc:", o["bbox_voc"])
```

Nếu `area` quá lớn (>100k px cho grape top-view, khi median ảnh nho-T
là ~58k) → override.

---

## 2. Công cụ & format

- **Annotator tool**: GIMP 3.2.4 (đã thử Labelme/CVAT — không cần thiết
  cho 1-2 ảnh đơn lẻ vì đường cong phức tạp)
- **Output**: 1 file PNG grayscale, **đen = nền, trắng = nho**
  (foreground = `pixel > 127`)
- **Path**: `data/annotation/manual_masks/<stem>_<class>.png`
- **Size**: khớp `image_shape` của `<stem>` (thường `612×816` cho ECUSTFD).
  Script tự resize nếu lệch, nhưng resize dễ làm mask bập bênh ở biên
  bbox, nên vẽ đúng size ngay từ đầu.

---

## 3. Quy trình vẽ thủ công (cho `grape001T(5)`)

### Bước A — Tạo canvas đen đúng size

Trong GIMP:

1. `File → New` (Ctrl+N)
2. Width `612`, Height `816` (nhập đúng theo `image_shape`,
   KHÔNG đảo — với ECUSTFD là width 612, height 816)
3. `Advanced Options → Fill with: Black`
4. `File → Export As` → `data/annotation/manual_masks/grape001T(5)_grape.png`

### Bước B — Mở ảnh gốc bên cạnh để tham chiếu

1. `File → Open` → `data/raw/ECUSTFD/JPEGImages/grape001T(5).JPG`
2. Kéo cửa sổ ảnh gốc sang bên phải, cửa sổ mask sang bên trái.

### Bước C — Vẽ vùng nho trong mask

**Phương pháp đã chọn**: không vẽ tay tự do (gặp 2 vấn đề:
- dễ tô tràn ra ngoài bbox VOC,
- nếu vẽ trên ảnh màu rồi export PNG, `pixel > 127` threshold
  sẽ gộp cả nền sáng làm foreground).

Thay vào đó, **dùng Otsu tự động trong bbox** (xem mục 4) — chỉ cần
PNG là canvas đen, script sẽ tính mask từ ảnh gốc trong bbox.

Nếu vẫn muốn vẽ tay để kiểm tra hoặc override ảnh không có bbox VOC:

1. Cửa sổ mask active, chọn `Fuzzy Select (U)`, **Threshold** = 50
2. Click vùng nho trên ảnh gốc → đường đứt bao quanh
3. Giữ `Shift` + click bổ sung đến khi đủ
4. `Edit → Fill with White` trong mask
5. Nếu sai: `Ctrl+Z` rồi làm lại
6. `File → Overwrite grape001T(5)_grape.png`

**Lưu ý kỹ thuật**: nếu export PNG từ GIMP thấy file gần đen → OK, đó
là mask tốt. Mở rộng threshold ảnh sẽ thấy vùng trắng nằm trong bbox.

---

## 4. Phương pháp xử lý tự động (script đã chạy)

Thay vì dùng Otsu trên toàn ảnh (tách sai vì nền tối trong bbox
giống màu nho), dùng **Otsu trong bbox + đảo ngược** (dark = grape).

Pipeline (`TMP/masks_patched/extract_grape_v6.py` hoặc one-liner):

```python
import cv2, numpy as np
from PIL import Image

jpg = r"E:\AI_Research\dlt8\data\raw\ECUSTFD\JPEGImages\grape001T(5).JPG"
out = r"E:\AI_Research\dlt8\data\annotation\manual_masks\grape001T(5)_grape.png"

x1, y1, x2, y2 = 262, 4, 686, 446   # bbox VOC của grape001T(5)
img = cv2.imread(jpg)
crop = img[y1:y2+1, x1:x2+1]

gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, bw = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# INVERT: Otsu trong bbox này tách phần sáng = nền, phần tối = nho
grape_mask = 255 - bw

# Lấp lỗ giữa các quả nho + giữ thành phần lớn nhất
grape_mask = cv2.morphologyEx(grape_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
grape_mask = cv2.morphologyEx(grape_mask, cv2.MORPH_CLOSE, np.ones((25,25), np.uint8))

num, labels, stats, _ = cv2.connectedComponentsWithStats(grape_mask)
largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
clean = (labels == largest).astype('uint8') * 255

full = np.zeros((612, 816), dtype='uint8')
full[y1:y2+1, x1:x2+1] = clean
Image.fromarray(full).save(out)
```

Kết quả: `fg% = 23.8%`, `area = 118862 px` — gần với median grape top-view
(`~11.7%`) và được xác nhận qua overlay `_v6.png`.

**Vì sao đảo ngược**: trong crop bbox của ảnh này, mean brightness = 86.7
(tối). Otsu mặc định lấy phần sáng = nền (đúng), đảo ngược → lấy phần
tối = nho.

---

## 5. Patch `.npy` in-place

Sau khi PNG xong, chạy:

```bash
python src/data_prep_SAM1/apply_masks/apply_mask_overrides.py --only "grape001T(5)"
```

Script sẽ:
1. Đọc `masks/grape001T(5).npy`
2. Tìm object có `class=grape`, `target_index=0`
3. Thay `mask` bằng PNG (resize về `(H,W)` với `INTER_NEAREST`)
4. Tính lại `area = mask.sum()`
5. Giữ `bbox_voc` cũ (vì PNG đã vẽ trong bbox đó)
6. Ghi lại `.npy` cùng tên + SHA-256 hash trước/sau

Output mẫu:

```
[OK ] grape001T(5)  class=grape  obj_idx=0  area: 9284 -> 118862
     bbox_voc: [262, 4, 686, 446] -> [262, 4, 686, 446]
     src: raw_gt_box_prompt -> manual_mask_override
     png=data/annotation/manual_masks/grape001T(5)_grape.png
     // SAM bbox quá rộng, mask bám background. Thay mask thủ công.
     // wrote in-place  sha256: 746d64a754c3.. -> 0b0a9b533a56..
```

---

## 6. Verify bằng overlay (khuyến nghị trước khi patch)

Tạo ảnh nho với vùng đỏ chồng lên để check nhanh mask có đúng không:

```python
import numpy as np, cv2
from PIL import Image

d = np.load("data/processed/sam_masks_full/masks/grape001T(5).npy",
            allow_pickle=True).item()
m = d["objects"][0]["mask"]
img = cv2.imread("data/raw/ECUSTFD/JPEGImages/grape001T(5).JPG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ov = img.copy()
ov[m] = [255, 0, 0]
blend = cv2.addWeighted(img, 0.5, ov, 0.5, 0)
Image.fromarray(blend).save("TMP/masks_patched/overlay_grape001T(5)_v6.png")
```

Vùng đỏ phải bao đúng chùm nho, không tràn ra ngoài bbox VOC.

---

## 7. Thêm ảnh override mới (nếu có)

> **Lưu ý**: hiện tại dataset chỉ có 1 ảnh lỗi là `grape001T(5)` đã xử lý.
> Mục này chỉ để khi SCAN tiếp tục phát hiện ảnh mới có mIoU < 0.5
> (kết quả ghi ở `data/processed/sam_masks_full/_poor_miou_report.csv`).

Trong `apply_mask_overrides.py`, thêm 1 entry vào dict `OVERRIDES`:

```python
"<stem>": {
    "class": "grape",                       # object class bị sai mask
    "target_index": 0,                      # nếu có nhiều object cùng class
    "mask_png": "data/annotation/manual_masks/<stem>_grape.png",
    "new_bbox_xyxy": None,                  # hoặc tuple 4 số VOC nếu đo lại
    "note": "lý do override ngắn gọn",
},
```

Sau đó:

```bash
# 1. Dry-run trước, đọc log xem có lỗi không
python src/data_prep_SAM1/apply_masks/apply_mask_overrides.py --dry-run --only "<stem>"

# 2. Ghi vào TMP để an toàn
python src/data_prep_SAM1/apply_masks/apply_mask_overrides.py --to-tmp --only "<stem>"

# 3. Khi OK mới patch in-place
python src/data_prep_SAM1/apply_masks/apply_mask_overrides.py --only "<stem>"

# 4. Regenerate viz + CSV
python src/data_prep_SAM1/sam_masks_full/visualize_results.py --clean
```

---

## 8. Checklist trước khi commit

- [ ] PNG `data/annotation/manual_masks/<stem>_<class>.png` tồn tại, fg% hợp lý
- [ ] Overlay (`TMP/masks_patched/overlay_*.png`) cho thấy mask đúng
- [ ] Entry trong `OVERRIDES` có `note` giải thích lý do
- [ ] `apply_mask_overrides.py --only "<stem>"` chạy OK (không skip)
- [ ] SHA-256 trước/sau khác nhau (chứng tỏ file thực sự đã ghi)
- [ ] `visualize_results.py --clean` chạy xong, không lỗi
- [ ] CSV `data/processed/sam_masks_full/*.csv` cập nhật `source=manual_mask_override`