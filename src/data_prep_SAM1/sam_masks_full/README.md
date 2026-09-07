# SAM1 box-prompt masks — research log

Quá trình chạy **SAM1 (vit_b) với box-prompt** toàn bộ ECUSTFD (2978 ảnh),
rồi visualize kết quả ra grid per-class. Code tách riêng với phần bbox để
không phụ thuộc lẫn nhau.

## Cấu trúc thư mục

```
src/data_prep_SAM1/sam_masks_full/
    README.md                       ← file này
    segment_sam1_box.py             ← chạy SAM1 box-prompt toàn dataset
    visualize_results.py            ← render grid kết quả + highlight mIoU thấp
```

## Cấu trúc output

```
data/processed/sam_masks_full/
    masks/<stem>.npy                ← 1 file / ảnh, dict pickle lưu masks + bbox
    images/<stem>.jpg               ← ảnh vẽ GT bbox + mask overlay
    grids/<class>_pNN_of_NN.png     ← grid theo class
    _index.csv                      ← tile-global index (6062 object tiles)
    _per_image_stats.csv            ← mIoU / area / leak / fill_ratio per ảnh
    _poor_miou_report.csv           ← ảnh có mean mIoU < threshold
    _segmentation_report.md         ← tổng kết
    run.log                         ← log quá trình
    README.txt                      ← auto-gen
```

## Format file `masks/<stem>.npy`

Mỗi file là một dict pickle-compatible:

```python
{
    "image_shape": [H, W, 3],
    "objects": [
        {"class": "tomato", "bbox_voc": [x1,y1,x2,y2],
         "mask": (H,W) bool, "score": float, "area": int},
        {"class": "coin",   "bbox_voc": [x1,y1,x2,y2],
         "mask": (H,W) bool, "score": float, "area": int},
        ...
    ],
    "source": "patched_gt_box_prompt" | "raw_gt_box_prompt",
}
```

- `source = "patched_gt_box_prompt"` cho 9 ảnh có XML trong `Annotations_patched/`,
  ngược lại `"raw_gt_box_prompt"` cho 2969 ảnh còn lại.
- Tất cả object (food + coin) của ảnh được segment và lưu trong cùng file.

## Bước 1 — `segment_sam1_box.py`

Mục đích: chạy SAM1 (vit_b) box-prompt cho toàn bộ 2978 ảnh ECUSTFD. Mỗi
object (food + coin) trong ảnh được predict với bbox từ VOC XML (ưu tiên
`Annotations_patched/` nếu có).

```bash
python src/data_prep_SAM1/sam_masks_full/segment_sam1_box.py --clean
```

Tham số:

- `--clean` xoá toàn bộ `sam_masks_full/` (trừ `run.log`, `README.md`) rồi
  build lại — đảm bảo không có mask cached từ XML cũ.
- `--workers N` (default `min(8, cpu_count)`) — chia dataset cho N worker,
  mỗi worker load SAM1 + xử lý 1 subset stem.

**Cache safety (theo yêu cầu):**

- Script đọc lại VOC XML cho từng stem; ưu tiên XML trong
  `data/processed/bbox_full/Annotations_patched/` (chỉ 9 file).
- Nếu `Annotations_patched/<stem>.xml` tồn tại → dùng bbox từ XML patched,
  `source = "patched_gt_box_prompt"`.
- Ngược lại → dùng bbox từ `data/raw/ECUSTFD/Annotations/<stem>.xml`,
  `source = "raw_gt_box_prompt"`.
- In summary ở cuối: `patched_stems: 9, raw_stems: 2969, total: 2978`.

**Quy ước tọa độ:** VOC XML là 1-indexed. Khi predict, trừ 1 cho mỗi cạnh
để ra 0-indexed xyxy theo convention của `SamPredictor.predict(box=...)`.
Đã verify convention từ `TMP/verify_9patched.py` (line 109).

## Bước 2 — `visualize_results.py`

Mục đích: render kết quả segment ra grid per-class, tính mIoU mask-vs-bbox
cho từng ảnh, đánh dấu ảnh kém chất lượng.

```bash
python src/data_prep_SAM1/sam_masks_full/visualize_results.py --clean
```

Tham số chính:

- `--clean` xoá `images/`, `grids/`, các `_*.csv` cũ (giữ `masks/` + log).
- `--per_page 80`, `--cols 8`, `--tile 240`.
- `--miou_threshold 0.5` — ảnh có `mean_mIoU < threshold` sẽ bị highlight.
- `--mask_alpha 0.5` — độ trong suốt của mask overlay.
- `--render_workers N` — song song hoá bước vẽ ảnh.

**Output per image (`images/<stem>.jpg`):**

Ảnh gốc | Ảnh vẽ GT bbox theo class color | Ảnh vẽ GT bbox + mask overlay
(3 panel ngang, có caption tên stem + mean mIoU).

**Output per class (`grids/<class>_pNN_of_NN.png`):**

Mỗi object 1 tile 240×240 crop quanh bbox, có mask overlay. Ảnh có
`mean_mIoU < threshold` → viền đỏ 4px quanh tile + text đỏ `mIoU=0.42`
trên tile (rất dễ thấy trong grid).

**Báo cáo:**

- `_index.csv` — 6062 object tiles với class, stem, bbox, page.
- `_per_image_stats.csv` — per-image: stem, source, num_objects,
  mean_miou, mask_area_total, leak_score, fill_ratio.
- `_poor_miou_report.csv` — chỉ chứa ảnh `mean_miou < threshold`,
  sort tăng dần theo `mean_miou`.
- `_segmentation_report.md` — tổng kết đẹp.

Cách dùng:

- Mở các grid `<class>_pNN_of_NN.png`.
- Tìm viền đỏ → xem ảnh trong `_poor_miou_report.csv`.
- Cung cấp danh sách stem cho mình để viết tiếp phần patch mask (round 2).

## Quá trình thực hiện

### Bước 1 — segment 2978 ảnh

Chạy single-process trên GPU (RTX 4050 / RTX 5090), SAM1 vit_b:

```bash
python src/data_prep_SAM1/sam_masks_full/segment_sam1_box.py --clean --workers 1
```

- **2978/2978** masks được tạo trong `masks/<stem>.npy`
- **37 ảnh** dùng `source = patched_gt_box_prompt` (đọc từ
  `Annotations_patched/`: 14 ảnh Bước 2/3 + 17 doughnut + 3 mango + 2 orange +
  1 mix), **2941 ảnh** còn lại dùng `source = raw_gt_box_prompt` (VOC XML gốc).
- Tổng object: 6062 (gồm cả coin) — mỗi ảnh có 2 objects (food + coin).
- Thời gian: ~19 phút (≈ 2.6 img/s), rate ổn định trên RTX 4050.
- Mỗi `.npy` chứa dict pickle-compatible với key:
  - `image_shape: [H, W, 3]`
  - `objects: List[{class, bbox_voc, mask, score, area}]` — food + coin
  - `source: "patched_gt_box_prompt" | "raw_gt_box_prompt"`

### Bước 2 — visualize & thống kê mIoU

```bash
python src/data_prep_SAM1/sam_masks_full/visualize_results.py --clean
```

- **2978/2978** 3-panel JPG (`images/<stem>.jpg`).
- **87 grid PNG** (per-class, có highlight đỏ cho ảnh mIoU < 0.5).
- **6062 tile** trong `_index.csv` (đầy đủ food + coin).
- Thời gian: ~95s tổng (render + grid build).

### Kết quả thực tế (mIoU threshold = 0.5)

- **Mean mIoU toàn dataset: 0.743** (trung bình của per-image mean mIoU).
- Phân bố:

  | mIoU range | Số ảnh |
  |------------|--------|
  | [0.00, 0.20) | 0 |
  | [0.20, 0.40) | 1 |
  | [0.40, 0.60) | 63 |
  | [0.60, 0.80) | 2734 |
  | [0.80, 1.01) | 180 |

- **97.9% ảnh** đạt `mean_mIoU ≥ 0.6`.
- **1/2978 ảnh** mIoU < 0.5 — xem `_poor_miou_report.csv`:

  | stem | source | mean mIoU | min mIoU | n |
  |---|---|---|---|---|
  | `grape001T(5)` | raw_gt_box_prompt | 0.377 | 0.043 | 2 |

  Ảnh có `min_mIoU = 0.043` ở object food (grape bị bbox lệch sang nền —
  mask bám vào background) → mask bị sai. Coin vẫn segment ổn
  (`max_mIoU ≈ 0.71`). Có thể cần round 2 — patch bbox grape bằng tay hoặc
  dùng SAM1 với point prompt.

### So với lần chạy trước (chỉ 9 ảnh patched)

| Chỉ số | Lần trước | Lần này |
|---|---|---|
| Patched stems | 9 | **37** |
| Raw stems | 2969 | 2941 |
| Poor (mIoU < 0.5) | **3** | **1** |
| Mean mIoU | 0.743 | 0.743 |

Hai ảnh `tomato004T(25)`, `tomato001S(4)` trước đó có `min_mIoU < 0.1` ở
object food do dùng bbox VOC sai (Bước 4 tiếp) — lần này dùng bbox đã sửa
trong `Annotations_patched/`, kết quả segment đạt `mean_mIoU ≥ 0.5`.

## Bước 3 — patch mask cho ảnh kém còn lại (round 2)

Tạo bản override kiểu `apply_bbox_overrides.py`:

1. Bạn mở 3 ảnh `images/<stem>.jpg` (panel 3 = bbox + mask overlay).
2. Xác nhận bbox GT có cần dịch chuyển không, hoặc cho mình stem nào
   cần predict lại mask bằng tay (dùng SAM1 với point prompt, hoặc
   tạo mask binary thủ công).
3. Mình sẽ viết `apply_mask_overrides.py` tương tự pattern của
   `visual_bbox/apply_bbox_overrides.py`, đọc danh sách override
   từ `data/processed/sam_masks_full/mask_overrides.json` rồi
   sinh mask mới cho các object trong `masks/<stem>.npy`. Visualization
   sẽ tự pick up mask mới khi chạy lại.

## Tổng kết

- Ảnh đã segment bằng SAM1 box-prompt: **2978/2978** (100%)
- Ảnh đạt `mean_mIoU ≥ 0.5`: **2977/2978** (99.97%)
- Ảnh `mean_mIoU < 0.5`: **1/2978** — `grape001T(5)` (đã liệt kê trong
  `_poor_miou_report.csv`)
- Object segment: **6062/6062** (food + coin, không miss object nào)
- Code giữ lại: chỉ `src/data_prep_SAM1/sam_masks_full/`
- Dữ liệu phụ: `data/processed/sam_masks_full/` (masks + viz + reports)
- Cache safety: `--clean` luôn wipe `masks/` + `images/` + `grids/` +
  CSVs → luôn đọc lại XML từ đầu, không có mask cached từ XML cũ.
- So với lần chạy trước (9 patched): giảm từ 3 ảnh kém xuống **1 ảnh**, nhờ
  bbox của 28 ảnh Bước 4 (17 doughnut + 3 mango + 2 orange + 1 mix) đã được
  sửa trong `Annotations_patched/`.
