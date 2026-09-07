# ECUSTFD bbox — research log

Quá trình chuẩn hóa ground-truth bounding box của ECUSTFD trước khi đưa vào
pipeline nghiên cứu chính. Repo này chỉ giữ code visualize bbox và tiện ích
sửa annotation lỗi; các bước chuẩn bị dữ liệu khác đã được dọn đi.

## Bước 1 — `vs_bbox`: visualize toàn bộ VOC ground-truth

Mục đích: render đủ `2978/2978` ảnh ECUSTFD kèm tất cả bounding box (gồm cả
`coin`), tạo grid theo class để kiểm tra nhãn bằng mắt.

Chạy:

```bash
python src/data_prep_SAM1/visual_bbox/visualize_bbox_full.py --clean
```

Tham số chính:

- `--per_page 80`, `--cols 8`, `--tile 240`, `--workers = min(8, cpu_count)`
- Đọc `data/raw/ECUSTFD/Annotations/*.xml`, ảnh nguồn `JPEGImages/*.JPG`
- Mỗi XML đều được parse; XML lỗi/ảnh thiếu được ghi vào
  `_invalid_xml.txt` / `_missing_images.txt`
- Xuất:
  - `data/processed/bbox_full/images/<stem>.jpg` — 2978 ảnh đã vẽ bbox
  - `data/processed/bbox_full/grids/<class>_pNN_of_NN.png` — grid theo class
  - `_image_index.csv`, `_index.csv`, `_per_class_counts.csv`, `README.txt`

Kết quả kiểm tra nhanh:

- Tổng XML: 2978, unique stem: 2978, không có stem trùng trong tập ECUSTFD
- Ảnh render thành công: 2978/2978, **0** missing/render_failed
- Tổng tile trong `_index.csv`: 6062 (gồm food + coin), phủ đủ 2978 stem

## Bước 2 — Sửa các bbox sai bằng mắt

Khi rà grid theo class, phát hiện **14 ảnh** có bounding box VOC rõ ràng lệch
khỏi vật thể. Danh sách và tọa độ đo lại (sai số ±2 px, tham chiếu trên ảnh
có vạch 50 px).

Chú thích về nhiều lần đo:
- Bước 1: lần đầu mình cố tình sửa cả food lẫn coin → nhận ra sai vì coin VOC
  đã đúng cho cả 14 ảnh.
- Bước 2 (lần trước): chỉ sửa food (tomato / qiwi) bbox, coin giữ nguyên VOC,
  được 9 ảnh (5 tomato + 4 qiwi). Lưu ý: 4 ảnh ``qiwi006S(7),(8) / qiwi007S(1) /
  qiwi007T(3)`` được đánh nhãn ``kiwi`` trong VOC gốc; đã sửa sang ``qiwi``
  để khớp ECUSTFD.
- Bước 3 (lần này): mở rộng thêm 5 ảnh mới — 1 pear + 2 plum + 2 tomato — phát
  hiện trong quá trình rà thêm các class khác.
- Bước 4 (lần này): phát hiện 17 ảnh doughnut có bbox coin sai (VOC copy bbox
  doughnut sang coin, nghĩa là coin bbox = doughnut bbox). Trong đó 16 file có
  bbox coin "to" dù vẫn = doughnut, và `doughnut006S(15)` coin bị đánh bbox
  cực nhỏ (2×3 = 6 px² — gần như chỉ là 1 điểm). Đã đo lại bằng mắt
  `(132,317)→(202,384)` (70×67 = 4690 px²) — sửa coin, giữ nguyên food.

Bbox mới ở bảng dưới là bản đã chốt sau Bước 3:

| Stem | Object | XML cũ | Bbox mới | Ghi chú |
|---|---|---|---|---|
| `tomato003T(19)` | tomato | (145,135)→(210,196) | (431,232)→(653,450) | plate region |
| `tomato003T(27)` | tomato | (125,180)→(192,238) | (419,215)→(630,428) | plate region |
| `tomato004S(18)` | tomato | (114,232)→(186,292) | (461,144)→(654,283) | plate region |
| `tomato004S(26)` | tomato | (118,231)→(186,291) | (460,198)→(656,337) | plate region |
| `tomato004T(4)`  | tomato | (71,239)→(141,309) | (420,197)→(650,425) | plate region |
| `tomato001S(4)`  | tomato | (119,373)→(178,434) | (403,254)→(602,404) | plate region (Bước 3) |
| `tomato004T(25)` | tomato | (121,264)→(192,328) | (389,212)→(603,420) | plate region (Bước 3) |
| `qiwi006S(7)` | qiwi | (440,360)→(515,425) | (179,264)→(378,413) | hand-side (round 2) |
| `qiwi006S(8)` | qiwi | (490,170)→(580,275) | (126,240)→(336,408) | hand-side (round 2) |
| `qiwi007S(1)` | qiwi | (370,375)→(450,475) | (328,257)→(546,423) | hand-side (round 2) |
| `qiwi007T(3)` | qiwi | (155,395)→(240,475) | (337,263)→(497,462) | hand-side (round 2) |
| `pear003T(7)`  | pear  | (159,333)→(231,398) | (417,232)→(615,425) | whole fruit (Bước 3) |
| `plum002T(16)` | plum  | (129,169)→(193,237) | (485,266)→(654,442) | whole fruit (Bước 3) |
| `plum003S(11)` | plum  | (89,156)→(170,231) | (438,250)→(600,404) | whole fruit (Bước 3) |

Về coin bbox: VOC coin đã đúng cho cả 14 ảnh. `apply_bbox_overrides.py`
in warning `[warn] <stem>: <class> override=(...) VOC=(...)` mỗi khi một
override khác với VOC; sau khi bỏ override coin (giữ nguyên VOC) thì không
còn warning coin nào xuất hiện nữa, chỉ còn warning food (tomato / qiwi /
pear / plum) đúng như dự đoán.

**Tỉ lệ: 14/14 ảnh đã tự sửa (100%).**

### Bước 4 — sửa coin bbox sai trong nhóm doughnut

Khi rà 2978 ảnh ECUSTFD, phát hiện **17 ảnh doughnut có bbox coin sai**:
<bbox của coin trùng hoàn toàn với bbox của doughnut> (cùng `(xmin,ymin)→(xmax,ymax)`).
Đây không phải lỗi pipeline mà là lỗi annotation gốc của ECUSTFD VOC — người
annotate đã copy/paste bbox doughnut sang coin.

Diện tích coin bbox VOC (px²) của 17 file:

| Stem | Coin VOC (xmin,ymin)→(xmax,ymax) | Diện tích coin |
|---|---|---|
| `doughnut004S(15)` | (417,317)→(678,403) | 261×86 = 22446 |
| `doughnut006T(7)` | (393,180)→(666,479) | 273×299 = 81627 |
| `doughnut006T(18)` | (138,144)→(206,214) | 68×70 = 4760 |
| `doughnut007S(14)` | (435,267)→(728,375) | 293×108 = 31644 |
| `doughnut007S(3)` | (447,278)→(692,373) | 245×95 = 23275 |
| `doughnut008S(3)` | (424,265)→(674,360) | 250×95 = 23750 |
| `doughnut008S(4)` | (432,232)→(718,340) | 286×108 = 30888 |
| `doughnut008T(12)` | (409,193)→(667,414) | 258×221 = 57018 |
| `doughnut008T(13)` | (387,178)→(644,450) | 257×272 = 69904 |
| `doughnut008T(8)` | (442,186)→(683,455) | 241×269 = 64829 |
| `doughnut009S(1)` | (453,244)→(719,339) | 266×95 = 25270 |
| `doughnut009S(2)` | (424,221)→(688,306) | 264×85 = 22440 |
| `doughnut009S(9)` | (467,242)→(747,341) | 280×99 = 27720 |
| `doughnut009T(10)` | (422,158)→(684,419) | 262×261 = 68382 |
| `doughnut009T(11)` | (445,254)→(696,505) | 251×251 = 63001 |
| `doughnut009T(6)` | (420,165)→(688,433) | 268×268 = 71824 |
| **`doughnut006S(15)`** | (128,316)→(130,319) | 2×3 = **6** (cực nhỏ, gần như 1 điểm) |

Trong 17 file này, 16 file có bbox coin "to" (thường 22k–82k px²) — coin
chỉ sai vì copy từ doughnut, không phải đường kính thật. **`doughnut006S(15)`
là trường hợp đặc biệt**: coin bbox VOC chỉ 2×3 = 6 px² (gần như chỉ là 1
điểm, sai nghiêm trọng nhất trong 17 file), do annotator đã vẽ bbox cực nhỏ
khác với 16 file còn lại.

Đo lại bằng mắt với cả 17 ảnh (sai số ±2 px), bbox coin thật như sau. Bbox
coin đúng có kích thước tương đương coin bình thường (~70 px đường kính trên
ảnh 816×612):

| Stem | Class | VOC cũ (coin) | Bbox mới (coin) | Ghi chú |
|---|---|---|---|---|
| `doughnut004S(15)` | coin | (417,317)→(678,403) | (97,354)→(167,420) | riêng (Bước 4) |
| `doughnut006T(7)` | coin | (393,180)→(666,479) | (126,54)→(194,121) | riêng (Bước 4) |
| `doughnut006T(18)` | coin | (138,144)→(206,214) | (127,116)→(192,180) | riêng (Bước 4) |
| `doughnut007S(14)` | coin | (435,267)→(728,375) | (46,316)→(121,388) | riêng (Bước 4) |
| `doughnut007S(3)` | coin | (447,278)→(692,373) | (57,324)→(126,390) | riêng (Bước 4) |
| `doughnut008S(3)` | coin | (424,265)→(674,360) | (70,314)→(140,387) | riêng (Bước 4) |
| `doughnut008S(4)` | coin | (432,232)→(718,340) | (72,282)→(140,354) | riêng (Bước 4) |
| `doughnut008T(12)` | coin | (409,193)→(667,414) | (152,138)→(213,202) | riêng (Bước 4) |
| `doughnut008T(13)` | coin | (387,178)→(644,450) | (127,108)→(195,178) | riêng (Bước 4) |
| `doughnut008T(8)` | coin | (442,186)→(683,455) | (141,128)→(210,195) | riêng (Bước 4) |
| `doughnut009S(1)` | coin | (453,244)→(719,339) | (27,292)→(103,364) | riêng (Bước 4) |
| `doughnut009S(2)` | coin | (424,221)→(688,306) | (61,257)→(137,332) | riêng (Bước 4) |
| `doughnut009S(9)` | coin | (467,242)→(747,341) | (25,283)→(100,357) | riêng (Bước 4) |
| `doughnut009T(10)` | coin | (422,158)→(684,419) | (98,190)→(167,254) | riêng (Bước 4) |
| `doughnut009T(11)` | coin | (445,254)→(696,505) | (151,220)→(218,283) | riêng (Bước 4) |
| `doughnut009T(6)` | coin | (420,165)→(688,433) | (140,171)→(209,238) | riêng (Bước 4) |
| `doughnut006S(15)` | coin | (128,316)→(130,319) | (132,317)→(202,384) | riêng (Bước 4) |

Bbox doughnut của 17 ảnh được **giữ nguyên VOC** (đã đúng). Cập nhật dict
`OVERRIDES` trong `apply_bbox_overrides.py` với trường `coin_xyxy` optional,
phục vụ cho override chỉ coin — khi muốn sửa thêm ảnh nào, chỉ cần thêm
entry mới với bbox coin đo được.

**Tỉ lệ Bước 4: 17/17 ảnh đã sửa (100%).**

### Bước 4 tiếp — sửa coin bbox sai trong nhóm mango

Khi rà tiếp các class khác, phát hiện **3 ảnh mango** có cùng lỗi: bbox coin
copy y hệt từ bbox mango trong VOC:

| Stem | Mango VOC | Coin VOC (xmin,ymin)→(xmax,ymax) | Diện tích coin |
|---|---|---|---|
| `mango008T(4)` | (441,270)→(589,502) | (441,270)→(589,502) | 148×232 = 34336 |
| `mango004S(7)` | (388,353)→(543,436) | (388,353)→(543,436) | 155×83 = 12865 |
| `mango004T(7)` | (481,228)→(623,438) | (481,228)→(623,438) | 142×210 = 29820 |

Đo lại bằng mắt với 3 ảnh (sai số ±2 px), bbox coin thật như sau. Bbox mango
của 3 ảnh này được **giữ nguyên VOC** (đã đúng):

| Stem | Class | VOC cũ (coin) | Bbox mới (coin) | Ghi chú |
|---|---|---|---|---|
| `mango008T(4)` | coin | (441,270)→(589,502) | (75,295)→(144,364) | riêng (Bước 4 tiếp) |
| `mango004S(7)` | coin | (388,353)→(543,436) | (90,396)→(150,463) | riêng (Bước 4 tiếp) |
| `mango004T(7)` | coin | (481,228)→(623,438) | (101,237)→(180,311) | riêng (Bước 4 tiếp) |

**Tỉ lệ Bước 4 tiếp: 3/3 ảnh đã sửa (100%).**

### Bước 4 tiếp — sửa coin bbox sai trong nhóm orange

Tiếp tục rà các class khác, phát hiện **2 ảnh orange** có cùng lỗi bbox coin
copy y hệt từ bbox orange VOC. Lưu ý 2 ảnh orange có size **504×378**
(khác 816×612 của doughnut/mango):

| Stem | Orange VOC | Coin VOC (xmin,ymin)→(xmax,ymax) | Diện tích coin |
|---|---|---|---|
| `orange012T(7)` | (244,134)→(343,246) | (244,134)→(343,246) | 99×112 = 11088 |
| `orange015S(6)` | (43,120)→(207,265) | (43,120)→(207,265) | 164×145 = 23780 |

Đo lại bằng mắt với 2 ảnh (sai số ±2 px), bbox coin thật như sau. Bbox orange
của 2 ảnh được **giữ nguyên VOC** (đã đúng):

| Stem | Class | VOC cũ (coin) | Bbox mới (coin) | Ghi chú |
|---|---|---|---|---|
| `orange012T(7)` | coin | (244,134)→(343,246) | (85,52)→(118,84) | riêng (Bước 4 tiếp) |
| `orange015S(6)` | coin | (43,120)→(207,265) | (414,225)→(479,288) | riêng (Bước 4 tiếp) |

**Tỉ lệ Bước 4 tiếp: 2/2 ảnh đã sửa (100%).**

### Bước 4 tiếp — sửa nhầm nhãn trong mix012T(4)

Khác với 22 ảnh trước (lỗi bbox coin copy từ food), ảnh `mix012T(4)` thuộc
nhóm mix có **nhiều loại quả** trong cùng ảnh, và **1 trong 2 nhãn food bị
ghi nhầm** chứ không phải sai bbox. VOC gốc có 2 food obj:

| Obj | Class (VOC) | Bbox | Ghi chú |
|---|---|---|---|
| 1 | `mango` | (329,76)→(519,273) | **nhầm**, thực tế là `orange` |
| 2 | `lemon` | (340,377)→(486,511) | đúng |
| 3 | `coin` | (103,174)→(155,227) | đúng (52×53 = 2756 px²) |

Sửa: đổi nhãn `mango` → `orange` cho obj 1, giữ nguyên `lemon` (obj 2) và
`coin` (obj 3). Bbox không đổi. Thêm field `class_changes` vào entry
`OVERRIDES` của stem này:

```python
"mix012T(4)": {"new_class": None, "xyxy": None, "class_changes": {"mango": "orange"}},
```

**Tỉ lệ Bước 4 tiếp: 1/1 ảnh đã sửa (100%).**

Cách tái hiện (2 bước):

```bash
# Bước 1: sinh XML patched cho 37 ảnh vào Annotations_patched/ (mặc định)
python src/data_prep_SAM1/visual_bbox/apply_bbox_overrides.py

# Bước 2: visualize lại toàn bộ 2978 ảnh từ VOC (ưu tiên dùng XML patched
# trong Annotations_patched/ nếu có). --clean xóa images/, grids/, *.csv
# trước khi render; giữ nguyên Annotations_patched/, run.log.
python src/data_prep_SAM1/visual_bbox/visualize_bbox_full.py --clean
```

`apply_bbox_overrides.py` mặc định ghi vào `Annotations_patched/` (để
`visualize_bbox_full.py` tự động dùng XML đã sửa). Dùng `--to-tmp` nếu
chỉ muốn ghi vào `/TMP/Annotations_patched_regen/` để verify trước
(không ảnh hưởng visualize).

Script visualize này:

1. Đọc `data/processed/bbox_full/bbox_overrides.json` (danh sách bbox mới).
2. Sinh VOC XML đã sửa vào `data/processed/bbox_full/Annotations_patched/`,
   đánh dấu `<folder>VOC2007_patched</folder>` và thêm `<path>...</path>` cho
   từng object để truy vết nguồn.
3. Gọi lại `visualize_bbox_full.py --clean`, truyền env
   `BBOX_FULL_OVERRIDES_DIR`/`BBOX_FULL_OVERRIDES_JSON` để visualizer ưu
   tiên dùng XML patched cho các stem trong override và giữ nguyên XML gốc
   cho các stem còn lại.

Sau khi render, 14 ảnh (Bước 2+3) đã được kiểm tra lại bằng mắt và **toàn bộ
bbox đều khít vật thể** (cà chua, qiwi, lê, mận, đồng xu đúng vị trí). Riêng
Bước 4, 17 ảnh doughnut + 3 ảnh mango + 2 ảnh orange + 1 ảnh mix (đổi
nhãn) đã được sinh patched XML mới với bbox đo lại bằng mắt (ghi vào
`/TMP/Annotations_patched_regen/`, chưa ghi đè `Annotations_patched/` để
chờ verify thêm).

## Tổng kết

- Ảnh đã visualize: **2978/2978** (100%)
- Ảnh có bbox sai và đã sửa: **14/14** (Bước 2+3, 100%) — 5 tomato + 4 qiwi
  + 2 tomato + 1 pear + 2 plum; **17/17** doughnut + **3/3** mango + **2/2**
  orange + **1/1** mix đổi nhãn (Bước 4, 100% cho mỗi nhóm)
- Code giữ lại: chỉ `src/data_prep_SAM1/visual_bbox/` (visualize +
  apply_bbox_overrides)
- Dữ liệu phụ: `data/processed/bbox_full/` (output visualize),
  `data/processed/bbox_full/Annotations_patched/` (XML đã sửa, 14 file),
  `TMP/Annotations_patched_regen/` (regen đầy đủ 37 file bao gồm 17
  doughnut + 3 mango + 2 orange + 1 mix Bước 4),
  `data/processed/bbox_full/bbox_overrides.json` (override)