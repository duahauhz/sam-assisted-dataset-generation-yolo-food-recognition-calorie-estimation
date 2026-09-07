# ECUSTFD-Seg: an instance-segmentation layer for ECUSTFD

This release provides **derived annotation artifacts only** for the ECUSTFD
food-calorie benchmark [1]: a 20-class YOLO-segmentation label set produced by
box-prompting a frozen Segment Anything Model (SAM ViT-B), the SAM binary
masks with full provenance, 37 quality-audited VOC XML corrections, audit
tables, and the evaluation splits used in the accompanying paper.

**It contains no ECUSTFD images.** The images are owned by the original
authors and must be obtained from the official repository
<https://github.com/Liang-yc/ECUSTFD-resized->. If you use this layer in
research, you must also cite the original ECUSTFD paper [1] (see
`CITATION.cff`).

## Contents

```
ecustfd-seg-release/
├── yolo_ecustfd_seg/
│   ├── ecustfd-seg.yaml        # 20 classes (0–19, alphabetical; coin = 4)
│   ├── labels/
│   │   ├── train/  (1,245)     # official trainval images
│   │   └── val/    (1,733)     # official test images (see note below)
│   └── sam_masks/
│       └── masks/              # 2,978 .npy files, one per image
├── patched_xml/                # 37 corrected VOC XMLs (drop-in replacements)
├── audit/
│   ├── _per_image_stats.csv   # per-image proxy-IoU, provenance, class list
│   ├── _per_class_counts.csv
│   ├── _index.csv              # per-instance tile index with VOC bbox + IoU
│   └── _segmentation_report.md
└── splits/                     # ImageSets-style file lists
    ├── trainval.txt (1,245)  train.txt (622)  val.txt (623)
    ├── test.txt (1,733)
    ├── test_tune.txt (805)    # tuning half of test (threshold sweep only)
    └── test_final.txt (928)   # reporting half of test (all paper numbers)
```

To train with Ultralytics, place the original images into
`yolo_ecustfd_seg/images/{train,val}/` (create the folders) with the file
names matching the labels — see the instructions in
`yolo_ecustfd_seg/ecustfd-seg.yaml`.

## Getting the images (one-time setup)

The original photographs are distributed by the ECUSTFD authors via their
repository:

```bash
git clone https://github.com/Liang-yc/ECUSTFD-resized-.git
# all 2,978 .JPG files are in ECUSTFD-resized-/JPEGImages/ (flat folder)
```

Then split them into the two training folders, following the file lists in
`splits/` (each line is one image file name):

```bash
cd yolo_ecustfd_seg
mkdir -p images/train images/val
while read -r f; do cp "/path/to/ECUSTFD-resized-/JPEGImages/$f.JPG" images/train/; done < ../splits/trainval.txt
while read -r f; do cp "/path/to/ECUSTFD-resized-/JPEGImages/$f.JPG" images/val/;   done < ../splits/test.txt
```

(On Windows, the same copy loop runs as-is in Git Bash; in PowerShell,
use `Get-Content ../splits/trainval.txt | ForEach-Object { Copy-Item ... }`.)

After this step `images/train` holds 1,245 files and `images/val` 1,733 —
the same counts as the label folders.

## Image-name mapping

Labels/masks are keyed by the **original ECUSTFD file names**
(e.g. `apple001S(1)`), so they overlay the official `JPEGImages/` directory
directly — copy the folders side by side and the matching is 1:1, no renaming.

Two name-list notes:

- `mix002T(2)` and `mix005S(4)` contain no calibration coin (known issue in
  the original release); they are retained for detector training only and are
  excluded from volume/calorie evaluation.
- The nine files `egg002s(1)`–`egg002s(9)` use a lower-case `s` in the
  original release and do not pair into (top, side) views; they are part of
  the label set but take no part in the pair-based evaluation.

## Split semantics (important)

The `labels/val` directory is the **official ECUSTFD test set**, not a
held-out validation set. In the accompanying paper:

- detector training uses `trainval` (1,245 images);
- detection-confidence operating points are tuned on `test_tune` (805 images)
  and frozen before any reporting;
- every reported metric is computed on `test_final` (928 images, 6,398
  top–side pairs) under a penalized protocol;
- the per-class volume-bias calibration (β) is fitted on a 50/50 item split of
  the training items, disjoint from `test_final`.

`test_tune.txt` and `test_final.txt` are complementary halves of
`test.txt` (union = 1,733, intersection = empty).

## Label format

Standard Ultralytics YOLO-segmentation format: one `.txt` per image, one line
per instance — `class_id x1 y1 x2 y2 … xn yn` with normalized coordinates.
Class ids are alphabetical: `0 apple … 4 coin … 19 tomato`.

## Mask format (`yolo_ecustfd_seg/sam_masks/masks/<stem>.npy`)

Each `.npy` is a Python object (0-d array → `.item()`) dict:

```
{
  "image_shape": [H, W, 3],
  "source": "raw_gt_box_prompt" | "patched_gt_box_prompt" | "manual_mask_override",
  "objects": [
     {"class": str, "bbox_voc": [xmin, ymin, xmax, ymax],
      "mask": H×W bool array, "score": float (SAM IoU token score), "area": int}
  ]
}
```

Provenance counts over the 2,978 files: 2,940 `raw_gt_box_prompt`,
37 `patched_gt_box_prompt`, 1 `manual_mask_override`.

## Checksums

`SHA256SUMS` covers every file in this release (except itself).

## License

The derived labels, masks, patched XMLs, and audit tables in this release are
licensed under **CC BY 4.0** (see `LICENSE`). The underlying ECUSTFD images
are **not** covered by this license and remain subject to the terms of the
original authors; the original release asks that you cite their paper.

## Citation

If you use this annotation layer, please cite both this work and the original
ECUSTFD paper — see `CITATION.cff`.

## References

[1] Y. Liang and J. Li, "Computer vision-based food calorie estimation:
dataset, method, and experiment," arXiv preprint arXiv:1705.07632, 2017.
