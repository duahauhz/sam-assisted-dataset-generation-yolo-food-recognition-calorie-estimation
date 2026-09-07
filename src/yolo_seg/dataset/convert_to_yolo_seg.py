# -*- coding: utf-8 -*-
"""Convert ECUSTFD SAM masks (.npy) to YOLO26-seg polygon labels.

Output structure (YOLO format):
    dataset_root/
    ├── images/
    │   ├── train/<stem>.JPG
    │   └── val/<stem>.JPG
    └── labels/
        ├── train/<stem>.txt   # one row per object: <cls> <x1> <y1> ... <xn> <yn>
        └── val/<stem>.txt

Run once:
    python src/yolo_seg/dataset/convert_to_yolo_seg.py

After conversion, train:
    python src/yolo_seg/train/train_yolo26_seg.py
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
import random
import argparse
import sys

# Force UTF-8 on Windows console (cp1252 default breaks non-ASCII output)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ─── Config ───────────────────────────────────────────────────────────────────
ROOT        = Path("E:/AI_Research/dlt8")
MASK_DIR    = ROOT / "data/processed/sam_masks_full/masks"
RAW_DIR     = ROOT / "data/raw/ECUSTFD/JPEGImages"
OUT_ROOT    = ROOT / "data/processed/yolo_ecustfd_seg"
# Use ECUSTFD's official trainval/test split (paper-faithful, no data
# leakage). trainval.txt = train + val (1245 stems); test.txt = 1733
# stems held out for evaluation.
IMAGESETS_DIR = ROOT / "data/raw/ECUSTFD/ImageSets/Main"
TRAIN_SPLIT_FILE = IMAGESETS_DIR / "trainval.txt"
TEST_SPLIT_FILE = IMAGESETS_DIR / "test.txt"
SEED        = 42

# ECUSTFD class names (alphabetical → index 0-19).
#
# ECUSTFD gốc (MATLAB) có **19 foods + coin = 20 classes** (xem
# ``ECUSTFD/faster_rcnn/grabcut_mex.cpp`` lines 9-28 và
# ``src/constants.py::FOOD_CLASSES``).
#
# Lưu ý: 4 ảnh ``qiwi006S(7)``, ``qiwi006S(8)``, ``qiwi007S(1)``,
# ``qiwi007T(3)`` được gán nhãn ``<name>kiwi</name>`` trong file XML đã
# patch (``apply_bbox_overrides.py`` ghi ``kiwi`` thay vì ``qiwi`` để đánh
# dấu đây là kiwi fruit). Để không làm mất nhãn food của 4 ảnh này khi
# convert sang YOLO-seg, mapping dưới đây nạp ``kiwi`` như một alias của
# ``qiwi`` (cùng class index 17).
CLASS_NAMES = sorted([
    "apple", "banana", "bread", "bun", "coin",
    "doughnut", "egg", "fired_dough_twist", "grape",
    "lemon", "litchi", "mango", "mooncake",
    "orange", "peach", "pear", "plum", "qiwi",
    "sachima", "tomato",
])
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Alias map: các nhãn xuất hiện trong file XML đã patch nhưng không nằm
# trong CLASS_NAMES — ánh xạ về class chính. Lý do: 4 ảnh kiwi được
# apply_bbox_overrides.py ghi với ``<name>kiwi</name>`` để đánh dấu khác
# biệt với các ảnh qiwi còn lại; trong YOLO-seg ta muốn giữ nguyên 1 lớp.
CLASS_ALIASES = {
    "kiwi": "qiwi",
}


# ─── Core conversion ──────────────────────────────────────────────────────────

def mask_to_polygon_yolo(mask: np.ndarray, simplify_tolerance: float = 1.5
                          ) -> list[tuple[float, ...]]:
    """Convert a bool mask → list of normalized (x, y) polygon points.

    Uses OpenCV findContours (RETR_EXTERNAL to skip holes inside berries).
    Applies approxPolyDP to reduce vertex count.
    Normalizes coords to [0, 1] relative to image width/height.
    Returns [] if no contour found.
    """
    H, W = mask.shape
    mask_u8 = mask.astype("uint8") * 255

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Take largest contour only
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 9:          # skip tiny noise (< 3×3 px)
        return []

    # Approximate to reduce vertex count while preserving shape
    epsilon = simplify_tolerance
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)

    # Convert (N,1,2) → flat (N,2) → normalize
    pts = approx.reshape(-1, 2).astype(float)
    norm = [(x / W, y / H) for x, y in pts]

    # Ensure at least 3 points and closed polygon
    if len(norm) < 3:
        return []

    return norm


def convert_one_npy(npy_path: Path, raw_dir: Path
                     ) -> tuple[list[str], bool]:
    """Load one .npy, extract polygons, find raw image.

    Returns:
        lines  : list of YOLO-format label lines (one per object)
        found  : True if raw JPG exists
    """
    stem = npy_path.stem          # e.g. "apple001(1)"
    d = np.load(str(npy_path), allow_pickle=True).item()
    H, W = d["image_shape"][:2]

    lines = []
    for obj in d["objects"]:
        cls_name = obj["class"]
        # Áp dụng alias: "kiwi" -> "qiwi" (vì 4 ảnh kiwi có nhãn gốc
        # là "kiwi" trong XML đã patch, không có trong CLASS_NAMES).
        cls_name = CLASS_ALIASES.get(cls_name, cls_name)
        if cls_name not in CLASS_TO_IDX:
            continue              # skip unknown classes silently
        cls_idx = CLASS_TO_IDX[cls_name]

        mask = obj["mask"]
        poly = mask_to_polygon_yolo(mask)

        if not poly:
            # Degenerate mask – skip this object
            continue

        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
        lines.append(f"{cls_idx} {coords}")

    # Find raw image (JPG or jpg)
    exts = [".JPG", ".jpg", ".jpeg", ".JPEG"]
    raw_path = None
    for ext in exts:
        p = raw_dir / f"{stem}{ext}"
        if p.exists():
            raw_path = p
            break

    return lines, raw_path is not None


def build_dataset_from_imagesets(
    train_split_file: Path,
    test_split_file: Path,
) -> tuple[list, list]:
    """Load the ECUSTFD official trainval/test split.

    Reads per-image entries from ``trainval.txt`` and ``test.txt``
    (e.g. ``apple001S(1)`` — image stem with view + index). Each entry
    maps 1-to-1 to a single ``.npy`` mask in ``MASK_DIR``.

    Returns:
        (train_npy_files, test_npy_files) — list of npy mask paths.
    """
    train_stems = {s.strip() for s in train_split_file.read_text(encoding="utf-8").splitlines() if s.strip()}
    test_stems = {s.strip() for s in test_split_file.read_text(encoding="utf-8").splitlines() if s.strip()}
    print(
        f"ECUSTFD splits: trainval={len(train_stems)} images, "
        f"test={len(test_stems)} images"
    )
    overlap = train_stems & test_stems
    if overlap:
        print(f"WARNING: {len(overlap)} entries appear in BOTH splits — checking split files")
    print(f"Union: {len(train_stems | test_stems)} unique images")

    # Build stem -> path map (one npy per image stem).
    by_stem: dict[str, Path] = {}
    for npy in MASK_DIR.glob("*.npy"):
        # If there are duplicates, keep the first; warn if found.
        if npy.stem in by_stem:
            print(f"WARN: duplicate npy stem {npy.stem}, keeping {by_stem[npy.stem]}")
        else:
            by_stem[npy.stem] = npy

    train_files = [by_stem[s] for s in train_stems if s in by_stem]
    test_files = [by_stem[s] for s in test_stems if s in by_stem]
    missing_train = train_stems - set(by_stem.keys())
    missing_test = test_stems - set(by_stem.keys())
    if missing_train:
        print(f"WARN: {len(missing_train)} trainval entries have no .npy mask (e.g. {sorted(missing_train)[:3]})")
    if missing_test:
        print(f"WARN: {len(missing_test)} test entries have no .npy mask (e.g. {sorted(missing_test)[:3]})")
    return train_files, test_files


def build_dataset(split_ratio: float, seed: int) -> tuple[list, list]:
    """Legacy random split (kept for compatibility testing).

    Splits all .npy files into train / val by hashing stem (stable split).
    **Do NOT use this for paper evaluation** — it leaks test images
    into the train set. Use ``build_dataset_from_imagesets`` instead.
    """
    import hashlib
    npy_files = sorted(MASK_DIR.glob("*.npy"))

    train, val = [], []
    random.seed(seed)

    for p in npy_files:
        stem_bytes = p.stem.encode()
        hash_val = int(hashlib.md5(stem_bytes).hexdigest(), 16)
        if hash_val % 100 < int(split_ratio * 100):
            val.append(p)
        else:
            train.append(p)

    return train, val


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                   help="Re-create output dirs even if they already exist")
    ap.add_argument("--dry-run", action="store_true",
                   help="Print stats without writing files")
    ap.add_argument("--use-imagesets-split", action="store_true", default=True,
                   help="Use ECUSTFD trainval/test.txt (paper-faithful, "
                        "no data leakage). Default: True. Pass "
                        "--no-use-imagesets-split for the legacy hash split.")
    ap.add_argument("--no-use-imagesets-split", action="store_false",
                   dest="use_imagesets_split")
    args = ap.parse_args()

    if args.use_imagesets_split:
        train_files, val_files = build_dataset_from_imagesets(
            TRAIN_SPLIT_FILE, TEST_SPLIT_FILE
        )
        print(f"Train (trainval): {len(train_files)}  |  Val (test): {len(val_files)}")
    else:
        train_files, val_files = build_dataset(SPLIT_RATIO, SEED)
        print(f"Train (random hash): {len(train_files)}  |  Val: {len(val_files)}")

    for split_name, file_list in [("train", train_files), ("val", val_files)]:
        img_dir = OUT_ROOT / "images" / split_name
        lbl_dir = OUT_ROOT / "labels" / split_name

        if args.force:
            shutil.rmtree(img_dir, ignore_errors=True)
            shutil.rmtree(lbl_dir, ignore_errors=True)

        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        skipped_missing_img = 0
        skipped_no_contour = 0
        total_objects = 0
        class_counts = {c: 0 for c in CLASS_NAMES}

        for npy_path in file_list:
            stem = npy_path.stem
            lines, img_found = convert_one_npy(npy_path, RAW_DIR)

            if not lines:
                skipped_no_contour += 1
                continue

            if not img_found:
                skipped_missing_img += 1
                continue

            # Write label .txt
            lbl_path = lbl_dir / f"{stem}.txt"
            lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            # Symlink or copy raw image → images/<split>/
            exts = [".JPG", ".jpg", ".jpeg", ".JPEG"]
            raw_src = None
            for ext in exts:
                p = RAW_DIR / f"{stem}{ext}"
                if p.exists():
                    raw_src = p
                    break
            if raw_src is not None:
                dst_img = img_dir / f"{stem}{raw_src.suffix}"
                if not dst_img.exists():
                    shutil.copy2(raw_src, dst_img)

            for line in lines:
                cls_idx = int(line.split()[0])
                class_counts[CLASS_NAMES[cls_idx]] += 1
                total_objects += 1

        print(f"\n--- {split_name.upper()} ---")
        print(f"  Images written   : {len(list(img_dir.glob('*')))}")
        print(f"  Labels written   : {len(list(lbl_dir.glob('*.txt')))}")
        print(f"  Skipped (no img) : {skipped_missing_img}")
        print(f"  Skipped (no poly): {skipped_no_contour}")
        print(f"  Total objects    : {total_objects}")
        for c, n in sorted(class_counts.items()):
            if n:
                print(f"    {c}: {n}")

    # Write dataset YAML
    yaml_path = OUT_ROOT / "ecustfd-seg.yaml"
    yaml_content = f"""# ECUSTFD Instance Segmentation Dataset for YOLO26-seg
# Auto-generated by convert_to_yolo_seg.py

path: {OUT_ROOT.as_posix()}  # dataset root
train: images/train
val: images/val

# Classes ({len(CLASS_NAMES)} total)
names:
"""
    for i, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"

    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\nDataset YAML written: {yaml_path}")
    print(f"\nConversion done. Train with:")
    print(f"   python src/data_prep_SAM1/sam_masks_full/train_yolo26_seg.py")


if __name__ == "__main__":
    main()
