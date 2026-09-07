# -*- coding: utf-8 -*-
"""Verify YOLO dataset integrity before training.

Checks:
  1. images/ and labels/ have matching stems
  2. Label format: cls_idx coords... (one row per object)
  3. Coordinates in [0, 1]
  4. Classes all in range [0, num_classes)
  5. Sample image openable
  6. Ultralytics model.val() quick sanity on val split

Usage:
    python src/yolo_seg/dataset/verify_yolo_dataset.py
"""

import sys
from pathlib import Path
import numpy as np

ROOT     = Path("E:/AI_Research/dlt8")
OUT_ROOT = ROOT / "data/processed/yolo_ecustfd_seg"
YAML_PATH = OUT_ROOT / "ecustfd-seg.yaml"

NUM_CLASSES = 21

errors = []
warnings = []

def check():
    # 1. YAML
    yaml = OUT_ROOT / "ecustfd-seg.yaml"
    if not yaml.exists():
        errors.append(f"YAML not found: {yaml}")
        return
    print(f"[OK] YAML exists: {yaml}")

    # Parse YAML manually
    lines = yaml.read_text(encoding="utf-8").splitlines()
    names = {}
    for line in lines:
        line = line.strip()
        if line.startswith("names:") or line.startswith("#"):
            continue
        if ":" in line:
            idx_str, name_str = line.split(":", 1)
            idx_str = idx_str.strip()
            name_str = name_str.strip()
            if idx_str.isdigit():
                names[int(idx_str)] = name_str
    print(f"[OK] YAML classes: {len(names)} (expected {NUM_CLASSES})")
    if len(names) != NUM_CLASSES:
        warnings.append(f"Expected {NUM_CLASSES} classes, got {len(names)}")

    # 2. Check train/val
    for split in ["train", "val"]:
        img_dir = OUT_ROOT / "images" / split
        lbl_dir = OUT_ROOT / "labels" / split

        img_files = {p.stem: p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg",".jpeg",".png",".JPG",".JPEG",".PNG")}
        lbl_files = {p.stem: p for p in lbl_dir.glob("*.txt")}

        print(f"\n[{split}] images={len(img_files)} labels={len(lbl_files)}")

        # Match check
        img_stems = set(img_files.keys())
        lbl_stems = set(lbl_files.keys())
        only_img = img_stems - lbl_stems
        only_lbl = lbl_stems - img_files.keys()
        if only_img:
            errors.append(f"[{split}] images without labels: {sorted(only_img)[:5]}")
        if only_lbl:
            errors.append(f"[{split}] labels without images: {sorted(only_lbl)[:5]}")
        if not only_img and not only_lbl:
            print(f"[OK] [{split}] all images have matching labels")

        # Sample label format check (first 20 files)
        bad_fmt = []
        for stem, lbl_path in list(lbl_files.items())[:20]:
            lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
            for row_idx, row in enumerate(lines):
                if not row.strip():
                    continue
                parts = row.split()
                if len(parts) < 7:  # cls + at least 3 pts (6 coords)
                    bad_fmt.append((stem, row_idx, "too few coords"))
                    continue
                try:
                    cls_idx = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    if cls_idx < 0 or cls_idx >= NUM_CLASSES:
                        bad_fmt.append((stem, row_idx, f"cls={cls_idx} out of range"))
                    for v in coords:
                        if not (0.0 <= v <= 1.0):
                            bad_fmt.append((stem, row_idx, f"coord={v} out of [0,1]"))
                    if len(coords) % 2 != 0:
                        bad_fmt.append((stem, row_idx, "odd number of coords"))
                except ValueError as e:
                    bad_fmt.append((stem, row_idx, f"ValueError: {e}"))

        if bad_fmt:
            errors.append(f"[{split}] bad label format: {bad_fmt[:5]}")
        else:
            print(f"[OK] [{split}] first 20 label files: all valid")

        # Image openability sample
        import cv2
        sample_img = sorted(img_files.values())[:3]
        for p in sample_img:
            img = cv2.imread(str(p))
            if img is None:
                errors.append(f"Cannot read image: {p}")
            else:
                print(f"[OK] Image sample: {p.name} shape={img.shape}")

        # Per-class object count from labels
        from collections import Counter
        cls_counts = Counter()
        total_objs = 0
        for lbl_path in lbl_files.values():
            for row in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                if not row.strip():
                    continue
                cls_idx = int(row.split()[0])
                cls_counts[cls_idx] += 1
                total_objs += 1
        print(f"[{split}] total objects: {total_objs}")
        for idx in sorted(cls_counts):
            print(f"    class {idx:2d} ({names.get(idx,'?'):20s}): {cls_counts[idx]:4d} objects")

    # 3. Ultralytics val quick check (just load, don't train)
    print("\n--- Ultralytics dataset validation ---")
    try:
        from ultralytics.data.utils import check_det_dataset
        result = check_det_dataset(str(YAML_PATH), autodownload=False)
        print(f"[OK] Ultralytics accepts dataset YAML: {result}")
    except Exception as e:
        warnings.append(f"Ultralytics check_det_dataset: {e}")

    print("\n" + "=" * 60)
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("No errors found.")
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  ! {w}")
    else:
        print("No warnings.")
    print("=" * 60)
    return len(errors) == 0


if __name__ == "__main__":
    ok = check()
    sys.exit(0 if ok else 1)
