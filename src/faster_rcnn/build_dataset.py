# -*- coding: utf-8 -*-
"""Build the ECUSTFD-derived bbox dataset for Faster R-CNN training.

Why this script exists
----------------------
The YOLO26-seg pipeline (``src/yolo_seg/dataset/convert_to_yolo_seg.py``)
already produces a YOLO-format dataset at
``data/processed/yolo_ecustfd_seg/`` that splits the 2978 ECUSTFD images
into 1245 trainval / 1733 test using the official
``data/raw/ECUSTFD/ImageSets/Main/{trainval,test}.txt`` files.  Because
YOLO26-seg already encodes the segmentation ground-truth via SAM masks,
its train/val/test split is the canonical one for the project.

Faster R-CNN is detection-only, so it needs a *bbox-only* dataset
formatted as COCO JSON (the format torchvision's detection API expects).
Faster R-CNN also has its own evaluation story (COCO-style mAP via
``torchvision``'s ``CocoEvaluator``), which we do not run inside this
build step — we only produce the dataset files here.  Evaluation is left
to the matching notebook ``06_faster_rcnn_eval.ipynb``.

Configuration
-------------
As of 2026-08-08 only one variant is supported:

* **"new"** (default): bbox ground-truth prefers the manually overwritten
  files in ``data/processed/bbox_full/Annotations_patched/<stem>.xml``
  when they exist; for the remaining stems we fall back to the raw XML.
  This mirrors the SAM pipeline (``segment_sam1_box.py``), which also
  prefers patched XML when present.

The previous ``"old"`` variant (raw VOC XML only, 2017-paper baseline)
has been removed from this script because the project no longer compares
against the raw XML baseline.

Outputs (mirrors the YOLO26 layout)
----------------------------------

    data/processed/faster_rcnn_seg/new/
        images/
            train/      <stem>.jpg  (1245 images)
            test/       <stem>.jpg  (1733 images)
        annotations/
            train.json  (COCO JSON)
            test.json   (COCO JSON)

The image files are *symlinks* on Unix and *hard-links* (or copies) on
Windows, never real duplicates: the train/test images are still the
original ECUSTFD JPEGs.  We never copy the raw bytes.

Run:

    # Build the (only) variant (default)
    python src/faster_rcnn/build_dataset.py

    # Limit to first 5 stems (smoke test)
    python src/faster_rcnn/build_dataset.py --max-samples 5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make sure both ``src.faster_rcnn`` (sub-package) and ``src.constants`` etc.
# (top-level package) are importable when this script is run as a subprocess.
#   - parents[1] = src/  -> needed by "from faster_rcnn import ..."
#   - parents[2] = project root  -> needed by "from src.faster_rcnn import ..."
#     and "from src.constants import ..." (inference.py / eval_pipeline.py).
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))          # project root
sys.path.insert(0, str(ROOT_DIR / "src"))  # src

from faster_rcnn.faster_rcnn_guard import (  # noqa: E402
    LOGS_DIR,
    ROOT,
    make_run_dir,
    make_timestamp,
    setup_logger,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 21 classes (19 foods + coin + 1 __background__).
# In torchvision Faster R-CNN, index 0 is reserved for '__background__'.
# Therefore, foreground category IDs MUST be 1-indexed (1..20).
# EXACT SAME ORDER as the YOLO26-seg ``ecustfd-seg.yaml``.
_CLASS_NAMES: List[str] = [
    "apple", "banana", "bread", "bun", "coin",
    "doughnut", "egg", "fried_dough_twist", "grape",
    "lemon", "litchi", "mango", "mooncake",
    "orange", "peach", "pear", "plum", "qiwi",
    "sachima", "tomato",
]
# 1-indexed category_id (1..20)
_CLASS_TO_IDX: Dict[str, int] = {name: i + 1 for i, name in enumerate(_CLASS_NAMES)}
# Handle legacy typo in some annotations
_CLASS_TO_IDX["fired_dough_twist"] = _CLASS_TO_IDX["fried_dough_twist"]

# ECUSTFD sources
IMG_SRC       = ROOT / "data/raw/ECUSTFD/JPEGImages"
RAW_VOC_DIR   = ROOT / "data/raw/ECUSTFD/Annotations"
PATCHED_VOC_DIR = ROOT / "data/processed/bbox_full/Annotations_patched"
IMAGESETS_DIR = ROOT / "data/raw/ECUSTFD/ImageSets/Main"

# SAM mask source (only used by variant="new").
#
# Each ``masks/<stem>.npy`` contains:
#     {
#       "image_shape": [H, W, 3],
#       "objects": [
#         {"class": str, "bbox_voc": [x1,y1,x2,y2],
#          "mask": (H,W) bool ndarray, "score": float, "area": int},
#         ...
#       ],
#       "source": "raw_gt_box_prompt" | "manual_mask_override" | ...
#     }
#
# Note: ``apply_mask_overrides.py`` may rewrite some entries in-place
# (the only one as of 2026-08-05 is ``grape001T(5)`` -> class=grape,
# target_index=0, source changes to ``manual_mask_override``).  We do
# NOT reproduce that patcher here — we just read what's already there.
SAM_MASKS_DIR = ROOT / "data/processed/sam_masks_full/masks"

# Output (mirrors yolo_ecustfd_seg layout)
OUT_ROOT = ROOT / "data/processed/faster_rcnn_seg"

# Which split file defines which subset
TRAIN_SPLIT_FILE = IMAGESETS_DIR / "trainval.txt"   # 1245 stems
TEST_SPLIT_FILE  = IMAGESETS_DIR / "test.txt"       # 1733 stems

# Variants → description shown in the summary
VARIANT_INFO: Dict[str, str] = {
    "new": "voc_to_yolo patched XML (data/processed/bbox_full/Annotations_patched)"
           " with fallback to raw — mirrors SAM pipeline",
}


# ---------------------------------------------------------------------------
# VOC XML parsing
# ---------------------------------------------------------------------------

def _parse_voc_xml(xml_path: Path) -> Tuple[int, int, List[Dict]]:
    """Parse a Pascal VOC XML file.

    Returns ``(width, height, objects)`` where each object is::

        {"class_name": str, "bbox": [x1, y1, x2, y2], "difficult": int}

    Skips objects whose class is not in ``_CLASS_TO_IDX`` (unknown
    label noise can't be turned into a useful training signal anyway).
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects: List[Dict] = []
    for obj in root.findall("object"):
        cls_name = str(obj.find("name").text).strip()
        if cls_name not in _CLASS_TO_IDX:
            continue

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        difficult_el = obj.find("difficult")
        difficult = int(difficult_el.text) if difficult_el is not None else 0

        # Skip degenerate boxes
        if xmax <= xmin or ymax <= ymin:
            continue

        objects.append({
            "class_name": cls_name,
            "bbox": [xmin, ymin, xmax, ymax],
            "difficult": difficult,
        })

    return width, height, objects


def _resolve_xml_for_stem(stem: str, variant: str) -> Path:
    """Return the XML file Faster R-CNN should use for *stem* in *variant*.

    For ``variant == "new"`` we prefer the patched file when it exists.
    For ``variant == "old"`` we always use the raw XML, so the only
    variable changing between the two detector runs is the bbox quality,
    not the split.
    """
    if variant == "new":
        patched = PATCHED_VOC_DIR / f"{stem}.xml"
        if patched.exists():
            return patched
    return RAW_VOC_DIR / f"{stem}.xml"


# ---------------------------------------------------------------------------
# SAM mask loading (variant="new" only)
# ---------------------------------------------------------------------------

def _load_sam_mask(stem: str, logger) -> Optional[Dict]:
    """Load ``sam_masks_full/masks/<stem>.npy``.

    Returns ``None`` if the file does not exist OR if its inner object
    list is empty.  Returns the full dict (with ``objects``, ``source``,
    ``image_shape``) so callers can decide per-object whether to attach
    a mask pointer.

    Object order in this dict matches the VOC XML object order, so we
    match by class name + position for ``variant="new"``.  Any class that
    does not appear in the SAM payload simply has no mask pointer.
    """
    p = SAM_MASKS_DIR / f"{stem}.npy"
    if not p.exists():
        return None
    try:
        import numpy as np
        payload = np.load(str(p), allow_pickle=True).item()
    except Exception as exc:
        logger.warning(f"  SAM mask unreadable for {stem}: {exc}")
        return None
    if not isinstance(payload, dict) or "objects" not in payload:
        return None
    if not payload["objects"]:
        return None
    return payload


def _build_sam_index(payload: Dict) -> Dict[str, List[int]]:
    """Group SAM object indices by class name → list of obj positions.

    Used to look up which SAM object corresponds to a given VOC
    annotation by class name + ``target_index`` (see ``_match_sam_obj``).
    """
    out: Dict[str, List[int]] = {}
    for i, o in enumerate(payload["objects"]):
        out.setdefault(o.get("class"), []).append(i)
    return out


def _match_sam_obj(
    sam_payload: Dict,
    sam_by_class: Dict[str, List[int]],
    xml_class: str,
    xml_pos_in_class: int,
) -> Optional[int]:
    """Return the SAM objects[] index matching the XML annotation.

    Args:
        sam_payload: the SAM dict (kept for future use).
        sam_by_class: pre-computed ``{class: [obj_idx,...]}`` index.
        xml_class: class name from the VOC XML.
        xml_pos_in_class: 0-based index among XML objects of this class
            (so two ``apple`` boxes in one image map to ``pos=0`` and
            ``pos=1``).

    Returns:
        An int into ``sam_payload["objects"]`` if found, else ``None``.
    """
    candidates = sam_by_class.get(xml_class, [])
    if xml_pos_in_class < len(candidates):
        return candidates[xml_pos_in_class]
    return None


# ---------------------------------------------------------------------------
# COCO JSON writer
# ---------------------------------------------------------------------------

def _build_coco(
    stems: List[str],
    variant: str,
    imageset_file: Path,
    logger,
) -> Dict:
    """Build a COCO-format dict for a list of stems."""
    images: List[Dict] = []
    annotations: List[Dict] = []
    categories: List[Dict] = [
        {"id": i + 1, "name": name} for i, name in enumerate(_CLASS_NAMES)
    ]

    next_image_id = 1
    next_annot_id = 1

    n_with_objects = 0
    n_files_missing = 0
    n_with_mask = 0        # only used by variant="new"; reset to 0 once per split

    for stem in stems:
        # Try both .JPG (raw) and .jpg (lowercase) — the raw dataset
        # actually uses .JPG on Windows but the yolo folder uses .jpg.
        img_candidates = [IMG_SRC / f"{stem}.JPG", IMG_SRC / f"{stem}.jpg"]
        img_path = next((p for p in img_candidates if p.exists()), None)
        if img_path is None or not img_path.exists():
            n_files_missing += 1
            logger.warning(f"  Missing image for stem '{stem}'")
            continue

        xml_path = _resolve_xml_for_stem(stem, variant)
        if not xml_path.exists():
            n_files_missing += 1
            logger.warning(f"  Missing XML for stem '{stem}' ({xml_path})")
            continue

        # XML dims come from the file we'll use as bbox ground-truth
        try:
            w, h, objs = _parse_voc_xml(xml_path)
        except Exception as exc:
            logger.warning(f"  Cannot parse XML {xml_path}: {exc}")
            continue

        # Variant "new" may carry a SAM mask per object.  We match by
        # class name + position inside the SAM payload so the annotation
        # dict can carry ``mask_object_index`` and ``mask_source``.
        sam_payload: Optional[Dict] = None
        sam_by_class: Dict[str, List[int]] = {}
        if variant == "new":
            sam_payload = _load_sam_mask(stem, logger)
            if sam_payload is not None:
                sam_by_class = _build_sam_index(sam_payload)

        # Add image record
        images.append({
            "id": next_image_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
            "stem": stem,
            "voc_xml_source": "patched" if "patched" in str(xml_path).lower() else "raw",
            "sam_mask_source": (
                sam_payload.get("source") if sam_payload is not None else None
            ),
            "sam_mask_path": (
                str(SAM_MASKS_DIR.relative_to(ROOT) / f"{stem}.npy")
                if sam_payload is not None else None
            ),
        })
        image_id = next_image_id
        next_image_id += 1

        if objs:
            n_with_objects += 1

        # Track per-class position in the XML objects list so we can map
        # back to the SAM payload by class + position.
        pos_in_class: Dict[str, int] = {}
        for obj in objs:
            x1, y1, x2, y2 = obj["bbox"]
            # Clip to image dims in case the XML is over-sized
            x1 = max(0.0, min(x1, w))
            y1 = max(0.0, min(y1, h))
            x2 = max(0.0, min(x2, w))
            y2 = max(0.0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            cls_name = obj["class_name"]
            this_pos = pos_in_class.get(cls_name, 0)
            pos_in_class[cls_name] = this_pos + 1

            # Look up SAM mask index (variant="new" only).
            mask_idx: Optional[int] = None
            mask_src: Optional[str] = None
            mask_path: Optional[str] = None
            if sam_payload is not None:
                mask_idx = _match_sam_obj(
                    sam_payload, sam_by_class, cls_name, this_pos
                )
                if mask_idx is not None:
                    n_with_mask += 1
                    mask_src = sam_payload.get("source")
                    mask_path = str(
                        SAM_MASKS_DIR.relative_to(ROOT) / f"{stem}.npy"
                    )

            annot = {
                "id": next_annot_id,
                "image_id": image_id,
                "category_id": _CLASS_TO_IDX[cls_name],
                "bbox": [x1, y1, x2 - x1, y2 - y1],   # COCO = [x, y, w, h]
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
                "difficult": obj["difficult"],
            }
            # Mask pointers stay as JSON-native fields (no ndarray in JSON).
            if mask_idx is not None:
                annot["mask_object_index"] = mask_idx
                annot["mask_source"] = mask_src
                annot["mask_path"] = mask_path
            annotations.append(annot)
            next_annot_id += 1

    if variant == "new":
        logger.info(
            f"  [{imageset_file.name}] {len(images)} images, "
            f"{len(annotations)} objects, "
            f"{n_with_objects} imgs with objects, "
            f"{n_with_mask} objects carry a SAM mask"
            + (f", {n_files_missing} missing" if n_files_missing else "")
        )
    else:
        logger.info(
            f"  [{imageset_file.name}] {len(images)} images, "
            f"{len(annotations)} objects, "
            f"{n_with_objects} imgs with objects, "
            f"{n_files_missing} missing"
        )

    return {
        "info": {
            "description": f"ECUSTFD Faster R-CNN dataset ({variant} variant)",
            "variant": variant,
            "n_classes": len(_CLASS_NAMES),
            "imageset_file": str(imageset_file.relative_to(ROOT)),
            "voc_sources": {
                "raw":    str(RAW_VOC_DIR.relative_to(ROOT)),
                "patched": str(PATCHED_VOC_DIR.relative_to(ROOT)),
            },
            "sam_masks_dir": (
                str(SAM_MASKS_DIR.relative_to(ROOT)) if variant == "new" else None
            ),
            "has_masks": variant == "new",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


# ---------------------------------------------------------------------------
# Image linking
# ---------------------------------------------------------------------------

def _link_image(src: Path, dst: Path, logger) -> None:
    """Link a single image from src to dst.

    Strategy:
        1. If ``dst`` exists with the same content, skip.
        2. Prefer a symlink (smallest on disk, works on macOS/Linux).
        3. Fall back to a hard link (works on same filesystem, allows
           both Linux and Windows).
        4. Last resort: byte-copy.

    We never use shutil.copytree here because the target file may
    already exist due to a previous run.
    """
    if dst.exists() or dst.is_symlink():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    # Try symlink first (works on Unix & Windows when developer mode is on).
    try:
        os.symlink(src, dst)
        return
    except (OSError, NotImplementedError):
        pass

    # Try hard link (same filesystem, instant).
    try:
        os.link(src, dst)
        return
    except OSError:
        pass

    # Last resort: byte copy
    shutil.copy2(src, dst)


def _link_split(stems: List[str], target_dir: Path, logger) -> None:
    """Symlink/copy every stem's image into ``target_dir``."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        src = IMG_SRC / f"{stem}.JPG"
        if not src.exists():
            src = IMG_SRC / f"{stem}.jpg"
        if not src.exists():
            logger.warning(f"  Cannot find image for stem '{stem}'")
            continue
        _link_image(src.resolve(), target_dir / src.name, logger)


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------

def build_variant(variant: str, logger) -> Dict:
    """Build the ECUSTFD bbox dataset for one variant end-to-end.

    Only the ``new`` variant is supported as of 2026-08-08 -- the project
    no longer compares against the raw 2017-paper XML baseline.

    Returns a metadata dict suitable for the run summary.
    """
    assert variant == "new", f"Unknown variant: {variant!r} (only 'new' is supported)"

    logger.info(f"=== Building variant '{variant}' ===")
    logger.info(f"  {VARIANT_INFO[variant]}")

    out_dir = OUT_ROOT / variant
    img_dir = out_dir / "images"
    ann_dir = out_dir / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Read stem lists
    train_stems = TRAIN_SPLIT_FILE.read_text(encoding="utf-8").splitlines()
    test_stems  = TEST_SPLIT_FILE.read_text(encoding="utf-8").splitlines()
    logger.info(f"  trainval.txt: {len(train_stems)} stems")
    logger.info(f"  test.txt    : {len(test_stems)} stems")

    # Symlink images
    logger.info("  Linking images into images/train/, images/test/")
    train_img_dir = img_dir / "train"
    test_img_dir  = img_dir / "test"
    _link_split(train_stems, train_img_dir, logger)
    _link_split(test_stems,  test_img_dir,  logger)

    # Build COCO JSONs
    logger.info("  Building COCO JSON annotations")
    train_coco = _build_coco(train_stems, variant, TRAIN_SPLIT_FILE, logger)
    test_coco  = _build_coco(test_stems,  variant, TEST_SPLIT_FILE,  logger)

    train_json = ann_dir / "train.json"
    test_json  = ann_dir / "test.json"
    train_json.write_text(
        json.dumps(train_coco, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    test_json.write_text(
        json.dumps(test_coco, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"  Wrote {train_json.relative_to(ROOT)}")
    logger.info(f"  Wrote {test_json.relative_to(ROOT)}")

    return {
        "variant": variant,
        "out_dir": str(out_dir.relative_to(ROOT)),
        "n_train_images": len(train_coco["images"]),
        "n_train_objects": len(train_coco["annotations"]),
        "n_test_images": len(test_coco["images"]),
        "n_test_objects": len(test_coco["annotations"]),
        "n_classes": len(_CLASS_NAMES),
        "train_json": str(train_json.relative_to(ROOT)),
        "test_json":  str(test_json.relative_to(ROOT)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_summary(run_dir: Path, meta: Dict, logger) -> None:
    """Write a human-readable summary file (mirrors 04_*_summary.txt)."""
    path = run_dir / "summary.txt"
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("05 -- BUILD FASTER R-CNN DATASET (new) -- SUMMARY")
    lines.append("=" * 70)
    ts = run_dir.name.rsplit("_", 1)[-1]
    lines.append(f"Run timestamp      : {ts}")
    lines.append(f"PROJECT_ROOT       : {ROOT}")
    lines.append(f"LOG_PATH           : {LOGS_DIR / f'05_build_{ts}.log'}")
    lines.append("")
    lines.append("Variants built:")
    for m in meta["variants"]:
        lines.append(
            f"  [{m['variant']}] {VARIANT_INFO[m['variant']]}"
        )
        lines.append(
            f"      train: {m['n_train_images']} images, "
            f"{m['n_train_objects']} objects"
        )
        lines.append(
            f"      test : {m['n_test_images']} images, "
            f"{m['n_test_objects']} objects"
        )
        lines.append(f"      train json: {m['train_json']}")
        lines.append(f"      test  json: {m['test_json']}")
        lines.append("")
    lines.append("Files written:")
    for m in meta["variants"]:
        lines.append(f"  {ROOT / m['out_dir']}")
    lines.append(f"  {path}")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Summary written: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--variant", choices=["new"], default="new",
        help="(kept for backward compatibility; only 'new' is supported)",
    )
    ap.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit to first N stems per split (smoke-test)",
    )
    args = ap.parse_args()

    ts = make_timestamp()
    logger = setup_logger("faster_rcnn_build", timestamp=ts, log_stem="05_build")
    run_dir = make_run_dir("05_build", timestamp=ts)

    logger.info("=" * 70)
    logger.info("=== 05 -- BUILD FASTER R-CNN DATASET (new) ===")
    logger.info("=" * 70)
    logger.info(f"PROJECT_ROOT = {ROOT}")
    logger.info(f"RUN_DIR      = {run_dir}")
    logger.info(f"Variant arg  = {args.variant}")

    variants = ["new"]
    meta_variants: List[Dict] = []
    for v in variants:
        # If --max-samples is set, take only the first N stems per split
        # BEFORE building (both for images and for COCO JSON).
        if args.max_samples is not None:
            train_stems_all = TRAIN_SPLIT_FILE.read_text(encoding="utf-8").splitlines()
            test_stems_all  = TEST_SPLIT_FILE.read_text(encoding="utf-8").splitlines()
            n = args.max_samples
            TAG = f".first{n}"
            TRAIN_SPLIT_FILE_TMP = TRAIN_SPLIT_FILE.with_name(TRAIN_SPLIT_FILE.stem + TAG + TRAIN_SPLIT_FILE.suffix)
            TEST_SPLIT_FILE_TMP = TEST_SPLIT_FILE.with_name(TEST_SPLIT_FILE.stem + TAG + TEST_SPLIT_FILE.suffix)
            TRAIN_SPLIT_FILE_TMP.write_text("\n".join(train_stems_all[:n]), encoding="utf-8")
            TEST_SPLIT_FILE_TMP.write_text("\n".join(test_stems_all[:n]),  encoding="utf-8")
            orig_train, orig_test = TRAIN_SPLIT_FILE, TEST_SPLIT_FILE
            # Re-point globals for this run only
            globals()["TRAIN_SPLIT_FILE"] = TRAIN_SPLIT_FILE_TMP
            globals()["TEST_SPLIT_FILE"]  = TEST_SPLIT_FILE_TMP
            try:
                meta = build_variant(v, logger)
            finally:
                globals()["TRAIN_SPLIT_FILE"] = orig_train
                globals()["TEST_SPLIT_FILE"]  = orig_test
        else:
            meta = build_variant(v, logger)
        meta_variants.append(meta)

    meta = {"variants": meta_variants, "run_dir": str(run_dir)}
    (run_dir / "dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary(run_dir, meta, logger)

    logger.info("=" * 70)
    logger.info("✅ BUILD DONE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
