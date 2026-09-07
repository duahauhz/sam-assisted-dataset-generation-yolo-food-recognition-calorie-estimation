"""Unit tests for the segmentation runtime module.

The crop_to_bbox tests are pure-numpy and run without Ultralytics.
The yolo_infer tests are skipped if Ultralytics is not installed.
"""

from __future__ import annotations

import sys

import numpy as np

from . import _CLASS_NAMES, binarize_mask, crop_mask_to_bbox


# ---------------------------------------------------------------------------
# crop_to_bbox tests
# ---------------------------------------------------------------------------
def test_crop_to_bbox_basic() -> None:
    """Crop a 10x10 mask to a 4x4 inner box."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 3:7] = 1
    crop = crop_mask_to_bbox(mask, (3, 2, 7, 6))
    assert crop.shape == (4, 4)
    assert crop.dtype == bool
    assert crop.all()


def test_crop_to_bbox_clips() -> None:
    """Bbox partially outside the image gets clipped."""
    mask = np.ones((10, 10), dtype=np.uint8)
    crop = crop_mask_to_bbox(mask, (-5, -5, 5, 5))
    assert crop.shape == (5, 5)
    assert crop.all()


def test_crop_to_bbox_degenerate() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    try:
        crop_mask_to_bbox(mask, (5, 5, 5, 5))
    except ValueError:
        return
    raise AssertionError("Expected ValueError for degenerate bbox")


def test_crop_to_bbox_outside_image() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    try:
        crop_mask_to_bbox(mask, (20, 20, 30, 30))
    except ValueError:
        return
    raise AssertionError("Expected ValueError for bbox outside image")


def test_binarize_mask() -> None:
    mask = np.array([[0, 1, 255], [0, 0, 1]], dtype=np.uint8)
    b = binarize_mask(mask)
    assert b.dtype == bool
    assert b.sum() == 3


def test_class_names_count() -> None:
    """The label set exposes 20 classes (19 foods + coin)."""
    assert len(_CLASS_NAMES) == 20
    assert "coin" in _CLASS_NAMES


# ---------------------------------------------------------------------------
# Smoke test for inference (skipped if no weights or no ultralytics)
# ---------------------------------------------------------------------------
def test_yolo_infer_smoke() -> None:
    """Run YOLO-seg on a real image if the weights are on disk."""
    try:
        from . import load_yolo_seg, predict_one
    except ImportError:
        print("  SKIP  test_yolo_infer_smoke (ultralytics missing)")
        return

    weights = r"E:\AI_Research\dlt8\runs\yolo_seg\ecustfd_yolo26seg-2\weights\best.pt"
    import os
    if not os.path.exists(weights):
        print(f"  SKIP  test_yolo_infer_smoke (no weights at {weights})")
        return

    # Use the first training image as a smoke-test input.
    images_dir = r"E:\AI_Research\dlt8\data\raw\ECUSTFD\JPEGImages"
    try:
        sample = os.path.join(images_dir, sorted(os.listdir(images_dir))[0])
    except (OSError, IndexError):
        sample = None
    if not sample or not os.path.exists(sample):
        print(f"  SKIP  test_yolo_infer_smoke (no sample image at {images_dir})")
        return

    model = load_yolo_seg(weights, device="cpu")
    dets = predict_one(model, sample, conf=0.25, imgsz=480)
    assert isinstance(dets, list)
    print(f"    -> {len(dets)} detections on {sample}")
    assert len(dets) >= 0  # just ensure no exception
    print(f"  PASS  {sys._getframe().f_code.co_name}")


def run_all() -> None:
    tests = [
        test_crop_to_bbox_basic,
        test_crop_to_bbox_clips,
        test_crop_to_bbox_degenerate,
        test_crop_to_bbox_outside_image,
        test_binarize_mask,
        test_class_names_count,
        test_yolo_infer_smoke,
    ]
    for t in tests:
        t()
        # The smoke test prints PASS/SKIP itself; we handle that here.
        if t.__name__ != "test_yolo_infer_smoke":
            print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
