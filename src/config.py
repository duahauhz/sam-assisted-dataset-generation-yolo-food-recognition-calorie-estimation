"""Global configuration for the food calorie estimation project."""

import os
from pathlib import Path

# Root directory of the project
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"
DATA_ROOT: Path = PROJECT_ROOT / "data"

# Raw dataset path
RAW_DATA_ROOT: Path = DATA_ROOT / "raw" / "ECUSTFD"
ANNOTATIONS_DIR: Path = RAW_DATA_ROOT / "Annotations"
IMAGES_DIR: Path = RAW_DATA_ROOT / "JPEGImages"
IMAGESETS_DIR: Path = RAW_DATA_ROOT / "ImageSets" / "Main"
DENSITY_XLS: Path = RAW_DATA_ROOT / "density.xls"

# Processed dataset paths
PROCESSED_ROOT: Path = DATA_ROOT / "processed"
SAM_MASKS_DIR: Path = PROCESSED_ROOT / "sam_masks"
AUGMENTED_DIR: Path = PROCESSED_ROOT / "augmented"
POINTCLOUD_DIR: Path = PROCESSED_ROOT / "pointclouds"

# YOLO format paths
YOLO_ROOT: Path = PROCESSED_ROOT / "yolo"
YOLO_IMAGES_DIR: Path = YOLO_ROOT / "images"
YOLO_LABELS_DIR: Path = YOLO_ROOT / "labels"

# Model checkpoints
MODELS_ROOT: Path = PROJECT_ROOT / "models"
YOLO_MODEL_DIR: Path = MODELS_ROOT / "yolo"
SAM_MODEL_DIR: Path = MODELS_ROOT / "sam"
DEPTH_MODEL_DIR: Path = MODELS_ROOT / "depth"

# Output paths
OUTPUT_ROOT: Path = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR: Path = OUTPUT_ROOT / "predictions"
VISUALIZATIONS_DIR: Path = OUTPUT_ROOT / "visualizations"
REPORTS_DIR: Path = OUTPUT_ROOT / "reports"

# YOLO13-seg model settings
YOLO_MODEL_NAME: str = "yolo13s-seg.pt"  # or yolo13m-seg.pt, yolo13l-seg.pt
YOLO_CONF_THRESH: float = 0.25
YOLO_IOU_THRESH: float = 0.7

# SAM model settings
SAM_MODEL_TYPE: str = "vit_b"  # vit_b, vit_l, vit_h
SAM_CHECKPOINT: str = "models/sam/sam_vit_b_01ec64.pth"

# MiDaS depth estimation settings
MIDAS_MODEL_TYPE: str = "DPT_Large"  # DPT_Large, DPT_Hybrid, MiDaS_small

# Volume estimation settings
DEPTH_DELTA_THRESHOLD: float = 0.05  # threshold for depth-based vs shape-prior switching

# Training settings
TRAIN_EPOCHS: int = 100
TRAIN_BATCH_SIZE: int = 16
TRAIN_IMG_SIZE: int = 640

# Ensure directories exist
for _dir in [
    MODELS_ROOT, YOLO_MODEL_DIR, SAM_MODEL_DIR, DEPTH_MODEL_DIR,
    SAM_MASKS_DIR, AUGMENTED_DIR, POINTCLOUD_DIR,
    YOLO_IMAGES_DIR, YOLO_LABELS_DIR,
    PREDICTIONS_DIR, VISUALIZATIONS_DIR, REPORTS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)


def get_data_paths(split: str = "train") -> dict:
    """Get YOLO format paths for a given split.

    Args:
        split: One of 'train', 'val', 'test'

    Returns:
        dict with 'images' and 'labels' Path objects
    """
    return {
        "images": YOLO_IMAGES_DIR / split,
        "labels": YOLO_LABELS_DIR / split,
    }
