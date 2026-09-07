"""ECUSTFD VOC-format annotations -> PyTorch Faster R-CNN dataset."""

from .voc_to_coco_dataset import (
    ECUSTFDDetectionDataset,
    VOCAnnotations,
    get_transform,
)

__all__ = [
    "ECUSTFDDetectionDataset",
    "VOCAnnotations",
    "get_transform",
]
