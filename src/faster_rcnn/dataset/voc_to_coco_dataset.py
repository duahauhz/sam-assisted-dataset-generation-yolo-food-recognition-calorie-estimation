# -*- coding: utf-8 -*-
"""ECUSTFD VOC XML annotations -> PyTorch Detection Dataset.

Architecture mirrors ``src/yolo_seg/dataset/convert_to_yolo_seg.py``:

  1. Parse raw VOC XML files into Python objects (VOCAnnotations).
  2. Wrap in ``torch.utils.data.Dataset`` subclass compatible with
     torchvision's ``make_global_color_transform`` + Faster R-CNN training.
  3. Export a ``get_transform`` helper so training scripts stay minimal.

Dataset layout expected on disk::

    data/raw/ECUSTFD/
        Annotations/   <- VOC XML files (already exist)
        JPEGImages/    <- source images (already exist)

Outputs are held entirely in memory / the DataLoader; no intermediate
files are written.

Class table (21 classes = 19 foods + coin) must stay in sync with
``src/constants.py`` and ``src/segmentation_runtime/yolo_infer.py``.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.utils.data
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import functional as F, InterpolationMode
from torchvision.transforms.v2 import Transform

# ---------------------------------------------------------------------------
# Class table — same order as YOLO26-seg ecustfd-seg.yaml
# ---------------------------------------------------------------------------
_CLASS_NAMES: List[str] = [
    "apple", "banana", "bread", "bun", "coin",
    "doughnut", "egg", "fired_dough_twist", "grape",
    "lemon", "litchi", "mango", "mooncake",
    "orange", "peach", "pear", "plum", "qiwi",
    "sachima", "tomato",
]
_CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(_CLASS_NAMES)}


# ---------------------------------------------------------------------------
# VOC XML parser
# ---------------------------------------------------------------------------

class VOCAnnotations:
    """Container for parsed VOC-style XML annotations."""

    def __init__(self, image_path: str, width: int, height: int,
                 depth: int, segmented: int, objects: List[Dict[str, Any]]):
        self.image_path = image_path
        self.width = width
        self.height = height
        self.depth = depth
        self.segmented = segmented
        self.objects = objects

    @staticmethod
    def from_xml(xml_path: Path, images_dir: Path) -> "VOCAnnotations":
        """Parse a single Pascal VOC XML annotation file."""
        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        filename = root.find("filename").text
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        depth = int(size.find("depth").text)
        segmented_el = root.find("segmented")
        segmented = int(segmented_el.text) if segmented_el is not None else 0

        objects: List[Dict[str, Any]] = []
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            cls_name = str(obj.find("name").text).strip()
            difficult = int(obj.find("difficult").text) if obj.find("difficult") is not None else 0

            objects.append({
                "class_name": cls_name,
                "bbox": [xmin, ymin, xmax, ymax],   # [x1, y1, x2, y2] in pixels
                "difficult": difficult,
            })

        image_path = str(images_dir / filename)
        return VOCAnnotations(image_path, width, height, depth, segmented, objects)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ECUSTFDDetectionDataset(torch.utils.data.Dataset):
    """PyTorch dataset compatible with torchvision Faster R-CNN training.

    Args:
        image_paths: list of image file paths.
        annotations: list of VOCAnnotations objects (same length as image_paths).
        transform: optional torchvision Transform applied after loading.
        min_area: skip boxes smaller than this many px² (removes noise).
    """

    def __init__(
        self,
        image_paths: List[Path],
        annotations: List[VOCAnnotations],
        transform: Optional[Transform] = None,
        min_area: float = 20.0,
    ):
        assert len(image_paths) == len(annotations), (
            f"Mismatch: {len(image_paths)} images vs {len(annotations)} annotations"
        )
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform
        self.min_area = min_area

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[tv_tensors.Image, Dict[str, Any]]:
        img_path = self.image_paths[idx]
        ann = self.annotations[idx]

        # Load image
        image = Image.open(ann.image_path).convert("RGB")
        w, h = image.size

        # Build box list
        boxes: List[List[float]] = []
        labels: List[int] = []

        for obj in ann.objects:
            cls = obj["class_name"]
            if cls not in _CLASS_TO_IDX:
                continue
            x1, y1, x2, y2 = obj["bbox"]
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_area:
                continue
            # Clip to image bounds
            x1 = max(0.0, min(x1, w))
            y1 = max(0.0, min(y1, h))
            x2 = max(0.0, min(x2, w))
            y2 = max(0.0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(_CLASS_TO_IDX[cls])

        if not boxes:
            # Empty image — create a dummy box so the DataLoader doesn't crash
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [0]

        boxes_t = tv_tensors.BoundingBoxes(
            boxes, format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(h, w)
        )
        labels_t = torch.tensor(labels, dtype=torch.int64)

        # Wrap as tv_tensors.Image
        image_t = tv_tensors.Image(F.pil_to_tensor(image))

        target: Dict[str, Any] = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": torch.tensor([(b[2]-b[0])*(b[3]-b[1]) for b in boxes], dtype=torch.float32),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }

        if self.transform is not None:
            image_t, target = self.transform(image_t, target)

        return image_t, target


# ---------------------------------------------------------------------------
# Transform helpers (match torchvision detection reference training style)
# ---------------------------------------------------------------------------

class ToTensor:
    """Convert PIL Image to tensor (0-1 float)."""
    def __call__(self, image: torch.Tensor, target: Dict[str, Any]):
        image = F.convert_image_dtype(image, dtype=torch.float32)
        return image, target


class Normalize:
    """ImageNet normalization + ToTensor combined."""
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __call__(self, image: torch.Tensor, target: Dict[str, Any]):
        image = F.convert_image_dtype(image, dtype=torch.float32)
        image = F.normalize(image, mean=self._MEAN, std=self._STD)
        return image, target


def get_transform(train: bool = False) -> Callable:
    """Return the torchvision-style transform pipeline for Faster R-CNN.

    Training augmentation mirrors what torchvision detection reference uses.
    """
    transforms: List[Callable] = [ToTensor()]
    if train:
        # Random horizontal flip
        transforms.append(_RandomHorizontalFlip())
    transforms.append(Normalize())
    return _ComposeTransforms(transforms)


class _RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: torch.Tensor, target: Dict[str, Any]):
        if torch.rand(1) < self.p:
            image = F.horizontal_flip(image)
            h, w = image.shape[-2:]
            boxes = target["boxes"]
            if isinstance(boxes, tv_tensors.BoundingBoxes):
                boxes_fmt = boxes.format
                canvas = boxes.canvas_size
                flipped = tv_tensors.BoundingBoxes(
                    torch.stack([w - boxes[..., 2],
                                 boxes[..., 1],
                                 w - boxes[..., 0],
                                 boxes[..., 3]], dim=-1),
                    format=boxes_fmt,
                    canvas_size=canvas,
                )
                target = {**target, "boxes": flipped}
        return image, target


class _ComposeTransforms:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, image: torch.Tensor, target: Dict[str, Any]):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# ---------------------------------------------------------------------------
# CLI for debugging / one-off dataset generation
# ---------------------------------------------------------------------------

def _main():
    import argparse, glob, random

    ap = argparse.ArgumentParser(description="Inspect ECUSTFD VOC annotations")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"],
                    help="Which ImageSet split to load")
    ap.add_argument("--max-samples", type=int, default=5)
    args = ap.parse_args()

    ROOT = Path("E:/AI_Research/dlt8")
    ANNOTATIONS_DIR = ROOT / "data/raw/ECUSTFD/Annotations"
    IMAGES_DIR = ROOT / "data/raw/ECUSTFD/JPEGImages"
    IMAGESETS_DIR = ROOT / "data/raw/ECUSTFD/ImageSets/Main"

    split_file = IMAGESETS_DIR / f"{args.split}.txt"
    stems = split_file.read_text(encoding="utf-8").splitlines()
    random.shuffle(stems)
    stems = stems[: args.max_samples]

    xml_paths = [ANNOTATIONS_DIR / f"{s}.xml" for s in stems]
    image_paths = [IMAGES_DIR / f"{s}.JPG" for s in stems]
    annotations = [VOCAnnotations.from_xml(x, IMAGES_DIR) for x in xml_paths]

    dataset = ECUSTFDDetectionDataset(image_paths, annotations, transform=get_transform(train=False))

    for idx in range(len(dataset)):
        img, target = dataset[idx]
        print(f"\n[{idx}] {image_paths[idx].name}")
        print(f"  Image shape: {img.shape}")
        print(f"  Boxes: {target['boxes'].shape[0]}")
        print(f"  Labels: {target['labels'].tolist()}")


if __name__ == "__main__":
    _main()
