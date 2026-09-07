"""YOLO-seg inference wrapper for the end-to-end pipeline.

Thin wrapper around Ultralytics that:
1. Loads the trained YOLO26-seg weights.
2. Runs ``model.predict`` on a single image (or a folder).
3. Returns a list of standardised detection dicts with keys
   ``class_name``, ``conf``, ``bbox`` (xyxy), ``mask`` (full-image
   boolean mask).

This is the single place that knows about Ultralytics. The rest of
the pipeline only consumes the plain dicts returned here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

try:
    from ultralytics import YOLO  # type: ignore
except ImportError as _exc:  # pragma: no cover
    _YOLO_IMPORT_ERROR = _exc
else:
    _YOLO_IMPORT_ERROR = None


_CLASS_NAMES: list[str] = [
    "apple", "banana", "bread", "bun", "coin",
    "doughnut", "egg", "fired_dough_twist", "grape",
    "lemon", "litchi", "mango", "mooncake",
    "orange", "peach", "pear", "plum", "qiwi",
    "sachima", "tomato",
]
# Note: the original ECUSTFD class set has 19 foods + coin (20 classes).
# Some of the user's bbox-override files use ``kiwi`` as a label for
# fruit that ECUSTFD calls ``qiwi``; both refer to kiwi fruit. We
# normalise to ``qiwi`` here (matches ``src/constants.py::FOOD_CLASSES``).


def _ensure_ultralytics() -> None:
    if _YOLO_IMPORT_ERROR is not None:
        raise ImportError(
            "ultralytics is required for YOLO-seg inference. "
            "Install it via `pip install ultralytics`. "
            f"Original error: {_YOLO_IMPORT_ERROR}"
        )


def load_yolo_seg(weights: str | Path, device: str | int | None = None):
    """Load a YOLO-seg model from the given weights path.

    Args:
        weights: path to a ``.pt`` file (e.g. ``runs/yolo_seg/.../best.pt``).
        device: forwarded to ``model.predict(device=...)`` on each call.
            Common values: ``0`` (GPU 0), ``"cpu"``. ``None`` lets
            Ultralytics auto-select.

    Returns:
        An ``ultralytics.YOLO`` model instance.
    """
    _ensure_ultralytics()
    model = YOLO(str(weights))
    # NOTE: we do NOT call ``model.to(device)`` here because Ultralytics
    # expects the device argument on ``predict()`` and ``.to("0")`` raises
    # in plain torch. Callers should pass ``device`` to ``predict_one``.
    _ = device
    return model

# Convert kết quả từ Ultralytics sang list of detection dicts
def _results_to_detections(
    result,# Đối tượng chứa kết quả dự đoán từ YOLO
    class_names: list[str],# Danh sách tên các lớp
    orig_w: int,# Chiều rộng ảnh gốc
    orig_h: int,# Chiều cao ảnh gốc
    imgsz: int,# Kích thước ảnh đầu vào mô hình (ví dụ: 480)
    stride: int = 32,# Bước nhảy của mô hình (thường là 32 cho YOLO)
    auto: bool = True,# Tự động điều chỉnh kích thước ảnh
) -> List[dict]:
    """Convert a single Ultralytics result to a list of detection dicts.

    Ultralytics returns bounding-box coordinates in *original image* pixel
    space, but the segmentation masks are in the *letterboxed* canvas
    space (the image is resized to fit ``imgsz`` × ``imgsz`` while
    preserving aspect ratio, then padded on the short side).  We replicate
    the exact ``ultralytics.data.augment.LetterBox.get_params`` transform
    here so that every bbox is converted into mask-space and
    ``crop_mask_to_bbox`` produces non-empty crops.

    For inference (``model.predict``), Ultralytics uses ``auto=True``
    which pads to the **nearest multiple of the model stride** instead
    of padding all the way to ``imgsz``.  This is what produces the
    observed 384x480 mask shape for an 816x612 input at ``imgsz=480``.

    Returns an empty list if the model found no objects.
    """
    if result.masks is None or result.boxes is None:
        return []

    boxes = result.boxes # Hộp giới hạn (bounding boxes)
    masks = result.masks # Mặt nạ phân đoạn
    n = len(boxes) # Số lượng vật thể phát hiện được

    # Letterbox transform — verbatim port of Ultralytics' LetterBox logic.
    r = min(imgsz / orig_h, imgsz / orig_w) # Tỷ lệ co giãn ảnh
    new_unpad_w = round(orig_w * r) # Chiều rộng ảnh mới sau khi co giãn
    new_unpad_h = round(orig_h * r) # Chiều cao ảnh mới sau khi co giãn
    dw = imgsz - new_unpad_w # Khoảng cách padding theo chiều rộng
    dh = imgsz - new_unpad_h # Khoảng cách padding theo chiều cao
    if auto: # Nếu tự động điều chỉnh kích thước ảnh
        dw = float(np.mod(dw, stride)) # Padding đến bội số gần nhất của stride
        dh = float(np.mod(dh, stride))
    top = dh / 2.0 # Padding trên cùng
    left = dw / 2.0 # Padding bên trái

    def _to_letterboxed(x: float, y: float) -> tuple[float, float]:
        return (x * r + left, y * r + top) # Chuyển đổi tọa độ từ ảnh gốc sang ảnh letterboxed

    out: List[dict] = []
    for i in range(n):
        cls_idx = int(boxes.cls[i].item()) # Chỉ số lớp
        conf = float(boxes.conf[i].item()) # Độ tin cậy
        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()] # Tọa độ hộp giới hạn
        # Convert from original-image coords to letterboxed mask coords.
        x1_lb, y1_lb = _to_letterboxed(x1, y1) # Chuyển đổi tọa độ từ ảnh gốc sang ảnh letterboxed
        x2_lb, y2_lb = _to_letterboxed(x2, y2) # Chuyển đổi tọa độ từ ảnh gốc sang ảnh letterboxed
        m = masks.data[i].cpu().numpy() # Mặt nạ phân đoạn
        out.append(
            {
                "class_name": class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx), # Tên lớp
                "conf": conf, # Độ tin cậy
                "bbox": (x1_lb, y1_lb, x2_lb, y2_lb), # Tọa độ hộp giới hạn
                "mask": m.astype(bool), # Mặt nạ phân đoạn
            }
        )
    return out


def predict_one(
    model, # Mô hình YOLO đã được load
    image_path: str | Path, # Đường dẫn đến ảnh
    conf: float = 0.25, # Ngưỡng tin cậy
    iou: float = 0.7, # Ngưỡng IoU
    imgsz: int = 480, # Kích thước ảnh đầu vào mô hình
    class_names: Optional[list[str]] = None, # Danh sách tên các lớp
) -> List[dict]:
    """Run inference on a single image.

    Args:
        model: an Ultralytics ``YOLO`` model (from ``load_yolo_seg``).
        image_path: path to the image file.
        conf: confidence threshold for inference.
        iou: NMS IoU threshold.
        imgsz: inference image size (the trained checkpoint used 480).
        class_names: optional override for the class list; defaults to
            the 21-class YOLO-seg set.

    Returns:
        List of detection dicts, one per detected instance.
        **Important:** ``bbox`` values are in *letterboxed* mask space
        (same coordinate system as the returned ``mask`` array), so callers
        can crop directly without any additional coordinate transform.
    """
    _ensure_ultralytics()
    from PIL import Image as _PILImage

    cn = class_names if class_names is not None else _CLASS_NAMES # Danh sách tên các lớp
    # Read original dimensions before inference so we can convert bboxes.
    with _PILImage.open(str(image_path)) as _img: # Mở ảnh
        orig_w, orig_h = _img.size # Lấy kích thước ảnh gốc
    results = model.predict(# Thực hiện dự đoán
        source=str(image_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )
    if not results: # Nếu không có kết quả
        return [] # Trả về danh sách rỗng
    return _results_to_detections(results[0], cn, orig_w, orig_h, imgsz) # Trả về danh sách các kết quả


def predict_many(
    model,# Mô hình YOLO đã được load
    image_paths: Iterable[str | Path], # Danh sách các đường dẫn đến ảnh
    conf: float = 0.25, # Ngưỡng tin cậy
    iou: float = 0.7, # Ngưỡng IoU
    imgsz: int = 480, # Kích thước ảnh đầu vào mô hình
    class_names: Optional[list[str]] = None, # Danh sách tên các lớp
) -> dict[str, List[dict]]:
    """Run inference on many images and return a dict keyed by path.

    The keys are the string forms of the input paths. Each value is a
    list of detection dicts.
    """
    _ensure_ultralytics()
    cn = class_names if class_names is not None else _CLASS_NAMES # Danh sách tên các lớp
    out: dict[str, List[dict]] = {} # Dictionary để lưu kết quả
    for p in image_paths: # Duyệt qua danh sách các đường dẫn đến ảnh
        dets = predict_one(model, p, conf=conf, iou=iou, imgsz=imgsz, class_names=cn) # Thực hiện dự đoán
        out[str(p)] = dets # Lưu kết quả vào dictionary
    return out # Trả về dictionary chứa kết quả


__all__ = [
    "load_yolo_seg",
    "predict_one",
    "predict_many",
    "_CLASS_NAMES",
]
