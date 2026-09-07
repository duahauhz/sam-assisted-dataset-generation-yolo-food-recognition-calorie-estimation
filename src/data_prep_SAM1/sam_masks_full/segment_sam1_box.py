"""Run SAM1 (vit_b) with a box prompt on every ECUSTFD image.

For every annotated object (food + coin) the script predicts a single mask
using its ground-truth bounding box from VOC XML as the prompt.  When a
patched XML exists in ``data/processed/bbox_full/Annotations_patched/`` it
takes priority so the 9 manually-corrected images are segment-ed with the
correct bboxes.

Output (under ``data/processed/sam_masks_full``):
    masks/<stem>.npy
        One file per image, a pickle-compatible dict with:
            {
                "image_shape": [H, W, 3],
                "objects": [
                    {"class": str, "bbox_voc": [x1,y1,x2,y2],
                     "mask": (H,W) bool, "score": float, "area": int},
                    ...
                ],
                "source": "patched_gt_box_prompt" | "raw_gt_box_prompt",
            }
        Every annotated object of the image is included - both food and
        coin - so the file is a complete per-image mask bundle.

Usage:
    python src/data_prep_SAM1/sam_masks_full/segment_sam1_box.py --clean
    python src/data_prep_SAM1/sam_masks_full/segment_sam1_box.py --workers 4
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_XML_DIR = PROJECT_ROOT / "data" / "raw" / "ECUSTFD" / "Annotations"
IMG_DIR = PROJECT_ROOT / "data" / "raw" / "ECUSTFD" / "JPEGImages"
PATCHED_XML_DIR = PROJECT_ROOT / "data" / "processed" / "bbox_full" / "Annotations_patched"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "sam_masks_full"
MASKS_DIR = OUT_DIR / "masks"
LOG_PATH = OUT_DIR / "run.log"

SAM_CHECKPOINT = PROJECT_ROOT / "models" / "sam" / "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

SOURCE_PATCHED = "patched_gt_box_prompt"
SOURCE_RAW = "raw_gt_box_prompt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_image(stem: str) -> Path | None:
    for ext in (".JPG", ".jpg", ".jpeg", ".png"):
        candidate = IMG_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def resolve_xml_path(stem: str) -> tuple[Path, str] | None:
    """Pick the XML for ``stem`` (patched takes priority) and report its source."""
    patched = PATCHED_XML_DIR / f"{stem}.xml"
    if patched.exists():
        return patched, SOURCE_PATCHED
    raw = RAW_XML_DIR / f"{stem}.xml"
    if raw.exists():
        return raw, SOURCE_RAW
    return None


def parse_voc_objects(xml_path: Path) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Parse VOC XML and return ``[(class, bbox_xyxy 1-indexed), ...]``.

    Bbox convention is **1-indexed** (matches VOC convention).
    Order follows XML ``<object>`` order.
    """
    root = ET.parse(str(xml_path)).getroot()
    out: list[tuple[str, tuple[int, int, int, int]]] = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        box_node = obj.find("bndbox")
        if name_node is None or box_node is None or not name_node.text:
            continue
        try:
            coords = tuple(int(float(box_node.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax"))
        except (AttributeError, ValueError, TypeError):
            continue
        out.append((name_node.text.strip(), coords))
    return out


# ---------------------------------------------------------------------------
# Worker process: each worker loads SAM1 once and processes an assigned batch
# ---------------------------------------------------------------------------

def _worker_init(model_type: str, checkpoint: str, device: str) -> None:
    """Initialize a SAM1 predictor inside each child process (once)."""
    global _PREDICTOR
    from segment_anything import sam_model_registry, SamPredictor  # imported lazily

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    _PREDICTOR = SamPredictor(sam)


def _process_one(stem: str, out_dir: Path) -> tuple[str, str, int, int]:
    """Process a single stem inside a worker.

    Returns ``(stem, source, n_objects, n_segments)`` for logging.  Any
    exception is caught and reported via the ``stem`` plus an error string.
    """
    global _PREDICTOR
    try:
        xml = resolve_xml_path(stem)
        if xml is None:
            return (stem, "<no-xml>", 0, 0)
        xml_path, source = xml
        objects = parse_voc_objects(xml_path)
        if not objects:
            return (stem, f"{source}:no_objects", 0, 0)

        image_path = find_image(stem)
        if image_path is None:
            return (stem, f"{source}:no_image", len(objects), 0)

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return (stem, f"{source}:read_fail", len(objects), 0)
        H, W = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        _PREDICTOR.set_image(image_rgb)

        results: list[dict] = []
        for class_name, (x1, y1, x2, y2) in objects:
            # Convert VOC 1-indexed bbox to SAM's 0-indexed xyxy and clamp.
            x1c = max(0, min(W - 1, x1 - 1))
            y1c = max(0, min(H - 1, y1 - 1))
            x2c = max(0, min(W, x2))      # +1 then clamp covers full pixel range
            y2c = max(0, min(H, y2))
            if x2c <= x1c or y2c <= y1c:
                # Box collapsed after clamping - skip rather than crash.
                continue
            box = np.array([x1c, y1c, x2c, y2c], dtype=np.float32)
            masks, scores, _ = _PREDICTOR.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )
            mask = masks[0].astype(bool)
            results.append({
                "class": class_name,
                "bbox_voc": [int(x1), int(y1), int(x2), int(y2)],
                "mask": mask,
                "score": float(scores[0]),
                "area": int(mask.sum()),
            })

        if not results:
            return (stem, f"{source}:all_box_collapsed", len(objects), 0)

        payload = {
            "image_shape": [int(H), int(W), 3],
            "objects": results,
            "source": source,
        }
        out_path = out_dir / f"{stem}.npy"
        np.save(str(out_path), payload, allow_pickle=True)
        return (stem, source, len(objects), len(results))
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        return (stem, f"error:{tb.splitlines()[-2] if tb else 'unknown'}", 0, 0)


def _worker_main(stems: list[str], out_dir: Path, log_lock, init_args) -> None:
    """Worker entry: init SAM1 once, then process assigned stems."""
    _worker_init(*init_args)
    for stem in stems:
        result = _process_one(stem, out_dir)
        with log_lock:
            with LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%H:%M:%S')} {result[0]:24s} {result[1]:32s} objects={result[2]} segs={result[3]}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true",
                        help="wipe sam_masks_full/ (except run.log and README.md) before running")
    parser.add_argument("--workers", type=int, default=max(1, min(8, mp.cpu_count())),
                        help="number of worker processes (each loads SAM1 separately)")
    parser.add_argument("--limit", type=int, default=0,
                        help="process only the first N stems (debug)")
    parser.add_argument("--cpu", action="store_true",
                        help="force CPU even if CUDA is available")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    device = "cpu" if args.cpu or not _cuda_available() else "cuda"

    # ------- Sanity checks ----------------------------------------------------
    if not SAM_CHECKPOINT.exists():
        print(f"[fatal] SAM checkpoint not found: {SAM_CHECKPOINT}", flush=True)
        return 2

    # ------- Optional clean ---------------------------------------------------
    if args.clean and OUT_DIR.exists():
        # Static artifacts we never want to wipe.
        preserve = {"run.log", "README.md"}
        # Per-run logs from the notebook's run_script_step helper live in
        # `logs/run_<timestamp>.log`. Preserve that directory too so
        # re-running this script does not delete the log it is currently
        # writing to.
        for entry in OUT_DIR.iterdir():
            if entry.name in preserve:
                continue
            if entry.name == "logs":
                continue
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink(missing_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    # ------- Enumerate stems --------------------------------------------------
    all_xml_paths = sorted(RAW_XML_DIR.glob("*.xml"))
    stems = [p.stem for p in all_xml_paths]
    if args.limit > 0:
        stems = stems[:args.limit]
    print(f"[scan] xml stems: {len(stems)}  workers: {args.workers}  device: {device}", flush=True)

    # ------- Source breakdown before run -------------------------------------
    patched_count = sum(1 for s in stems if (PATCHED_XML_DIR / f"{s}.xml").exists())
    raw_count = len(stems) - patched_count
    print(f"[scan] patched source: {patched_count}  raw source: {raw_count}", flush=True)

    # ------- Log header -------------------------------------------------------
    if not LOG_PATH.exists():
        LOG_PATH.write_text("", encoding="utf-8")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n--- segment run  workers={args.workers} device={device} "
            f"clean={args.clean} limit={args.limit} ts={time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
        )

    if not stems:
        return 0

    # ------- Run workers ------------------------------------------------------
    init_args = (SAM_MODEL_TYPE, str(SAM_CHECKPOINT), device)

    if args.workers == 1:
        # Single-process path: no spawn overhead, simpler logging.
        _worker_init(*init_args)
        log_lock = _FakeLock()
        t_start = time.time()
        for idx, stem in enumerate(stems, start=1):
            result = _process_one(stem, MASKS_DIR)
            stem_name, source, n_obj, n_segs = result
            line = (
                f"  [{idx:>4d}/{len(stems)}] {stem_name:24s} src={source:32s} "
                f"obj={n_obj} seg={n_segs}"
            )
            print(line, flush=True)
            with LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(
                    f"{time.strftime('%H:%M:%S')} {stem_name:24s} src={source:32s} "
                    f"objects={n_obj} segs={n_segs}\n"
                )
            if idx % 50 == 0 or idx == len(stems):
                elapsed = time.time() - t_start
                rate = idx / max(elapsed, 0.001)
                eta = (len(stems) - idx) / max(rate, 0.001)
                print(
                    f"        progress {idx}/{len(stems)}  "
                    f"elapsed={elapsed:.1f}s rate={rate:.2f} img/s eta={eta:.1f}s",
                    flush=True,
                )
    else:
        # Multi-process path: split stems across workers; each loads its own SAM1.
        chunks = [stems[i::args.workers] for i in range(args.workers)]
        chunks = [c for c in chunks if c]
        log_lock = mp.Lock()
        procs = []
        for chunk in chunks:
            p = mp.Process(
                target=_worker_main,
                args=(chunk, MASKS_DIR, log_lock, init_args),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    # ------- Summary ----------------------------------------------------------
    n_masks = sum(1 for _ in MASKS_DIR.glob("*.npy"))
    print(f"\n[done] masks generated: {n_masks}/{len(stems)}", flush=True)
    print(f"[done] output dir:      {OUT_DIR}", flush=True)
    print(f"[done] log:             {LOG_PATH}", flush=True)
    return 0


class _FakeLock:
    """No-op context manager used when workers=1."""
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _cuda_available() -> bool:
    try:
        import torch  # noqa: F401
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
