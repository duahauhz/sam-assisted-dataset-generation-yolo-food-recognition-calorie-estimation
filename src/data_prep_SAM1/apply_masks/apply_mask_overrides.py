"""Apply manually-drawn mask overrides for a subset of ECUSTFD images.

Background
----------
``segment_sam1_box.py`` produces ``masks/<stem>.npy`` with a payload:

    {
        "image_shape": [H, W, 3],
        "objects": [
            {"class": str, "bbox_voc": [x1,y1,x2,y2],
             "mask": (H,W) bool, "score": float, "area": int},
            ...
        ],
        "source": "patched_gt_box_prompt" | "raw_gt_box_prompt" | ...
    }

When SAM1 produces a poor mask for a single object in an image (e.g. bbox
covers background so SAM bakes the background), we want to keep every other
object in the file untouched and replace only the failing object's ``mask``
(and recompute ``area`` and ``bbox_voc`` if the user provided one).

This script reads an ``OVERRIDES`` table analogous to
``visual_bbox/apply_bbox_overrides.py``. Each entry targets ONE stem and ONE
object (identified by ``class``) within that stem's mask file. The manual
mask comes from a PNG that the annotator produced with an external tool
(Labelme / CVAT / GIMP / etc.) so that:

  * the masking interface is a tool the paper reviewer will recognise;
  * display settings (opacity, zoom, bbox reference) are documented by the
    chosen tool rather than re-implemented in this codebase.

Convention
----------
* The PNG path is **either** the exact ``image_shape`` of the stem **or**
  any size; the loader resizes with ``cv2.INTER_NEAREST`` so the bool grid
  stays sharp.
* The PNG is read as grayscale; ``pixel > 127`` => foreground. This matches
  how Labelme exports a "label PNG" (foreground = white).
* Only ``mask`` / ``area`` / ``bbox_voc`` (optional) / ``score`` are touched.
  Other fields in the object dict, plus the rest of the file's objects,
  are preserved as-is.
* ``source`` is rewritten to ``"manual_mask_override"`` so downstream viz
  can flag it; other stems keep their original source untouched.
* The patched ``.npy`` is written bytewise via ``np.save(..., allow_pickle=True)``
  so it re-loads identically. A SHA-256 hash before/after is printed for
  every modified file.

Usage
-----
    # dry-run: print what would change, don't write
    python src/data_prep_SAM1/apply_masks/apply_mask_overrides.py --dry-run

    # write patched masks into /TMP/masks_patched/ (safe to diff)
    python src/data_prep_SAM1/apply_masks/apply_mask_overrides.py --to-tmp

    # in-place: overwrite data/processed/sam_masks_full/masks/<stem>.npy
    python src/data_prep_SAM1/apply_masks/apply_mask_overrides.py

    # run a single stem only
    python src/data_prep_SAM1/apply_masks/apply_mask_overrides.py --only grape001T(5)

After in-place run, regenerate the visualization/CSV layer:
    python src/data_prep_SAM1/sam_masks_full/visualize_results.py --clean
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# Force UTF-8 on stdout so Unicode in pipeline messages survives on cp1252
# Windows consoles (default for PowerShell / cmd.exe).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ----------------------------------------------------------------------------
# Paths (mirror apply_bbox_overrides.py layout)
# ----------------------------------------------------------------------------
ROOT = Path("E:/AI_Research/dlt8")
MASKS_DIR = ROOT / "data/processed/sam_masks_full/masks"
DEFAULT_OUTPUT_DIR = ROOT / "data/processed/sam_masks_full/masks"  # in-place
TMP_OUTPUT_DIR = ROOT / "TMP/masks_patched"  # safe scratch dir

OVERRIDE_SOURCE_TAG = "manual_mask_override"


# ----------------------------------------------------------------------------
# OVERRIDES table
#
# Each entry:
#   stem          : str       — e.g. "grape001T(5)"
#   class         : str       — the object class to override inside the stem
#                                (e.g. "grape"). Must match exactly one
#                                entry in objects[]. If multiple objects
#                                of the same class exist, target_index
#                                disambiguates.
#   target_index  : int       — (optional, default 0) index inside
#                                objects[] filtered by class.
#   mask_png      : str       — relative-to-ROOT path to the manual
#                                grayscale PNG (foreground = white > 127).
#                                PNG can be any size; loader resizes.
#   new_bbox_xyxy : tuple     — (optional) 1-indexed VOC xyxy. If provided,
#                                overrides the existing bbox_voc on the same
#                                object so visualizations use the correct
#                                box. Use None to keep the existing bbox.
#   note          : str       — (optional) free-text comment.
# ----------------------------------------------------------------------------
OVERRIDES: dict[str, dict] = {
    # ----- grape001T(5): SAM mask bám background vì bbox VOC quá rộng -----
    # Ảnh raw:  data/raw/ECUSTFD/JPEGImages/grape001T(5).JPG  (612x816)
    # Bbox VOC cũ (grape): (262, 4) -> (686, 446)   — rộng, trải từ mép trên
    # Bbox coin giữ nguyên VOC: (119, 90) -> (177, 152)
    # User vẽ PNG mask bằng GIMP 3.2.4, xuất thành:
    #   data/annotation/manual_masks/grape001T(5)_grape.png
    "grape001T(5)": {
        "class": "grape",
        "target_index": 0,
        "mask_png": "data/annotation/manual_masks/grape001T(5)_grape.png",
        "new_bbox_xyxy": None,  # nếu đo lại được thì ghi 4 số VOC ở đây
        "note": "SAM bbox quá rộng, mask bám background. Thay mask thủ công.",
    },
    # Thêm ảnh mới bằng cách copy entry trên + đổi stem / mask_png / note.
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _sha256(arr: np.ndarray) -> str:
    """Stable hash for a numpy payload (pickled bytes)."""
    return hashlib.sha256(arr.tobytes() if isinstance(arr, np.ndarray) else b"").hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "<missing>"


def _load_payload(stem: str) -> dict:
    """Read ``masks/<stem>.npy`` and assert it's a well-formed payload."""
    p = MASKS_DIR / f"{stem}.npy"
    obj = np.load(str(p), allow_pickle=True).item()
    if not isinstance(obj, dict) or "objects" not in obj or "image_shape" not in obj:
        raise ValueError(f"{p}: not a SAM payload (missing image_shape/objects)")
    return obj


def _resolve_object_index(payload: dict, cls: str, target_index: int) -> int:
    """Return the absolute index in ``payload['objects']`` for the override target."""
    hits = [i for i, o in enumerate(payload["objects"]) if o.get("class") == cls]
    if not hits:
        raise ValueError(f"class={cls!r} not present in objects: {[o.get('class') for o in payload['objects']]}")
    if target_index < 0 or target_index >= len(hits):
        raise ValueError(
            f"target_index={target_index} out of range for class={cls!r} "
            f"(matched {len(hits)} object(s): indices {hits})"
        )
    return hits[target_index]


def _load_manual_mask(png_rel_path: str, target_hw: Tuple[int, int]) -> np.ndarray:
    """Load a user-drawn grayscale PNG and return a bool array of shape target_hw.

    - Reads as grayscale.
    - ``pixel > 127`` => foreground (matches Labelme "label PNG" convention).
    - Resizes with ``cv2.INTER_NEAREST`` so the foreground stays discrete.
    - Flattens anything with >1 channel defensively.
    """
    abs_path = ROOT / png_rel_path
    if not abs_path.exists():
        raise FileNotFoundError(f"manual mask PNG not found: {abs_path}")
    img = cv2.imread(str(abs_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cv2 failed to read: {abs_path}")
    H, W = target_hw
    if img.shape != (H, W):
        print(
            f"[resize] {abs_path.name}: {img.shape} -> ({H},{W}) (INTER_NEAREST)",
            flush=True,
        )
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    if img.ndim > 2:
        img = img[..., 0]
    return img > 127


# ----------------------------------------------------------------------------
# Patch a single stem
# ----------------------------------------------------------------------------
def patch_one(stem: str, dry_run: bool, in_place: bool) -> Tuple[bool, str]:
    spec = OVERRIDES[stem]
    cls = spec["class"]
    target_index = int(spec.get("target_index", 0))
    png_rel = spec["mask_png"]
    new_bbox = spec.get("new_bbox_xyxy")
    note = spec.get("note", "")

    src_path = MASKS_DIR / f"{stem}.npy"
    if not src_path.exists():
        return False, f"[SKIP] {stem}: source mask missing: {src_path}"
    if in_place:
        out_path = src_path
    else:
        out_path = TMP_OUTPUT_DIR / f"{stem}.npy"

    # ---- read source payload ----
    payload = _load_payload(stem)
    H, W = int(payload["image_shape"][0]), int(payload["image_shape"][1])

    # ---- resolve target object ----
    obj_idx = _resolve_object_index(payload, cls, target_index)
    obj_old = payload["objects"][obj_idx]

    # ---- build new mask + bbox ----
    new_mask = _load_manual_mask(png_rel, (H, W))
    new_area = int(new_mask.sum())
    if new_area == 0:
        return False, f"[SKIP] {stem}: manual mask is empty (all zeros)"

    new_bbox_voc = list(obj_old["bbox_voc"])
    if new_bbox is not None and len(new_bbox) == 4:
        new_bbox_voc = [int(v) for v in new_bbox]

    # ---- build patched payload (do not mutate original dict copy semantics) ----
    patched = {
        "image_shape": payload["image_shape"],
        "objects": [
            {**o, "mask": new_mask.astype(bool)
                       if i == obj_idx else o["mask"].astype(bool),
             "area": new_area if i == obj_idx else int(o["area"]),
             "bbox_voc": new_bbox_voc if i == obj_idx else list(o["bbox_voc"])}
            for i, o in enumerate(payload["objects"])
        ],
        "source": OVERRIDE_SOURCE_TAG,
    }
    # score is SAM's own confidence and does not apply to a manual mask.
    # We keep the original score so the visualisation layer doesn't crash on
    # missing keys, but tag it via ``source`` so the report can flag it.

    msg = (
        f"[OK ] {stem}  class={cls}  obj_idx={obj_idx}  "
        f"area: {obj_old['area']} -> {new_area}  "
        f"bbox_voc: {obj_old['bbox_voc']} -> {patched['objects'][obj_idx]['bbox_voc']}  "
        f"src: {payload.get('source','?')} -> {OVERRIDE_SOURCE_TAG}  "
        f"png={png_rel}"
    )
    if note:
        msg += f"  // {note}"

    if dry_run:
        # Sanity: write a temp copy and round-trip read it.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tf:
            tmp = Path(tf.name)
        np.save(str(tmp), patched, allow_pickle=True)
        rt = np.load(str(tmp), allow_pickle=True).item()
        ok_roundtrip = (
            isinstance(rt, dict)
            and len(rt["objects"]) == len(patched["objects"])
            and rt["objects"][obj_idx]["mask"].shape == (H, W)
            and rt["objects"][obj_idx]["mask"].sum() == new_area
            and rt["source"] == OVERRIDE_SOURCE_TAG
        )
        tmp.unlink()
        suffix = "  [dry-run round-trip OK]" if ok_roundtrip else "  [dry-round-trip FAIL]"
        return (ok_roundtrip, msg + suffix)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pre_hash = _file_sha256(src_path) if in_place else _file_sha256(out_path)
    np.save(str(out_path), patched, allow_pickle=True)
    post_hash = _file_sha256(out_path)
    location = "in-place" if in_place else str(out_path.parent)
    msg += f"  // wrote {location}  sha256: {pre_hash[:12]}.. -> {post_hash[:12]}.."
    return True, msg


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't write; round-trip the patched payload in memory.")
    ap.add_argument("--to-tmp", action="store_true",
                    help=f"Write to {TMP_OUTPUT_DIR} instead of in-place.")
    ap.add_argument("--only", default=None,
                    help="Process only one stem, e.g. --only grape001T(5)")
    args = ap.parse_args()

    targets = list(OVERRIDES.keys())
    if args.only:
        if args.only not in OVERRIDES:
            print(f"[ERR] --only={args.only!r} not in OVERRIDES", file=sys.stderr)
            return 1
        targets = [args.only]

    in_place = not args.to_tmp
    if args.dry_run:
        mode = "DRY-RUN"
    elif in_place:
        mode = f"WRITE-IN-PLACE -> {MASKS_DIR}"
    else:
        mode = f"WRITE -> {TMP_OUTPUT_DIR}"

    print(f"== apply_mask_overrides.py | mode={mode} | n={len(targets)} ==", flush=True)
    n_ok = n_skip = 0
    for stem in targets:
        ok, msg = patch_one(stem, args.dry_run, in_place)
        print(msg, flush=True)
        if ok:
            n_ok += 1
        else:
            n_skip += 1
    print(f"---\nSummary: ok={n_ok}  skip={n_skip}", flush=True)

    if not args.dry_run and in_place and n_ok:
        print(
            "\n[next] regenerate images / grids / CSVs:\n"
            "  python src/data_prep_SAM1/sam_masks_full/visualize_results.py --clean\n",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
