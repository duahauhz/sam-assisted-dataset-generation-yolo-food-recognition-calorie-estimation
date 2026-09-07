"""Visualize every VOC ground-truth bounding box in the ECUSTFD dataset.

The XML annotations are the only source used for boxes. SAM masks are not
loaded, so every annotated object, including coin, is included.

Outputs under ``data/processed/bbox_full``:
    images/<stem>.jpg
        Every source image with all of its ground-truth boxes and class labels.
    grids/<class>_pNN_of_NN.png
        Object-oriented contact sheets. An image may appear more than once
        when it contains multiple objects; the current object is highlighted.
    _image_index.csv
    _index.csv
    _per_class_counts.csv
    README.txt
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
XML_DIR = PROJECT_ROOT / "data" / "raw" / "ECUSTFD" / "Annotations"
IMG_DIR = PROJECT_ROOT / "data" / "raw" / "ECUSTFD" / "JPEGImages"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "bbox_full"
IMAGE_OUT_DIR = OUT_DIR / "images"
GRID_OUT_DIR = OUT_DIR / "grids"
OVERRIDES_DIR = Path(
    __import__("os").environ.get(
        "BBOX_FULL_OVERRIDES_DIR",
        str(OUT_DIR / "Annotations_patched"),
    )
)
OVERRIDES_JSON = Path(
    __import__("os").environ.get(
        "BBOX_FULL_OVERRIDES_JSON",
        str(OUT_DIR / "bbox_overrides.json"),
    )
)

CLASS_ORDER = [
    "apple",
    "banana",
    "bread",
    "bun",
    "coin",
    "doughnut",
    "egg",
    "fired_dough_twist",
    "grape",
    "lemon",
    "litchi",
    "mango",
    "mooncake",
    "orange",
    "peach",
    "pear",
    "plum",
    "qiwi",
    "sachima",
    "tomato",
]

# RGB values keep the palette consistent with the existing SAM viewer.
CLASS_COLORS_RGB = [
    (255, 82, 82),
    (255, 213, 79),
    (211, 153, 102),
    (188, 170, 154),
    (180, 180, 180),
    (244, 67, 54),
    (255, 255, 255),
    (255, 152, 0),
    (156, 39, 176),
    (255, 235, 59),
    (233, 30, 99),
    (255, 193, 7),
    (121, 85, 72),
    (255, 112, 67),
    (255, 171, 145),
    (156, 204, 101),
    (103, 58, 183),
    (139, 195, 74),
    (175, 122, 197),
    (229, 57, 53),
]
CLASS_COLORS_BGR = [tuple(reversed(color)) for color in CLASS_COLORS_RGB]

ObjectRecord = tuple[str, tuple[int, int, int, int]]
AnnotationRecord = tuple[str, list[ObjectRecord]]


def find_image(stem: str) -> Path | None:
    for extension in (".JPG", ".jpg", ".jpeg", ".png"):
        image_path = IMG_DIR / f"{stem}{extension}"
        if image_path.exists():
            return image_path
    return None


def parse_annotation(xml_path: Path) -> AnnotationRecord | None:
    try:
        root = ET.parse(xml_path).getroot()
        objects: list[ObjectRecord] = []
        for object_node in root.findall("object"):
            name_node = object_node.find("name")
            box_node = object_node.find("bndbox")
            if name_node is None or box_node is None or not name_node.text:
                continue
            values = []
            for key in ("xmin", "ymin", "xmax", "ymax"):
                value_node = box_node.find(key)
                if value_node is None or value_node.text is None:
                    raise ValueError(f"missing {key}")
                values.append(int(float(value_node.text)))
            objects.append((name_node.text.strip(), tuple(values)))
        return xml_path.stem, objects
    except (ET.ParseError, OSError, TypeError, ValueError):
        return None


def class_color(class_name: str) -> tuple[int, int, int]:
    try:
        index = CLASS_ORDER.index(class_name)
    except ValueError:
        index = len(CLASS_ORDER)
    return CLASS_COLORS_BGR[index % len(CLASS_COLORS_BGR)]


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width - 1, x2)),
        max(0, min(height - 1, y2)),
    )


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    x = max(0, x)
    y = max(text_height + baseline + 2, y)
    top = y - text_height - baseline - 4
    right = min(image.shape[1] - 1, x + text_width + 6)
    cv2.rectangle(image, (x, top), (right, y), color, -1)
    cv2.putText(image, text, (x + 3, y - baseline - 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def render_image(
    image_path: Path,
    objects: list[ObjectRecord],
    output_path: Path,
    highlight_idx: int | None = None,
    tile_size: int | None = None,
) -> bool:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return False

    height, width = image.shape[:2]
    for object_idx, (class_name, raw_box) in enumerate(objects):
        box = clamp_box(raw_box, width, height)
        x1, y1, x2, y2 = box
        color = class_color(class_name)
        is_highlighted = object_idx == highlight_idx
        thickness = 4 if is_highlighted else 2
        if is_highlighted:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), thickness + 2, cv2.LINE_AA)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        draw_label(image, f"{class_name} [{object_idx}]", (x1, y1), color)

    if tile_size is not None:
        image = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    return bool(cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]))


def render_full_image(args: tuple[Path, list[ObjectRecord], Path]) -> tuple[str, bool]:
    image_path, objects, output_path = args
    return output_path.name, render_image(image_path, objects, output_path)


def render_tile(
    args: tuple[Path, list[ObjectRecord], int, int, int, str]
) -> tuple[np.ndarray | None, int, bool]:
    image_path, objects, highlight_idx, tile_size, tile_number, stem = args
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None, highlight_idx, False

    height, width = image.shape[:2]
    for object_idx, (class_name, raw_box) in enumerate(objects):
        x1, y1, x2, y2 = clamp_box(raw_box, width, height)
        color = class_color(class_name)
        is_highlighted = object_idx == highlight_idx
        if is_highlighted:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 6, cv2.LINE_AA)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4 if is_highlighted else 2, cv2.LINE_AA)
        draw_label(image, f"{class_name} [{object_idx}]", (x1, y1), color)

    image = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    badge = f"#{tile_number:04d}"
    cv2.rectangle(image, (0, 0), (90, 25), (30, 30, 220), -1)
    cv2.putText(image, badge, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(image, (0, tile_size - 22), (min(tile_size - 1, 8 * len(stem) + 12), tile_size - 1), (0, 0, 0), -1)
    cv2.putText(image, stem, (5, tile_size - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return image, highlight_idx, True


def ordered_classes(classes: Iterable[str]) -> list[str]:
    available = set(classes)
    known = [class_name for class_name in CLASS_ORDER if class_name in available]
    return known + sorted(available.difference(CLASS_ORDER))


def make_grid(tiles: list[np.ndarray | None], tile_size: int, columns: int, page: int, total_pages: int) -> np.ndarray:
    rows = max(1, (len(tiles) + columns - 1) // columns)
    header_height = 28
    grid = np.full((header_height + rows * tile_size, columns * tile_size, 3), 32, dtype=np.uint8)
    cv2.putText(grid, f"page {page}/{total_pages}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    for index, tile in enumerate(tiles):
        if tile is None:
            tile = np.full((tile_size, tile_size, 3), (48, 0, 0), dtype=np.uint8)
            cv2.putText(tile, "IMAGE NOT FOUND", (8, tile_size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        row, column = divmod(index, columns)
        y = header_height + row * tile_size
        x = column * tile_size
        grid[y:y + tile_size, x:x + tile_size] = tile
    return grid


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per_page", type=int, default=80, help="objects per class grid page")
    parser.add_argument("--cols", type=int, default=8, help="grid columns")
    parser.add_argument("--tile", type=int, default=240, help="grid tile size in pixels")
    parser.add_argument("--workers", type=int, default=max(1, min(8, mp.cpu_count())), help="render worker processes")
    parser.add_argument("--clean", action="store_true", help="remove images/, grids/, and CSV outputs under bbox_full before rendering (preserves Annotations_patched/, bbox_overrides.json, run.log)")
    args = parser.parse_args()

    if args.per_page < 1 or args.cols < 1 or args.tile < 32 or args.workers < 1:
        parser.error("per_page, cols, workers must be positive and tile must be at least 32")

    if args.clean and OUT_DIR.exists():
        # Static artifacts we never want to wipe.
        preserve = {"Annotations_patched", "bbox_overrides.json", "run.log", "README.md"}
        # Per-run logs from the notebook's run_script_step helper live in
        # `logs/run_<timestamp>.log`. Preserve that directory AND any
        # top-level file matching the same prefix so re-running this
        # script does not delete the log it is currently appending to.
        for entry in OUT_DIR.iterdir():
            if entry.name in preserve:
                continue
            if entry.name == "logs":
                continue
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink(missing_ok=True)
    IMAGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    GRID_OUT_DIR.mkdir(parents=True, exist_ok=True)

    parsed: list[AnnotationRecord] = []
    invalid_xml: list[str] = []

    override_stems: set[str] = set()
    if OVERRIDES_DIR.exists():
        # Any patched XML under OVERRIDES_DIR counts as an override,
        # regardless of whether bbox_overrides.json is also present.
        for xml_path in OVERRIDES_DIR.glob("*.xml"):
            override_stems.add(xml_path.stem)

    def _xml_path_for(stem: str) -> Path:
        patched = OVERRIDES_DIR / f"{stem}.xml"
        if patched.exists():
            return patched
        return XML_DIR / f"{stem}.xml"

    xml_paths: list[Path] = []
    seen_stems: set[str] = set()
    for xml_path in sorted(XML_DIR.glob("*.xml")):
        stem = xml_path.stem
        if stem in override_stems:
            patched = OVERRIDES_DIR / f"{stem}.xml"
            if patched.exists():
                xml_paths.append(patched)
                seen_stems.add(stem)
                continue
        xml_paths.append(xml_path)
        seen_stems.add(stem)

    for xml_path in xml_paths:
        record = parse_annotation(xml_path)
        if record is None:
            invalid_xml.append(xml_path.name)
        else:
            parsed.append(record)

    print(f"[scan] XML annotations: {len(parsed)} | invalid: {len(invalid_xml)}", flush=True)
    image_jobs = []
    image_rows: list[dict[str, object]] = []
    buckets: dict[str, list[tuple[str, int]]] = defaultdict(list)
    missing_images: list[str] = []

    for stem, objects in parsed:
        image_path = find_image(stem)
        output_path = IMAGE_OUT_DIR / f"{stem}.jpg"
        image_rows.append({
            "stem": stem,
            "source_image": image_path.name if image_path else "",
            "output_image": str(Path("images") / output_path.name) if image_path else "",
            "n_objects": len(objects),
            "status": "pending" if image_path else "missing_image",
        })
        if image_path is None:
            missing_images.append(stem)
            continue
        image_jobs.append((image_path, objects, output_path))
        for object_idx, (class_name, _box) in enumerate(objects):
            buckets[class_name].append((stem, object_idx))

    print(f"[scan] images: {len(image_jobs)} | missing: {len(missing_images)}", flush=True)
    print(f"[scan] annotated objects: {sum(len(items) for items in buckets.values())}", flush=True)

    with mp.Pool(processes=args.workers) as pool:
        image_results = pool.map(render_full_image, image_jobs)
    rendered_names = {name for name, success in image_results if success}
    for row in image_rows:
        if row["status"] == "pending":
            row["status"] = "rendered" if row["output_image"] and Path(str(row["output_image"])).name in rendered_names else "render_failed"

    class_names = ordered_classes(buckets)
    counts_rows = []
    index_rows: list[dict[str, object]] = []
    global_tile = 0

    annotation_by_stem = {stem: objects for stem, objects in parsed}
    for class_index, class_name in enumerate(class_names):
        items = sorted(buckets[class_name])
        total_pages = (len(items) + args.per_page - 1) // args.per_page
        counts_rows.append({"class": class_name, "n_objects": len(items), "n_pages": total_pages})
        print(f"  {class_name}: {len(items)} objects", flush=True)

        for page in range(1, total_pages + 1):
            chunk = items[(page - 1) * args.per_page:page * args.per_page]
            tile_jobs = []
            page_global_start = global_tile + 1
            for stem, object_idx in chunk:
                image_path = find_image(stem)
                objects = annotation_by_stem[stem]
                tile_number = global_tile + 1
                tile_jobs.append(
                    (image_path, objects, object_idx, args.tile, tile_number, stem)
                    if image_path
                    else (Path(""), objects, object_idx, args.tile, tile_number, stem)
                )
                global_tile = tile_number

            with mp.Pool(processes=args.workers) as pool:
                tile_results = pool.map(render_tile, tile_jobs)
            tiles = [result[0] for result in tile_results]

            for page_local, ((stem, object_idx), result) in enumerate(zip(chunk, tile_results), start=1):
                class_for_object, box = annotation_by_stem[stem][object_idx]
                global_number = page_global_start + page_local - 1
                index_rows.append({
                    "tile_global": global_number,
                    "class": class_for_object,
                    "page": page,
                    "page_local": page_local,
                    "stem": stem,
                    "obj_idx": object_idx,
                    "bbox_xmin": box[0],
                    "bbox_ymin": box[1],
                    "bbox_xmax": box[2],
                    "bbox_ymax": box[3],
                    "image": str(Path("images") / f"{stem}.jpg"),
                    "rendered": result[2],
                })

            grid = make_grid(tiles, args.tile, args.cols, page, total_pages)
            grid_path = GRID_OUT_DIR / f"{class_index:02d}_{class_name}_p{page:02d}_of_{total_pages:02d}.png"
            cv2.imwrite(str(grid_path), grid, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            print(f"  saved {grid_path.name} ({len(tiles)} tiles)", flush=True)

    write_csv(
        OUT_DIR / "_image_index.csv",
        ["stem", "source_image", "output_image", "n_objects", "status"],
        image_rows,
    )
    write_csv(
        OUT_DIR / "_index.csv",
        [
            "tile_global", "class", "page", "page_local", "stem", "obj_idx",
            "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax", "image", "rendered",
        ],
        index_rows,
    )
    write_csv(OUT_DIR / "_per_class_counts.csv", ["class", "n_objects", "n_pages"], counts_rows)

    readme = OUT_DIR / "README.txt"
    readme.write_text(
        "VOC ground-truth bounding-box visualization for the complete ECUSTFD dataset.\n\n"
        "Source: data/raw/ECUSTFD/Annotations/*.xml and JPEGImages/*.JPG.\n"
        "No SAM mask is used. Every XML object is drawn, including coin.\n\n"
        "Output:\n"
        "  images/  - one annotated image for every source image with a matching XML.\n"
        "  grids/   - all annotated objects grouped by class, 80 tiles per page by default.\n"
        "  _image_index.csv - image-level coverage and render status.\n"
        "  _index.csv - tile number, class, source stem, object index, and real VOC bbox.\n"
        "  _per_class_counts.csv - object and page counts by class.\n\n"
        "Box colors identify classes. The object selected by a grid tile has a thick white outline;\n"
        "the other real boxes in the same image remain visible.\n",
        encoding="utf-8",
    )
    if invalid_xml:
        (OUT_DIR / "_invalid_xml.txt").write_text("\n".join(invalid_xml) + "\n", encoding="utf-8")
    if missing_images:
        (OUT_DIR / "_missing_images.txt").write_text("\n".join(missing_images) + "\n", encoding="utf-8")

    print(f"[done] annotated images: {len(rendered_names)}", flush=True)
    print(f"[done] object tiles: {len(index_rows)}", flush=True)
    print(f"[done] output: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
