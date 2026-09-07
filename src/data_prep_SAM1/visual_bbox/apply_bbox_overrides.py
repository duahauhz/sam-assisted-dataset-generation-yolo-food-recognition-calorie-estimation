"""
apply_bbox_overrides.py — Tái sinh code đã mất: ghi đè bbox thực phẩm bằng
tọa độ chính xác mới, đồng thời giữ nguyên bbox coin từ VOC (hoặc override
nếu coin bbox trong VOC sai). Mỗi object ghi ra đều được đánh dấu bằng tag
<path>apply_bbox_overrides.py:OVERRIDES</path> và folder=VOC2007_patched.

Cấu trúc mỗi entry trong OVERRIDES:
  - new_class   : (Bước 2/3) tên nhãn thực phẩm mới (giữ nguyên qiwi).
  - xyxy        : (Bước 2/3) bbox thực phẩm mới (xmin, ymin, xmax, ymax).
  - order       : "food_then_coin" (mặc định) hoặc "coin_then_food".
  - coin_xyxy   : (tuỳ chọn, Bước 4) bbox coin mới khi VOC sai (copy từ food).
                  Nếu không có, giữ bbox coin từ VOC.
  - class_changes : (tuỳ chọn, Bước 4 đổi nhãn) {old_class: new_class}. Áp
                  dụng khi VOC có nhiều food obj (vd mix012T(4) đổi
                  mango -> orange), giữ nguyên bbox + các obj khác.

Cách dùng:
  python apply_bbox_overrides.py --dry-run   # so sánh byte-for-byte, không ghi
  python apply_bbox_overrides.py             # ghi 37 file vào PATCHED_DIR
                                             # (mặc định, pipeline visualize)
  python apply_bbox_overrides.py --to-tmp    # ghi 37 file vào /TMP/Annotations_patched_regen
  python apply_bbox_overrides.py --only pear003T(7)

Khi chạy --dry-run:
  [OK   ] = byte-for-byte identical với file patch hiện tại.
  [NEW]  = file patch chưa tồn tại (sẽ tạo mới khi chạy thật).
  [DIFF ] = file patch khác nội dung.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from lxml import etree


# ----------------------------------------------------------------------------
# Đường dẫn
# ----------------------------------------------------------------------------
ROOT = Path("E:/AI_Research/dlt8")
RAW_VOC = ROOT / "data/raw/ECUSTFD/Annotations"
PATCHED_DIR = ROOT / "data/processed/bbox_full/Annotations_patched"
# Mặc định ghi vào /TMP để không ghi đè 14 file patch hiện tại một cách vô tình.
# Khi đã verify đúng, dùng --in-place để ghi đè PATCHED_DIR.
DEFAULT_OUTPUT_DIR = ROOT / "TMP/Annotations_patched_regen"
SEG_FULL_DIR = ROOT / "data/processed/sam_masks_full"
IMAGES_FULL_DIR = ROOT / "data/processed/bbox_full/images"

OVERRIDE_TAG = "apply_bbox_overrides.py:OVERRIDES"


# ----------------------------------------------------------------------------
# Bảng OVERRIDES — khôi phục từ 14 file patch hiện có + bổ sung sau Bước 4
# Cấu trúc mỗi entry:
#   - new_class: tên nhãn thực phẩm mới (giữ nguyên qiwi)
#   - xyxy: bbox thực phẩm mới (xmin, ymin, xmax, ymax)
#   - order: "food_then_coin" hoặc "coin_then_food" — đúng thứ tự file patch
#   - coin_xyxy: (tuỳ chọn) nếu trùng với bbox thực phẩm VOC (coin sai), ghi đè
#     bằng bbox mới. Nếu không có, giữ bbox coin từ VOC.
#   - class_changes: (tuỳ chọn, Bước 4 đổi nhãn) {old_class: new_class}. Áp
#     dụng khi VOC có nhiều food obj và cần đổi nhãn cho từng obj riêng lẻ,
#     giữ nguyên bbox + các obj khác (vd mix012T(4) đổi mango -> orange).
# ----------------------------------------------------------------------------
OVERRIDES: dict[str, dict] = {
    # 14 file đã có patch gốc (Bước 2 + Bước 3)
    "pear003T(7)":   {"new_class": "pear",   "xyxy": (417, 232, 615, 425), "order": "food_then_coin"},
    "plum002T(16)":  {"new_class": "plum",   "xyxy": (485, 266, 654, 442), "order": "food_then_coin"},
    "plum003S(11)":  {"new_class": "plum",   "xyxy": (438, 250, 600, 404), "order": "food_then_coin"},
    "qiwi006S(7)":   {"new_class": "kiwi",   "xyxy": (179, 264, 378, 413), "order": "coin_then_food"},
    "qiwi006S(8)":   {"new_class": "kiwi",   "xyxy": (126, 240, 336, 408), "order": "coin_then_food"},
    "qiwi007S(1)":   {"new_class": "kiwi",   "xyxy": (328, 257, 546, 423), "order": "coin_then_food"},
    "qiwi007T(3)":   {"new_class": "kiwi",   "xyxy": (337, 263, 497, 462), "order": "coin_then_food"},
    "tomato001S(4)": {"new_class": "tomato", "xyxy": (403, 254, 602, 404), "order": "food_then_coin"},
    "tomato003T(19)":{"new_class": "tomato", "xyxy": (431, 232, 653, 450), "order": "coin_then_food"},
    "tomato003T(27)":{"new_class": "tomato", "xyxy": (419, 215, 630, 428), "order": "coin_then_food"},
    "tomato004S(18)":{"new_class": "tomato", "xyxy": (461, 144, 654, 283), "order": "coin_then_food"},
    "tomato004S(26)":{"new_class": "tomato", "xyxy": (460, 198, 656, 337), "order": "coin_then_food"},
    "tomato004T(25)":{"new_class": "tomato", "xyxy": (389, 212, 603, 420), "order": "food_then_coin"},
    "tomato004T(4)": {"new_class": "tomato", "xyxy": (420, 197, 650, 425), "order": "coin_then_food"},
    # Bước 4 — sửa coin sai (VOC copy bbox doughnut sang coin). Ngoài sửa
    # coin, giữ nguyên food bbox (nếu có) từ VOC. Tổng cộng có 17 ảnh
    # doughnut có coin bbox sai:
    # - 16 ảnh này: coin bbox VOC copy y hệt từ doughnut bbox, sửa coin về
    #   bbox thật (đo bằng mắt, sai số ±2 px).
    # - doughnut006S(15): coin VOC bị đánh bbox cực nhỏ (2×3 = 6 px², gần
    #   như 1 điểm) — đo lại được (132,317)→(202,384), 70×67 = 4690 px².
    "doughnut004S(15)": {"new_class": None, "xyxy": None, "coin_xyxy": (97, 354, 167, 420)},
    "doughnut006T(18)": {"new_class": None, "xyxy": None, "coin_xyxy": (127, 116, 192, 180)},
    "doughnut006T(7)":  {"new_class": None, "xyxy": None, "coin_xyxy": (126, 54, 194, 121)},
    "doughnut007S(14)": {"new_class": None, "xyxy": None, "coin_xyxy": (46, 316, 121, 388)},
    "doughnut007S(3)":  {"new_class": None, "xyxy": None, "coin_xyxy": (57, 324, 126, 390)},
    "doughnut008S(3)":  {"new_class": None, "xyxy": None, "coin_xyxy": (70, 314, 140, 387)},
    "doughnut008S(4)":  {"new_class": None, "xyxy": None, "coin_xyxy": (72, 282, 140, 354)},
    "doughnut008T(12)": {"new_class": None, "xyxy": None, "coin_xyxy": (152, 138, 213, 202)},
    "doughnut008T(13)": {"new_class": None, "xyxy": None, "coin_xyxy": (127, 108, 195, 178)},
    "doughnut008T(8)":  {"new_class": None, "xyxy": None, "coin_xyxy": (141, 128, 210, 195)},
    "doughnut009S(1)":  {"new_class": None, "xyxy": None, "coin_xyxy": (27, 292, 103, 364)},
    "doughnut009S(2)":  {"new_class": None, "xyxy": None, "coin_xyxy": (61, 257, 137, 332)},
    "doughnut009S(9)":  {"new_class": None, "xyxy": None, "coin_xyxy": (25, 283, 100, 357)},
    "doughnut009T(10)": {"new_class": None, "xyxy": None, "coin_xyxy": (98, 190, 167, 254)},
    "doughnut009T(11)": {"new_class": None, "xyxy": None, "coin_xyxy": (151, 220, 218, 283)},
    "doughnut009T(6)":  {"new_class": None, "xyxy": None, "coin_xyxy": (140, 171, 209, 238)},
    "doughnut006S(15)":  {"new_class": None, "xyxy": None, "coin_xyxy": (132, 317, 202, 384)},
    # Bước 4 tiếp — mango: cùng lỗi copy bbox mango sang coin. Tổng 3 file.
    "mango008T(4)": {"new_class": None, "xyxy": None, "coin_xyxy": (75, 295, 144, 364)},
    "mango004S(7)": {"new_class": None, "xyxy": None, "coin_xyxy": (90, 396, 150, 463)},
    "mango004T(7)": {"new_class": None, "xyxy": None, "coin_xyxy": (101, 237, 180, 311)},
    # Bước 4 tiếp — orange: cùng lỗi copy bbox orange sang coin. Tổng 2 file.
    "orange012T(7)": {"new_class": None, "xyxy": None, "coin_xyxy": (85, 52, 118, 84)},
    "orange015S(6)": {"new_class": None, "xyxy": None, "coin_xyxy": (414, 225, 479, 288)},
    # Bước 4 tiếp — đổi nhãn nhầm trong mix. mix012T(4) có 2 food obj trong
    # VOC: mango (cần đổi -> orange) và lemon (giữ nguyên). Coin bbox VOC đã
    # đúng (52×53), không sửa — chỉ đổi class.
    "mix012T(4)": {"new_class": None, "xyxy": None, "class_changes": {"mango": "orange"}},
}


# ----------------------------------------------------------------------------
# Hàm xử lý 1 file
# ----------------------------------------------------------------------------
def _load_voc_objects(xml_path: Path) -> Tuple[List[Tuple[str, int, int, int, int]], List[Tuple[str, int, int, int, int]]]:
    """Trả về (food_objects, coin_objects) đọc từ VOC. Mỗi obj: (class, xmin, ymin, xmax, ymax)."""
    tree = etree.parse(str(xml_path))
    food, coin = [], []
    for obj in tree.getroot().findall("object"):
        cls = (obj.find("name").text or "").strip()
        b = obj.find("bndbox")
        xyxy = tuple(int(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        if cls == "coin":
            coin.append((cls, *xyxy))
        else:
            food.append((cls, *xyxy))
    return food, coin


def _build_patched_xml(
    voc_path: Path,
    food_class: Optional[str],
    food_xyxy: Optional[Tuple[int, int, int, int]],
    coin_xyxy: Tuple[int, int, int, int],
    order: str,
    class_changes: Optional[dict] = None,
) -> bytes:
    """Tạo nội dung XML patch từ VOC gốc, áp dụng override + thêm <path>.

    Hai trường hợp:
    - food_class/xyxy đầy đủ: dùng làm bbox thực phẩm (Bước 2/3).
    - food_class=None, food_xyxy=None: giữ nguyên food bbox từ VOC; chỉ ghi
      đè coin bbox (Bước 4 — coin bị annotate sai).

    Khi VOC có nhiều food obj và Bước 4 cần đổi nhãn từng obj, truyền
    class_changes={old_cls: new_cls}. Ví dụ {"mango": "orange"} đổi tất cả
    food obj có name="mango" sang name="orange", giữ nguyên bbox + các obj
    khác.
    """
    # parse VOC, strip sạch whitespace để lxml sinh lại indent 2 spaces sạch
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(str(voc_path), parser)
    root = tree.getroot()

    # đổi folder -> VOC2007_patched
    folder_el = root.find("folder")
    folder_el.text = "VOC2007_patched"

    # xóa hết object cũ, build lại theo thứ tự
    for obj in list(root.findall("object")):
        root.remove(obj)

    def _make_object(cls: str, x1: int, y1: int, x2: int, y2: int) -> etree._Element:
        obj = etree.SubElement(root, "object")
        etree.SubElement(obj, "name").text = cls
        etree.SubElement(obj, "pose").text = "Unspecified"
        etree.SubElement(obj, "truncated").text = "0"
        etree.SubElement(obj, "difficult").text = "0"
        bb = etree.SubElement(obj, "bndbox")
        etree.SubElement(bb, "xmin").text = str(x1)
        etree.SubElement(bb, "ymin").text = str(y1)
        etree.SubElement(bb, "xmax").text = str(x2)
        etree.SubElement(bb, "ymax").text = str(y2)
        # đánh dấu nguồn override
        etree.SubElement(obj, "path").text = OVERRIDE_TAG
        return obj

    if food_class is not None and food_xyxy is not None:
        # override food bbox (Bước 2/3)
        if order == "food_then_coin":
            _make_object(food_class, *food_xyxy)
            _make_object("coin", *coin_xyxy)
        elif order == "coin_then_food":
            _make_object("coin", *coin_xyxy)
            _make_object(food_class, *food_xyxy)
        else:
            raise ValueError(f"unknown order: {order}")
    else:
        # chỉ override coin bbox (Bước 4): giữ nguyên food + sửa coin
        food_objs, _ = _load_voc_objects(voc_path)
        if not food_objs:
            raise ValueError(f"{voc_path.name}: no food obj in VOC")
        # Nếu có class_changes: áp dụng đổi nhãn cho từng food obj (giữ bbox)
        changes = class_changes or {}
        # ghi theo thứ tự xuất hiện trong VOC, sau đó coin
        for f_cls, fx1, fy1, fx2, fy2 in food_objs:
            new_cls = changes.get(f_cls, f_cls)
            _make_object(new_cls, fx1, fy1, fx2, fy2)
        _make_object("coin", *coin_xyxy)

    # serialize với indent 2 spaces (closing tag đúng indent)
    etree.indent(tree, space="  ")
    out_bytes = etree.tostring(
        tree,
        xml_declaration=True,
        encoding="utf-8",
        pretty_print=False,  # indent đã được áp dụng ở trên
    )
    # Decode -> string để xử lý indent bug + CRLF trên text thuần, tránh
    # quirk của re.sub trên bytes với MULTILINE.
    out = out_bytes.decode("utf-8")

    # Tái hiện bug indent của code cũ: tất cả closing tag (`</xyz>`) đều bị
    # thụt vào thêm 1 level (+2 spaces) so với opening tag tương ứng — chỉ
    # áp dụng cho dòng chỉ chứa closing tag, không phải inline `</x>...</x>`.
    out = _RE_CLOSING_TAG_INDENT_BUG.sub(r"\1  \2", out)
    # Convert \n -> \r\n cho khớp file patch (CRLF line endings)
    out = out.replace("\n", "\r\n")
    return out.encode("utf-8")


# Closing tag của một element: `^\s*</\w+>\s*$` (chỉ dòng chỉ chứa closing tag)
# Bug cũ: thêm 2 spaces vào trước closing tag.
import re as _re
_RE_CLOSING_TAG_INDENT_BUG = _re.compile(r"^(\s*)(</[A-Za-z0-9_]+>)\s*$", _re.MULTILINE)


def patch_one(stem: str, dry_run: bool, in_place: bool) -> Tuple[bool, str]:
    """Patch 1 file. Trả về (ok, message)."""
    spec = OVERRIDES[stem]
    voc_path = RAW_VOC / f"{stem}.xml"
    # Luôn so sánh với file patch hiện tại (PATCHED_DIR) để verify.
    check_path = PATCHED_DIR / f"{stem}.xml"
    # - chạy thường (không --in-place) => ghi qua TMP, KHÔNG ghi đè
    # - chạy --in-place => ghi đè PATCHED_DIR
    out_path = check_path if in_place else (DEFAULT_OUTPUT_DIR / f"{stem}.xml")
    if not voc_path.exists():
        return False, f"[SKIP] VOC not found: {voc_path}"

    food_list, coin_list = _load_voc_objects(voc_path)
    if len(coin_list) != 1:
        return False, f"[SKIP] {stem}: expected 1 coin in VOC, got {len(coin_list)}"
    _, cx1, cy1, cx2, cy2 = coin_list[0]

    # Nếu có coin_xyxy override thì dùng nó, không thì lấy từ VOC
    if "coin_xyxy" in spec and spec["coin_xyxy"] is not None:
        coin_xyxy = spec["coin_xyxy"]
    else:
        coin_xyxy = (cx1, cy1, cx2, cy2)

    # Order mặc định khi không có (chỉ override coin, giữ food VOC)
    order = spec.get("order", "food_then_coin")

    new_bytes = _build_patched_xml(
        voc_path=voc_path,
        food_class=spec["new_class"],
        food_xyxy=spec["xyxy"],
        coin_xyxy=coin_xyxy,
        order=order,
        class_changes=spec.get("class_changes"),
    )

    if dry_run:
        if not check_path.exists():
            return False, f"[NEW] {stem}: no existing patched file (would create)"
        old_bytes = check_path.read_bytes()
        if old_bytes == new_bytes:
            return True, f"[OK   ] {stem}: byte-for-byte identical"
        diff_msg = _short_diff(old_bytes, new_bytes)
        return False, f"[DIFF ] {stem}:\n{diff_msg}"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(new_bytes)
        return True, f"[WRITE] {stem}: written to {out_path} ({len(new_bytes)} bytes)"


def _short_diff(a: bytes, b: bytes, ctx: int = 60) -> str:
    """In ngắn vị trí khác biệt đầu tiên giữa a và b."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            sa = a[max(0, i - ctx):i + ctx]
            sb = b[max(0, i - ctx):i + ctx]
            return f"  first diff @ byte {i}\n  old: ...{sa!r}...\n  new: ...{sb!r}..."
    if len(a) != len(b):
        return f"  length differs: old={len(a)} new={len(b)} (prefix matches {n} bytes)"
    return "  no diff"


def main():
    ap = argparse.ArgumentParser(description=(
        "Ghi đè bbox thực phẩm bằng tọa độ chính xác mới, giữ bbox coin từ VOC. "
        "Mặc định ghi vào Annotations_patched/ để visualize_bbox_full.py dùng "
        "được; dùng --to-tmp để ghi vào /TMP/Annotations_patched_regen (an toàn "
        "để verify trước)."
    ))
    ap.add_argument("--dry-run", action="store_true",
                    help="Không ghi file, chỉ so sánh byte-for-byte với file patch hiện tại")
    ap.add_argument("--to-tmp", action="store_true",
                    help="Ghi vào /TMP/Annotations_patched_regen thay vì Annotations_patched/")
    ap.add_argument("--only", default=None,
                    help="Chỉ chạy 1 stem, vd: pear003T(7)")
    args = ap.parse_args()

    targets = list(OVERRIDES.keys())
    if args.only:
        if args.only not in OVERRIDES:
            print(f"[ERR] --only={args.only} not in OVERRIDES", file=sys.stderr)
            sys.exit(1)
        targets = [args.only]

    in_place = not args.to_tmp

    if args.dry_run:
        mode = "DRY-RUN"
    elif in_place:
        mode = f"WRITE -> PATCHED_DIR ({PATCHED_DIR})"
    else:
        mode = f"WRITE -> TMP ({DEFAULT_OUTPUT_DIR})"
    print(f"== apply_bbox_overrides.py | mode={mode} | n={len(targets)} ==")
    n_ok = n_diff = n_err = 0
    for stem in targets:
        ok, msg = patch_one(stem, args.dry_run, in_place)
        print(msg)
        if ok:
            n_ok += 1
        elif msg.startswith("[DIFF"):
            n_diff += 1
        elif msg.startswith("[SKIP") or msg.startswith("[ERR"):
            n_err += 1
    print(f"---\nSummary: ok={n_ok}  diff={n_diff}  err={n_err}")


if __name__ == "__main__":
    main()
