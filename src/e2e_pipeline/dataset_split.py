"""ECUSTFD image-pair pairing.

ECUSTFD filenames follow the pattern ``{class}{item_id}{S|T}({i}).JPG``
where ``S`` is the side view and ``T`` is the top view. The original
``ImageSets/Main/{test,train,val}.txt`` lists each image individually,
so we must group them by ``{class}{item_id}`` stem before pairing.

Why a separate folder/wrapper:
- The dataset ships with one line per image, but the pipeline needs
  one row per (top, side) pair. Keeping the grouping logic isolated
  makes it testable without loading Ultralytics.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
# Biểu thức chính quy để phân tích tên file
_FILENAME_RE = re.compile(r"^(?P<stem>.+?)(?P<view>[ST])\((?P<idx>\d+)\)\.[jJ][pP][gG]$")

# Phân tích tên file
def parse_filename(name: str) -> Optional[Tuple[str, str, int]]:
    """Parse ``apple001S(1).JPG`` into ``(stem='apple001', view='S', idx=1)``.

    Returns ``None`` for files that don't match the ECUSTFD pattern.
    """
    p = Path(name)# Chuyển đổi tên file sang Path
    m = _FILENAME_RE.match(p.name) # Tìm kiếm trong tên file
    if not m: # Nếu không tìm thấy tên file
        return None # Trả về None
    return m.group("stem"), m.group("view"), int(m.group("idx")) # Trả về stem, view, và index

# Load file text chứa danh sách các ảnh
def load_split(path: str | Path) -> List[str]:
    """Load a ``test.txt``/``train.txt``/``val.txt`` file as image stems.

    Each line is treated as a stem (no extension). Stems that don't
    exist on disk are silently filtered out.
    """
    p = Path(path) # Chuyển đổi đường dẫn sang Path
    if not p.exists(): # Nếu không tìm thấy file text
        raise FileNotFoundError(p) # Raise lỗi
    out: List[str] = [] # Tạo list rỗng để chứa các stem
    for raw in p.read_text(encoding="utf-8").splitlines(): # Đọc file text và tách thành các dòng
        s = raw.strip() # Loại bỏ khoảng trắng ở đầu và cuối dòng
        if not s: # Nếu không tìm thấy stem
            continue # Bỏ qua
        out.append(s) # Thêm stem vào list
    return out

# Resolve đường dẫn ảnh
def resolve_image_paths(
    stems: List[str], # Danh sách các stem
    images_dir: str | Path, # Đường dẫn đến thư mục chứa ảnh
    extensions: Tuple[str, ...] = (".JPG", ".jpg", ".jpeg", ".JPEG"), # Các đuôi file ảnh
) -> List[Path]:
    """Resolve each stem to an existing path on disk.

    Tries each extension in turn. Returns only the stems that were
    resolved; missing ones are dropped.
    """
    d = Path(images_dir) # Chuyển đổi đường dẫn sang Path
    out: List[Path] = [] # Tạo list rỗng để chứa các đường dẫn ảnh
    for s in stems: # Duyệt qua các stem
        for ext in extensions: # Duyệt qua các đuôi file ảnh
            candidate = d / (s + ext) # Tạo đường dẫn ảnh
            if candidate.exists(): # Nếu tìm thấy ảnh
                out.append(candidate) # Thêm đường dẫn ảnh vào list
                break # Bỏ qua các đuôi file khác
    return out

# Group ảnh theo stem và split thành top/side lists
def group_top_side(
    image_paths: List[Path], # Danh sách các đường dẫn ảnh
) -> List[Dict[str, List[Path]]]:
    """Group image paths by stem and split into top/side lists.

    Returns a list of dicts, one per stem, with keys ``"top"`` and
    ``"side"``. Each value is a list of Paths (a single item may
    appear in multiple shots, e.g. ``apple001S(1).JPG``,
    ``apple001S(2).JPG``).

    Stems that have only one view are still kept (with the missing
    key absent) so the orchestrator can report them.
    """
    groups: Dict[str, Dict[str, List[Path]]] = defaultdict(# Tạo dict để chứa các stem
        lambda: {"top": [], "side": []} # Tạo list rỗng để chứa top và side
    )
    for p in image_paths: # Duyệt qua các đường dẫn ảnh
        parsed = parse_filename(p.name) # Phân tích tên file
        if parsed is None: # Nếu không tìm thấy tên file
            continue # Bỏ qua
        stem, view, _idx = parsed # Lấy stem, view và index
        key = "top" if view == "T" else "side" # Lấy top hoặc side
        groups[stem][key].append(p) # Thêm đường dẫn ảnh vào dict

    # Stable order: sorted by stem.
    return [groups[k] for k in sorted(groups)]

# Make a flat list of (top, side) pairs
def make_pairs(
    groups: List[Dict[str, List[Path]]], # List chứa các nhóm ảnh
) -> List[Tuple[Path, Path]]:
    """Make a flat list of (top, side) pairs.

    If a stem has N top views and M side views, this returns N*M pairs
    (cartesian product). The orchestrator can then decide whether to
    deduplicate or take a particular index.
    """
    pairs: List[Tuple[Path, Path]] = [] # Tạo list rỗng để chứa các cặp ảnh
    for g in groups: # Duyệt qua các nhóm ảnh
        tops = g.get("top") or [] # Lấy danh sách các ảnh top
        sides = g.get("side") or [] # Lấy danh sách các ảnh side
        for t in tops: # Duyệt qua các ảnh top
            for s in sides: # Duyệt qua các ảnh side
                pairs.append((t, s)) # Thêm cặp ảnh vào list
    return pairs # Trả về list các cặp ảnh

# Split ảnh theo tỷ lệ 50/50
def get_paper_50_50_split(
    image_paths: List[Path], # Danh sách các đường dẫn ảnh
) -> Tuple[List[Path], List[Path]]:
    """Split image paths 50/50 per class matching `xls_results_analysis.m`.

    For each class, the image files are grouped by item ID (stem), sorted,
    and the first 50% of items are placed in train, the remaining 50% in test.
    """
    by_class: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list)) # Tạo dict để chứa các nhóm ảnh
    for p in image_paths: # Duyệt qua các đường dẫn ảnh
        parsed = parse_filename(p.name) # Phân tích tên file
        if parsed is None: # Nếu không tìm thấy tên file
            continue # Bỏ qua
        stem, _view, _idx = parsed # Lấy stem, view và index
        # stem is e.g. "apple001" -> class is "apple"
        m = re.match(r"^([a-zA-Z_]+)\d+", stem) # Tìm kiếm trong tên file
        cls_name = m.group(1) if m else stem # Lấy tên class
        by_class[cls_name][stem].append(p) # Thêm đường dẫn ảnh vào dict

    train_paths: List[Path] = [] # Tạo list rỗng để chứa các đường dẫn ảnh train
    test_paths: List[Path] = [] # Tạo list rỗng để chứa các đường dẫn ảnh test

    for cls_name, stems_dict in sorted(by_class.items()): # Duyệt qua các nhóm ảnh
        sorted_stems = sorted(stems_dict.keys()) # Sắp xếp các stem theo thứ tự bảng chữ cái
        mid = len(sorted_stems) // 2 # Tính điểm chia đôi
        train_stems = set(sorted_stems[:mid]) # Tạo set các stem train
        for stem, paths in stems_dict.items(): # Duyệt qua các stem và đường dẫn ảnh
            if stem in train_stems: # Nếu stem nằm trong set train
                train_paths.extend(paths) # Thêm đường dẫn ảnh vào list train
            else: # Nếu stem không nằm trong set train
                test_paths.extend(paths) # Thêm đường dẫn ảnh vào list test

    return train_paths, test_paths # Trả về list các đường dẫn ảnh train và test


__all__ = [
    "parse_filename",
    "load_split",
    "resolve_image_paths",
    "group_top_side",
    "make_pairs",
    "get_paper_50_50_split",
]

