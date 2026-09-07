# -*- coding: utf-8 -*-
"""Helper utilities for YOLO26-seg evaluation pipeline.

Provides dual logging, schema validation, ground-truth aliasing fixes,
smoke testing, speed latency profiling, and Markdown report rendering.
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from src.constants import FOOD_CLASSES

# Canonical 20-class list for ECUSTFD YOLO models (index 0..19)
CANONICAL_YOLO_CLASSES: List[str] = [
    "apple", "banana", "bread", "bun", "coin",
    "doughnut", "egg", "fired_dough_twist", "grape",
    "lemon", "litchi", "mango", "mooncake",
    "orange", "peach", "pear", "plum", "qiwi",
    "sachima", "tomato",
]


def setup_eval_logging(
    session_name: str = "04_e2e_yolo_seg_eval",
    project_root: Optional[Union[str, Path]] = None,
) -> Tuple[logging.Logger, Path, Path, str]:
    """Setup dual logger streaming to console and file with timestamped run directory."""
    if project_root is None:
        project_root = Path("E:/AI_Research/dlt8").resolve()
    project_root = Path(project_root)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = project_root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{session_name}_{ts}.log"

    run_dir = project_root / "outputs" / "predictions" / f"{session_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    log = logging.getLogger(session_name)
    log.info("=" * 70)
    log.info("=== %s -- EVALUATION SESSION STARTED ===", session_name.upper())
    log.info("=" * 70)
    log.info("PROJECT_ROOT : %s", project_root)
    log.info("LOG_PATH     : %s", log_path)
    log.info("RUN_DIR      : %s", run_dir)
    log.info("TIMESTAMP    : %s", ts)

    return log, log_path, run_dir, ts


def sanity_check_label_mapping(
    class_list: Sequence[str] = CANONICAL_YOLO_CLASSES,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Verify class mapping contract."""
    log = logger or logging.getLogger(__name__)
    log.info("[Sanity Check] Verifying YOLO-seg class mapping contract...")

    assert len(class_list) == 20, f"Expected 20 classes, got {len(class_list)}"
    assert class_list[0] == "apple", f"Expected idx 0='apple', got {class_list[0]!r}"
    assert class_list[4] == "coin", f"Expected idx 4='coin', got {class_list[4]!r}"
    assert class_list[19] == "tomato", f"Expected idx 19='tomato', got {class_list[19]!r}"

    log.info("  [PASS] 20 YOLO-seg classes validated: idx 0='apple', idx 4='coin', idx 19='tomato'.")
    log.info("  [PASS] No off-by-one label shift detected.")
    return True


def fix_ground_truth_aliasing(
    gt_by_class: Dict[str, Dict[str, Tuple[float, float]]],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Ensure both 'fired_dough_twist' and 'fried_dough_twist' exist in GT lookup dict."""
    log = logger or logging.getLogger(__name__)
    out = dict(gt_by_class)

    if "fried_dough_twist" in out and "fired_dough_twist" not in out:
        out["fired_dough_twist"] = out["fried_dough_twist"]
        log.debug("Aliased GT 'fried_dough_twist' -> 'fired_dough_twist'")
    elif "fired_dough_twist" in out and "fried_dough_twist" not in out:
        out["fried_dough_twist"] = out["fired_dough_twist"]
        log.debug("Aliased GT 'fired_dough_twist' -> 'fried_dough_twist'")

    return out


def run_smoke_test(
    predict_fn: Callable[[Path, float], List[dict]],
    sample_images: Sequence[Path],
    conf: float = 0.25,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Execute predictor on representative images to verify no crashes."""
    log = logger or logging.getLogger(__name__)
    log.info("[Smoke Test] Testing YOLO-seg predictor on %d sample images...", len(sample_images))

    for i, p in enumerate(sample_images, 1):
        if not p.exists():
            log.warning("  Sample [%d/%d] NOT FOUND: %s", i, len(sample_images), p)
            continue
        dets = predict_fn(p, conf)
        cls_counts = {}
        for d in dets:
            c = d.get("class_name", "unknown")
            cls_counts[c] = cls_counts.get(c, 0) + 1
        log.info("  Sample [%d/%d] %s: %d detections, classes=%s", i, len(sample_images), p.name, len(dets), cls_counts)

    log.info("[Smoke Test] PASS -- All smoke test samples processed without crash.")


def compute_speed_report(
    times_list: List[float],# Danh sách thời gian chạy từng ảnh (giây), ví dụ: [0.015, 0.018, 0.014...]
    ndets_list: List[int],# Số lượng vật thể phát hiện được ở mỗi ảnh, ví dụ: [2, 3, 1...]
    nmasks_list: List[int],# Số lượng mask được gọi ở mỗi ảnh, ví dụ: [2, 3, 1...]
    config_dict: Dict[str, Any],# Cấu hình mô hình, ví dụ: {"model_type": "yolo26seg", "conf_thresh": 0.25, "iou_thresh": 0.45, "topk_dets": 100, "topk_masks": 100, "input_size": [640, 640], "use_half": True, "image_size": 640, "model_path": "outputs/models/ecustfd_yolo26seg_best.pt"}
    run_dir: Path,# Thư mục lưu kết quả
    backend_title: str = "YOLO26-seg (Instance Segmentation)",# Tên backend
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Compute per-image latency distribution and write speed_per_image.json."""
    log = logger or logging.getLogger(__name__)
    n = len(times_list)# Số lượng ảnh đã xử lý

    if n == 0:
        log.warning("No timed samples recorded.")
        return {}

    sorted_times = sorted(times_list)
    total_sec = sum(times_list)

    def _p(pct: float) -> float:
        idx = int(round((pct / 100.0) * (n - 1)))
        return sorted_times[min(max(0, idx), n - 1)]

    mean_sec = statistics.mean(times_list)
    median_sec = statistics.median(times_list)
    p90_sec = _p(90)
    p95_sec = _p(95)
    p99_sec = _p(99)
    min_sec = sorted_times[0]
    max_sec = sorted_times[-1]
    stdev_sec = statistics.stdev(times_list) if n > 1 else 0.0

    stats = {
        "config": config_dict,
        "metric": "end_to_end_per_image_seconds",
        "backend": backend_title,
        "n_images": n,
        "total_seconds": total_sec,
        "images_per_second": n / total_sec if total_sec > 0 else 0.0,
        "mean": mean_sec,
        "median": median_sec,
        "p90": p90_sec,
        "p95": p95_sec,
        "p99": p99_sec,
        "min": min_sec,
        "max": max_sec,
        "stdev": stdev_sec,
        "avg_n_detections": sum(ndets_list) / n if n > 0 else 0.0,
        "avg_n_mask_calls": sum(nmasks_list) / n if n > 0 else 0.0,
    }

    out_json = run_dir / "speed_per_image.json"
    out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("[timer] Speed report saved -> %s", out_json)

    print("\n" + "=" * 70)
    print(f"  PER-IMAGE END-TO-END SPEED ({backend_title})")
    print("=" * 70)
    print(f"  n_images         : {n}")
    print(f"  total wall time  : {total_sec:.2f}s ({total_sec/60:.2f} min)")
    print(f"  throughput       : {stats['images_per_second']:.3f} images/sec")
    print("  Per-image latency:")
    print(f"    mean   : {mean_sec*1000:.1f} ms  ({mean_sec:.4f}s)")
    print(f"    median : {median_sec*1000:.1f} ms  ({median_sec:.4f}s)")
    print(f"    p90    : {p90_sec*1000:.1f} ms  ({p90_sec:.4f}s)")
    print(f"    p95    : {p95_sec*1000:.1f} ms  ({p95_sec:.4f}s)")
    print(f"    p99    : {p99_sec*1000:.1f} ms  ({p99_sec:.4f}s)")
    print(f"    min/max: {min_sec*1000:.1f} ms / {max_sec*1000:.1f} ms")
    print(f"    stdev  : {stdev_sec*1000:.1f} ms")
    print(f"  avg #dets / #masks per img: {stats['avg_n_detections']:.2f} / {stats['avg_n_mask_calls']:.2f}")
    print("=" * 70 + "\n")

    return stats

# Hàm format và in ra báo cáo
def format_and_print_report(
    json_path: Path,# Đường dẫn đến file JSON chứa báo cáo
    title: str = "Per-class ME_vol / ME_mass",# Tiêu đề báo cáo
) -> Tuple[pd.DataFrame, dict]:
    """Load evaluation JSON report and format as a Markdown table."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8")) # Load file JSON chứa báo cáo
    per_class = data.get("per_class", []) # Lấy danh sách các lớp

    rows = [] # Tạo list rỗng để chứa các dòng báo cáo
    for r in per_class: # Duyệt qua các lớp
        rows.append({ # Thêm vào list rows
            "Class": r["class_name"], # Tên lớp
            "n": r["n_samples"], # Số lượng mẫu
            "ME_vol (%)": r["me_volume_pct"], # ME_vol (%)
            "|ME_vol| (%)": r["abs_me_volume_pct"], # |ME_vol| (%)
            "ME_mass (%)": r["me_mass_pct"], # ME_mass (%)
        })

    df = pd.DataFrame(rows) # Tạo DataFrame từ list rows
    if not df.empty: # Nếu DataFrame không rỗng
        df = df.sort_values("|ME_vol| (%)").reset_index(drop=True) # Sắp xếp theo |ME_vol| (%) và reset index

    print(f"\n=== {title} ===") # In tiêu đề báo cáo
    if not df.empty: # Nếu DataFrame không rỗng
        try:
            print(df.to_markdown(index=False, floatfmt=".2f")) # In DataFrame theo định dạng Markdown
        except (ImportError, AttributeError):
            print(df.to_string(index=False))
    else:
        print("No samples evaluated.")

    overall = data.get("overall", {}) # Lấy overall metrics từ data
    print("\n=== Overall Metrics ===") # In tiêu đề overall metrics
    for k, v in overall.items(): # Duyệt qua overall metrics
        val_str = f"{v:.4f}" if isinstance(v, float) else str(v) # Format value
        print(f"  {k:20s}: {val_str}") # In overall metrics

    return df, data
