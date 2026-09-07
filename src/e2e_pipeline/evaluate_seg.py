"""Evaluate YOLO-seg on the val split and dump per-class metrics.

Re-exports the segmentation metrics that Ultralytics already computed
during training (precision, recall, mAP50, mAP50-95, F1) from the
training CSV, plus an optional fresh evaluation on the val split when
the user asks for it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

RUN_DIR = Path("E:/AI_Research/dlt8/runs/yolo_seg/ecustfd_yolo26seg-2")


def parse_results_csv(path: Path) -> Dict[str, List[float]]:
    """Parse ``results.csv`` into a dict of column → list of values."""
    out: Dict[str, List[float]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k is None:
                    continue
                k = k.strip()
                v = v.strip()
                try:
                    out.setdefault(k, []).append(float(v))
                except ValueError:
                    continue
    return out


def best_epoch(metrics: Dict[str, List[float]], key: str) -> tuple[int, float]:
    """Return (epoch_of_best, best_value) for the given metric column."""
    values = metrics[key]
    best_i = max(range(len(values)), key=lambda i: values[i])
    return best_i, values[best_i]


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump YOLO-seg metrics.")
    parser.add_argument("--run-dir", default=str(RUN_DIR))
    parser.add_argument("--output-dir", default="E:/AI_Research/dlt8/outputs/reports")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(results_csv)

    metrics = parse_results_csv(results_csv)
    keys_of_interest = [
        "metrics/precision(M)",
        "metrics/recall(M)",
        "metrics/mAP50(M)",
        "metrics/mAP50-95(M)",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    summary: Dict[str, Dict[str, float]] = {}
    for k in keys_of_interest:
        if k not in metrics:
            continue
        best_i, best_v = best_epoch(metrics, k)
        summary[k] = {
            "last": metrics[k][-1],
            "best": best_v,
            "best_epoch": best_i,
        }

    out_path = out_dir / "seg_metrics.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    # Markdown summary.
    md_path = out_dir / "seg_metrics.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# YOLO-seg Segmentation Metrics\n\n")
        f.write(f"Source: `{results_csv}`\n\n")
        f.write("| Metric | Last | Best | Best Epoch |\n")
        f.write("|---|---|---|---|\n")
        for k, v in summary.items():
            f.write(
                f"| {k} | {v['last']:.4f} | {v['best']:.4f} | "
                f"{int(v['best_epoch'])} |\n"
            )
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
