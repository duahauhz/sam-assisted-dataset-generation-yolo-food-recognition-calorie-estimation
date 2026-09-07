"""Aggregate per-config JSON reports into a single Markdown report.

Combines the JSON outputs of ``run_e2e.py`` and the segmentation
metrics JSON from ``evaluate_seg.py`` into a single human-readable
Markdown file at ``outputs/reports/e2e_report.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.1f}%"


def fmt_abs(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.1f}%"


def render_per_class_section(title: str, reports: List[dict]) -> str:
    if not reports:
        return f"## {title}\n\n_No data._\n"

    # Gather all class names.
    all_classes = sorted(
        {r["class_name"] for rep in reports for r in rep["per_class"]}
    )
    # Build header.
    headers = ["Class"]
    for rep in reports:
        cfg = rep["config"]
        tag = f"{cfg['split']}@conf{cfg['conf_threshold']}"
        headers.extend([f"{tag} n", f"{tag} ME_vol", f"{tag} |ME_vol|", f"{tag} ME_mass"])
    lines = ["## " + title, "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    for cls in all_classes:
        row = [cls]
        for rep in reports:
            entry = next((r for r in rep["per_class"] if r["class_name"] == cls), None)
            if entry is None:
                row.extend(["-", "-", "-", "-"])
            else:
                row.append(str(entry["n_samples"]))
                row.append(fmt_pct(entry.get("me_volume_pct")))
                row.append(fmt_abs(entry.get("abs_me_volume_pct")))
                row.append(fmt_pct(entry.get("me_mass_pct")))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def render_overall(reports: List[dict]) -> str:
    lines = ["## Overall", "", "| Config | n_pairs | n_samples | n_classes | mean ME_vol (%) | mean |ME_vol| (%) | mean ME_mass (%) |", "|---|---|---|---|---|---|---|"]
    for rep in reports:
        cfg = rep["config"]
        ov = rep["overall"]
        tag = f"{cfg['split']}@conf{cfg['conf_threshold']}"
        lines.append(
            f"| {tag} | {cfg['n_pairs']} | {cfg['n_samples']} | {ov['n_classes']} | "
            f"{fmt_pct(ov.get('mean_me_volume_pct'))} | "
            f"{fmt_abs(ov.get('mean_abs_me_volume_pct'))} | "
            f"{fmt_pct(ov.get('mean_me_mass_pct'))} |"
        )
    return "\n".join(lines) + "\n"


def render_seg(seg: dict | None) -> str:
    if not seg:
        return "## YOLO-seg metrics\n\n_Not computed._\n"
    out = ["## YOLO-seg metrics (from training run)", ""]
    out.append("| Metric | Last | Best | Best Epoch |")
    out.append("|---|---|---|---|")
    for k, v in seg.items():
        out.append(
            f"| {k} | {v['last']:.4f} | {v['best']:.4f} | {int(v['best_epoch'])} |"
        )
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-dir",
        default="E:/AI_Research/dlt8/outputs/predictions",
    )
    parser.add_argument(
        "--seg-metrics",
        default="E:/AI_Research/dlt8/outputs/reports/seg_metrics.json",
    )
    parser.add_argument(
        "--output",
        default="E:/AI_Research/dlt8/outputs/reports/e2e_report.md",
    )
    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reports: List[dict] = []
    for p in sorted(pred_dir.glob("report_*.json")):
        d = load_json(p)
        if d is not None:
            reports.append(d)

    seg = load_json(Path(args.seg_metrics))

    sections: List[str] = []
    sections.append("# End-to-end Calorie Pipeline Report\n")
    sections.append(
        "This report aggregates the JSON outputs of "
        "`src.e2e_pipeline.run_e2e` for the requested split/conf "
        "configurations. All numbers are **signed** Mean Error (paper §4) "
        "expressed as percentages.\n"
    )
    sections.append(render_seg(seg))
    sections.append(render_overall(reports))
    sections.append(render_per_class_section("Per-class Mean Error", reports))

    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
