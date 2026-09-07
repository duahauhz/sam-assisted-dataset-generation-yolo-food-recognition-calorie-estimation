"""End-to-end pipeline package.

Public entry points:

- ``run_e2e.main``           — full pipeline run
- ``evaluate_seg.main``      — re-export YOLO-seg metrics
- ``render_report.main``     — aggregate JSON reports into Markdown
"""

from .run_e2e import main
from .evaluate_seg import main as evaluate_seg_main
from .render_report import main as render_report_main

__all__ = ["main", "evaluate_seg_main", "render_report_main"]
