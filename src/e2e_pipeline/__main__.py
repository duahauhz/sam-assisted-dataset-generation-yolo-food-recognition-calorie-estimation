"""``python -m src.e2e_pipeline`` entry point.

Chooses a sub-command based on the first positional argument:

    python -m src.e2e_pipeline run     --split val --conf 0.5
    python -m src.e2e_pipeline seg      --run-dir runs/yolo_seg/...
    python -m src.e2e_pipeline report
"""

import sys

from .evaluate_seg import main as seg_main
from .render_report import main as report_main
from .run_e2e import main as run_main


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sub = sys.argv.pop(1)
    if sub == "run":
        run_main()
    elif sub == "seg":
        seg_main()
    elif sub == "report":
        report_main()
    else:
        print(f"Unknown sub-command '{sub}'. Use 'run', 'seg', or 'report'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
