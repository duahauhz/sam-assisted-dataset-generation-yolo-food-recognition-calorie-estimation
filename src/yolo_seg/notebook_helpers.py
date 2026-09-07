"""Helper utilities shared by yolo_seg notebooks (Tee + _header)."""

from pathlib import Path
from typing import Sequence


class Tee:
    """Split stdout between an original stream (cell output) and a log file.

    - Cell output: raw text (preserves ``\\r`` for live progress bar overwrites)
    - Log file:    sanitized (``\\r`` → ``\\n``) so each update is its own line
    """

    def __init__(self, cell_stream, log_fh):
        self.cell_stream = cell_stream
        self.log_fh = log_fh

    def write(self, s):
        self.cell_stream.write(s)
        if s:
            self.log_fh.write(s.replace('\r', '\n'))
        return len(s)

    def flush(self):
        for t in (self.cell_stream, self.log_fh):
            try:
                t.flush()
            except Exception:
                pass

    def isatty(self):
        return getattr(self.cell_stream, 'isatty', lambda: False)()


PROJ = Path('E:/AI_Research/dlt8')
DATA_DIR = PROJ / 'data' / 'processed' / 'yolo_ecustfd_seg'
LOG_DIR = DATA_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _header(run_label: str, lines: Sequence[str]) -> str:
    """Build the canonical ``[run ] …`` header block.

    Matches the format used by notebook 03 so logs are grep-compatible.
    """
    sep = '-' * 70
    body = '\n'.join(f'[{k:<6}] {v}' for k, v in (('run ', run_label), *lines))
    return f'{body}\n{sep}\n'


if __name__ == '__main__':
    print('This is a helper module, not a CLI.  Import it from a notebook cell.')
