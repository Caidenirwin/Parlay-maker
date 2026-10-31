"""CSV locations for the scrapers, models, and dashboard.

Scripts keep using a bare filename (`"odds.csv"`). `csv_path` stores that file
under `data/` next to this module, so the repo root stays limited to code.
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")


def csv_path(name):
    """Return the path to a CSV in `data/`. `name` is a filename, not a directory."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, os.path.basename(str(name).replace("\\", "/")))
