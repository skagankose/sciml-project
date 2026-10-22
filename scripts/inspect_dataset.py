"""Thin wrapper: `python scripts/inspect_dataset.py ...` == `python -m rbbench inspect-dataset ...` (see --help)."""

import sys

from rbbench.cli import main

if __name__ == "__main__":
    sys.exit(main(["inspect-dataset", *sys.argv[1:]]))
