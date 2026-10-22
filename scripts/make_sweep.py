"""Thin wrapper: `python scripts/make_sweep.py ...` == `python -m rbbench make-sweep ...` (see --help)."""

import sys

from rbbench.cli import main

if __name__ == "__main__":
    sys.exit(main(["make-sweep", *sys.argv[1:]]))
