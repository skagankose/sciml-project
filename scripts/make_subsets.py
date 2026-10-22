"""Thin wrapper: `python scripts/make_subsets.py ...` == `python -m rbbench make-subsets ...` (see --help)."""

import sys

from rbbench.cli import main

if __name__ == "__main__":
    sys.exit(main(["make-subsets", *sys.argv[1:]]))
