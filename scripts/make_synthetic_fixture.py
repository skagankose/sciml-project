"""Thin wrapper: `python scripts/make_synthetic_fixture.py ...` == `python -m rbbench fixture ...` (see --help)."""

import sys

from rbbench.cli import main

if __name__ == "__main__":
    sys.exit(main(["fixture", *sys.argv[1:]]))
