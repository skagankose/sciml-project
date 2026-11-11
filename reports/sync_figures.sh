#!/bin/bash
# Rebuild the report's figures (reports/figures/fig_*.pdf) from the pulled outputs, at their printed size.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python scripts/findings.py > /dev/null      # runs.csv and the evaluated-run list the figures read
.venv/bin/python scripts/report_figures.py
