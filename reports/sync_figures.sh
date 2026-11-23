#!/bin/bash
# Rebuild the reports' figures (mid-project: reports/figures/fig_*.pdf; final: reports/figures/final/) from the pulled outputs, at their printed size.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python scripts/findings.py > /dev/null      # runs.csv and the evaluated-run list the figures read
.venv/bin/python scripts/report_figures.py
.venv/bin/python scripts/final_figures.py            # final report: reports/figures/final/
