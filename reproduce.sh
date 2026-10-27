#!/bin/bash
# One command to rebuild every table, figure and number in the paper from the raw outputs.
# Needs: python3, numpy, pandas, scipy, matplotlib, and a LaTeX install (only for the last step).
set -e
cd "$(dirname "$0")"
export ARTIFACT_ROOT="$(pwd)"
[ -f raw/rows.jsonl ] || gunzip -k raw/rows.jsonl.gz
# 1. recompute statistics from the raw outputs (no GPU, no model is run)
CTITRIAGE_DRY_RUN=1 CTITRIAGE_GEN_STOP=0 CTITRIAGE_ROOT="$ARTIFACT_ROOT" python3 code/run_experiment_colab.py > statistics_run.log 2>&1
# 2. extra analyses, figures, tables, paper numbers
cd analysis
mkdir -p figs tabs
python3 extra_analysis.py > /dev/null
python3 build_assets.py > /dev/null
python3 build_paper.py
echo "Done. Tables are in analysis/tabs, figures in analysis/figs, paper text in analysis/paper.tex"
