#!/bin/bash
# End-to-end test of the experiment script with a MOCK model (no GPU, no downloads):
# builds the benchmark, runs every tier, computes statistics, makes figures and ZIPs.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
T="$(mktemp -d)"; ROOT="$T/run"
CTITRIAGE_DRY_RUN=1 CTITRIAGE_GEN_STOP=5000 CTITRIAGE_CKPT_EVERY=999999 CTITRIAGE_ROOT="$ROOT" \
  python3 "$HERE/code/run_experiment_colab.py" > "$T/log.txt" 2>&1
N=$(wc -l < "$ROOT/raw/rows.jsonl")
[ "$N" -ge 19000 ] || { echo "FAIL: only $N rows"; exit 1; }
[ -f "$ROOT/statistics/coverage.csv" ] && [ -f "$ROOT/statistics/pooled.csv" ] || { echo "FAIL: statistics missing"; exit 1; }
[ -f "$T/CTITRIAGE_V3_COMPLETE.zip" ] || { echo "FAIL: final ZIP missing"; exit 1; }
echo "PASS: mock pipeline produced $N rows, statistics and final ZIP"
