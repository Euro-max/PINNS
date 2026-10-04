#!/usr/bin/env bash
# Train the lambda-sweep models used by E6 (3 at a time).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SEED=${SEED:-0}
EXTRA=${EXTRA:-}
mkdir -p results/logs
run() { $PY -m pinc.train --seed "$SEED" --run-id "ab_lam$1_s$SEED" --set loss.lam="$1" $EXTRA > "results/logs/ab_lam$1_s$SEED.log" 2>&1; }
run 0 & run 0.01 & run 0.1 & wait
run 1 & run 10 & run 100 & wait
echo "lambda sweep done"
