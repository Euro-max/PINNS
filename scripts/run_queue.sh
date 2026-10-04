#!/usr/bin/env bash
# Full-size experiment queue (sequential so that E4 timing runs on an idle machine).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p results/logs
run() { local name=$1; shift; echo "== $name start $(date +%T)"; /usr/bin/time -f "$name wall %e s maxrss %M KB" "$@" > "results/logs/$name.log" 2>&1; echo "== $name exit $? $(date +%T)"; }
run e1 $PY -m experiments.e1_open_loop --run-id e1
run e4 $PY -m experiments.e4_timing --run-id e4
run e3 $PY -m experiments.e3_closed_loop --run-id e3
run e5 $PY -m experiments.e5_robustness --run-id e5
run e6 $PY -m experiments.e6_ablations --run-id e6
run e2 $PY -m experiments.e2_data_efficiency --run-id e2
echo "== queue done $(date +%T)"
