#!/usr/bin/env bash
# Full rerun with the increment-scaled loss (configs/default.yaml).  Old results are kept under the v1 run ids.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
M="--pinc-model results/models/pinc_v2_s0 --blackbox-model results/models/blackbox_v2_s0"
mkdir -p results/logs
run() { local name=$1; shift; echo "== $name start $(date +%T)"; /usr/bin/time -f "$name wall %e s" "$@" > "results/logs/$name.log" 2>&1; echo "== $name exit $? $(date +%T)"; }
echo "== models start $(date +%T)"
$PY -m pinc.train --seed 0 --run-id pinc_v2_s0 > results/logs/train_pinc_v2_s0.log 2>&1 &
$PY -m pinc.train --seed 0 --run-id blackbox_v2_s0 --set loss.lam=0 > results/logs/train_blackbox_v2_s0.log 2>&1 &
wait; echo "== models exit $(date +%T)"
run e1_v2 $PY -m experiments.e1_open_loop --run-id e1_v2 $M
run e4_v2 $PY -m experiments.e4_timing --run-id e4_v2 $M
run e3_v2 $PY -m experiments.e3_closed_loop --run-id e3_v2 $M
run e5_v2 $PY -m experiments.e5_robustness --run-id e5_v2 $M
run e6_v2 $PY -m experiments.e6_ablations --run-id e6_v2
run e2_v2 $PY -m experiments.e2_data_efficiency --run-id e2_v2
echo "== queue done $(date +%T)"
