#!/usr/bin/env bash
# Reproduce every artifact in results/ from a clean checkout.
#   QUICK=1 scripts/run_all.sh     -> reduced sizes (smoke run, results labelled quick_*)
#   scripts/run_all.sh             -> full sizes (see README for expected runtimes)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
Q=""; SUF=""
if [[ "${QUICK:-0}" == "1" ]]; then Q="--quick"; SUF="_quick"; fi
mkdir -p results/logs

echo "== tests"
$PY -m pytest -q -p no:warnings

echo "== scales (no-op unless the vehicle box changed)"
$PY scripts/compute_scales.py

echo "== E7 architecture study (depth/width/dropout/residual/layernorm x lambda; informs configs/default.yaml)"
$PY -m experiments.e7_architecture $Q --run-id "e7${SUF}"

echo "== E6 ablations (lambda sweep + ablations at the config lambda)"
$PY -m experiments.e6_ablations $Q --run-id "e6${SUF}"

echo "== default models (model and lambda from configs/default.yaml, chosen from E7/E6 on the validation set)"
$PY -m pinc.train --seed 0 --run-id pinc_default_s0
$PY -m pinc.train --seed 0 --run-id blackbox_default_s0 --set loss.lam=0

echo "== E1 open loop";        $PY -m experiments.e1_open_loop $Q --run-id "e1${SUF}"
echo "== E2 data efficiency";  $PY -m experiments.e2_data_efficiency $Q --run-id "e2${SUF}"
echo "== E4 timing";           $PY -m experiments.e4_timing $Q --run-id "e4${SUF}"
echo "== E3 closed loop";      $PY -m experiments.e3_closed_loop $Q --run-id "e3${SUF}"
echo "== E5 robustness";       $PY -m experiments.e5_robustness $Q --run-id "e5${SUF}"
echo "all done; see results/MANIFEST.md"
