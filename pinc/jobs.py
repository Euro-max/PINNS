"""
Run many `python -m pinc.train` jobs as subprocesses on a set of device slots.

Training is bound by float64 arithmetic, which a consumer GPU (RTX 5080 Laptop) does
no faster than the 24-thread CPU of the same machine; running one job on the GPU and
one on the CPU at the same time roughly doubles throughput, while stacking several jobs
on one device only slows each of them down (README, "Hardware and runtimes").

A slot is "gpu" or "cpu"; e.g. slots "gpu,cpu" run two jobs at once.  CPU jobs are hidden
from the GPU (CUDA_VISIBLE_DEVICES=-1) and get `cpu_threads` intra-/inter-op threads each.
Finished runs (results/models/<run_id>/summary.json exists) are skipped, so an interrupted
batch can simply be restarted.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from .config import RESULTS_DIR, ROOT

DEFAULT_SLOTS = "gpu,cpu"


def summary(run_id: str, exp: str = "models"):
    p = os.path.join(RESULTS_DIR, exp, run_id, "summary.json")
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return None


def parse_slots(slots) -> list:
    if isinstance(slots, str):
        slots = [s.strip() for s in slots.split(",") if s.strip()]
    bad = [s for s in slots if s not in ("gpu", "cpu")]
    if bad or not slots:
        raise ValueError(f"slots must be a non-empty list of 'gpu' / 'cpu', got {slots!r}")
    return list(slots)


def default_cpu_threads(slots) -> int:
    """Split the logical CPUs between the CPU slots, keeping two for the GPU job(s) and the OS."""
    n_cpu = sum(1 for s in slots if s == "cpu")
    if n_cpu == 0:
        return 0
    return max(1, ((os.cpu_count() or 4) - 2*any(s == "gpu" for s in slots))//n_cpu)


def run_jobs(jobs, slots=DEFAULT_SLOTS, config: str | None = None, overrides=(), cpu_threads: int | None = None,
             exp: str = "models", poll: float = 2.0, verbose: bool = True) -> dict:
    """jobs: list of (run_id, seed, {dotted key: value}).  Returns {run_id: (slot, wall seconds)} for the
    jobs run now.  Raises if any job fails (after the others in flight have finished)."""
    slots = parse_slots(slots)
    cpu_threads = cpu_threads or default_cpu_threads(slots)
    pending = [j for j in jobs if summary(j[0], exp) is None]
    if verbose:
        print(f"  {len(jobs) - len(pending)} reused, {len(pending)} to train on slots {slots}"
              + (f" ({cpu_threads} threads per CPU job)" if "cpu" in slots else ""), flush=True)
    logdir = os.path.join(RESULTS_DIR, "logs")
    os.makedirs(logdir, exist_ok=True)
    free = list(range(len(slots)))
    running, done, failed = [], {}, []
    while pending or running:
        while pending and free and not failed:
            i = free.pop(0)
            rid, seed, ov = pending.pop(0)
            cmd = [sys.executable, "-m", "pinc.train", "--seed", str(seed), "--run-id", rid, "--exp", exp]
            if config:
                cmd += ["--config", config]
            for k, v in ov.items():
                cmd += ["--set", f"{k}={v}"]
            for o in overrides:
                cmd += ["--set", o]
            env = dict(os.environ, PYTHONPATH=ROOT)
            if slots[i] == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = "-1"
                cmd += ["--threads", str(cpu_threads)]
            log = open(os.path.join(logdir, rid + ".log"), "w")
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
            running.append((rid, i, p, log, time.time()))
            if verbose:
                print(f"  started  {rid} on {slots[i]}", flush=True)
        for item in list(running):
            rid, i, p, log, t0 = item
            if p.poll() is None:
                continue
            log.close()
            running.remove(item)
            free.append(i)
            wall = time.time() - t0
            if p.returncode != 0:
                failed.append(rid)
            else:
                done[rid] = (slots[i], wall)
            if verbose:
                print(f"  finished {rid} on {slots[i]} (exit {p.returncode}, {wall:.0f} s)", flush=True)
        if failed and not running:
            break
        time.sleep(poll)
    if failed:
        raise RuntimeError(f"runs failed: {failed}; see results/logs/<run_id>.log")
    return done


def add_slot_args(parser):
    parser.add_argument("--slots", default=DEFAULT_SLOTS,
                        help="comma list of device slots for training jobs, e.g. gpu,cpu (one job per slot)")
    parser.add_argument("--cpu-threads", type=int, default=None, help="threads per CPU job (default: split the CPUs)")
    return parser
