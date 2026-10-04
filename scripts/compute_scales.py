"""Compute S_f = std of each component of f(s, u) over the training box and
write it into configs/default.yaml.  Run once; the value is then fixed."""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pinc import plant  # noqa: E402
from pinc.config import load_config, DEFAULT_CONFIG_PATH  # noqa: E402
from pinc.data import sample_box, sample_inputs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200000)
ap.add_argument("--write", action="store_true")
a = ap.parse_args()

cfg = load_config()
rng = np.random.default_rng(12345)
s0 = sample_box(a.n, cfg.box_train, rng)
u = sample_inputs(a.n, cfg, rng)
x = np.concatenate([s0, np.zeros((a.n, 2))], axis=1)
F = plant.f(x, u, cfg.params)[:, :4]
S_f = np.std(F, axis=0)
print("std of f over training box:", np.round(S_f, 4))
print("rms of f over training box:", np.round(np.sqrt(np.mean(F**2, axis=0)), 4))
if a.write:
    with open(DEFAULT_CONFIG_PATH) as fh:
        txt = fh.read()
    new = "  S_f: [" + ", ".join(f"{v:.4g}" for v in S_f) + "]   # std of f over the training box (scripts/compute_scales.py, n=%d, seed 12345)" % a.n
    txt, n = re.subn(r"^  S_f: .*$", new, txt, flags=re.M)
    assert n == 1
    with open(DEFAULT_CONFIG_PATH, "w") as fh:
        fh.write(txt)
    print("wrote", DEFAULT_CONFIG_PATH)
