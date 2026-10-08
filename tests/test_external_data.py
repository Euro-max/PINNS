"""External trajectories (E30): training reads the train / val sets and the starting-state pool of a data file."""
import os

import numpy as np

from pinc.config import ROOT, load_config
from pinc.data import make_splits, sample_collocation, sample_ic


def test_external_data_file(tmp_path):
    cfg = load_config(os.path.join(ROOT, "configs", "default.yaml"))
    rng = np.random.default_rng(0)
    n_s, n_u = len(cfg.S_x), len(cfg.S_u)
    arr = lambda n: dict(t=rng.uniform(0, cfg.T, n), s0=rng.normal(size=(n, n_s)), u=rng.normal(size=(n, n_u)), s=rng.normal(size=(n, n_s)))
    tr, va = arr(50), arr(20)
    pool = rng.normal(size=(30, n_s)) + 100.0                       # recognisable values
    f = tmp_path / "ext.npz"
    np.savez(f, pool=pool, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()})
    cfg = cfg.with_overrides({"train.data_file": os.path.relpath(f, ROOT), "train.n_data": 10})
    sp = make_splits(cfg)
    assert np.array_equal(sp["train"]["s0"], tr["s0"][:10])          # nested subset: the first n_data
    assert np.array_equal(sp["val"]["s"], va["s"]) and np.array_equal(sp["test"]["t"], va["t"])
    for s0 in (sample_ic(40, 1, cfg)["s0"], sample_collocation(40, 2, cfg)["s0"]):
        assert np.all(s0 > 90.0)                                     # drawn from the pool
