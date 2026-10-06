"""
Training: Adam (cosine / exponential decay) followed by L-BFGS (SciPy),
collocation points resampled every epoch, model selection on the validation
loss with the best weights restored before saving (fixes D12, D13, D15).

CLI:  python -m pinc.train --config configs/default.yaml --seed 0 [--set key=value ...]
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

from . import loss as L
from .config import Config, add_config_args, load_config
from .data import make_splits, sample_collocation, sample_ic, scale_inputs
from .metrics import nrmse, rmse
from .model import build_model
from .runinfo import make_run_dir, manifest_append, save_json, write_meta
from .tfsetup import setup


def _lr_schedule(cfg: Config, steps_per_epoch: int):
    tr = cfg.train
    total = max(1, tr.epochs*steps_per_epoch)
    if tr.lr_decay == "cosine":
        return tf.keras.optimizers.schedules.CosineDecay(tr.lr, total, alpha=tr.lr_final_frac)
    if tr.lr_decay == "exponential":
        return tf.keras.optimizers.schedules.ExponentialDecay(tr.lr, total, tr.lr_final_frac, staircase=False)
    if tr.lr_decay == "none":
        return tr.lr
    raise ValueError(tr.lr_decay)


def _tensors(d, cfg, dtype):
    z = tf.constant(scale_inputs(d["t"], d["s0"], d["u"], cfg), dtype)
    s = tf.constant(d["s"], dtype) if "s" in d else None
    return z, s


def evaluate(net, z_d, s_d, z_ic, z_c, cfg) -> dict:
    out = L.total_loss(net, z_d, s_d, z_ic, z_c, cfg)
    return {k: float(v) for k, v in out.items()}


def test_metrics(net, split, cfg) -> dict:
    z, s = _tensors(split, cfg, cfg.dtype)
    pred = net.physical(net(z)).numpy()
    return dict(rmse=rmse(pred, split["s"]).tolist(), nrmse=nrmse(pred, split["s"], cfg.S_x).tolist(),
                n=int(len(split["t"])))


def train(cfg: Config, seed: int, run_id: str | None = None, exp: str = "models",
          verbose: bool = True, splits=None, threads: int | None = None) -> dict:
    dtype = setup(seed, cfg.dtype, threads=threads)
    tr = cfg.train
    run_id = run_id or f"pinc_lam{cfg.loss.lam:g}_n{tr.n_data}_s{seed}"
    run_dir = make_run_dir(exp, run_id)
    t_start = time.time()

    splits = splits or make_splits(cfg)
    if getattr(cfg.model, "greybox", False):
        if cfg.loss.lam != 0:
            raise ValueError("the grey-box model is trained on data only (loss.lam = 0)")
        from .greybox import to_residual
        splits = {k: to_residual(v, cfg) for k, v in splits.items()}
    z_tr, s_tr = _tensors(splits["train"], cfg, dtype)
    z_val, s_val = _tensors(splits["val"], cfg, dtype)
    n_ic = min(tr.n_data, 4000)
    z_ic_tr, _ = _tensors(sample_ic(n_ic, cfg.seeds.train + 1000, cfg), cfg, dtype)
    z_ic_val, _ = _tensors(sample_ic(min(tr.n_val, 2000), cfg.seeds.val + 1000, cfg), cfg, dtype)
    z_c_val, _ = _tensors(sample_collocation(tr.n_val, cfg.seeds.val + 2000, cfg), cfg, dtype)

    net = build_model(cfg)
    n_train = int(z_tr.shape[0])
    steps_per_epoch = int(np.ceil(n_train/tr.batch_data))
    if tr.steps > 0:                                   # fixed optimiser budget regardless of n_data
        tr.epochs = int(np.ceil(tr.steps/steps_per_epoch))
    opt = tf.keras.optimizers.Adam(learning_rate=_lr_schedule(cfg, steps_per_epoch))

    @tf.function
    def train_step(z_d, s_d, z_ic, z_c):
        with tf.GradientTape() as tape:
            out = L.total_loss(net, z_d, s_d, z_ic, z_c, cfg, training=True)
        grads = tape.gradient(out["total"], net.trainable_variables)
        opt.apply_gradients(zip(grads, net.trainable_variables))
        return out

    @tf.function
    def eval_step(z_d, s_d, z_ic, z_c):
        return L.total_loss(net, z_d, s_d, z_ic, z_c, cfg)

    def val_loss():
        return {k: float(v) for k, v in eval_step(z_val, s_val, z_ic_val, z_c_val).items()}

    select = getattr(tr, "select_on", "total")
    history, best = [], dict(val=np.inf, epoch=-1, weights=None, stage="init")

    def consider(v, epoch, stage):
        if v[select] < best["val"]:
            best.update(val=v[select], epoch=epoch, weights=net.get_flat_weights(), stage=stage)

    rng = np.random.default_rng(seed)
    step = 0
    for epoch in range(tr.epochs):
        colloc = sample_collocation(tr.n_colloc, cfg.seeds.colloc + 7919*seed + epoch, cfg)   # resampled every epoch
        z_c_all, _ = _tensors(colloc, cfg, dtype)
        perm = rng.permutation(n_train)
        perm_c = rng.permutation(int(z_c_all.shape[0]))
        perm_ic = rng.permutation(int(z_ic_tr.shape[0]))
        acc = dict(total=0.0, data=0.0, ic=0.0, phys=0.0)
        for b in range(steps_per_epoch):
            idx = perm[b*tr.batch_data:(b + 1)*tr.batch_data]
            idc = perm_c[(b*tr.batch_colloc) % len(perm_c):][:tr.batch_colloc]
            if len(idc) < tr.batch_colloc:
                idc = np.concatenate([idc, perm_c[:tr.batch_colloc - len(idc)]])
            idi = perm_ic[(b*256) % len(perm_ic):][:256]
            out = train_step(tf.gather(z_tr, idx), tf.gather(s_tr, idx), tf.gather(z_ic_tr, idi), tf.gather(z_c_all, idc))
            for k in acc:
                acc[k] += float(out[k])
            step += 1
        for k in acc:
            acc[k] /= steps_per_epoch
        if not np.isfinite(acc["total"]):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}: {acc}")
        row = dict(epoch=epoch, stage="adam", step=step, lr=float(opt.learning_rate) if not callable(opt.learning_rate) else float(opt.learning_rate(step)),
                   time=time.time() - t_start, **{f"train_{k}": v for k, v in acc.items()})
        if (epoch + 1) % tr.val_every == 0 or epoch == tr.epochs - 1:
            v = val_loss()
            row.update({f"val_{k}": vv for k, vv in v.items()})
            consider(v, epoch, "adam")
        history.append(row)
        if verbose and ((epoch + 1) % tr.log_every == 0 or epoch == 0):
            print(f"[{run_id}] epoch {epoch+1}/{tr.epochs} train {acc['total']:.3e} (data {acc['data']:.2e} phys {acc['phys']:.2e}) "
                  f"val {row.get('val_total', float('nan')):.3e} best {best['val']:.3e}@{best['epoch']} {row['time']:.0f}s", flush=True)

    # ---- L-BFGS stage (full batch, fixed collocation set) ------------------
    if tr.lbfgs_iters > 0:
        if best["weights"] is not None:
            net.set_flat_weights(best["weights"])
        colloc = sample_collocation(tr.n_colloc, cfg.seeds.colloc + 7919*seed + tr.epochs, cfg)
        z_c_fix, _ = _tensors(colloc, cfg, dtype)
        lbfgs_training = False        # deterministic objective for L-BFGS (dropout off); Adam uses dropout

        @tf.function
        def loss_and_grad():
            with tf.GradientTape() as tape:
                out = L.total_loss(net, z_tr, s_tr, z_ic_tr, z_c_fix, cfg, training=lbfgs_training)
            g = tape.gradient(out["total"], net.trainable_variables)
            return out["total"], tf.concat([tf.reshape(gi, [-1]) for gi in g], 0)

        it = dict(n=0, last_total=np.nan)

        def fun(w):
            net.set_flat_weights(w)
            tot, g = loss_and_grad()
            if not np.isfinite(float(tot)):
                raise FloatingPointError("non-finite loss in L-BFGS")
            it["last_total"] = float(tot)
            return float(tot), g.numpy().astype(np.float64)

        def cb(w):
            it["n"] += 1
            if it["n"] % tr.val_every == 0:
                net.set_flat_weights(w)
                v = val_loss()
                consider(v, tr.epochs + it["n"], "lbfgs")
                history.append(dict(epoch=tr.epochs + it["n"], stage="lbfgs", step=step + it["n"], lr=0.0,
                                    time=time.time() - t_start, train_total=it["last_total"],
                                    **{f"val_{k}": vv for k, vv in v.items()}))
                if verbose and it["n"] % tr.log_every == 0:
                    print(f"[{run_id}] lbfgs {it['n']}/{tr.lbfgs_iters} train {it['last_total']:.3e} val {v['total']:.3e} best {best['val']:.3e}", flush=True)

        w0 = net.get_flat_weights().astype(np.float64)
        minimize(fun, w0, jac=True, method="L-BFGS-B", callback=cb,
                 options=dict(maxiter=tr.lbfgs_iters, maxfun=tr.lbfgs_iters*2, ftol=1e-15, gtol=1e-12, maxcor=50))

    # ---- restore best, evaluate, save --------------------------------------
    assert best["weights"] is not None
    net.set_flat_weights(best["weights"])
    v = val_loss()
    net.save_to(run_dir)
    with open(os.path.join(run_dir, "loss_curve.csv"), "w", newline="") as fh:
        keys = sorted({k for r in history for k in r}, key=lambda k: (k != "epoch", k))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in history:
            w.writerow(r)
    summary = dict(run_id=run_id, seed=seed, best_val=best["val"], best_epoch=best["epoch"], best_stage=best["stage"],
                   val=v, test=test_metrics(net, splits["test"], cfg),
                   test_extrap=test_metrics(net, splits["test_extrap"], cfg),
                   n_params=int(sum(int(np.prod(x.shape)) for x in net.trainable_variables)),
                   train_seconds=time.time() - t_start, lam=cfg.loss.lam, n_data=tr.n_data,
                   theta={k: float(v) for k, v in net.theta().items()} or None)
    save_json(summary, os.path.join(run_dir, "summary.json"))
    write_meta(run_dir, cfg, seed, dict(run_id=run_id, exp=exp))
    manifest_append([os.path.join(run_dir, "summary.json"), os.path.join(run_dir, "weights.weights.h5")])
    if verbose:
        print(f"[{run_id}] done: best val {best['val']:.3e} ({best['stage']} @ {best['epoch']}), "
              f"test NRMSE {np.round(summary['test']['nrmse'], 4)}, extrap NRMSE {np.round(summary['test_extrap']['nrmse'], 4)}")
    summary["run_dir"] = run_dir
    return summary


def main(argv=None):
    ap = add_config_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--exp", default="models")
    ap.add_argument("--threads", type=int, default=None, help="intra/inter-op threads (1 for bit-reproducible runs)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    train(cfg, a.seed, a.run_id, a.exp, threads=a.threads)


if __name__ == "__main__":
    main()
