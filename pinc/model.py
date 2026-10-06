"""
PINCNet: maps scaled inputs z = [t/T, s0/S_x, u/S_u] (1 + n_s + n_u) to the
scaled state s(t)/S_x (n_s); n_s = len(S_x), n_u = len(S_u) (4 and 2 for the
bicycle system).  No clipping anywhere (fixes D14).

Hard initial condition (config `model.hard_ic`, default True):
    s_hat(t) = s0_hat + (t/T) * NN(t, s0, u)
so s(0) = s0 exactly.  With `hard_ic: false` the raw NN output is used and
the IC is enforced by the soft loss term instead (ablated in E6).
"""
from __future__ import annotations

import json
import os

import numpy as np
import tensorflow as tf

from .config import Config, ModelCfg


class PINCNet(tf.keras.Model):
    def __init__(self, mcfg: ModelCfg, S_x, S_u, T: float, dtype: str = "float64", S_f=None, theta0=None, **kw):
        super().__init__(**kw)
        # learnable prior parameters (plan decision 2d): log-parametrised (positive), trained with the weights;
        # used only by the physics residual (pinc/loss.py)
        self.theta0 = dict(theta0) if theta0 else None
        if getattr(mcfg, "learn_theta", False):
            if not self.theta0:
                raise ValueError("learn_theta needs the nominal parameters theta0")
            self.theta_names = tuple(sorted(self.theta0))
            init = np.log([self.theta0[k] for k in self.theta_names])
            # add_weight, not tf.Variable: Keras must track it, or the optimisers never update it and it is not saved
            self.log_theta = self.add_weight(shape=(len(init),), initializer=tf.keras.initializers.Constant(init),
                                             dtype=dtype, name="log_theta", trainable=True)
        else:
            self.theta_names, self.log_theta = (), None
        self._S_f = None if S_f is None else np.asarray(S_f, dtype=float)
        self.mcfg = mcfg
        self._S_x = np.asarray(S_x, dtype=float)
        self._S_u = np.asarray(S_u, dtype=float)
        self._T = float(T)
        self._dtype_str = dtype
        self.n_s, self.n_u = self._S_x.size, self._S_u.size
        init = tf.keras.initializers.GlorotUniform(seed=0)
        res = mcfg.residual
        if isinstance(res, bool):
            res = "skip" if res else "none"
        if res not in ("none", "skip", "block"):
            raise ValueError(f"model.residual must be none|skip|block, got {res!r}")
        self.residual = res
        self.hidden = []
        for i in range(mcfg.depth):
            if res == "block" and i > 0:
                self.hidden.append((tf.keras.layers.Dense(mcfg.width, activation=mcfg.activation, kernel_initializer=init, dtype=dtype, name=f"b{i}a"),
                                    tf.keras.layers.Dense(mcfg.width, activation=None, kernel_initializer=init, dtype=dtype, name=f"b{i}b")))
            else:
                self.hidden.append(tf.keras.layers.Dense(mcfg.width, activation=mcfg.activation, kernel_initializer=init, dtype=dtype, name=f"h{i}"))
        self.norms = [tf.keras.layers.LayerNormalization(dtype=dtype, name=f"ln{i}") for i in range(mcfg.depth)] if mcfg.layernorm else None
        self.drop = tf.keras.layers.Dropout(float(mcfg.dropout), seed=0, dtype=dtype) if mcfg.dropout > 0 else None
        self.out = tf.keras.layers.Dense(self.n_s, activation=None, kernel_initializer=init, dtype=dtype, name="out")
        self.hard_ic = bool(mcfg.hard_ic)
        self.S_x_t = tf.constant(self._S_x, dtype=dtype)
        self.S_u_t = tf.constant(self._S_u, dtype=dtype)
        self.T_t = tf.constant(self._T, dtype=dtype)
        inc = np.ones(self.n_s)
        if getattr(mcfg, "increment_scaling", False):
            if self._S_f is None:
                raise ValueError("increment_scaling needs S_f")
            # O(1) output per channel; capped at 1 for fast states whose change over T is bounded by S_x
            # (e.g. wheel slip, which settles within milliseconds).  Bicycle values are all < 1: unchanged.
            inc = np.minimum(self._S_f*self._T/self._S_x, 1.0)
        self.inc_t = tf.constant(inc, dtype=dtype)
        self.build((None, 1 + self.n_s + self.n_u))

    def build(self, input_shape):
        h = tf.keras.Input(shape=(1 + self.n_s + self.n_u,), dtype=self._dtype_str)
        _ = self.call(h)
        super().build(input_shape)

    def call(self, z, training=False):
        h = None
        for i, layer in enumerate(self.hidden):
            if i == 0:
                h = layer(z)
            elif self.residual == "skip":
                h = h + layer(h)
            elif self.residual == "block":
                a, b = layer
                h = h + b(a(h))
            else:
                h = layer(h)
            if self.norms is not None:
                h = self.norms[i](h)
            if self.drop is not None:
                h = self.drop(h, training=training)
        nn = self.out(h)
        if self.hard_ic:
            return z[:, 1:1 + self.n_s] + z[:, 0:1]*nn*self.inc_t
        return nn

    def theta(self):
        """Current prior parameters {name: tensor} (empty unless learn_theta)."""
        if self.log_theta is None:
            return {}
        v = tf.exp(self.log_theta)
        return {k: v[i] for i, k in enumerate(self.theta_names)}

    # ---- unit helpers ------------------------------------------------
    def physical(self, s_hat):
        return s_hat*self.S_x_t

    def scale_inputs(self, t, s0, u):
        t = tf.reshape(tf.cast(t, self.S_x_t.dtype), (-1, 1))
        s0 = tf.cast(s0, self.S_x_t.dtype)
        u = tf.cast(u, self.S_x_t.dtype)
        return tf.concat([t/self.T_t, s0/self.S_x_t, u/self.S_u_t], axis=1)

    def predict_physical(self, t, s0, u):
        return self.physical(self(self.scale_inputs(t, s0, u)))

    # ---- persistence -------------------------------------------------
    def save_to(self, d: str):
        os.makedirs(d, exist_ok=True)
        meta = dict(model=self.mcfg.__dict__, S_x=self._S_x.tolist(), S_u=self._S_u.tolist(),
                    T=self._T, dtype=self._dtype_str, S_f=None if self._S_f is None else self._S_f.tolist(),
                    theta0=self.theta0)
        if self.log_theta is not None:
            meta["theta"] = {k: float(v) for k, v in self.theta().items()}
        with open(os.path.join(d, "model.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        self.save_weights(os.path.join(d, "weights.weights.h5"))

    @classmethod
    def load_from(cls, d: str) -> "PINCNet":
        with open(os.path.join(d, "model.json")) as fh:
            meta = json.load(fh)
        net = cls(ModelCfg(**meta["model"]), meta["S_x"], meta["S_u"], meta["T"], meta["dtype"], S_f=meta.get("S_f"),
                  theta0=meta.get("theta0"))
        net.load_weights(os.path.join(d, "weights.weights.h5"))
        return net

    def get_flat_weights(self) -> np.ndarray:
        return np.concatenate([v.numpy().ravel() for v in self.trainable_variables])

    def set_flat_weights(self, flat: np.ndarray):
        i = 0
        for v in self.trainable_variables:
            n = int(np.prod(v.shape))
            v.assign(tf.reshape(tf.cast(flat[i:i+n], v.dtype), v.shape))
            i += n
        assert i == flat.size


def build_model(cfg: Config) -> PINCNet:
    theta0 = None
    if getattr(cfg.model, "learn_theta", False):
        from .system import get_system
        sysm = get_system(cfg)
        if not hasattr(sysm, "theta_nominal"):
            raise ValueError(f"system {cfg.system!r} has no learnable prior parameters")
        theta0 = sysm.theta_nominal()
    return PINCNet(cfg.model, cfg.S_x, cfg.S_u, cfg.T, cfg.dtype, S_f=cfg.S_f, theta0=theta0)
