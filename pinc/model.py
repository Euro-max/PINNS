"""
PINCNet: maps scaled inputs z = [t/T, s0/S_x, u/S_u] (7-D) to the scaled
state s(t)/S_x (4-D).  No clipping anywhere (fixes D14).

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
    def __init__(self, mcfg: ModelCfg, S_x, S_u, T: float, dtype: str = "float64", S_f=None, **kw):
        super().__init__(**kw)
        self._S_f = None if S_f is None else np.asarray(S_f, dtype=float)
        self.mcfg = mcfg
        self._S_x = np.asarray(S_x, dtype=float)
        self._S_u = np.asarray(S_u, dtype=float)
        self._T = float(T)
        self._dtype_str = dtype
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
        self.out = tf.keras.layers.Dense(4, activation=None, kernel_initializer=init, dtype=dtype, name="out")
        self.hard_ic = bool(mcfg.hard_ic)
        self.S_x_t = tf.constant(self._S_x, dtype=dtype)
        self.S_u_t = tf.constant(self._S_u, dtype=dtype)
        self.T_t = tf.constant(self._T, dtype=dtype)
        inc = np.ones(4)
        if getattr(mcfg, "increment_scaling", False):
            if self._S_f is None:
                raise ValueError("increment_scaling needs S_f")
            inc = self._S_f*self._T/self._S_x
        self.inc_t = tf.constant(inc, dtype=dtype)
        self.build((None, 7))

    def build(self, input_shape):
        h = tf.keras.Input(shape=(7,), dtype=self._dtype_str)
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
            return z[:, 1:5] + z[:, 0:1]*nn*self.inc_t
        return nn

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
                    T=self._T, dtype=self._dtype_str, S_f=None if self._S_f is None else self._S_f.tolist())
        with open(os.path.join(d, "model.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        self.save_weights(os.path.join(d, "weights.weights.h5"))

    @classmethod
    def load_from(cls, d: str) -> "PINCNet":
        with open(os.path.join(d, "model.json")) as fh:
            meta = json.load(fh)
        net = cls(ModelCfg(**meta["model"]), meta["S_x"], meta["S_u"], meta["T"], meta["dtype"], S_f=meta.get("S_f"))
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
    return PINCNet(cfg.model, cfg.S_x, cfg.S_u, cfg.T, cfg.dtype, S_f=cfg.S_f)
