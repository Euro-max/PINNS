"""
PINCNet: maps scaled inputs z = [t/T, s0/S_x, u/S_u] (1 + n_s + n_u) to the
scaled state s(t)/S_x (n_s); n_s = len(S_x), n_u = len(S_u) (4 and 2 for the
bicycle system).  No clipping anywhere (fixes D14).

Hard initial condition (config `model.hard_ic`, default True):
    s_hat(t) = s0_hat + (t/T) * NN(t, s0, u)
so s(0) = s0 exactly.  With `hard_ic: false` the raw NN output is used and
the IC is enforced by the soft loss term instead (ablated in E6).

Architectures (config `model.arch`, E21; `mlp` is the network of every result before E21, unchanged):
  mlp           depth x width tanh MLP (options residual / layernorm / dropout, E7)
  adaptive      the same MLP with tanh(a_l x), a learnable slope a_l per layer (Jagtap et al. 2020)
  fourier       random Fourier features [z, sin(2 pi z B), cos(2 pi z B)] in front of the MLP
  modified_mlp  two input encoders U, V gate every hidden layer, h = (1 - Z) U + Z V (Wang, Teng, Perdikaris 2021)
  split         shared trunk, separate heads for the body, actuator and wheel states
  chebykan      Kolmogorov-Arnold layers with Chebyshev edge functions
  time_basis    s = s0 + D sum_k c_k(s0, u) phi_k(t/T), phi = {tau, tau^2, tau^3, exponentials with model.time_scales}
  deeponet      s = s0 + D sum_p b_p(s0, u) tau trunk_p(tau)  (branch / trunk networks)
  anchored      s = anchor(t; s0, u) + tau D NN, anchor = one cheap step of the physics prior (system.anchor_tf)
All of them satisfy s(0) = s0 exactly (with hard_ic) and take the time derivative by the same forward-mode JVP.
"""
from __future__ import annotations

import json
import os

import numpy as np
import tensorflow as tf

from .config import Config, ModelCfg

ARCHS = ("mlp", "adaptive", "fourier", "modified_mlp", "split", "chebykan", "time_basis", "deeponet", "anchored")
_MLP_CORE = ("mlp", "adaptive", "fourier", "modified_mlp", "anchored")


class PINCNet(tf.keras.Model):
    def __init__(self, mcfg: ModelCfg, S_x, S_u, T: float, dtype: str = "float64", S_f=None, theta0=None,
                 system=None, **kw):
        super().__init__(**kw)
        self.arch = getattr(mcfg, "arch", "mlp")
        if self.arch not in ARCHS:
            raise ValueError(f"model.arch must be one of {ARCHS}, got {self.arch!r}")
        if self.arch == "anchored" and system is None:
            raise ValueError("the anchored architecture needs the system (its physics prior)")
        self.system = system
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
        act = None if self.arch == "adaptive" else mcfg.activation      # adaptive: tanh(a_l x) applied in call
        for i in range(mcfg.depth):
            if res == "block" and i > 0:
                self.hidden.append((tf.keras.layers.Dense(mcfg.width, activation=act, kernel_initializer=init, dtype=dtype, name=f"b{i}a"),
                                    tf.keras.layers.Dense(mcfg.width, activation=None, kernel_initializer=init, dtype=dtype, name=f"b{i}b")))
            else:
                self.hidden.append(tf.keras.layers.Dense(mcfg.width, activation=act, kernel_initializer=init, dtype=dtype, name=f"h{i}"))
        self.norms = [tf.keras.layers.LayerNormalization(dtype=dtype, name=f"ln{i}") for i in range(mcfg.depth)] if mcfg.layernorm else None
        self.drop = tf.keras.layers.Dropout(float(mcfg.dropout), seed=0, dtype=dtype) if mcfg.dropout > 0 else None
        n_out = self.n_s
        if self.arch == "time_basis":
            self.time_scales = [float(x) for x in mcfg.time_scales]
            n_out = self.n_s*(3 + len(self.time_scales))
        elif self.arch == "deeponet":
            n_out = self.n_s*int(mcfg.n_basis)
        self.out = tf.keras.layers.Dense(n_out, activation=None, kernel_initializer=init, dtype=dtype, name="out")
        self._make_arch_parts(mcfg, dtype)
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

    def _make_arch_parts(self, mcfg, dtype):
        """Weights of the non-default architectures (created after the MLP's, so `mlp` is unchanged)."""
        a, w = self.arch, mcfg.width
        init = tf.keras.initializers.GlorotUniform(seed=1)
        n_in = 1 + self.n_s + self.n_u
        if a == "adaptive":
            self.slopes = self.add_weight(shape=(mcfg.depth,), initializer="ones", dtype=dtype, name="slopes", trainable=True)
        elif a == "fourier":
            B = np.random.default_rng(0).normal(0.0, float(mcfg.fourier_sigma), (n_in, int(mcfg.fourier_m)))
            self.B = tf.constant(B, dtype=dtype)                         # fixed, regenerated identically on load
        elif a == "modified_mlp":
            self.enc_u = tf.keras.layers.Dense(w, activation=mcfg.activation, kernel_initializer=init, dtype=dtype, name="enc_u")
            self.enc_v = tf.keras.layers.Dense(w, activation=mcfg.activation, kernel_initializer=init, dtype=dtype, name="enc_v")
        elif a == "split":
            groups = [[0, 1, 2, 3], [4, 5], [6, 7, 8, 9]] if self.n_s == 10 else [list(range(self.n_s))]
            self.groups = groups
            self.n_trunk = max(1, mcfg.depth - 2)
            self.heads = [[tf.keras.layers.Dense(w, activation=mcfg.activation, kernel_initializer=init, dtype=dtype, name=f"g{j}a"),
                           tf.keras.layers.Dense(w, activation=mcfg.activation, kernel_initializer=init, dtype=dtype, name=f"g{j}b"),
                           tf.keras.layers.Dense(len(g), activation=None, kernel_initializer=init, dtype=dtype, name=f"g{j}o")]
                          for j, g in enumerate(groups)]
            self.order = tf.constant(np.argsort(np.concatenate(groups)), dtype=tf.int32)
        elif a == "deeponet":
            p = int(mcfg.n_basis)
            self.trunk = [tf.keras.layers.Dense(32, activation="tanh", kernel_initializer=init, dtype=dtype, name="tr0"),
                          tf.keras.layers.Dense(32, activation="tanh", kernel_initializer=init, dtype=dtype, name="tr1"),
                          tf.keras.layers.Dense(p, activation=None, kernel_initializer=init, dtype=dtype, name="tr2")]
        elif a == "chebykan":
            deg = int(mcfg.kan_degree)
            dims = [n_in] + [w]*(int(mcfg.kan_layers) - 1) + [self.n_s]
            rng = np.random.default_rng(0)
            self.kan = []
            for l, (di, do) in enumerate(zip(dims[:-1], dims[1:])):
                C = self.add_weight(shape=(di, do, deg + 1), dtype=dtype, name=f"kan{l}", trainable=True,
                                    initializer=tf.keras.initializers.Constant(rng.normal(0.0, 1.0/np.sqrt(di*(deg + 1)), (di, do, deg + 1))))
                b = self.add_weight(shape=(do,), initializer="zeros", dtype=dtype, name=f"kanb{l}", trainable=True)
                self.kan.append((C, b))

    def build(self, input_shape):
        if self.arch == "mlp":
            h = tf.keras.Input(shape=(1 + self.n_s + self.n_u,), dtype=self._dtype_str)
            _ = self.call(h)
        else:                                       # TF ops (sin, exp, the prior) need a concrete tensor
            _ = self.call(tf.zeros((1, 1 + self.n_s + self.n_u), dtype=self._dtype_str))
        super().build(input_shape)

    def call(self, z, training=False):
        a = self.arch
        if a in ("time_basis", "deeponet"):
            return self._basis_output(z, training)
        if a == "split":
            nn = self._split_core(z, training)
        elif a == "chebykan":
            nn = self._kan_core(z)
        else:
            nn = self._mlp_core(z, training)
        if a == "anchored":
            return self._anchored_output(z, nn)
        if self.hard_ic:
            return z[:, 1:1 + self.n_s] + z[:, 0:1]*nn*self.inc_t
        return nn

    def _mlp_core(self, z, training=False, x=None, layers=None, out=True):
        a = self.arch
        x = z if x is None else x
        if a == "fourier":
            zb = 2.0*np.pi*tf.matmul(x, self.B)
            x = tf.concat([x, tf.sin(zb), tf.cos(zb)], axis=1)
        if a == "modified_mlp":
            U, V = self.enc_u(x), self.enc_v(x)
        layers = self.hidden if layers is None else layers
        h = None
        for i, layer in enumerate(layers):
            if i == 0:
                h = layer(x)
            elif self.residual == "skip":
                h = h + layer(h)
            elif self.residual == "block":
                a, b = layer
                h = h + b(a(h))
            else:
                h = layer(h)
            if a == "adaptive":
                h = tf.tanh(self.slopes[i]*h)
            elif a == "modified_mlp":
                h = (1.0 - h)*U + h*V
            if self.norms is not None:
                h = self.norms[i](h)
            if self.drop is not None:
                h = self.drop(h, training=training)
        return self.out(h) if out else h

    def _split_core(self, z, training=False):
        h = self._mlp_core(z, training, layers=self.hidden[:self.n_trunk], out=False)
        outs = []
        for d1, d2, o in self.heads:
            outs.append(o(d2(d1(h))))
        return tf.gather(tf.concat(outs, axis=1), self.order, axis=1)

    def _kan_core(self, z):
        x = z
        for C, b in self.kan:
            xt = tf.tanh(x)
            T = [tf.ones_like(xt), xt]
            for _ in range(2, C.shape[2]):
                T.append(2.0*xt*T[-1] - T[-2])
            x = tf.einsum("bik,ijk->bj", tf.stack(T[:C.shape[2]], axis=-1), C) + b
        return x

    def _basis_output(self, z, training=False):
        tau, x = z[:, 0:1], z[:, 1:]
        c = self._mlp_core(z, training, x=x)
        if self.arch == "time_basis":
            T = float(self._T)
            phi = [tau, tau**2, tau**3] + [(1.0 - tf.exp(-tau*T/ts))/(1.0 - np.exp(-T/ts)) for ts in self.time_scales]
            phi = tf.concat(phi, axis=1)                                    # (B, K), all zero at tau = 0
        else:
            h = tau
            for layer in self.trunk:
                h = layer(h)
            phi = tau*h                                                     # (B, p)
        c = tf.reshape(c, (-1, self.n_s, phi.shape[1]))
        return z[:, 1:1 + self.n_s] + tf.einsum("bik,bk->bi", c, phi)*self.inc_t

    def _anchored_output(self, z, nn):
        tau = z[:, 0:1]
        s0 = z[:, 1:1 + self.n_s]*self.S_x_t
        u = z[:, 1 + self.n_s:]*self.S_u_t
        anchor = self.system.anchor_tf(s0, u, tau*self.T_t, self.theta())/self.S_x_t   # learned theta, if any
        return anchor + tau*nn*self.inc_t

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
        if self.system is not None:
            meta["system"] = self.system.name if self.system.name == "bicycle" else f"hf_{self.system.variant.lower()}"
        if self.log_theta is not None:
            meta["theta"] = {k: float(v) for k, v in self.theta().items()}
        with open(os.path.join(d, "model.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        self.save_weights(os.path.join(d, "weights.weights.h5"))

    @classmethod
    def load_from(cls, d: str) -> "PINCNet":
        with open(os.path.join(d, "model.json")) as fh:
            meta = json.load(fh)
        system = None
        if meta.get("system"):
            from .system import get_system
            system = get_system(meta["system"])
        net = cls(ModelCfg(**meta["model"]), meta["S_x"], meta["S_u"], meta["T"], meta["dtype"], S_f=meta.get("S_f"),
                  theta0=meta.get("theta0"), system=system)
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
    system = None
    if getattr(cfg.model, "arch", "mlp") == "anchored":
        from .system import get_system
        system = get_system(cfg)
    return PINCNet(cfg.model, cfg.S_x, cfg.S_u, cfg.T, cfg.dtype, S_f=cfg.S_f, theta0=theta0, system=system)
