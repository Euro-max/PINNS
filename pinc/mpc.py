"""
One MPC class with a pluggable prediction model (fixes D8, D10, D16, D17, D19).

    J = sum_{k=1..N} (z_k - z_ref,k)^T Q (z_k - z_ref,k)          (k < N)
      + (z_N - z_ref,N)^T P (z_N - z_ref,N)                       (terminal)
      + sum_{k=0..N-1} u~_k^T R u~_k + du~_k^T R_D du~_k           (u~ = u / S_u)
      + w_rmax * sum_k max(0, |r_k| - r_max)^2                     (soft yaw-rate limit)

z = [vx, vy, r, psi, X, Y]; the predictor gives s = [vx, vy, r, psi] and X, Y
are integrated from s outside the network with the trapezoidal rule.  Cost
and gradient come from one TF call; SciPy SLSQP with `jac=True`, identical
options and warm start (previous solution shifted, last input repeated) for
every predictor.  The controller only ever receives the NOMINAL parameters.
"""
from __future__ import annotations

import time

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

from .config import Config
from .system import get_system


# ---------------------------------------------------------------------------
class Predictor:
    """Maps (s0 (n_s,), u (N, n_u) physical) -> s (N, n_s) physical, differentiable."""
    name = "base"
    n_extra = 0

    def prepare(self, s0: np.ndarray, u_warm: np.ndarray):
        """Per-solve hook (used by the linearised predictor); returns extra tensors."""
        return ()

    def rollout(self, s0, u, extra=()):
        raise NotImplementedError


class PINCPredictor(Predictor):
    """One network call per control step; time input = T/T = 1.0."""
    name = "pinc"

    def __init__(self, model, cfg: Config, name="pinc"):
        self.model, self.cfg, self.name = model, cfg, name
        self.S_x = tf.constant(cfg.S_x, cfg.dtype)
        self.S_u = tf.constant(cfg.S_u, cfg.dtype)

    def step(self, s, u):
        one = tf.ones_like(s[:, :1])
        z = tf.concat([one, s/self.S_x, u/self.S_u], axis=1)
        return self.model(z)*self.S_x

    def rollout(self, s0, u, extra=()):
        s = tf.reshape(s0, (1, -1))
        out = []
        for k in range(u.shape[0]):
            s = self.step(s, u[k:k + 1])
            out.append(s)
        return tf.concat(out, axis=0)

    def rollout_batch(self, s0, u):
        """Batched chained prediction: s0 (B, n_s), u (B, N, n_u) -> (B, N, n_s)."""
        s = s0
        out = []
        for k in range(u.shape[1]):
            s = self.step(s, u[:, k])
            out.append(s)
        return tf.stack(out, axis=1)


class RK4Predictor(Predictor):
    """TF RK4 with the nominal parameters, fixed substep cfg.mpc.dt_pred.  model='prior' is the system's
    physics model (for the single-track system: the exact plant); model='true' is the true plant in
    network coordinates (high-fidelity system only: the NMPC-HF reference arm)."""
    name = "rk4"

    def __init__(self, cfg: Config, params=None, name="rk4", model="prior"):
        self.cfg, self.name = cfg, name
        self.params = params or cfg.params           # NOMINAL by construction
        self.dt, self.T = cfg.mpc.dt_pred, cfg.T
        self.tyre = "linear"
        self.sys = get_system(cfg)
        if model not in ("prior", "true"):
            raise ValueError(model)
        self._step = self.sys.rk4_step_s_tf if model == "prior" else self.sys.rk4_step_s_true_tf

    def step(self, s, u):
        """One control period of RK4 substeps as a tf.while_loop (keeps the
        traced graph small so XLA compiles in seconds, not minutes)."""
        n = int(round(self.T/self.dt))
        p, dt, tyre = self.params, self.dt, self.tyre

        def body(i, s):
            return i + 1, self._step(s, u, dt, p, tyre)
        _, s = tf.while_loop(lambda i, s: i < n, body, (tf.constant(0), s), maximum_iterations=n)
        return s

    def rollout(self, s0, u, extra=()):
        s = tf.reshape(s0, (1, -1))
        if getattr(self.cfg.mpc, "loop_rollout", False):
            # Horizon as a tf.while_loop: the traced graph holds ONE control step whatever N is.  The Python loop
            # below copies all N steps (each with its substep loop) into the graph; for the high-fidelity true model
            # that costs ~2.3 GB of XLA compile memory per step.
            N = u.shape[0]
            ta = tf.TensorArray(s.dtype, size=N)

            def body(k, s, ta):
                s = self.step(s, tf.reshape(u[k], (1, -1)))
                return k + 1, s, ta.write(k, s[0])
            _, _, ta = tf.while_loop(lambda k, s, ta: k < N, body, (tf.constant(0), s, ta), maximum_iterations=N)
            return ta.stack()
        out = []
        for k in range(u.shape[0]):
            s = self.step(s, u[k:k + 1])
            out.append(s)
        return tf.concat(out, axis=0)

    def rollout_batch(self, s0, u):
        s = s0
        out = []
        for k in range(u.shape[1]):
            s = self.step(s, u[:, k])
            out.append(s)
        return tf.stack(out, axis=1)


class GreyBoxPredictor(PINCPredictor):
    """Prior RK4 over one control period plus the network's learned correction (pinc/greybox.py)."""
    name = "greybox"

    def __init__(self, model, cfg: Config, name="greybox"):
        super().__init__(model, cfg, name)
        self.kind = getattr(model.mcfg, "greybox_prior", "full")
        if self.kind == "qs":                        # quasi-steady wheels: 10 RK4 steps of 10 ms per period
            from .greybox import QS_DT
            self.n_qs = int(round(cfg.T/QS_DT))
            self.sys = get_system(cfg)
        else:
            self.prior = RK4Predictor(cfg, name="greybox_prior", model="prior")

    def prior_step(self, s, u):
        if self.kind == "qs":
            return self.sys.prior_qs_step_tf(s, u, self.cfg.T/self.n_qs, self.n_qs)
        return self.prior.step(s, u)

    def step(self, s, u):
        return self.prior_step(s, u) + super().step(s, u) - s

    def rollout(self, s0, u, extra=()):
        if getattr(self.cfg.mpc, "loop_rollout", False):
            N = u.shape[0]
            ta = tf.TensorArray(s0.dtype, size=N)

            def body(k, s, ta):
                s = self.step(s, tf.reshape(u[k], (1, -1)))
                return k + 1, s, ta.write(k, s[0])
            _, _, ta = tf.while_loop(lambda k, s, ta: k < N, body, (tf.constant(0), tf.reshape(s0, (1, -1)), ta),
                                     maximum_iterations=N)
            return ta.stack()
        return super().rollout(s0, u, extra)


def make_predictor(model, cfg: Config, name="pinc"):
    """PINC / data-only network, or the grey-box model when the network was trained as one."""
    if getattr(model.mcfg, "greybox", False):
        return GreyBoxPredictor(model, cfg, name)
    return PINCPredictor(model, cfg, name)


class LinearPredictor(Predictor):
    """LTV model: the RK4 one-step map linearised along the warm-start
    trajectory (re-linearised once per solve)."""
    name = "ltv"
    n_extra = 4

    def __init__(self, cfg: Config, params=None, name="ltv"):
        self.cfg, self.name = cfg, name
        self.base = RK4Predictor(cfg, params)
        self.dtype = cfg.dtype

        @tf.function
        def _lin(s0, u_nom):
            s_nom = [tf.reshape(s0, (1, -1))]
            s = s_nom[0]
            for k in range(u_nom.shape[0] - 1):
                s = self.base.step(s, u_nom[k:k + 1])
                s_nom.append(s)
            s_nom = tf.concat(s_nom, axis=0)                 # (N, n_s) states at which each step is linearised
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(s_nom)
                tape.watch(u_nom)
                s_next = self.base.step(s_nom, u_nom)
            A = tape.batch_jacobian(s_next, s_nom)
            B = tape.batch_jacobian(s_next, u_nom)
            del tape
            return s_nom, s_next, tf.concat([A, B], axis=2)
        self._lin = _lin

    def prepare(self, s0, u_warm):
        u_nom = tf.constant(np.asarray(u_warm, float), self.dtype)
        s_nom, s_next, AB = self._lin(tf.constant(np.asarray(s0, float), self.dtype), u_nom)
        return (s_nom, u_nom, s_next, AB)

    def rollout(self, s0, u, extra):
        s_nom, u_nom, s_next, AB = extra
        n_s = s_nom.shape[1]
        A, B = AB[:, :, :n_s], AB[:, :, n_s:]
        s = tf.reshape(s0, (1, -1))
        out = []
        for k in range(u.shape[0]):
            ds = s - s_nom[k:k + 1]
            du = u[k:k + 1] - u_nom[k:k + 1]
            s = s_next[k:k + 1] + tf.matmul(ds, A[k], transpose_b=True) + tf.matmul(du, B[k], transpose_b=True)
            out.append(s)
        return tf.concat(out, axis=0)


# ---------------------------------------------------------------------------
def integrate_xy(XY0, s_prev, s, T, sysm=None):
    """Trapezoidal integration of X, Y from the predicted s sequence."""
    sysm = sysm or get_system("bicycle")

    def g(s):
        vx, vy, psi = sysm.planar_velocity(s)
        return tf.stack([vx*tf.cos(psi) - vy*tf.sin(psi), vx*tf.sin(psi) + vy*tf.cos(psi)], axis=1)
    inc = 0.5*T*(g(s_prev) + g(s))
    return tf.reshape(XY0, (1, 2)) + tf.cumsum(inc, axis=0)


class MPC:
    def __init__(self, predictor: Predictor, cfg: Config, Q=None, P=None):
        self.pred, self.cfg = predictor, cfg
        self.sys = sysm = get_system(cfg)
        self.n_u = n_u = sysm.n_u
        m = cfg.mpc
        self.N, self.T = m.N, cfg.T
        self.dtype = cfg.dtype
        self.S_u = np.asarray(cfg.S_u)
        self.Q = tf.constant(np.asarray(Q if Q is not None else m.Q, float), self.dtype)
        self.P = tf.constant(np.asarray(P if P is not None else m.P, float), self.dtype)
        self.R = tf.constant(np.asarray(m.R, float), self.dtype)
        self.R_D = tf.constant(np.asarray(m.R_delta, float), self.dtype)
        self.r_max = float(m.r_max)
        self.w_rmax = float(m.w_rmax)
        self.lo = np.asarray(cfg.u_min)/self.S_u
        self.hi = np.asarray(cfg.u_max)/self.S_u
        self.bounds = [(self.lo[j], self.hi[j]) for _ in range(self.N) for j in range(n_u)]
        self.u_tilde_prev = None                     # last applied (normalised) input
        self.u_seq = None                            # last solution (N, n_u) normalised
        self.n_calls = 0
        S_u_t = tf.constant(self.S_u, self.dtype)
        T_t = tf.constant(self.T, self.dtype)

        @tf.function(jit_compile=bool(getattr(m, "jit", True)))
        def cost_and_grad(s0, XY0, u_flat, u_prev, ref, *extra):
            with tf.GradientTape() as tape:
                tape.watch(u_flat)
                u_t = tf.reshape(u_flat, (self.N, n_u))
                u = u_t*S_u_t
                s = self.pred.rollout(s0, u, extra)                       # (N, n_s)
                s_prev = tf.concat([tf.reshape(s0, (1, -1)), s[:-1]], axis=0)
                XY = integrate_xy(XY0, s_prev, s, T_t, sysm)
                z = sysm.track_vector(s, XY)
                e = z - ref
                J_track = tf.reduce_sum(tf.square(e[:-1])*self.Q) + tf.reduce_sum(tf.square(e[-1])*self.P)
                du = u_t - tf.concat([tf.reshape(u_prev, (1, n_u)), u_t[:-1]], axis=0)
                J_u = tf.reduce_sum(tf.square(u_t)*self.R) + tf.reduce_sum(tf.square(du)*self.R_D)
                viol = tf.nn.relu(tf.abs(s[:, 2]) - self.r_max)
                J_r = self.w_rmax*tf.reduce_sum(tf.square(viol))
                J = J_track + J_u + J_r
            g = tape.gradient(J, u_flat)
            return J, g, z
        self._cost_and_grad = cost_and_grad

    # ---- helpers ---------------------------------------------------------
    def reference_sequence(self, t, ref) -> np.ndarray:
        """Reference at t + (k+1) T for k = 0..N-1 (the state AFTER each step)."""
        return ref(t + self.T*np.arange(1, self.N + 1))

    def warm_start(self, x_meas):
        if self.u_seq is None:
            u0 = np.zeros((self.N, self.n_u))
            from .plant import trim_force
            u0[:, 0] = trim_force(max(x_meas[0], 0.5), self.cfg.params)/self.S_u[0]
            return np.clip(u0, self.lo, self.hi)
        return np.concatenate([self.u_seq[1:], self.u_seq[-1:]], axis=0)

    def cost(self, x, u_tilde_flat, ref_seq, u_prev=None, extra=None):
        """Cost, gradient and predicted z for given normalised inputs (numpy)."""
        s0 = tf.constant(np.asarray(self.sys.from_full(x), float), self.dtype)
        XY0 = tf.constant(np.asarray(self.sys.xy(x), float), self.dtype)
        u_prev = np.zeros(self.n_u) if u_prev is None else u_prev
        if extra is None:
            extra = self.pred.prepare(self.sys.from_full(x), np.reshape(u_tilde_flat, (self.N, self.n_u))*self.S_u)
        J, g, z = self._cost_and_grad(s0, XY0, tf.constant(np.asarray(u_tilde_flat, float), self.dtype),
                                      tf.constant(np.asarray(u_prev, float), self.dtype),
                                      tf.constant(np.asarray(ref_seq, float), self.dtype), *extra)
        return float(J), g.numpy(), z.numpy()

    # ---- solve -------------------------------------------------------------
    def solve(self, x_meas, ref_seq):
        m = self.cfg.mpc
        t0 = time.perf_counter()
        u_ws = self.warm_start(x_meas)
        u_prev = self.u_tilde_prev if self.u_tilde_prev is not None else u_ws[0]
        s_meas = self.sys.from_full(x_meas)
        extra = self.pred.prepare(s_meas, u_ws*self.S_u)
        s0 = tf.constant(np.asarray(s_meas, float), self.dtype)
        XY0 = tf.constant(np.asarray(self.sys.xy(x_meas), float), self.dtype)
        u_prev_t = tf.constant(np.asarray(u_prev, float), self.dtype)
        ref_t = tf.constant(np.asarray(ref_seq, float), self.dtype)
        n_eval = [0]

        def fun(u_flat):
            n_eval[0] += 1
            J, g, _ = self._cost_and_grad(s0, XY0, tf.constant(u_flat, self.dtype), u_prev_t, ref_t, *extra)
            J, g = float(J), g.numpy()
            if not np.isfinite(J) or not np.all(np.isfinite(g)):
                raise FloatingPointError("non-finite MPC cost/gradient")
            return J, g

        res = minimize(fun, u_ws.ravel(), jac=True, method=m.solver, bounds=self.bounds,
                       options=dict(maxiter=m.maxiter, ftol=m.ftol))
        u_seq = np.clip(res.x.reshape(self.N, self.n_u), self.lo, self.hi)
        self.u_seq = u_seq
        self.u_tilde_prev = u_seq[0].copy()
        self.n_calls += 1
        info = dict(solve_time=time.perf_counter() - t0, nit=int(res.nit), nfev=int(n_eval[0]),
                    success=bool(res.success), cost=float(res.fun), status=int(res.status))
        return u_seq[0]*self.S_u, info

    def __call__(self, t, x_meas, ref):
        return self.solve(x_meas, self.reference_sequence(t, ref))

    def reset(self):
        self.u_tilde_prev, self.u_seq, self.n_calls = None, None, 0


# ---------------------------------------------------------------------------
def make_controller(arm: str, cfg: Config, models: dict | None = None, Q=None, P=None) -> MPC:
    """arms: nmpc_rk4 | nmpc_true | pinc | blackbox | greybox | ltv.  `models` maps arm -> loaded PINCNet."""
    models = models or {}
    if arm == "nmpc_rk4":
        pred = RK4Predictor(cfg)
    elif arm == "nmpc_true":
        pred = RK4Predictor(cfg, name="nmpc_true", model="true")
    elif arm == "pinc":
        pred = PINCPredictor(models["pinc"], cfg, name="pinc")
    elif arm == "blackbox":
        pred = PINCPredictor(models["blackbox"], cfg, name="blackbox")
    elif arm == "greybox":
        pred = GreyBoxPredictor(models["greybox"], cfg, name="greybox")
    elif arm == "ltv":
        pred = LinearPredictor(cfg)
    else:
        raise ValueError(arm)
    return MPC(pred, cfg, Q, P)


ARMS = ("nmpc_rk4", "pinc", "blackbox", "ltv")
ARM_LABELS = dict(nmpc_rk4="NMPC-RK4", pinc="PINC-MPC", blackbox="Black-box-MPC", ltv="LTV-MPC", nmpc_true="NMPC-true", greybox="Grey-box-MPC")
