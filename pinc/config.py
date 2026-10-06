"""
Single source of truth for parameters, bounds, scales, horizon and seeds
(ground rule 5).  Training and inference both import from here.

Values are loaded from `configs/default.yaml`; any entry can be overridden
from the command line with `--set a.b.c=value`.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(ROOT, "configs", "default.yaml")
RESULTS_DIR = os.path.join(ROOT, "results")

STATE_NAMES = ("vx", "vy", "r", "psi")          # PINC state s
FULL_STATE_NAMES = ("vx", "vy", "r", "psi", "X", "Y")
INPUT_NAMES = ("Fx", "delta")


@dataclass
class VehicleCfg:
    m: float = 1500.0
    Iz: float = 2500.0
    lf: float = 1.4
    lr: float = 1.4
    Caf: float = 50000.0
    Car: float = 50000.0
    Cd: float = 0.3
    A: float = 2.2
    rho: float = 1.225
    Frr: float = 300.0
    mu: float = 1.0
    Fz: float = 1500.0*9.81/2.0

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class BoxCfg:
    """Uniform sampling box for initial states (and the input box)."""
    vx: list = field(default_factory=lambda: [5.0, 25.0])
    vy: list = field(default_factory=lambda: [-1.5, 1.5])
    r: list = field(default_factory=lambda: [-0.6, 0.6])
    psi: list = field(default_factory=lambda: [-0.5, 0.5])

    def lo(self):
        return np.array([self.vx[0], self.vy[0], self.r[0], self.psi[0]])

    def hi(self):
        return np.array([self.vx[1], self.vy[1], self.r[1], self.psi[1]])


@dataclass
class ScalesCfg:
    S_x: list = field(default_factory=lambda: [30.0, 1.5, 0.6, 0.5])
    S_u: list = field(default_factory=lambda: [6000.0, 0.3])
    # characteristic rate of each state, used to normalise the physics
    # residual: std of f(s, u) over the training box (scripts/compute_scales.py)
    S_f: list = field(default_factory=lambda: [2.035, 9.41, 5.296, 0.3466])


@dataclass
class SeedsCfg:
    train: int = 0
    val: int = 1
    test: int = 2
    test_extrap: int = 3
    colloc: int = 4
    weights: int = 5


@dataclass
class ModelCfg:
    depth: int = 8
    width: int = 64
    activation: str = "tanh"
    hard_ic: bool = True
    residual: str = "none"            # none | skip (h += layer(h)) | block (ResNet: h += W2 tanh(W1 h)); depth counts blocks
    dropout: float = 0.0              # dropout rate after every hidden layer (training only)
    layernorm: bool = False           # LayerNormalization after every hidden layer
    increment_scaling: bool = False   # hard IC: s_hat = s0_hat + (t/T) * NN * (S_f*T/S_x), i.e. O(1) network output per channel
    learn_theta: bool = False         # learnable physical parameters of the prior (HF system: Caf, Car, C_kappa, tau_F, tau_delta)
    greybox: bool = False             # the network learns a correction to the prior's own prediction (pinc/greybox.py)


@dataclass
class LossCfg:
    lam: float = 0.01                 # physics weight lambda (see configs/default.yaml)
    w_ic: float = 1.0                 # IC loss weight (only matters if soft IC)
    residual_mask: list = field(default_factory=lambda: [1, 1, 1, 1])


@dataclass
class TrainCfg:
    n_data: int = 20000
    n_val: int = 4000
    n_test: int = 4000
    n_colloc: int = 20000
    batch_data: int = 1024
    batch_colloc: int = 1024
    epochs: int = 300
    steps: int = 0                    # if > 0, overrides epochs so that Adam runs this many steps
    lr: float = 1.0e-3
    lr_decay: str = "cosine"          # cosine | exponential | none
    lr_final_frac: float = 0.02
    lbfgs_iters: int = 500            # 0 disables the L-BFGS stage
    val_every: int = 1
    log_every: int = 10
    select_on: str = "total"          # validation quantity for model selection: total | data
    colloc_log_frac: float = 0.0      # fraction of collocation times drawn log-uniformly in [1e-3 T, T] (fast transients near t = 0)
    distill_from: str = ""            # run id of a trained grey-box model whose predictions label extra training samples
    n_distill: int = 20000            # number of samples it labels (unlabelled states from the collocation pool)


@dataclass
class MPCCfg:
    N: int = 10
    Q: list = field(default_factory=lambda: [1.0, 0.0, 0.0, 10.0, 0.0, 1.0])   # [vx,vy,r,psi,X,Y]
    P: list = field(default_factory=lambda: [1.0, 0.0, 0.0, 10.0, 0.0, 1.0])
    R: list = field(default_factory=lambda: [0.05, 0.05])                       # on u/S_u
    R_delta: list = field(default_factory=lambda: [0.5, 0.5])                   # on d(u/S_u)
    r_max: float = 0.5
    w_rmax: float = 100.0
    dt_pred: float = 0.01              # RK4Predictor substep
    solver: str = "SLSQP"
    maxiter: int = 100
    ftol: float = 1.0e-8
    jit: bool = True                   # XLA-compile the cost+gradient function (same for every predictor)
    loop_rollout: bool = False         # RK4 predictor horizon as a tf.while_loop (constant graph size; needed for the HF true model)


@dataclass
class SimCfg:
    duration: float = 10.0
    dt_plant: float = 1.0e-3
    tyre: str = "linear"
    # Measurement noise std per channel [vx, vy, r, psi, X, Y].
    # vx, vy: 0.05 m/s  (GNSS/INS velocity accuracy class, e.g. u-blox
    #                    ZED-F9P datasheet: velocity accuracy 0.05 m/s)
    # r:      0.002 rad/s (automotive MEMS gyro, ~0.1 deg/s rms noise class,
    #                    e.g. Bosch SMI230 datasheet)
    # psi:    0.005 rad  (dual-antenna GNSS heading, ~0.3 deg)
    # X, Y:   0.05 m     (RTK-GNSS position, cm-level, with outages)
    noise_sigma: list = field(default_factory=lambda: [0.05, 0.05, 0.002, 0.005, 0.05, 0.05])
    x0_sigma: list = field(default_factory=lambda: [0.5, 0.0, 0.0, 0.0, 0.0, 0.0])


@dataclass
class RefsCfg:
    v0: float = 20.0
    sin_amp: float = 2.0
    sin_omega: float = 0.5
    step_dv: float = 3.0
    step_time: float = 2.0
    slc_offset: float = 3.5
    slc_length: float = 40.0
    slc_start: float = 20.0
    dlc_speed: float = 12.0
    dlc_offset: float = 3.5


@dataclass
class Config:
    T: float = 0.1                     # control period [s]
    system: str = "bicycle"            # vehicle system (pinc/system.py)
    dtype: str = "float64"
    actuator_lag: bool = False         # v1: zero-order hold on commands
    vehicle: VehicleCfg = field(default_factory=VehicleCfg)
    box_train: BoxCfg = field(default_factory=BoxCfg)
    box_full: BoxCfg = field(default_factory=lambda: BoxCfg(vx=[5.0, 30.0]))
    box_extrap: BoxCfg = field(default_factory=lambda: BoxCfg(vx=[25.0, 30.0]))
    u_min: list = field(default_factory=lambda: [-6000.0, -0.3])
    u_max: list = field(default_factory=lambda: [3000.0, 0.3])
    scales: ScalesCfg = field(default_factory=ScalesCfg)
    seeds: SeedsCfg = field(default_factory=SeedsCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    mpc: MPCCfg = field(default_factory=MPCCfg)
    sim: SimCfg = field(default_factory=SimCfg)
    refs: RefsCfg = field(default_factory=RefsCfg)

    # ---- convenience -------------------------------------------------
    @property
    def S_x(self) -> np.ndarray:
        return np.asarray(self.scales.S_x, dtype=float)

    @property
    def S_u(self) -> np.ndarray:
        return np.asarray(self.scales.S_u, dtype=float)

    @property
    def S_f(self) -> np.ndarray:
        return np.asarray(self.scales.S_f, dtype=float)

    @property
    def params(self) -> dict:
        return self.vehicle.as_dict()

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, path: str):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    def copy(self) -> "Config":
        return copy.deepcopy(self)

    def with_overrides(self, overrides: dict | list | None) -> "Config":
        cfg = self.copy()
        for k, v in _as_pairs(overrides):
            _set_dotted(cfg, k, v)
        cfg.validate()
        return cfg

    def validate(self):
        assert self.T > 0 and self.mpc.N >= 1
        from .system import get_system
        sysm = get_system(self.system)
        assert len(self.scales.S_x) == len(self.scales.S_f) == sysm.n_s, "S_x / S_f must match the system state"
        assert len(self.scales.S_u) == len(self.u_min) == len(self.u_max) == sysm.n_u, "S_u / u bounds must match the system input"
        assert len(self.loss.residual_mask) == sysm.n_s
        assert np.all(np.asarray(self.u_min) < np.asarray(self.u_max))
        assert abs(round(self.T/self.mpc.dt_pred)*self.mpc.dt_pred - self.T) < 1e-9
        assert abs(round(self.T/self.sim.dt_plant)*self.sim.dt_plant - self.T) < 1e-9
        assert self.dtype in ("float32", "float64")
        assert self.sim.tyre in ("linear", "fiala")


# ---------------------------------------------------------------------------
_SUB = {
    "vehicle": VehicleCfg, "box_train": BoxCfg, "box_full": BoxCfg,
    "box_extrap": BoxCfg, "scales": ScalesCfg, "seeds": SeedsCfg,
    "model": ModelCfg, "loss": LossCfg, "train": TrainCfg, "mpc": MPCCfg,
    "sim": SimCfg, "refs": RefsCfg,
}


def from_dict(d: dict) -> Config:
    cfg = Config()
    for k, v in d.items():
        if k in _SUB:
            if not hasattr(cfg, k):
                raise KeyError(k)
            sub = _SUB[k]()
            for kk, vv in v.items():
                if not hasattr(sub, kk):
                    raise KeyError(f"{k}.{kk}")
                setattr(sub, kk, vv)
            setattr(cfg, k, sub)
        else:
            if not hasattr(cfg, k):
                raise KeyError(k)
            setattr(cfg, k, v)
    cfg.validate()
    return cfg


def _parse_value(v: str) -> Any:
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def _as_pairs(overrides):
    if not overrides:
        return []
    if isinstance(overrides, dict):
        return list(overrides.items())
    pairs = []
    for item in overrides:
        if isinstance(item, str):
            k, _, v = item.partition("=")
            pairs.append((k.strip(), _parse_value(v.strip())))
        else:
            pairs.append(tuple(item))
    return pairs


def _set_dotted(cfg, key: str, value):
    parts = key.split(".")
    obj = cfg
    for p in parts[:-1]:
        if not hasattr(obj, p):
            raise KeyError(key)
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        raise KeyError(key)
    old = getattr(obj, parts[-1])
    if isinstance(old, bool):
        value = bool(value) if not isinstance(value, str) else value.lower() in ("1", "true", "yes")
    elif isinstance(old, int) and not isinstance(value, bool):
        value = int(value)
    elif isinstance(old, float):
        value = float(value)
    elif isinstance(old, list) and isinstance(value, str):
        value = yaml.safe_load(value)
    setattr(obj, parts[-1], value)


def load_config(path: str | None = None, overrides=None) -> Config:
    """Load `configs/default.yaml` (or `path`), then apply dotted overrides."""
    path = path or DEFAULT_CONFIG_PATH
    with open(path) as fh:
        d = yaml.safe_load(fh) or {}
    cfg = from_dict(d)
    return cfg.with_overrides(overrides)


def add_config_args(parser):
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override a config entry, e.g. --set loss.lam=0")
    parser.add_argument("--seed", type=int, default=0)
    return parser
