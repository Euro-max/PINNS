"""
Magic Formula 6.1 tyre forces (Pacejka, *Tire and Vehicle Dynamics*, 3rd ed., 2012, sec. 4.3.2):
pure and combined longitudinal / lateral slip, zero camber, no turn slip, nominal inflation
pressure (every pressure term vanishes) and no speed dependence of friction.

The coefficients are read from a parameter file of `NAME = value` lines (exported from the
MathWorks Vehicle Dynamics Blockset by `scripts/export_tyre_params.m` into data/tyre/, which is
not redistributed).  Scaling factors (LFZO, LMUX, LKY, ...) are not in that file and default to 1;
`mu_scale` scales both friction coefficients for road-surface scenarios.

Sign convention.  MF uses the ISO-W axis system, in which a positive slip angle produces a
negative lateral force (PKY1 < 0).  The plant (pinc/plant.py) uses alpha = -atan(v_s / v_l), for
which a positive slip angle produces a positive lateral force.  `forces()` takes the plant's
alpha and returns the plant's (Fx, Fy): the MF formulas are evaluated at alpha_MF = -alpha and Fy
is used as is (both systems have y to the left), so the force always opposes lateral sliding.

The same code runs on NumPy arrays and TensorFlow tensors: pass `xp=NP` or `xp=TF`.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

NP = SimpleNamespace(sin=np.sin, cos=np.cos, atan=np.arctan, exp=np.exp, sign=np.sign, abs=np.abs,
                     minimum=np.minimum, maximum=np.maximum)


def _tf_ns():
    import tensorflow as tf
    return SimpleNamespace(sin=tf.sin, cos=tf.cos, atan=tf.atan, exp=tf.exp, sign=tf.sign, abs=tf.abs,
                           minimum=tf.minimum, maximum=tf.maximum)


class _LazyTF(SimpleNamespace):
    def __getattr__(self, name):
        ns = _tf_ns()
        self.__dict__.update(vars(ns))
        return getattr(ns, name)


TF = _LazyTF()

DEFAULT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tyre",
                            "mf_235_45R18_params.txt")
SCALES = ("LFZO", "LCX", "LMUX", "LEX", "LKX", "LHX", "LVX", "LCY", "LMUY", "LEY", "LKY", "LHY", "LVY",
          "LXAL", "LYKA", "LVYKA")
EPS = 1e-6


def load_params(path: str = DEFAULT_FILE) -> dict:
    """Read `NAME = value` lines; non-numeric values (tyre-type labels) are skipped."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run scripts/export_tyre_params.m in MATLAB and copy its "
                                "output to data/tyre/ (see docs/PLAN_HIGH_FIDELITY.md)")
    p = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("%") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            try:
                p[k.strip()] = float(v.strip())
            except ValueError:
                pass
    for s in SCALES:
        p.setdefault(s, 1.0)
    return p


def _g(p, k):
    return p.get(k, 0.0)


def forces(kappa, alpha, Fz, p: dict, mu_scale=1.0, xp=NP, mirror=False):
    """Combined-slip tyre forces in the plant's convention.

    kappa: longitudinal slip (R w - v_l) / |v_l|;  alpha: slip angle, plant convention (rad);
    Fz: vertical load (N, > 0).  Returns (Fx, Fy) in N.
    mirror: the same tyre mounted on the other side of the car, Fy(alpha) -> -Fy(-alpha), so that its
    built-in lateral offsets (ply-steer, conicity: PHY1, PVY1, RHY1, ...) cancel across an axle.
    """
    if mirror:
        Fx, Fy = forces(kappa, -alpha, Fz, p, mu_scale, xp)
        return Fx, -Fy
    a = -alpha                                            # MF (ISO-W) slip angle
    Fz0 = p["FNOMIN"]*p["LFZO"]
    dfz = (Fz - Fz0)/Fz0
    lmux, lmuy = p["LMUX"]*mu_scale, p["LMUY"]*mu_scale

    # ---- pure longitudinal slip
    SHx = (_g(p, "PHX1") + _g(p, "PHX2")*dfz)*p["LHX"]
    kx = kappa + SHx
    Cx = p["PCX1"]*p["LCX"]
    mux = (p["PDX1"] + _g(p, "PDX2")*dfz)*lmux
    Dx = mux*Fz
    Ex = xp.minimum((_g(p, "PEX1") + _g(p, "PEX2")*dfz + _g(p, "PEX3")*dfz**2)*(1.0 - _g(p, "PEX4")*xp.sign(kx))*p["LEX"], 1.0)
    Kxk = Fz*(p["PKX1"] + _g(p, "PKX2")*dfz)*xp.exp(_g(p, "PKX3")*dfz)*p["LKX"]
    Bx = Kxk/(Cx*Dx + EPS)
    SVx = Fz*(_g(p, "PVX1") + _g(p, "PVX2")*dfz)*p["LVX"]*lmux
    Fx0 = Dx*xp.sin(Cx*xp.atan(Bx*kx - Ex*(Bx*kx - xp.atan(Bx*kx)))) + SVx

    # ---- pure lateral slip
    SHy = (_g(p, "PHY1") + _g(p, "PHY2")*dfz)*p["LHY"]
    ay = a + SHy
    Cy = p["PCY1"]*p["LCY"]
    muy = (p["PDY1"] + _g(p, "PDY2")*dfz)*lmuy
    Dy = muy*Fz
    Ey = xp.minimum((_g(p, "PEY1") + _g(p, "PEY2")*dfz)*(1.0 - _g(p, "PEY3")*xp.sign(ay))*p["LEY"], 1.0)
    Kya = p["PKY1"]*Fz0*xp.sin(p["PKY4"]*xp.atan(Fz/(p["PKY2"]*Fz0)))*p["LKY"]
    By = Kya/(Cy*Dy + EPS)
    SVy = Fz*(_g(p, "PVY1") + _g(p, "PVY2")*dfz)*p["LVY"]*lmuy
    Fy0 = Dy*xp.sin(Cy*xp.atan(By*ay - Ey*(By*ay - xp.atan(By*ay)))) + SVy

    # ---- combined slip: weighting functions (cosine form)
    SHxa = _g(p, "RHX1")
    As = a + SHxa
    Bxa = p["RBX1"]*xp.cos(xp.atan(p["RBX2"]*kappa))*p["LXAL"]
    Cxa = p["RCX1"]
    Exa = xp.minimum(_g(p, "REX1") + _g(p, "REX2")*dfz, 1.0)
    Gxa0 = xp.cos(Cxa*xp.atan(Bxa*SHxa - Exa*(Bxa*SHxa - xp.atan(Bxa*SHxa))))
    Gxa = xp.cos(Cxa*xp.atan(Bxa*As - Exa*(Bxa*As - xp.atan(Bxa*As))))/Gxa0
    Fx = Gxa*Fx0

    SHyk = _g(p, "RHY1") + _g(p, "RHY2")*dfz
    Ks = kappa + SHyk
    Byk = p["RBY1"]*xp.cos(xp.atan(p["RBY2"]*(a - _g(p, "RBY3"))))*p["LYKA"]
    Cyk = p["RCY1"]
    Eyk = xp.minimum(_g(p, "REY1") + _g(p, "REY2")*dfz, 1.0)
    DVyk = muy*Fz*(_g(p, "RVY1") + _g(p, "RVY2")*dfz)*xp.cos(xp.atan(_g(p, "RVY4")*a))
    SVyk = DVyk*xp.sin(_g(p, "RVY5")*xp.atan(_g(p, "RVY6")*kappa))*p["LVYKA"]
    Gyk0 = xp.cos(Cyk*xp.atan(Byk*SHyk - Eyk*(Byk*SHyk - xp.atan(Byk*SHyk))))
    Gyk = xp.cos(Cyk*xp.atan(Byk*Ks - Eyk*(Byk*Ks - xp.atan(Byk*Ks))))/Gyk0
    Fy = Gyk*Fy0 + SVyk
    return Fx, Fy


def cornering_stiffness(Fz, p: dict) -> float:
    """|dFy/dalpha| at zero slip (pure lateral), N/rad."""
    Fz0 = p["FNOMIN"]*p["LFZO"]
    return abs(p["PKY1"]*Fz0*np.sin(p["PKY4"]*np.arctan(Fz/(p["PKY2"]*Fz0)))*p["LKY"])


def slip_stiffness(Fz, p: dict) -> float:
    """dFx/dkappa at zero slip (pure longitudinal), N per unit slip."""
    Fz0 = p["FNOMIN"]*p["LFZO"]
    dfz = (Fz - Fz0)/Fz0
    return Fz*(p["PKX1"] + _g(p, "PKX2")*dfz)*np.exp(_g(p, "PKX3")*dfz)*p["LKX"]
