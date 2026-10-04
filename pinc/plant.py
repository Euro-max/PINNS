"""
Single-track (bicycle) vehicle model + fixed-step RK4 ground truth.

This is the ONE plant used by every controller in every experiment (ground
rule 2).  It is the corrected model from `cav_plant.py`, extended so that the
same functions accept a batch of states (shape (B, 6)) as well as a single
state (shape (6,)).  Semantics are unchanged.

State x = [vx, vy, r, psi, X, Y]
  vx  longitudinal velocity in body frame [m/s]
  vy  lateral velocity in body frame      [m/s]
  r   yaw rate                            [rad/s]
  psi yaw angle                           [rad]
  X,Y inertial position                   [m]

Input u = [Fx, delta]   traction force [N], road-wheel steer angle [rad]

  alpha_f = delta - atan2(vy + lf r, vx)
  alpha_r =       - atan2(vy - lr r, vx)
  m (dvx - vy r) = Fx - 0.5 rho Cd A vx^2 - Frr tanh(vx/0.1) - Fyf sin(delta)
  m (dvy + vx r) = Fyf cos(delta) + Fyr
  Iz dr          = lf Fyf cos(delta) - lr Fyr          # rear OPPOSES
  dpsi = r ;  dX = vx cos psi - vy sin psi ;  dY = vx sin psi + vy cos psi

Tyre: `linear` (default) or `fiala` (saturating, mu*Fz limit).
"""
import numpy as np

DEFAULT_PARAMS = dict(
    m=1500.0, Iz=2500.0, lf=1.4, lr=1.4,
    Caf=50000.0, Car=50000.0,          # cornering stiffness [N/rad]
    Cd=0.3, A=2.2, rho=1.225, Frr=300.0,
    mu=1.0, Fz=1500.0*9.81/2.0,        # for the saturating tyre
)
VX_MIN = 0.5
STATE_NAMES = ("vx", "vy", "r", "psi", "X", "Y")
INPUT_NAMES = ("Fx", "delta")


def perturbed(p, **changes):
    """Copy of parameter dict `p` with the given entries replaced."""
    q = dict(p)
    for k, v in changes.items():
        if k not in q:
            raise KeyError(f"unknown vehicle parameter {k!r}")
        q[k] = float(v)
    return q


def tyre_force(alpha, C, mu=1.0, Fz=7357.5, model="linear"):
    """Lateral force. `linear` for analysis, `fiala` for a saturating tyre."""
    if model == "linear":
        return C * alpha
    if model != "fiala":
        raise ValueError(f"unknown tyre model {model!r}")
    Fmax = mu * Fz
    a_sl = np.arctan(3.0 * Fmax / C)
    a = np.clip(alpha, -a_sl, a_sl)
    t = np.tan(a)
    F = C*t - (C**2/(3*Fmax))*np.abs(t)*t + (C**3/(27*Fmax**2))*t**3
    return np.where(np.abs(alpha) >= a_sl, np.sign(alpha)*Fmax, F)


def slip_angles(vx, vy, r, delta, p):
    vxs = np.maximum(vx, VX_MIN)
    alpha_f = delta - np.arctan2(vy + p['lf']*r, vxs)
    alpha_r = -np.arctan2(vy - p['lr']*r, vxs)
    return alpha_f, alpha_r


def f(x, u, p, tyre="linear"):
    """Continuous-time dynamics dx/dt.  x: (6,) or (B, 6); u: (2,) or (B, 2)."""
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    vx, vy, r, psi = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    Fx, delta = u[..., 0], u[..., 1]
    vxs = np.maximum(vx, VX_MIN)

    alpha_f, alpha_r = slip_angles(vx, vy, r, delta, p)
    Fyf = tyre_force(alpha_f, p['Caf'], p['mu'], p['Fz'], tyre)
    Fyr = tyre_force(alpha_r, p['Car'], p['mu'], p['Fz'], tyre)

    F_drag = 0.5*p['rho']*p['Cd']*p['A']*vxs**2
    F_roll = p['Frr']*np.tanh(vx/0.1)          # smooth, sign-correct at vx~0

    dvx = (Fx - F_drag - F_roll - Fyf*np.sin(delta))/p['m'] + vy*r
    dvy = (Fyf*np.cos(delta) + Fyr)/p['m'] - vx*r
    dr = (p['lf']*Fyf*np.cos(delta) - p['lr']*Fyr)/p['Iz']   # rear OPPOSES
    dpsi = r
    dX = vx*np.cos(psi) - vy*np.sin(psi)
    dY = vx*np.sin(psi) + vy*np.cos(psi)
    return np.stack([dvx, dvy, dr, dpsi, dX, dY], axis=-1)


def rk4_step(x, u, dt, p, tyre="linear"):
    k1 = f(x, u, p, tyre)
    k2 = f(x + 0.5*dt*k1, u, p, tyre)
    k3 = f(x + 0.5*dt*k2, u, p, tyre)
    k4 = f(x + dt*k3, u, p, tyre)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def _n_steps(T, dt):
    n = int(round(T/dt))
    if abs(n*dt - T) > 1e-9*max(1.0, abs(T)):
        raise ValueError(f"dt={dt} does not divide T={T}")
    return n


def simulate(x0, u, T, dt, p, tyre="linear"):
    """Zero-order-hold `u` over [0,T]. Returns final state; dt must divide T.
    Works for a single state (6,) or a batch (B, 6) with u (2,) or (B, 2)."""
    n = _n_steps(T, dt)
    x = np.array(x0, dtype=float, copy=True)
    u = np.asarray(u, dtype=float)
    for _ in range(n):
        x = rk4_step(x, u, dt, p, tyre)
    return x


def simulate_traj(x0, u, T, dt, p, tyre="linear"):
    """Like `simulate` but returns (t, x) with x of shape (n+1, 6) (or (n+1, B, 6))."""
    n = _n_steps(T, dt)
    x = np.array(x0, dtype=float, copy=True)
    u = np.asarray(u, dtype=float)
    out = [x.copy()]
    for _ in range(n):
        x = rk4_step(x, u, dt, p, tyre)
        out.append(x.copy())
    return np.arange(n + 1)*dt, np.stack(out, axis=0)


def check_finite(x, what="state"):
    if not np.all(np.isfinite(x)):
        raise FloatingPointError(f"non-finite {what}: {x}")
    return x


def lateral_eigs(vx, p):
    """Eigenvalues of the linearised (vy, r) subsystem at speed vx."""
    m, Iz, lf, lr, Caf, Car = (p['m'], p['Iz'], p['lf'], p['lr'],
                               p['Caf'], p['Car'])
    A = np.array([
        [-(Caf+Car)/(m*vx),        -(lf*Caf - lr*Car)/(m*vx) - vx],
        [-(lf*Caf - lr*Car)/(Iz*vx), -(lf**2*Caf + lr**2*Car)/(Iz*vx)],
    ])
    return np.linalg.eigvals(A)


def understeer_gradient(p):
    """K_us > 0 understeer, = 0 neutral, < 0 oversteer. [rad/(m/s^2)]"""
    m, lf, lr, L = p['m'], p['lf'], p['lr'], p['lf']+p['lr']
    return m*(lr/(L*p['Caf']) - lf/(L*p['Car']))


def trim_force(vx, p):
    """Longitudinal force that holds `vx` constant on the straight."""
    return 0.5*p['rho']*p['Cd']*p['A']*vx**2 + p['Frr']*np.tanh(vx/0.1)


if __name__ == "__main__":
    p = DEFAULT_PARAMS
    print("=== Validation of the corrected model ===\n")

    print("Lateral eigenvalues (must have negative real part):")
    for vx in (5.0, 10.0, 20.0, 30.0, 40.0):
        e = lateral_eigs(vx, p)
        print(f"  vx={vx:5.1f} m/s : {e[0]:+.4f} , {e[1]:+.4f}"
              f"   stable={np.all(np.real(e) < 0)}")
    print(f"\nUndersteer gradient K_us = {understeer_gradient(p):+.3e} "
          f"rad/(m/s^2)  (0 => neutral steer, as expected for lf=lr, Caf=Car)")

    print("\nFree yaw response, r0=0.01 rad/s, delta=0 (must decay to 0):")
    Fx_trim = trim_force(20.0, p)
    x = np.array([20.0, 0.0, 0.01, 0.0, 0.0, 0.0])
    for t in range(6):
        print(f"  t={t}s  vy={x[1]:+.6f}  r={x[2]:+.6f}")
        x = simulate(x, [Fx_trim, 0.0], 1.0, 0.001, p)

    print("\nSteady-state step steer, delta=0.02 rad at 20 m/s:")
    x = np.array([20.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x = simulate(x, [Fx_trim, 0.02], 8.0, 0.001, p)
    L = p['lf']+p['lr']
    r_ack = 20.0*0.02/(L + understeer_gradient(p)*20.0**2)
    print(f"  simulated r_ss = {x[2]:.6f} rad/s")
    print(f"  analytic  r_ss = {r_ack:.6f} rad/s  "
          f"(rel. err {abs(x[2]-r_ack)/r_ack:.2e})")

    print("\nRK4 step-size convergence (should be ~4th order):")
    x0 = np.array([20.0, 0.2, 0.05, 0.0, 0.0, 0.0])
    ref = simulate(x0, [Fx_trim, 0.03], 0.5, 1e-5, p)
    prev = None
    for dt in (0.05, 0.025, 0.0125, 0.00625):
        err = np.linalg.norm(simulate(x0, [Fx_trim, 0.03], 0.5, dt, p) - ref)
        rate = "" if prev is None else f"  order~{np.log2(prev/err):.2f}"
        print(f"  dt={dt:8.5f}  err={err:.3e}{rate}")
        prev = err

    print("\nSaturating (Fiala) tyre, aggressive steer delta=0.15 rad:")
    x = np.array([20.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x = simulate(x, [Fx_trim, 0.15], 2.0, 0.001, p, tyre="fiala")
    af, ar = slip_angles(x[0], x[1], x[2], 0.15, p)
    print(f"  vx={x[0]:.3f} vy={x[1]:+.3f} r={x[2]:+.4f} "
          f"| alpha_f={af:+.4f} alpha_r={ar:+.4f} rad")
