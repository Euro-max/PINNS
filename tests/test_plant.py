"""Phase 1 acceptance: the plant is physically sane (guards D5, D6)."""
import numpy as np
import pytest
import tensorflow as tf

from pinc import plant
from pinc.plant import DEFAULT_PARAMS, simulate, lateral_eigs, understeer_gradient, trim_force, perturbed
from pinc import plant_tf


@pytest.mark.parametrize("vx", [5.0, 10.0, 20.0, 30.0, 40.0])
def test_lateral_eigenvalues_stable(vx):
    e = lateral_eigs(vx, DEFAULT_PARAMS)
    assert np.all(np.real(e) < 0), e


def test_free_yaw_decays():
    p = DEFAULT_PARAMS
    x = np.array([20.0, 0.0, 0.01, 0.0, 0.0, 0.0])
    x = simulate(x, [trim_force(20.0, p), 0.0], 2.0, 1e-3, p)
    assert abs(x[2]) < 1e-4
    assert abs(x[1]) < 1e-3


def test_step_steer_steady_yaw_rate():
    p = DEFAULT_PARAMS
    x = np.array([20.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x = simulate(x, [trim_force(20.0, p), 0.02], 8.0, 1e-3, p)
    L = p['lf'] + p['lr']
    vx = x[0]                          # allow for the speed drop
    r_ack = vx*0.02/(L + understeer_gradient(p)*vx**2)
    assert abs(x[2] - r_ack)/abs(r_ack) < 0.05


def test_rk4_convergence_order():
    p = DEFAULT_PARAMS
    x0 = np.array([20.0, 0.2, 0.05, 0.0, 0.0, 0.0])
    u = [trim_force(20.0, p), 0.03]
    ref = simulate(x0, u, 0.5, 1e-5, p)
    errs = [np.linalg.norm(simulate(x0, u, 0.5, dt, p) - ref) for dt in (0.05, 0.025, 0.0125)]
    orders = [np.log2(errs[i]/errs[i+1]) for i in range(len(errs)-1)]
    for o in orders:
        assert 3.8 <= o <= 4.2, orders


def test_asymmetric_wheelbase_stable():
    """Guards against D5 returning: yaw damping must not depend on lf == lr."""
    p = perturbed(DEFAULT_PARAMS, lf=1.2, lr=1.6)
    for vx in (5.0, 20.0, 40.0):
        assert np.all(np.real(lateral_eigs(vx, p)) < 0)
    x = np.array([20.0, 0.0, 0.05, 0.0, 0.0, 0.0])
    x = simulate(x, [trim_force(20.0, p), 0.0], 3.0, 1e-3, p)
    assert abs(x[2]) < 1e-3 and abs(x[1]) < 1e-2


def test_lateral_velocity_is_a_state():
    """Guards against D6: vy must evolve and couple into yaw."""
    p = DEFAULT_PARAMS
    x = np.array([20.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x = simulate(x, [trim_force(20.0, p), 0.05], 1.0, 1e-3, p)
    assert abs(x[1]) > 1e-3


def test_batched_matches_single():
    p = DEFAULT_PARAMS
    rng = np.random.default_rng(0)
    X0 = np.column_stack([rng.uniform(5, 30, 8), rng.uniform(-1.5, 1.5, 8), rng.uniform(-.6, .6, 8),
                          rng.uniform(-.5, .5, 8), rng.uniform(-10, 10, 8), rng.uniform(-10, 10, 8)])
    U = np.column_stack([rng.uniform(-6000, 3000, 8), rng.uniform(-.3, .3, 8)])
    Xb = simulate(X0, U, 0.1, 1e-3, p)
    for i in range(8):
        xi = simulate(X0[i], U[i], 0.1, 1e-3, p)
        np.testing.assert_allclose(Xb[i], xi, rtol=0, atol=1e-12)


@pytest.mark.parametrize("tyre", ["linear", "fiala"])
def test_numpy_and_tf_plants_agree(tyre):
    p = DEFAULT_PARAMS
    rng = np.random.default_rng(1)
    X0 = np.column_stack([rng.uniform(5, 30, 16), rng.uniform(-1.5, 1.5, 16), rng.uniform(-.6, .6, 16),
                          rng.uniform(-.5, .5, 16), rng.uniform(-10, 10, 16), rng.uniform(-10, 10, 16)])
    U = np.column_stack([rng.uniform(-6000, 3000, 16), rng.uniform(-.3, .3, 16)])
    x_np = simulate(X0, U, 1.0, 0.01, p, tyre=tyre)
    x_tf = plant_tf.simulate_tf(tf.constant(X0, tf.float64), tf.constant(U, tf.float64), 1.0, 0.01, p, tyre=tyre).numpy()
    assert np.max(np.abs(x_np - x_tf)) < 1e-5
    # also the 4-state (network) variant of f
    f4 = plant_tf.f_tf(tf.constant(X0[:, :4], tf.float64), tf.constant(U, tf.float64), p, tyre).numpy()
    f6 = plant.f(X0, U, p, tyre)
    np.testing.assert_allclose(f4, f6[:, :4], rtol=1e-12, atol=1e-9)


def test_fiala_saturates():
    p = DEFAULT_PARAMS
    F = plant.tyre_force(np.array([0.05, 0.3, 1.0]), p['Caf'], p['mu'], p['Fz'], "fiala")
    assert F[0] < p['Caf']*0.05
    assert abs(F[2]) <= p['mu']*p['Fz'] + 1e-9
    assert abs(F[1] - F[2]) < abs(F[0] - F[1])


def test_controller_and_plant_params_can_differ():
    """Ground rule 3 support: perturbed params only change the plant."""
    q = perturbed(DEFAULT_PARAMS, m=1800.0)
    assert q['m'] == 1800.0 and DEFAULT_PARAMS['m'] == 1500.0
    with pytest.raises(KeyError):
        perturbed(DEFAULT_PARAMS, mass=1.0)


def test_nonlinear_f_linearises_to_bicycle_model():
    """Numerical Jacobian of f w.r.t. (vy, r) at straight driving must equal the
    closed-form linear bicycle model used by `lateral_eigs` (checks signs of
    slip angles, tyre forces and the yaw moment)."""
    p = perturbed(DEFAULT_PARAMS, lf=1.2, lr=1.6, Caf=60000.0, Car=45000.0)   # asymmetric on purpose
    for vx in (8.0, 20.0):
        x0 = np.array([vx, 0.0, 0.0, 0.0, 0.0, 0.0])
        u = np.array([trim_force(vx, p), 0.0])
        eps = 1e-6
        J = np.zeros((2, 2))
        for j, idx in enumerate((1, 2)):
            xp, xm = x0.copy(), x0.copy()
            xp[idx] += eps
            xm[idx] -= eps
            J[:, j] = ((plant.f(xp, u, p) - plant.f(xm, u, p))/(2*eps))[[1, 2]]
        m, Iz, lf, lr, Caf, Car = p['m'], p['Iz'], p['lf'], p['lr'], p['Caf'], p['Car']
        A = np.array([[-(Caf + Car)/(m*vx), -(lf*Caf - lr*Car)/(m*vx) - vx],
                      [-(lf*Caf - lr*Car)/(Iz*vx), -(lf**2*Caf + lr**2*Car)/(Iz*vx)]])
        np.testing.assert_allclose(J, A, rtol=1e-6, atol=1e-6)
        # steering input column: d(dvy)/d(delta) = Caf/m, d(dr)/d(delta) = lf*Caf/Iz at delta = 0
        up, um = u.copy(), u.copy()
        up[1] += eps
        um[1] -= eps
        B = ((plant.f(x0, up, p) - plant.f(x0, um, p))/(2*eps))[[1, 2]]
        np.testing.assert_allclose(B, [Caf/m, lf*Caf/Iz], rtol=1e-6)


def test_steady_turn_force_balance():
    """In a steady left turn (delta > 0): r > 0, and the lateral acceleration
    vx*r + dvy/dt equals (Fyf cos(delta) + Fyr)/m; dvx/dt balances the trim
    force minus the steering drag Fyf sin(delta)."""
    p = DEFAULT_PARAMS
    delta = 0.03
    x = np.array([15.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x = simulate(x, [trim_force(15.0, p), delta], 6.0, 1e-3, p)
    dx = plant.f(x, [trim_force(15.0, p), delta], p)
    assert x[2] > 0 and abs(dx[2]) < 1e-2 and abs(dx[1]) < 1e-2      # near-steady (vx drifts slowly: steering drag)
    af, ar = plant.slip_angles(x[0], x[1], x[2], delta, p)
    Fyf, Fyr = p['Caf']*af, p['Car']*ar
    ay = x[0]*x[2] + dx[1]
    np.testing.assert_allclose(ay, (Fyf*np.cos(delta) + Fyr)/p['m'], rtol=1e-9)
    np.testing.assert_allclose(p['Iz']*dx[2], p['lf']*Fyf*np.cos(delta) - p['lr']*Fyr, atol=1e-9)
