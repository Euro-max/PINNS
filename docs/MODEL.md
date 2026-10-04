# Vehicle model used everywhere (`pinc/plant.py`, `pinc/plant_tf.py`)

Single-track ("bicycle") model with a body-fixed frame at the centre of gravity.
State `x = [vx, vy, r, psi, X, Y]`, input `u = [Fx, delta]`.

## Newton–Euler in the rotating body frame

```
m (dvx/dt − vy r) = Fx − ½ ρ Cd A vx² − Frr tanh(vx/0.1) − Fyf sin δ
m (dvy/dt + vx r) = Fyf cos δ + Fyr
Iz dr/dt          = lf Fyf cos δ − lr Fyr
dψ/dt = r
dX/dt = vx cos ψ − vy sin ψ
dY/dt = vx sin ψ + vy cos ψ
```

The terms `−vy r` and `+vx r` are the Coriolis terms of the rotating frame. `Fx` is the total
longitudinal tyre force, applied along the body x-axis (no longitudinal slip modelled). The
front lateral force acts perpendicular to the steered wheel plane, hence the `cos δ` / `sin δ`
projections; the rear wheel is not steered. Rolling resistance uses `tanh(vx/0.1)` so the sign
is right through zero speed; `vx` is floored at 0.5 m/s inside the slip angles and the drag.

## Slip angles and tyre forces

Front-axle velocity in the body frame is `(vx, vy + lf r)`, rear-axle `(vx, vy − lr r)`:

```
α_f = δ − atan2(vy + lf r, vx)
α_r =   − atan2(vy − lr r, vx)
Fy  = C α                       (linear)      or   Fiala with saturation at μ Fz   (E5)
```

With this sign convention a positive slip angle gives a positive (leftward) lateral force.

## Consistency checks (all in `tests/test_plant.py`)

- Linearising `f` numerically about straight driving reproduces the textbook linear bicycle
  model (`test_nonlinear_f_linearises_to_bicycle_model`, with `lf ≠ lr`, `Caf ≠ Car`):

  ```
  d/dt [vy]   [ −(Caf+Car)/(m vx)          −(lf Caf − lr Car)/(m vx) − vx ] [vy]   [  Caf/m   ]
       [ r] = [ −(lf Caf − lr Car)/(Iz vx)  −(lf² Caf + lr² Car)/(Iz vx)  ] [ r] + [ lf Caf/Iz ] δ
  ```

  The yaw damping `−(lf² Caf + lr² Car)/(Iz vx)` is negative for any parameters. The legacy
  model had `Iz ṙ = lf Fyf + lr Fyr` with `α_r = +lr r / v`, whose damping is
  `−(lf² Caf − lr² Car)/(Iz vx)` = **0** for the symmetric car: no directional stability (D5).
- Lateral eigenvalues have negative real parts at 5–40 m/s; free yaw decays; a step steer settles to
  `r = vx δ / (L + K_us vx²)` with `K_us = m (lr/Caf − lf/Car)/L` within 5 %.
- Steady turn: `vx r + dvy/dt = (Fyf cos δ + Fyr)/m` and `Iz ṙ = lf Fyf cos δ − lr Fyr` hold exactly
  (`test_steady_turn_force_balance`).
- RK4 converges at order 4; NumPy and TF implementations agree to 1e-5 over 1 s.

## Assumptions to state in the paper

Flat road, no load transfer (static axle loads `Fz = m g / 2` in the Fiala tyre), no longitudinal
tyre slip or combined-slip coupling, no actuator dynamics in v1 (zero-order hold; config flag
`actuator_lag`), aerodynamic drag from `vx` only, small steering angles not assumed (full
`cos δ`, `sin δ`, `atan2` kept).

## What the network predicts

`s = [vx, vy, r, psi]`; the first four rows of `f` are closed in `s` (they do not depend on X, Y).
The physics residual is `R = (ds/dt − f(s, u)) / S_f` with `ds/dt` from forward-mode autodiff along
the time input (`pinc/loss.py`) and `S_f` the standard deviation of each component of `f` over the
training box (`scripts/compute_scales.py`). `X, Y` are integrated from `s` outside the network
(trapezoidal rule in `pinc/mpc.py`, exact RK4 in the plant).
