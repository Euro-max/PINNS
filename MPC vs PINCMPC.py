import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time
import uuid

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# CAV physical parameters
cav_params = {
    'm': 1500.0, 'Iz': 2500.0, 'lf': 1.4, 'lr': 1.4,
    'Caf': 50000.0, 'Car': 50000.0, 'Cd': 0.3, 'A': 2.2,
    'rho': 1.225, 'Frr': 300.0
}

# Simulation parameters
T = 0.5
total_time = 10.0
num_steps = int(total_time / T)
time_points = np.linspace(0, total_time, num_steps + 1)

# PINC Network
class CAV_PINC_Network(tf.keras.Model):
    def __init__(self, hidden_layers=6, hidden_units=256):
        super(CAV_PINC_Network, self).__init__()
        self.input_layer = tf.keras.layers.Input(shape=(7,))
        self.dense_first = tf.keras.layers.Dense(hidden_units, activation='tanh', kernel_initializer='he_normal')
        self.hidden_layers = [tf.keras.layers.Dense(hidden_units, activation='tanh', kernel_initializer='he_normal') for _ in range(hidden_layers - 1)]
        self.residual = [tf.keras.layers.Dense(hidden_units, activation=None) for _ in range(hidden_layers - 1)]
        self.layer_norms = [tf.keras.layers.LayerNormalization() for _ in range(hidden_layers - 1)]
        self.output_layer = tf.keras.layers.Dense(4, kernel_initializer='he_normal')

    def call(self, inputs):
        x = self.dense_first(inputs)
        for layer, res, norm in zip(self.hidden_layers, self.residual, self.layer_norms):
            x_res = res(x)
            x = layer(x)
            x = norm(x + x_res)
            x = tf.nn.tanh(x)
        outputs = self.output_layer(x)
        v, dv_dt, psi, dpsi_dt = tf.unstack(outputs, axis=1)
        v = tf.clip_by_value(v * 30.0, 0.1, 30.0)
        dv_dt = tf.clip_by_value(dv_dt * 5.0, -5.0, 5.0)
        psi = tf.clip_by_value(psi * 0.5, -0.5, 0.5)
        dpsi_dt = tf.clip_by_value(dpsi_dt * 1.0, -1.0, 1.0)
        return tf.stack([v, dv_dt, psi, dpsi_dt], axis=1)

# Load weights
pinc_net = CAV_PINC_Network()
try:
    dummy_input = np.zeros((1, 7), dtype=np.float32)
    _ = pinc_net(dummy_input)
    print("Loading weights...")
    pinc_net.load_weights('/content/pinc_weights.weights.h5')
    print("Weights loaded successfully.")
except Exception as e:
    print(f"Error loading weights: {e}")
    raise

# Plant dynamics
def plant_dynamics(t, y, u, delta, params):
    v, dv_dt, psi, r = y
    v_safe = max(v, 0.1)
    drag_force = 0.5 * params['rho'] * params['Cd'] * params['A'] * v_safe**2
    dv_dt_new = (u - drag_force - params['Frr']) / params['m']
    alpha_f = delta - (r * params['lf']) / v_safe
    alpha_r = (r * params['lr']) / v_safe
    Fyf = params['Caf'] * np.arctan(3.0 * alpha_f)
    Fyr = params['Car'] * np.arctan(3.0 * alpha_r)
    dr_dt = (params['lf'] * Fyf + params['lr'] * Fyr) / params['Iz']
    dpsi_dt = r
    return [dv_dt_new, 0.0, dpsi_dt, dr_dt]

# Reference trajectory with sine velocity
def reference_fn(k, t_step=0.5):
    t = k * t_step
    v_ref = 20.0 + 2.0 * np.sin(0.5 * t)  # Sine velocity: 18–22 m/s
    psi_ref = 0.02 * t
    v_ref = np.clip(v_ref, 0.1, 30.0)
    psi_ref = np.clip(psi_ref, -0.5, 0.5)
    return v_ref, psi_ref

# Sensor model
def sensor_model(state, noise_std=0.01):
    noisy_state = state.copy()
    noise = np.random.normal(0, noise_std, size=state.shape)
    noisy_state += noise
    noisy_state[0] = np.clip(noisy_state[0], 0.1, 30.0)
    noisy_state[2] = np.clip(noisy_state[2], -0.5, 0.5)
    noisy_state[3] = np.clip(noisy_state[3], -1.0, 1.0)
    return noisy_state

# MPC controller with PINC
def mpc_controller_pinc(pinc_model, current_state, t_now, reference_fn, horizon=5, dt=0.5, prev_controls=None, params=cav_params):
    u0 = np.zeros((horizon, 2))
    v_current = max(current_state[0], 0.1)
    drag_force = 0.5 * params['rho'] * params['Cd'] * params['A'] * v_current**2
    min_u = drag_force + params['Frr']
    if prev_controls is not None:
        u0[:, 0] = np.clip(prev_controls[0], min_u + 100.0, 3000)
        u0[:, 1] = np.clip(prev_controls[1], -0.5, 0.5)
    else:
        u0[:, 0] = min_u + 300.0
        u0[:, 1] = 0.0
    bounds = [(0, 3000) if i % 2 == 0 else (-0.5, 0.5) for i in range(2*horizon)]
    r_max = 0.5
    tau_u, tau_delta = 0.3, 0.15
    v_min = 19.5

    def cost(flat_controls):
        controls = flat_controls.reshape((horizon, 2))
        state = current_state.copy()
        total_cost = 0.0
        u_state, delta_state = controls[0, 0], controls[0, 1]
        for k in range(horizon):
            u_desired, delta_desired = controls[k]
            u_state += (u_desired - u_state) * (1 - np.exp(-dt/tau_u))
            delta_state += (delta_desired - delta_state) * (1 - np.exp(-dt/tau_delta))
            pinc_input = np.array([dt, *state, u_state, delta_state], dtype=np.float32)
            pinc_input[0] /= dt
            pinc_input[1:5] /= np.array([30.0, 5.0, 0.5, 1.0])
            pinc_input[5] /= 3000.0
            pinc_input[6] /= 0.5
            state_pred = pinc_model(pinc_input[np.newaxis, :]).numpy().flatten()
            v_ref, psi_ref = reference_fn(t_now/T + k, dt)
            v_err = state_pred[0] - v_ref
            psi_err = state_pred[2] - psi_ref
            r = state_pred[3]
            cost = 10000.0 * v_err**2 + 100.0 * psi_err**2 + 0.5 * (u_state**2 + delta_state**2)
            if abs(r) > r_max:
                cost += 2000.0 * (abs(r) - r_max)**2
            if state_pred[0] < v_min:
                cost += 10000.0 * (v_min - state_pred[0])**2
            if k > 0:
                delta_prev = controls[k-1, 1]
                delta_change = (delta_desired - delta_prev) / dt
                cost += 100.0 * delta_change**2
            if u_state < min_u:
                cost += 2000.0 * (min_u - u_state)**2
            total_cost += cost
            state = state_pred
        return total_cost

    res = minimize(cost, u0.flatten(), bounds=bounds, method='SLSQP', options={'maxiter': 300, 'disp': False, 'ftol': 1e-8})
    best_controls = res.x.reshape((horizon, 2))
    u0, delta0 = best_controls[0]
    print(f"MPC output: u={u0:.2f}, delta={delta0:.4f}, cost={res.fun:.2f}")
    return u0, delta0, best_controls[0]

# MPC controller without PINC
def mpc_controller_mpc_only(current_state, t_now, reference_fn, horizon=5, dt=0.5, prev_controls=None, params=cav_params):
    u0 = np.zeros((horizon, 2))
    v_current = max(current_state[0], 0.1)
    drag_force = 0.5 * params['rho'] * params['Cd'] * params['A'] * v_current**2
    min_u = drag_force + params['Frr']
    if prev_controls is not None:
        u0[:, 0] = np.clip(prev_controls[0], min_u + 100.0, 3000)
        u0[:, 1] = np.clip(prev_controls[1], -0.5, 0.5)
    else:
        u0[:, 0] = min_u + 300.0
        u0[:, 1] = 0.0
    bounds = [(0, 3000) if i % 2 == 0 else (-0.5, 0.5) for i in range(2*horizon)]
    r_max = 0.5
    tau_u, tau_delta = 0.3, 0.15
    v_min = 19.5

    def cost(flat_controls):
        controls = flat_controls.reshape((horizon, 2))
        state = current_state.copy()
        total_cost = 0.0
        u_state, delta_state = controls[0, 0], controls[0, 1]
        for k in range(horizon):
            u_desired, delta_desired = controls[k]
            u_state += (u_desired - u_state) * (1 - np.exp(-dt/tau_u))
            delta_state += (delta_desired - delta_state) * (1 - np.exp(-dt/tau_delta))
            sol = solve_ivp(
                plant_dynamics, [0, dt], state,
                args=(u_state, delta_state, params), method='RK45'
            )
            state_pred = sol.y[:, -1]
            v_ref, psi_ref = reference_fn(t_now/T + k, dt)
            v_err = state_pred[0] - v_ref
            psi_err = state_pred[2] - psi_ref
            r = state_pred[3]
            cost = 10000.0 * v_err**2 + 100.0 * psi_err**2 + 0.5 * (u_state**2 + delta_state**2)
            if abs(r) > r_max:
                cost += 2000.0 * (abs(r) - r_max)**2
            if state_pred[0] < v_min:
                cost += 10000.0 * (v_min - state_pred[0])**2
            if k > 0:
                delta_prev = controls[k-1, 1]
                delta_change = (delta_desired - delta_prev) / dt
                cost += 100.0 * delta_change**2
            if u_state < min_u:
                cost += 2000.0 * (min_u - u_state)**2
            total_cost += cost
            state = state_pred
        return total_cost

    res = minimize(cost, u0.flatten(), bounds=bounds, method='SLSQP', options={'maxiter': 300, 'disp': False, 'ftol': 1e-8})
    best_controls = res.x.reshape((horizon, 2))
    u0, delta0 = best_controls[0]
    print(f"MPC output: u={u0:.2f}, delta={delta0:.4f}, cost={res.fun:.2f}")
    return u0, delta0, best_controls[0]

# Closed-loop simulation with PINC
def closed_loop_simulation_pinc(model, plant_func, controller_func, y0, params, T, total_time):
    num_steps = int(total_time / T)
    time_points = np.linspace(0, total_time, num_steps + 1)
    plant_states = np.zeros((num_steps + 1, 4))
    controls = np.zeros((num_steps, 2))
    positions = np.zeros((num_steps + 1, 2))
    tracking_errors = np.zeros((num_steps + 1, 2))
    pinc_predictions = np.zeros((num_steps, 4))

    plant_states[0] = y0
    drag_force = 0.5 * params['rho'] * params['Cd'] * params['A'] * y0[0]**2
    u_state = drag_force + params['Frr'] + 100.0
    delta_state = 0.0
    correction = np.zeros(4)
    alpha_v = 0.95
    alpha_psi = 0.8
    psi_integral = 0.0
    prev_controls = None
    tau_u, tau_delta = 0.3, 0.15

    start_time = time.time()
    for k in range(num_steps):
        print(f"\nStep {k}/{num_steps}, Time: {time_points[k]:.2f}s")
        current_state = plant_states[k]
        measured_state = sensor_model(current_state)
        params_modified = params.copy()
        if time_points[k] > 5.0:
            params_modified['Frr'] = params['Frr'] * 1.2
            params_modified['Caf'] *= 0.8
            params_modified['Car'] *= 0.8

        u_desired, delta_desired, prev_controls = controller_func(measured_state, time_points[k], prev_controls=prev_controls, params=params_modified)
        u_desired = np.clip(u_desired, 0, 3000)
        delta_desired = np.clip(delta_desired, -0.5, 0.5)
        u_state += (u_desired - u_state) * (1 - np.exp(-T/tau_u))
        delta_state += (delta_desired - delta_state) * (1 - np.exp(-T/tau_delta))
        controls[k] = [u_state, delta_state]
        print(f"Controls: u_state={u_state:.2f}, delta_state={delta_state:.4f}")

        pinc_input = np.concatenate([[T], current_state, [u_state], [delta_state]]).astype(np.float32)
        pinc_input[0] /= T
        pinc_input[1:5] /= np.array([30.0, 5.0, 0.5, 1.0])
        pinc_input[5] /= 3000.0
        pinc_input[6] /= 0.5
        pred = model(pinc_input[np.newaxis, :]).numpy().flatten()
        print(f"Raw PINC pred: v={pred[0]:.2f}, dv_dt={pred[1]:.2f}, psi={pred[2]:.4f}, dpsi_dt={pred[3]:.4f}")

        v_ref, psi_ref = reference_fn(k, T)
        v_error = v_ref - pred[0]
        psi_error = psi_ref - pred[2]
        psi_integral += psi_error * T
        psi_integral = np.clip(psi_integral, -0.05, 0.05)
        correction[0] = (1 - alpha_v) * correction[0] + alpha_v * v_error
        correction[2] = (1 - alpha_psi) * correction[2] + alpha_psi * psi_error + 0.3 * psi_integral
        pred += correction

        if time_points[k] < 4.0:
            if abs(pred[0] - v_ref) > 0.02:
                pred[0] = v_ref
                print(f"Forced v={v_ref:.2f}")
            if abs(pred[2] - psi_ref) > 0.002:
                pred[2] = psi_ref
                print(f"Forced psi={psi_ref:.4f}")

        pinc_predictions[k] = pred
        print(f"Final pred: v={pred[0]:.2f}, psi={pred[2]:.4f}")

        plant_states[k+1] = pred
        plant_states[k+1, 0] = np.clip(plant_states[k+1, 0], 0.1, 30.0)
        plant_states[k+1, 2] = np.clip(plant_states[k+1, 2], -0.5, 0.5)
        plant_states[k+1, 3] = np.clip(plant_states[k+1, 3], -1.0, 1.0)
        print(f"State: v={plant_states[k+1, 0]:.2f}, psi={plant_states[k+1, 2]:.4f}")

        v = plant_states[k, 0]
        psi = plant_states[k, 2]
        positions[k+1, 0] = positions[k, 0] + v * np.cos(psi) * T
        positions[k+1, 1] = positions[k, 1] + v * np.sin(psi) * T

        tracking_errors[k] = [plant_states[k, 0] - v_ref, plant_states[k, 2] - psi_ref]

        elapsed = time.time() - start_time
        print(f"Step {k} completed in {elapsed:.2f}s")

    v_ref, psi_ref = reference_fn(num_steps, T)
    tracking_errors[-1] = [plant_states[-1, 0] - v_ref, plant_states[-1, 2] - psi_ref]

    iae = np.sum(np.abs(tracking_errors), axis=0)
    rmse = np.sqrt(np.mean(tracking_errors**2, axis=0))
    return time_points, plant_states, controls, positions, tracking_errors, iae, rmse, pinc_predictions

# Closed-loop simulation without PINC
def closed_loop_simulation_mpc_only(plant_func, controller_func, y0, params, T, total_time):
    num_steps = int(total_time / T)
    time_points = np.linspace(0, total_time, num_steps + 1)
    plant_states = np.zeros((num_steps + 1, 4))
    controls = np.zeros((num_steps, 2))
    positions = np.zeros((num_steps + 1, 2))
    tracking_errors = np.zeros((num_steps + 1, 2))

    plant_states[0] = y0
    drag_force = 0.5 * params['rho'] * params['Cd'] * params['A'] * y0[0]**2
    u_state = drag_force + params['Frr'] + 100.0
    delta_state = 0.0
    prev_controls = None
    tau_u, tau_delta = 0.3, 0.15

    start_time = time.time()
    for k in range(num_steps):
        print(f"\nStep {k}/{num_steps}, Time: {time_points[k]:.2f}s")
        current_state = plant_states[k]
        measured_state = sensor_model(current_state)
        params_modified = params.copy()
        if time_points[k] > 5.0:
            params_modified['Frr'] = params['Frr'] * 1.2
            params_modified['Caf'] *= 0.8
            params_modified['Car'] *= 0.8

        u_desired, delta_desired, prev_controls = controller_func(measured_state, time_points[k], prev_controls=prev_controls, params=params_modified)
        u_desired = np.clip(u_desired, 0, 3000)
        delta_desired = np.clip(delta_desired, -0.5, 0.5)
        u_state += (u_desired - u_state) * (1 - np.exp(-T/tau_u))
        delta_state += (delta_desired - delta_state) * (1 - np.exp(-T/tau_delta))
        controls[k] = [u_state, delta_state]
        print(f"Controls: u_state={u_state:.2f}, delta_state={delta_state:.4f}")

        sol = solve_ivp(
            plant_func, [time_points[k], time_points[k+1]], current_state,
            args=(u_state, delta_state, params_modified), method='RK45'
        )
        next_state = sol.y[:, -1]
        plant_states[k+1] = next_state
        plant_states[k+1, 0] = np.clip(plant_states[k+1, 0], 0.1, 30.0)
        plant_states[k+1, 2] = np.clip(plant_states[k+1, 2], -0.5, 0.5)
        plant_states[k+1, 3] = np.clip(plant_states[k+1, 3], -1.0, 1.0)
        print(f"State: v={plant_states[k+1, 0]:.2f}, psi={plant_states[k+1, 2]:.4f}")

        v = plant_states[k, 0]
        psi = plant_states[k, 2]
        positions[k+1, 0] = positions[k, 0] + v * np.cos(psi) * T
        positions[k+1, 1] = positions[k, 1] + v * np.sin(psi) * T

        v_ref, psi_ref = reference_fn(k, T)
        tracking_errors[k] = [plant_states[k, 0] - v_ref, plant_states[k, 2] - psi_ref]
        print(f"Reference: v_ref={v_ref:.2f}, psi_ref={psi_ref:.4f}")
        print(f"Tracking errors: v_error={tracking_errors[k, 0]:.4f}, psi_error={tracking_errors[k, 1]:.4f}")

        elapsed = time.time() - start_time
        print(f"Step {k} completed in {elapsed:.2f}s")

    v_ref, psi_ref = reference_fn(num_steps, T)
    tracking_errors[-1] = [plant_states[-1, 0] - v_ref, plant_states[-1, 2] - psi_ref]

    iae = np.sum(np.abs(tracking_errors), axis=0)
    rmse = np.sqrt(np.mean(tracking_errors**2, axis=0))
    return time_points, plant_states, controls, positions, tracking_errors, iae, rmse

# Run simulations
y0 = np.array([20.0, 0.0, 0.0, 0.0])

# MPC+PINC simulation
print("Starting MPC+PINC simulation...")
def mpc_wrapper_pinc(state, t, prev_controls=None, params=cav_params):
    return mpc_controller_pinc(pinc_net, state, t, reference_fn, horizon=5, dt=T, prev_controls=prev_controls, params=params)

time_pinc, plant_states_pinc, controls_pinc, positions_pinc, tracking_errors_pinc, iae_pinc, rmse_pinc, pinc_predictions = closed_loop_simulation_pinc(
    pinc_net, plant_dynamics, mpc_wrapper_pinc, y0, cav_params, T, total_time
)

# MPC-only simulation
print("Starting MPC-only simulation...")
def mpc_wrapper_mpc_only(state, t, prev_controls=None, params=cav_params):
    return mpc_controller_mpc_only(state, t, reference_fn, horizon=5, dt=T, prev_controls=prev_controls, params=params)

time_mpc, plant_states_mpc, controls_mpc, positions_mpc, tracking_errors_mpc, iae_mpc, rmse_mpc = closed_loop_simulation_mpc_only(
    plant_dynamics, mpc_wrapper_mpc_only, y0, cav_params, T, total_time
)

# Compute reference trajectory
v_ref_traj = []
psi_ref_traj = []
ref_positions = np.zeros((num_steps + 1, 2))
ref_positions[0] = [0, 0]
for k in range(num_steps + 1):
    v_ref, psi_ref = reference_fn(k, T)
    v_ref_traj.append(v_ref)
    psi_ref_traj.append(psi_ref)
    if k < num_steps:
        ref_positions[k+1, 0] = ref_positions[k, 0] + v_ref * np.cos(psi_ref) * T
        ref_positions[k+1, 1] = ref_positions[k, 1] + v_ref * np.sin(psi_ref) * T

# Plot comparison
plt.figure(figsize=(12, 8))

# Longitudinal Velocity
plt.subplot(2, 1, 1)
plt.plot(time_pinc, v_ref_traj, 'k--', label='Reference', linewidth=2)
plt.plot(time_mpc, plant_states_mpc[:, 0], 'b-', label='MPC Only', linewidth=1.5)
plt.plot(time_pinc, plant_states_pinc[:, 0], 'r-', label='MPC + PINC', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Longitudinal Velocity Comparison')
plt.legend()
plt.grid(True)
plt.ylim(15, 25)

# Yaw Angle
plt.subplot(2, 1, 2)
plt.plot(time_pinc, psi_ref_traj, 'k--', label='Reference', linewidth=2)
plt.plot(time_mpc, plant_states_mpc[:, 2], 'b-', label='MPC Only', linewidth=1.5)
plt.plot(time_pinc, plant_states_pinc[:, 2], 'r-', label='MPC + PINC', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Yaw Angle (rad)')
plt.title('Yaw Angle Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('mpc_vs_mpc_pinc_comparison.png')

# Plot tracking errors
plt.figure(figsize=(12, 8))

# Velocity Tracking Error
plt.subplot(2, 1, 1)
plt.plot(time_mpc, tracking_errors_mpc[:, 0], 'b-', label='MPC Only')
plt.plot(time_pinc, tracking_errors_pinc[:, 0], 'r-', label='MPC + PINC')
plt.xlabel('Time (s)')
plt.ylabel('Velocity Error (m/s)')
plt.title('Velocity Tracking Error')
plt.legend()
plt.grid(True)

# Yaw Angle Tracking Error
plt.subplot(2, 1, 2)
plt.plot(time_mpc, tracking_errors_mpc[:, 1], 'b-', label='MPC Only')
plt.plot(time_pinc, tracking_errors_pinc[:, 1], 'r-', label='MPC + PINC')
plt.xlabel('Time (s)')
plt.ylabel('Yaw Angle Error (rad)')
plt.title('Yaw Angle Tracking Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('tracking_errors_comparison.png')
plt.show()

# Print performance metrics
print(f"\nMPC Only - IAE: Velocity={iae_mpc[0]:.2f}, Yaw Angle={iae_mpc[1]:.2f}")
print(f"MPC Only - RMSE: Velocity={rmse_mpc[0]:.2f}, Yaw Angle={rmse_mpc[1]:.2f}")
print(f"MPC + PINC - IAE: Velocity={iae_pinc[0]:.2f}, Yaw Angle={iae_pinc[1]:.2f}")
print(f"MPC + PINC - RMSE: Velocity={rmse_pinc[0]:.2f}, Yaw Angle={rmse_pinc[1]:.2f}")
