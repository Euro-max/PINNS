import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models, optimizers
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import gc

class CAV_PINC_Network(tf.keras.Model):
    def __init__(self, hidden_layers=6, hidden_units=256):
        """
        Physics-Informed Neural Network for CAV Dynamics
        
        Args:
            hidden_layers: Number of hidden layers
            hidden_units: Number of units per hidden layer
        """
        super(CAV_PINC_Network, self).__init__()
        
        # Define the input layer explicitly
        self.input_layer = layers.Input(shape=(7,))  # Expecting [T, v, dv_dt, psi, r, u, delta]
        
        # First dense layer after the input
        self.dense_first = layers.Dense(
            hidden_units, 
            activation='tanh',
            kernel_initializer='he_normal',
            bias_initializer='zeros'
        )
        
        self.hidden_layers = [layers.Dense(
            hidden_units, 
            activation='tanh',
            kernel_initializer='he_normal',
            bias_initializer='zeros')
            for _ in range(hidden_layers - 1)]
        
        # Added residual connections
        self.residual = [layers.Dense(hidden_units, activation=None) 
                       for _ in range(hidden_layers - 1)]
        
        self.layer_norms = [layers.LayerNormalization() 
                          for _ in range(hidden_layers - 1)]
        
        self.output_layer = layers.Dense(
            4,  # Outputs: [v, dv/dt, ψ, dψ/dt]
            kernel_initializer='he_normal',
            bias_initializer='zeros')
        
        self.output_names = ['v', 'dv_dt', 'psi', 'dpsi_dt']

    def call(self, inputs):
        """
        Forward pass of the CAV PINC network with residual connections
        
        Args:
            inputs: concatenated tensor of [t, y0, u, δ]
                   t: time (scalar)
                   y0: initial state [v, dv/dt, ψ, dψ/dt]
                   u: longitudinal control input
                   δ: lateral control input
                   
        Returns:
            y(t): full state at time t [v, dv/dt, ψ, dψ/dt]
        """
        x = self.dense_first(inputs)
        for layer, res, norm in zip(self.hidden_layers, self.residual, self.layer_norms):
            x_res = res(x)
            x = layer(x)
            x = norm(x + x_res)  # Residual connection
            x = tf.nn.tanh(x)
        outputs = self.output_layer(x)
        # Scale outputs to match normalized ranges with safety checks
        v, dv_dt, psi, dpsi_dt = tf.unstack(outputs, axis=1)
        v = tf.clip_by_value(v * 30.0, 0.1, 30.0)  # De-normalize velocity with bounds
        dv_dt = tf.clip_by_value(dv_dt * 5.0, -5.0, 5.0)  # De-normalize acceleration
        psi = tf.clip_by_value(psi * 0.5, -0.5, 0.5)  # De-normalize yaw angle
        dpsi_dt = tf.clip_by_value(dpsi_dt * 1.0, -1.0, 1.0)  # De-normalize yaw rate
        return tf.stack([v, dv_dt, psi, dpsi_dt], axis=1)

@tf.function
def cav_pinc_loss(model, t_data, y0_data, u_data, y_true, t_colloc, y0_colloc, u_colloc, 
                 m, Iz, lf, lr, Caf, Car, Cd, A, rho, Frr, lambda_reg=1.0):
    """
    Enhanced physics-informed loss function for CAV PINC with:
    - Bicycle model dynamics
    - Aerodynamic drag
    - Rolling resistance
    - Numerical safeguards
    
    Args:
        model: CAV PINC network
        t_data: time points for data loss (can be empty tensor)
        y0_data: initial states for data loss (can be empty tensor)
        u_data: control inputs for data loss (can be empty tensor)
        y_true: true states (can be empty tensor)
        t_colloc: time points for collocation loss
        y0_colloc: initial states for collocation
        u_colloc: control inputs for collocation
        m: vehicle mass (kg)
        Iz: yaw moment of inertia (kg·m²)
        lf: distance from CG to front axle (m)
        lr: distance from CG to rear axle (m)
        Caf: front tire cornering stiffness (N/rad)
        Car: rear tire cornering stiffness (N/rad)
        Cd: aerodynamic drag coefficient
        A: frontal area (m²)
        rho: air density (kg/m³)
        Frr: rolling resistance force (N)
        lambda_reg: physics loss weight
        
    Returns:
        Tuple of (total_loss, mse_physics) where:
        - total_loss: mse_data + lambda_reg * mse_physics
        - mse_physics: raw physics loss (unscaled)
    """
    # Data loss (skip computation if inputs are empty)
    def compute_data_loss():
        inputs_data = tf.concat([t_data, y0_data, u_data], axis=1)
        y_pred_data = model(inputs_data)
        return tf.reduce_mean(tf.square(y_pred_data - y_true))
    
    has_data = tf.size(t_data) > 0
    mse_data = tf.cond(
        has_data,
        compute_data_loss,
        lambda: tf.constant(0.0, dtype=tf.float32)
    )
    
    # Physics loss (always computed)
    def compute_physics_loss():
        inputs_colloc = tf.concat([t_colloc, y0_colloc, u_colloc], axis=1)
        
        with tf.GradientTape(persistent=True) as outer_tape:
            outer_tape.watch(inputs_colloc)
            y_pred = model(inputs_colloc)
            v, dv_dt, psi, r = tf.unstack(y_pred, axis=1)
            u_long, delta = tf.unstack(u_colloc, axis=1)
            
            # Numerical safeguards
            v_safe = tf.maximum(v, 0.3)  # Minimum velocity 0.1 m/s
            delta_clipped = tf.clip_by_value(delta, -0.3, 0.3)  # Realistic steering bounds
            
            # First derivatives (dv/dt and dr/dt)
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(inputs_colloc)
                y_pred_inner = model(inputs_colloc)
                v_inner, _, psi_inner, r_inner = tf.unstack(y_pred_inner, axis=1)
            
            dy_dt = inner_tape.gradient(y_pred_inner, inputs_colloc)
            dv_dt_pred = dy_dt[:, 0]  # dv/dt
            dr_dt_pred = dy_dt[:, 3]  # dr/dt
            dpsi_dt_pred = dy_dt[:, 2]  # dpsi/dt
        
        # Enhanced bicycle model dynamics with actuator dynamics
        tau_u = 0.2  # Throttle time constant
        u_actual = u_long * (1 - tf.exp(-t_colloc/tau_u))  # First-order response
        
        # Longitudinal dynamics with drag and rolling resistance
        drag_force = 0.5 * rho * Cd * A * tf.square(v_safe)
        residual_long = (m * dv_dt_pred - (u_actual * 3000.0 - drag_force - Frr)) / 1e3
        
        # Enhanced lateral dynamics with Pacejka-like tire model
        alpha_f = tf.clip_by_value(delta_clipped * 0.3 - (r * lf) / v_safe, -0.5, 0.5)
        alpha_r = tf.clip_by_value((r * lr) / v_safe, -0.5, 0.5)
        Fyf = tf.clip_by_value(Caf * tf.math.atan(3.0 * alpha_f), -Caf*1.5, Caf*1.5)
        Fyr = tf.clip_by_value(Car * tf.math.atan(3.0 * alpha_r), -Car*1.5, Car*1.5)
        residual_yaw = (Iz * dr_dt_pred - (lf * Fyf + lr * Fyr)) / 1e3
        
        # Yaw angle consistency
        residual_psi = (dpsi_dt_pred - r) / 1e1
        
        # Apply clipping and combine losses
        mse_physics = (
            tf.reduce_mean(tf.square(tf.clip_by_value(residual_long, -1e3, 1e3))) +
            tf.reduce_mean(tf.square(tf.clip_by_value(residual_yaw, -1e3, 1e3))) +
            tf.reduce_mean(tf.square(tf.clip_by_value(residual_psi, -1e3, 1e3)))
        )
        del outer_tape
        return mse_physics
    
    mse_physics = compute_physics_loss()
    mse_physics = tf.where(tf.math.is_finite(mse_physics), mse_physics, 1e6)
    total_loss = mse_data + lambda_reg * mse_physics
    total_loss = tf.where(tf.math.is_finite(total_loss), total_loss, 1e6)
    
    return total_loss, mse_physics

def generate_cav_training_data(num_data_points, num_colloc_points, T, dt=0.1,
                             v_range=(0.5, 30), psi_range=(-0.5, 0.5),
                             u_range=(0, 3000), delta_range=(-0.3, 0.3),
                             m=1500.0, Iz=2500.0, lf=1.4, lr=1.4, 
                             Caf=50000.0, Car=50000.0, Cd=0.3, A=2.2, 
                             rho=1.225, Frr=300.0):
    """
    Enhanced training data generator with:
    - Realistic control sequences
    - Actuator dynamics simulation
    - Safety bounds on generated states
    
    Args:
        num_data_points: number of data points (initial conditions)
        num_colloc_points: number of collocation points
        T: time horizon for collocation points
        dt: time step for Euler integration
        v_range: tuple of (min, max) velocity (m/s)
        psi_range: tuple of (min, max) yaw angle (rad)
        u_range: tuple of (min, max) longitudinal control (N)
        delta_range: tuple of (min, max) steering angle (rad)
        m: vehicle mass (kg)
        Iz: yaw moment of inertia (kg·m²)
        lf: distance from CG to front axle (m)
        lr: distance from CG to rear axle (m)
        Caf: front tire cornering stiffness (N/rad)
        Car: rear tire cornering stiffness (N/rad)
        Cd: aerodynamic drag coefficient
        A: frontal area (m²)
        rho: air density (kg/m³)
        Frr: rolling resistance force (N)
        
    Returns:
        Tuple of (t_data, y0_data, u_data, y_true, t_colloc, y0_colloc, u_colloc)
    """
    # Time points with jitter for robustness
    t_data = tf.random.uniform((num_data_points, 1), -0.1*dt, 0.1*dt)
    
    # Initial states with safe bounds
    v_init = tf.random.uniform((num_data_points, 1), *v_range) / 30.0
    a_init = tf.random.uniform((num_data_points, 1), -3, 3) / 5.0  # Reduced max acceleration
    psi_init = tf.random.uniform((num_data_points, 1), *psi_range) / 0.5
    r_init = tf.random.uniform((num_data_points, 1), -0.5, 0.5) / 1.0  # Reduced yaw rate
    
    # Control inputs with realistic profiles
    t_span = tf.linspace(0.0, 10.0, num_data_points)
    u_base = 1000.0 * (1 + 0.5*tf.sin(2*np.pi*t_span/5.0))  # Varying throttle
    u_rand = tf.random.uniform((num_data_points, 1), *u_range)
    u = 0.7*u_base[:,tf.newaxis] + 0.3*u_rand  # Blend patterns
    u = u / 3000.0
    
    delta = 0.1 * tf.sin(2*np.pi*t_span/3.0)  # Smooth steering
    delta = tf.reshape(delta, (-1, 1))  # Now delta is (2000, 1)
    delta += 0.05 * tf.random.normal((num_data_points, 1))
    delta = tf.clip_by_value(delta, *delta_range) / 0.3
    
    # Enhanced dynamics integration with actuator delay
    tau_u, tau_delta = 0.2, 0.15  # Actuator time constants
    u_actual = u * (1 - tf.exp(-t_span[:,tf.newaxis]/tau_u))
    delta_actual = delta * (1 - tf.exp(-t_span[:,tf.newaxis]/tau_delta))
    
    # Longitudinal dynamics
    v_denorm = v_init * 30.0
    drag_force = 0.5 * rho * Cd * A * tf.square(v_denorm)
    a_next = (u_actual * 3000.0 - drag_force - Frr) / m
    v_next = tf.clip_by_value(v_init + (a_next * dt / 30.0), 0.1/30.0, 1.0)
    
    # Lateral dynamics with nonlinear tires
    v_safe = tf.maximum(v_denorm, 0.5)
    alpha_f = delta_actual * 0.3 - (r_init * 1.0 * lf) / v_safe
    alpha_r = (r_init * 1.0 * lr) / v_safe
    Fyf = Caf * tf.math.atan(3.0 * alpha_f)
    Fyr = Car * tf.math.atan(3.0 * alpha_r)
    r_dot = (lf * Fyf + lr * Fyr) / Iz
    r_next = tf.clip_by_value(r_init + (r_dot * dt / 1.0), -0.5/1.0, 0.5/1.0)
    psi_next = tf.clip_by_value(psi_init + (r_next * dt / 0.5), -1.0, 1.0)
    
    # Construct datasets
    y0_data = tf.concat([v_next, a_next / 5.0, psi_next, r_next], axis=1)
    u_data = tf.concat([u, delta], axis=1)
    y_true = y0_data
    
    # Collocation points with broader coverage
    t_colloc = tf.random.uniform((num_colloc_points, 1), 0.0, 2*T) / T  # Extended range
    
    v_colloc = tf.random.uniform((num_colloc_points, 1), *v_range) / 30.0
    a_colloc = tf.random.uniform((num_colloc_points, 1), -3, 3) / 5.0
    psi_colloc = tf.random.uniform((num_colloc_points, 1), *psi_range) / 0.5
    r_colloc = tf.random.uniform((num_colloc_points, 1), -0.5, 0.5) / 1.0
    
    t_colloc_span = tf.linspace(0.0, 10.0, num_colloc_points)
    u_colloc_base = 1000.0 * (1 + 0.5*tf.sin(2*np.pi*t_colloc_span/5.0))
    u_colloc_rand = tf.random.uniform((num_colloc_points, 1), *u_range)
    u_colloc = 0.7*u_colloc_base[:,tf.newaxis] + 0.3*u_colloc_rand
    u_colloc = u_colloc / 3000.0
    
    delta_colloc = 0.1 * tf.sin(2*np.pi*t_colloc_span/3.0)[:, tf.newaxis]
    delta_colloc += 0.05 * tf.random.normal((num_colloc_points, 1))
    delta_colloc = tf.clip_by_value(delta_colloc, *delta_range) / 0.3
    
    u_colloc_actual = u_colloc * (1 - tf.exp(-t_colloc_span[:,tf.newaxis]/tau_u))
    delta_colloc_actual = delta_colloc * (1 - tf.exp(-t_colloc_span[:,tf.newaxis]/tau_delta))
    
    v_colloc_denorm = v_colloc * 30.0
    drag_force_colloc = 0.5 * rho * Cd * A * tf.square(v_colloc_denorm)
    a_colloc_next = (u_colloc_actual * 3000.0 - drag_force_colloc - Frr) / m
    v_colloc_next = tf.clip_by_value(v_colloc + (a_colloc_next * dt / 30.0), 0.1/30.0, 1.0)
    
    v_colloc_safe = tf.maximum(v_colloc_denorm, 0.5)
    alpha_f_colloc = delta_colloc_actual * 0.3 - (r_colloc * 1.0 * lf) / v_colloc_safe
    alpha_r_colloc = (r_colloc * 1.0 * lr) / v_colloc_safe
    Fyf_colloc = Caf * tf.math.atan(3.0 * alpha_f_colloc)
    Fyr_colloc = Car * tf.math.atan(3.0 * alpha_r_colloc)
    r_dot_colloc = (lf * Fyf_colloc + lr * Fyr_colloc) / Iz
    r_colloc_next = tf.clip_by_value(r_colloc + (r_dot_colloc * dt / 1.0), -0.5/1.0, 0.5/1.0)
    psi_colloc_next = tf.clip_by_value(psi_colloc + (r_colloc_next * dt / 0.5), -1.0, 1.0)

    y0_colloc = tf.concat([v_colloc_next, a_colloc_next / 5.0, psi_colloc_next, r_colloc_next], axis=1)
    u_colloc = tf.concat([u_colloc, delta_colloc], axis=1)
   
    return t_data, y0_data, u_data, y_true, t_colloc, y0_colloc, u_colloc

def train_cav_adam(model, data, colloc, params, lambda_reg=1.0, num_epochs=1000, initial_lr=5e-4, batch_size=256):
    """
    Enhanced ADAM training with best model tracking
    
    Args:
        model: CAV PINC network
        data: tuple of (t_data, y0_data, u_data, y_true)
        colloc: tuple of (t_colloc, y0_colloc, u_colloc)
        params: dictionary of physical parameters
        lambda_reg: physics loss weight
        num_epochs: number of training epochs
        initial_lr: initial learning rate
        batch_size: size of the batch for training
        
    Returns:
        loss_history: List of loss values
        best_weights: Best model weights found
        best_loss: Minimum loss achieved
    """
    # Learning rate schedule with decay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )
    optimizer = optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    
    loss_history = []
    best_weights = None
    best_loss = float('inf')
    
    t_data, y0_data, u_data, y_true = data
    t_colloc, y0_colloc, u_colloc = colloc
    
    # Verify input shapes
    assert t_data.shape[1] == 1, f"Expected t_data shape (num_points, 1), got {t_data.shape}"
    assert y0_data.shape[1] == 4, f"Expected y0_data shape (num_points, 4), got {y0_data.shape}"
    assert u_data.shape[1] == 2, f"Expected u_data shape (num_points, 2), got {u_data.shape}"
    assert y_true.shape[1] == 4, f"Expected y_true shape (num_points, 4), got {y_true.shape}"
    
    # Convert to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(t_data, tf.float32),
         tf.cast(y0_data, tf.float32),
         tf.cast(u_data, tf.float32),
         tf.cast(y_true, tf.float32))
    ).shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    colloc_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(t_colloc, tf.float32),
         tf.cast(y0_colloc, tf.float32),
         tf.cast(u_colloc, tf.float32))
    ).batch(1024).prefetch(tf.data.AUTOTUNE)
    
    colloc_batches = [batch for batch in colloc_dataset.take(10)]
    
    @tf.function
    def train_step(data_batch):
        t_batch, y0_batch, u_batch, y_true_batch = data_batch
    
        with tf.GradientTape() as tape:
            # Data loss
            inputs_data = tf.concat([t_batch, y0_batch, u_batch], axis=1)
            # Verify concatenated shape
            assert inputs_data.shape[1] == 7, f"Expected inputs_data shape (batch_size, 7), got {inputs_data.shape}"
            y_pred_data = model(inputs_data)
            mse_data = tf.reduce_mean(tf.square(y_pred_data - y_true_batch))
        
            # Physics loss (batched)
            mse_physics = 0.0
            for t_phys, y0_phys, u_phys in colloc_batches:
                _, batch_loss = cav_pinc_loss(
                    model=model,
                    t_data=tf.zeros((0, 1)),  # Dummy for data loss
                    y0_data=tf.zeros((0, y0_phys.shape[1])),
                    u_data=tf.zeros((0, u_phys.shape[1])),
                    y_true=tf.zeros((0, 4)),  # Dummy
                    t_colloc=t_phys,
                    y0_colloc=y0_phys,
                    u_colloc=u_phys,
                    **params,
                    lambda_reg=1.0
                )
                mse_physics += batch_loss
        
            total_loss = mse_data + lambda_reg * (mse_physics / len(colloc_batches))
    
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
        return total_loss, mse_data

    # Training loop with best model tracking
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        num_batches = 0
        
        for batch in train_dataset:
            batch_loss, batch_mse = train_step(batch)
            if tf.math.is_nan(batch_loss) or tf.math.is_inf(batch_loss):
                print(f"Warning: Invalid loss at epoch {epoch+1}")
                break
            
            epoch_loss += float(batch_loss)
            epoch_mse += float(batch_mse)
            num_batches += 1
            
            # Update best weights if we find a new minimum
            if batch_loss < best_loss:
                best_loss = float(batch_loss)
                best_weights = [w.numpy() for w in model.trainable_variables]
        
        if num_batches > 0:
            epoch_loss /= num_batches
            epoch_mse /= num_batches
            loss_history.append(epoch_loss)
        
        if (epoch + 1) % 25 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}, MSE: {epoch_mse:.6f}, Best: {best_loss:.6f}')
        
        if loss_history and (np.isnan(loss_history[-1]) or np.isinf(loss_history[-1])):
            print("Stopping training due to invalid loss")
            break
    
    return loss_history, best_weights, best_loss
    

if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # CAV physical parameters (bicycle model)
    cav_params = {
        'm': 1500.0,
        'Iz': 2500.0,
        'lf': 1.4,
        'lr': 1.4,
        'Caf': 50000.0,
        'Car': 50000.0,
        'Cd': 0.3,
        'A': 2.2,
        'rho': 1.225,
        'Frr': 300.0
    }
    
    # Training parameters
    T = 0.5
    num_data_points = 20000
    num_colloc_points = 100000
    total_time = 100.0
    
    # Create network
    pinc_net = CAV_PINC_Network()
    
    # Build model with correct dummy input
    dummy_input = np.zeros((1, 7), dtype=np.float32)  # [T, v, dv_dt, psi, r, u, delta]
    _ = pinc_net(dummy_input)

    # Generate training data
    train_data = generate_cav_training_data(
        num_data_points, num_colloc_points, T)
    
    # Training phase
    print("Running ADAM training...")
    adam_loss, best_weights, best_loss = train_cav_adam(
        pinc_net, train_data[:4], train_data[4:], cav_params,
        lambda_reg=1.0, num_epochs=2000, initial_lr=3e-4)
    
    pinc_net.save_weights('pinc_weights.weights.h5')

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.semilogy(hist['time'], hist['loss'])
    ax1.set(xlabel='Time (s)', ylabel='Loss', title='L-BFGS Loss')
    ax2.semilogy(hist['time'], hist['grad_norm'])
    ax2.set(xlabel='Time (s)', ylabel='Gradient Norm', title='Gradient Norm History')
    plt.tight_layout()
    plt.show()
    print(f"Final loss: {final_loss:.2e}")
