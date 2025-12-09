"""
Quadcopter Dynamics Model
=========================

6-DOF Quadcopter state-space representation with:
- Non-linear equations of motion
- Drag coefficients and rotor inertia
- Actuator saturation (PWM clipping)
- Sensor noise (Ornstein-Uhlenbeck process)

State Vector: x = [z, z_dot, phi, phi_dot, theta, theta_dot, psi, psi_dot]
              (altitude, roll, pitch, yaw and their derivatives)

For PID tuning, we focus on altitude control as the primary demonstration.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class QuadcopterParams:
    """Physical parameters of the quadcopter."""
    mass: float = 1.5  # kg
    g: float = 9.81  # m/s^2
    
    # Inertia matrix (diagonal approximation)
    Ixx: float = 0.0142  # kg*m^2
    Iyy: float = 0.0142  # kg*m^2
    Izz: float = 0.0284  # kg*m^2
    
    # Rotor parameters
    k_thrust: float = 2.98e-6  # thrust coefficient
    k_torque: float = 1.14e-7  # torque coefficient
    arm_length: float = 0.225  # m
    rotor_inertia: float = 3.357e-5  # kg*m^2
    
    # Drag coefficients
    drag_coef_xy: float = 0.25  # N/(m/s)^2
    drag_coef_z: float = 0.50   # N/(m/s)^2
    
    # Actuator limits
    pwm_min: float = 0.0
    pwm_max: float = 1.0
    max_thrust: float = 30.0  # N (total)
    min_thrust: float = 0.0   # N
    
    # Motor dynamics (first-order lag)
    motor_time_constant: float = 0.02  # s


@dataclass
class SensorNoiseParams:
    """Ornstein-Uhlenbeck process parameters for sensor noise."""
    theta: float = 0.15  # mean reversion rate
    mu: float = 0.0      # long-term mean
    sigma: float = 0.05  # volatility
    

class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck process for realistic sensor drift modeling.
    
    dx = θ(μ - x)dt + σdW
    """
    
    def __init__(self, params: SensorNoiseParams, dt: float, seed: Optional[int] = None):
        self.theta = params.theta
        self.mu = params.mu
        self.sigma = params.sigma
        self.dt = dt
        self.x = 0.0
        self.rng = np.random.default_rng(seed)
        
    def sample(self) -> float:
        """Generate next noise sample."""
        dx = self.theta * (self.mu - self.x) * self.dt + \
             self.sigma * np.sqrt(self.dt) * self.rng.normal()
        self.x += dx
        return self.x
    
    def reset(self):
        """Reset noise state."""
        self.x = 0.0


class QuadcopterDynamics:
    """
    Non-linear 6-DOF Quadcopter Dynamics Model.
    
    Simplified to altitude control for PID tuning demonstration,
    but includes full state for extensibility.
    
    State equations for altitude (z-axis):
        ż = v_z
        v̇_z = (T/m)cos(φ)cos(θ) - g - (D_z/m)|v_z|v_z
        
    Where:
        T = total thrust
        D_z = vertical drag coefficient
        φ, θ = roll and pitch angles
    """
    
    def __init__(
        self, 
        params: Optional[QuadcopterParams] = None,
        dt: float = 0.001,
        enable_noise: bool = True,
        noise_params: Optional[SensorNoiseParams] = None,
        seed: Optional[int] = None
    ):
        self.params = params or QuadcopterParams()
        self.dt = dt
        self.enable_noise = enable_noise
        
        # Initialize state [z, vz, phi, vphi, theta, vtheta, psi, vpsi]
        self.state = np.zeros(8)
        self.motor_state = 0.0  # Current motor thrust (with dynamics)
        
        # Sensor noise
        if enable_noise:
            noise_params = noise_params or SensorNoiseParams()
            self.altitude_noise = OrnsteinUhlenbeckNoise(noise_params, dt, seed)
            self.velocity_noise = OrnsteinUhlenbeckNoise(
                SensorNoiseParams(theta=0.2, mu=0.0, sigma=0.02), dt, 
                seed + 1 if seed else None
            )
        
        # Time tracking
        self.time = 0.0
        
    def reset(self, initial_altitude: float = 0.0, initial_velocity: float = 0.0):
        """Reset quadcopter to initial conditions."""
        self.state = np.zeros(8)
        self.state[0] = initial_altitude  # z
        self.state[1] = initial_velocity  # vz
        self.motor_state = self.params.mass * self.params.g  # Hover thrust
        self.time = 0.0
        
        if self.enable_noise:
            self.altitude_noise.reset()
            self.velocity_noise.reset()
    
    def saturate_thrust(self, thrust: float) -> float:
        """Apply actuator saturation limits."""
        return np.clip(thrust, self.params.min_thrust, self.params.max_thrust)
    
    def motor_dynamics(self, thrust_command: float) -> float:
        """
        First-order motor dynamics (lag).
        
        τ * dT_actual/dt = T_command - T_actual
        """
        tau = self.params.motor_time_constant
        alpha = self.dt / (tau + self.dt)
        self.motor_state = (1 - alpha) * self.motor_state + alpha * thrust_command
        return self.motor_state
    
    def compute_drag(self, velocity: float) -> float:
        """Compute aerodynamic drag force."""
        return self.params.drag_coef_z * np.abs(velocity) * velocity
    
    def step(self, thrust_command: float) -> Tuple[float, float]:
        """
        Advance simulation by one time step.
        
        Args:
            thrust_command: Desired total thrust (N)
            
        Returns:
            (measured_altitude, measured_velocity) - with sensor noise if enabled
        """
        p = self.params
        
        # Apply actuator saturation
        thrust_saturated = self.saturate_thrust(thrust_command)
        
        # Motor dynamics
        actual_thrust = self.motor_dynamics(thrust_saturated)
        
        # Extract current state
        z, vz = self.state[0], self.state[1]
        phi, theta = self.state[2], self.state[4]  # roll, pitch
        
        # Compute acceleration (Newton's second law)
        # Thrust acts along body z-axis, transformed to world frame
        thrust_world_z = actual_thrust * np.cos(phi) * np.cos(theta)
        
        # Drag force
        drag = self.compute_drag(vz)
        
        # Net acceleration
        az = (thrust_world_z / p.mass) - p.g - (drag / p.mass)
        
        # Euler integration (RK4 could be used for higher accuracy)
        self.state[1] += az * self.dt  # vz
        self.state[0] += self.state[1] * self.dt  # z
        
        # Ground constraint (can't go below ground)
        if self.state[0] < 0:
            self.state[0] = 0
            self.state[1] = max(0, self.state[1])
        
        # Update time
        self.time += self.dt
        
        # Add sensor noise if enabled
        if self.enable_noise:
            measured_z = self.state[0] + self.altitude_noise.sample()
            measured_vz = self.state[1] + self.velocity_noise.sample()
        else:
            measured_z = self.state[0]
            measured_vz = self.state[1]
            
        return measured_z, measured_vz
    
    def get_true_state(self) -> Tuple[float, float]:
        """Get true (noise-free) altitude and velocity."""
        return self.state[0], self.state[1]
    
    @property
    def hover_thrust(self) -> float:
        """Thrust required for hover."""
        return self.params.mass * self.params.g
    

def simulate_step_response(
    quad: QuadcopterDynamics,
    controller,  # PID controller
    setpoint: float,
    duration: float,
    disturbance_time: Optional[float] = None,
    disturbance_magnitude: float = 0.0
) -> dict:
    """
    Simulate closed-loop step response.
    
    Returns:
        Dictionary with time, position, velocity, control, error histories
    """
    n_steps = int(duration / quad.dt)
    
    # Preallocate arrays
    time_history = np.zeros(n_steps)
    position_history = np.zeros(n_steps)
    velocity_history = np.zeros(n_steps)
    control_history = np.zeros(n_steps)
    error_history = np.zeros(n_steps)
    setpoint_history = np.zeros(n_steps)
    true_position_history = np.zeros(n_steps)
    
    # Reset systems
    quad.reset(initial_altitude=0.0)
    controller.reset()
    
    for i in range(n_steps):
        t = i * quad.dt
        
        # Get measurement
        measured_z, measured_vz = quad.get_true_state()
        if quad.enable_noise:
            measured_z += quad.altitude_noise.sample() if hasattr(quad, 'altitude_noise') else 0
        
        # Compute error
        error = setpoint - measured_z
        
        # Compute control action (thrust delta from hover)
        control_delta = controller.compute(error, quad.dt)
        thrust = quad.hover_thrust + control_delta
        
        # Apply disturbance if specified
        if disturbance_time and t >= disturbance_time:
            thrust += disturbance_magnitude
        
        # Step simulation
        quad.step(thrust)
        true_z, true_vz = quad.get_true_state()
        
        # Record history
        time_history[i] = t
        position_history[i] = measured_z
        true_position_history[i] = true_z
        velocity_history[i] = measured_vz if 'measured_vz' in dir() else true_vz
        control_history[i] = thrust
        error_history[i] = error
        setpoint_history[i] = setpoint
    
    return {
        'time': time_history,
        'position': position_history,
        'true_position': true_position_history,
        'velocity': velocity_history,
        'control': control_history,
        'error': error_history,
        'setpoint': setpoint_history,
        'dt': quad.dt
    }
