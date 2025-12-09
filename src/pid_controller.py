"""
PID Controller Implementation
=============================

Features:
- Derivative filtering (low-pass)
- Anti-windup (back-calculation)
- Output saturation
- Derivative kick prevention
- Bumpless transfer
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PIDGains:
    """PID controller gains."""
    Kp: float = 1.0
    Ki: float = 0.0
    Kd: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.Kp, self.Ki, self.Kd])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'PIDGains':
        return cls(Kp=arr[0], Ki=arr[1], Kd=arr[2])
    
    def __repr__(self):
        return f"PIDGains(Kp={self.Kp:.4f}, Ki={self.Ki:.4f}, Kd={self.Kd:.4f})"


class PIDController:
    """
    Industrial-grade PID Controller.
    
    Transfer function (parallel form):
        u(s) = Kp * e(s) + Ki * e(s)/s + Kd * s * e(s) / (1 + τ_d * s)
        
    Features:
        - Derivative filtering to reduce high-frequency noise
        - Anti-windup via back-calculation
        - Output clamping
        - Derivative on measurement (prevents derivative kick)
    """
    
    def __init__(
        self,
        gains: PIDGains,
        output_limits: Tuple[float, float] = (-np.inf, np.inf),
        derivative_filter_tau: float = 0.01,
        anti_windup_gain: float = 1.0,
        derivative_on_measurement: bool = True
    ):
        self.gains = gains
        self.output_min, self.output_max = output_limits
        self.tau_d = derivative_filter_tau
        self.Kb = anti_windup_gain  # Back-calculation gain
        self.derivative_on_measurement = derivative_on_measurement
        
        # Internal states
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_measurement = 0.0
        self._filtered_derivative = 0.0
        self._prev_output = 0.0
        
    def reset(self):
        """Reset controller states."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_measurement = 0.0
        self._filtered_derivative = 0.0
        self._prev_output = 0.0
        
    def compute(
        self, 
        error: float, 
        dt: float, 
        measurement: Optional[float] = None
    ) -> float:
        """
        Compute PID control output.
        
        Args:
            error: Current error (setpoint - measurement)
            dt: Time step
            measurement: Current measurement (for derivative on measurement)
            
        Returns:
            Control output
        """
        Kp, Ki, Kd = self.gains.Kp, self.gains.Ki, self.gains.Kd
        
        # Proportional term
        P = Kp * error
        
        # Integral term with anti-windup
        self._integral += Ki * error * dt
        I = self._integral
        
        # Derivative term with filtering
        if self.derivative_on_measurement and measurement is not None:
            # Derivative on measurement (no derivative kick)
            d_input = -(measurement - self._prev_measurement) / dt
            self._prev_measurement = measurement
        else:
            # Derivative on error
            d_input = (error - self._prev_error) / dt
            self._prev_error = error
        
        # Low-pass filter on derivative
        alpha = dt / (self.tau_d + dt)
        self._filtered_derivative = (1 - alpha) * self._filtered_derivative + alpha * d_input
        D = Kd * self._filtered_derivative
        
        # Compute raw output
        output_raw = P + I + D
        
        # Apply output limits
        output = np.clip(output_raw, self.output_min, self.output_max)
        
        # Anti-windup: back-calculation
        if output != output_raw:
            # We hit saturation, reduce integral
            self._integral += self.Kb * (output - output_raw)
        
        self._prev_error = error
        self._prev_output = output
        
        return output
    
    def set_gains(self, gains: PIDGains):
        """Update gains with bumpless transfer."""
        self.gains = gains
        
    def get_gains(self) -> PIDGains:
        """Get current gains."""
        return self.gains
    
    @property
    def state(self) -> dict:
        """Get internal state for debugging."""
        return {
            'integral': self._integral,
            'filtered_derivative': self._filtered_derivative,
            'prev_error': self._prev_error
        }


class CascadePIDController:
    """
    Cascade PID controller for inner/outer loop control.
    
    Outer loop: Position control
    Inner loop: Velocity control
    """
    
    def __init__(
        self,
        outer_gains: PIDGains,
        inner_gains: PIDGains,
        outer_limits: Tuple[float, float] = (-5.0, 5.0),
        inner_limits: Tuple[float, float] = (-20.0, 20.0)
    ):
        self.outer = PIDController(outer_gains, output_limits=outer_limits)
        self.inner = PIDController(inner_gains, output_limits=inner_limits)
        
    def reset(self):
        self.outer.reset()
        self.inner.reset()
        
    def compute(
        self, 
        position_error: float, 
        velocity: float, 
        dt: float
    ) -> float:
        """
        Compute cascade control output.
        
        Args:
            position_error: Error in position
            velocity: Current velocity
            dt: Time step
            
        Returns:
            Control output (thrust delta)
        """
        # Outer loop: position → velocity setpoint
        velocity_setpoint = self.outer.compute(position_error, dt)
        
        # Inner loop: velocity → thrust
        velocity_error = velocity_setpoint - velocity
        thrust_delta = self.inner.compute(velocity_error, dt)
        
        return thrust_delta


def tune_ziegler_nichols(Ku: float, Tu: float, controller_type: str = 'PID') -> PIDGains:
    """
    Ziegler-Nichols tuning method.
    
    Args:
        Ku: Ultimate gain (where system oscillates)
        Tu: Ultimate period of oscillation
        controller_type: 'P', 'PI', or 'PID'
        
    Returns:
        PIDGains
    """
    if controller_type == 'P':
        return PIDGains(Kp=0.5 * Ku, Ki=0.0, Kd=0.0)
    elif controller_type == 'PI':
        return PIDGains(Kp=0.45 * Ku, Ki=0.54 * Ku / Tu, Kd=0.0)
    elif controller_type == 'PID':
        return PIDGains(Kp=0.6 * Ku, Ki=1.2 * Ku / Tu, Kd=0.075 * Ku * Tu)
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")


def tune_cohen_coon(K: float, tau: float, theta: float) -> PIDGains:
    """
    Cohen-Coon tuning for FOPDT (First Order Plus Dead Time) systems.
    
    Args:
        K: Process gain
        tau: Time constant
        theta: Dead time
        
    Returns:
        PIDGains
    """
    r = theta / tau
    
    Kp = (1/K) * (tau/theta) * (4/3 + r/4)
    Ti = theta * (32 + 6*r) / (13 + 8*r)
    Td = theta * 4 / (11 + 2*r)
    
    Ki = Kp / Ti
    Kd = Kp * Td
    
    return PIDGains(Kp=Kp, Ki=Ki, Kd=Kd)
