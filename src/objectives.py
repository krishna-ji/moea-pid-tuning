"""
Tri-Objective Fitness Functions
================================

Objective 1: ITAE (Integral Time-weighted Absolute Error) - Tracking Accuracy
    J₁ = ∫₀ᵀ t|e(t)|dt
    
Objective 2: ICS (Integral Control Squared) - Energy Efficiency  
    J₂ = ∫₀ᵀ u(t)²dt
    
Objective 3: TV (Total Variation) - Control Smoothness
    J₃ = Σ|uₖ - uₖ₋₁|
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ObjectiveWeights:
    """Weights for scalarizing multi-objective into single objective."""
    w_itae: float = 1.0
    w_ics: float = 1.0
    w_tv: float = 1.0


def compute_itae(time: np.ndarray, error: np.ndarray) -> float:
    """
    Compute Integral of Time-weighted Absolute Error (ITAE).
    
    ITAE = ∫₀ᵀ t|e(t)|dt
    
    This metric penalizes errors that persist over time more heavily,
    encouraging fast settling with minimal steady-state error.
    
    Args:
        time: Time array
        error: Error array (same length as time)
        
    Returns:
        ITAE value (lower is better)
    """
    if len(time) < 2:
        return 0.0
    
    dt = np.mean(np.diff(time))
    integrand = time * np.abs(error)
    
    # Trapezoidal integration
    itae = np.trapezoid(integrand, time)
    
    return itae


def compute_ics(control: np.ndarray, dt: float) -> float:
    """
    Compute Integral of Control Squared (ICS).
    
    ICS = ∫₀ᵀ u(t)²dt
    
    This metric measures total control energy expenditure.
    Lower values mean better energy efficiency and less actuator stress.
    
    Args:
        control: Control signal array
        dt: Time step
        
    Returns:
        ICS value (lower is better)
    """
    if len(control) < 2:
        return 0.0
    
    # Trapezoidal integration
    time = np.arange(len(control)) * dt
    ics = np.trapezoid(control**2, time)
    
    return ics


def compute_tv(control: np.ndarray) -> float:
    """
    Compute Total Variation (TV) of control signal.
    
    TV = Σ|uₖ - uₖ₋₁|
    
    This metric measures the smoothness of the control signal.
    Lower values indicate smoother control without high-frequency jitter.
    
    Args:
        control: Control signal array
        
    Returns:
        TV value (lower is better)
    """
    if len(control) < 2:
        return 0.0
    
    tv = np.sum(np.abs(np.diff(control)))
    
    return tv


def compute_iae(error: np.ndarray, dt: float) -> float:
    """
    Compute Integral of Absolute Error (IAE).
    
    IAE = ∫₀ᵀ |e(t)|dt
    """
    time = np.arange(len(error)) * dt
    return np.trapezoid(np.abs(error), time)


def compute_ise(error: np.ndarray, dt: float) -> float:
    """
    Compute Integral of Squared Error (ISE).
    
    ISE = ∫₀ᵀ e(t)²dt
    """
    time = np.arange(len(error)) * dt
    return np.trapezoid(error**2, time)


def compute_settling_time(
    time: np.ndarray, 
    position: np.ndarray, 
    setpoint: float,
    tolerance: float = 0.02
) -> float:
    """
    Compute settling time (2% criterion by default).
    
    Args:
        time: Time array
        position: Position response array
        setpoint: Target setpoint
        tolerance: Settling tolerance (default 2%)
        
    Returns:
        Settling time in seconds
    """
    if len(time) < 2:
        return np.inf
    
    error_band = tolerance * abs(setpoint)
    within_band = np.abs(position - setpoint) <= error_band
    
    # Find last time we exit the band
    if within_band[-1]:
        # Find first index where we stay within band forever
        for i in range(len(within_band) - 1, -1, -1):
            if not within_band[i]:
                return time[i + 1] if i + 1 < len(time) else time[-1]
        return 0.0  # Always within band
    else:
        return np.inf  # Never settles


def compute_overshoot(position: np.ndarray, setpoint: float) -> float:
    """
    Compute percentage overshoot.
    
    Args:
        position: Position response array
        setpoint: Target setpoint
        
    Returns:
        Overshoot percentage (0-100+)
    """
    if setpoint == 0:
        return 0.0
    
    max_pos = np.max(position)
    if max_pos > setpoint:
        return 100.0 * (max_pos - setpoint) / setpoint
    return 0.0


def compute_rise_time(
    time: np.ndarray, 
    position: np.ndarray, 
    setpoint: float,
    low_pct: float = 0.1,
    high_pct: float = 0.9
) -> float:
    """
    Compute rise time (10% to 90% by default).
    
    Args:
        time: Time array
        position: Position response array
        setpoint: Target setpoint
        low_pct, high_pct: Rise time bounds
        
    Returns:
        Rise time in seconds
    """
    if len(time) < 2 or setpoint == 0:
        return np.inf
    
    low_val = low_pct * setpoint
    high_val = high_pct * setpoint
    
    # Find crossings
    low_idx = np.argmax(position >= low_val)
    high_idx = np.argmax(position >= high_val)
    
    if low_idx == 0 and position[0] < low_val:
        return np.inf
    if high_idx == 0 and position[0] < high_val:
        return np.inf
        
    return time[high_idx] - time[low_idx]


def evaluate_controller(
    sim_result: dict,
    setpoint: float,
    normalize: bool = True,
    normalization_factors: Optional[Tuple[float, float, float]] = None
) -> Tuple[float, float, float]:
    """
    Evaluate PID controller performance using tri-objective metrics.
    
    Args:
        sim_result: Dictionary from simulate_step_response
        setpoint: Target setpoint for normalization
        normalize: Whether to normalize objectives
        normalization_factors: (itae_ref, ics_ref, tv_ref) for normalization
        
    Returns:
        (ITAE, ICS, TV) tuple - all should be minimized
    """
    time = sim_result['time']
    error = sim_result['error']
    control = sim_result['control']
    dt = sim_result['dt']
    
    # Compute raw objectives
    itae = compute_itae(time, error)
    ics = compute_ics(control, dt)
    tv = compute_tv(control)
    
    # Normalize if requested
    if normalize and normalization_factors:
        itae_ref, ics_ref, tv_ref = normalization_factors
        itae /= itae_ref if itae_ref > 0 else 1.0
        ics /= ics_ref if ics_ref > 0 else 1.0
        tv /= tv_ref if tv_ref > 0 else 1.0
    
    return itae, ics, tv


def scalarize(
    objectives: Tuple[float, float, float],
    weights: ObjectiveWeights = ObjectiveWeights()
) -> float:
    """
    Scalarize multi-objective to single objective using weighted sum.
    
    Args:
        objectives: (ITAE, ICS, TV) tuple
        weights: Objective weights
        
    Returns:
        Scalar fitness value
    """
    itae, ics, tv = objectives
    return weights.w_itae * itae + weights.w_ics * ics + weights.w_tv * tv


def get_performance_summary(sim_result: dict, setpoint: float) -> dict:
    """
    Get comprehensive performance summary.
    
    Args:
        sim_result: Simulation result dictionary
        setpoint: Target setpoint
        
    Returns:
        Dictionary of performance metrics
    """
    time = sim_result['time']
    position = sim_result['true_position']
    error = sim_result['error']
    control = sim_result['control']
    dt = sim_result['dt']
    
    return {
        'itae': compute_itae(time, error),
        'ics': compute_ics(control, dt),
        'tv': compute_tv(control),
        'iae': compute_iae(error, dt),
        'ise': compute_ise(error, dt),
        'settling_time': compute_settling_time(time, position, setpoint),
        'overshoot': compute_overshoot(position, setpoint),
        'rise_time': compute_rise_time(time, position, setpoint),
        'steady_state_error': abs(position[-1] - setpoint)
    }
