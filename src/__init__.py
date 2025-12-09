# Multi-Objective Neuro-Evolutionary PID Tuning for High-Dynamic Systems
# =======================================================================
#
# This package implements a Tri-Objective Pareto-Optimal PID Controller
# for high-dynamic systems (e.g., quadcopter altitude control).
#
# Objectives:
#   J1: ITAE (Integral Time-weighted Absolute Error) - Tracking Accuracy
#   J2: ICS (Integral Control Squared) - Energy Efficiency
#   J3: TV (Total Variation) - Control Smoothness

from .quadcopter import QuadcopterDynamics
from .pid_controller import PIDController
from .nsga2 import NSGA2Optimizer
from .objectives import compute_itae, compute_ics, compute_tv
from .visualization import ParetoVisualizer

__all__ = [
    'QuadcopterDynamics',
    'PIDController', 
    'NSGA2Optimizer',
    'compute_itae',
    'compute_ics',
    'compute_tv',
    'ParetoVisualizer'
]
