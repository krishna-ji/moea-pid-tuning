"""
Multi-Objective Neuro-Evolutionary PID Tuning
=============================================

Main simulation pipeline for Tri-Objective Pareto-Optimal PID Controller
optimization for high-dynamic systems (quadcopter altitude control).

This script:
1. Sets up the quadcopter simulation environment
2. Runs NSGA-II optimization to find the Pareto front
3. Analyzes the trade-offs between objectives
4. Performs Monte Carlo robustness testing
5. Generates all publication-quality figures
"""

import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import asdict
import json

# Local imports
from src.quadcopter import QuadcopterDynamics, QuadcopterParams, simulate_step_response
from src.pid_controller import PIDController, PIDGains
from src.nsga2 import NSGA2Optimizer, OptimizationBounds, Individual
from src.objectives import (
    compute_itae, compute_ics, compute_tv,
    evaluate_controller, get_performance_summary
)
from src.visualization import ParetoVisualizer


def create_objective_function(
    quad_params: QuadcopterParams,
    dt: float = 0.001,
    simulation_duration: float = 5.0,
    setpoint: float = 5.0,
    control_limits: Tuple[float, float] = (-15.0, 15.0)
):
    """
    Create the tri-objective fitness function for NSGA-II.
    
    Returns a function that takes PID gains and returns (ITAE, ICS, TV).
    """
    def objective_function(genes: np.ndarray) -> np.ndarray:
        """Evaluate PID gains and return objective values."""
        Kp, Ki, Kd = genes
        
        # Create fresh instances for each evaluation
        quad = QuadcopterDynamics(
            params=quad_params,
            dt=dt,
            enable_noise=False,  # Deterministic evaluation
            seed=42
        )
        
        controller = PIDController(
            gains=PIDGains(Kp=Kp, Ki=Ki, Kd=Kd),
            output_limits=control_limits,
            derivative_filter_tau=0.01,
            anti_windup_gain=1.0
        )
        
        # Reset systems
        quad.reset(initial_altitude=0.0)
        controller.reset()
        
        n_steps = int(simulation_duration / dt)
        
        # Preallocate
        time_arr = np.zeros(n_steps)
        error_arr = np.zeros(n_steps)
        control_arr = np.zeros(n_steps)
        
        # Simulate
        for i in range(n_steps):
            t = i * dt
            z, vz = quad.get_true_state()
            
            error = setpoint - z
            control_delta = controller.compute(error, dt)
            thrust = quad.hover_thrust + control_delta
            
            quad.step(thrust)
            
            time_arr[i] = t
            error_arr[i] = error
            control_arr[i] = thrust
        
        # Compute objectives
        itae = compute_itae(time_arr, error_arr)
        ics = compute_ics(control_arr, dt)
        tv = compute_tv(control_arr)
        
        # Check for instability (penalize heavily)
        if np.any(np.isnan([itae, ics, tv])) or np.any(np.isinf([itae, ics, tv])):
            return np.array([1e6, 1e6, 1e6])
        
        return np.array([itae, ics, tv])
    
    return objective_function


def simulate_with_gains(
    gains: PIDGains,
    quad_params: QuadcopterParams,
    dt: float = 0.001,
    duration: float = 5.0,
    setpoint: float = 5.0,
    enable_noise: bool = False,
    control_limits: Tuple[float, float] = (-15.0, 15.0)
) -> dict:
    """Run a simulation with specific PID gains."""
    quad = QuadcopterDynamics(
        params=quad_params,
        dt=dt,
        enable_noise=enable_noise,
        seed=42
    )
    
    controller = PIDController(
        gains=gains,
        output_limits=control_limits
    )
    
    return simulate_step_response(quad, controller, setpoint, duration)


def run_monte_carlo_robustness(
    gains_dict: Dict[str, PIDGains],
    quad_params: QuadcopterParams,
    n_samples: int = 50,
    setpoint: float = 5.0,
    seed: int = 42
) -> Dict[str, List[dict]]:
    """
    Run Monte Carlo robustness testing with varying parameters.
    
    Varies:
    - Mass: ±20%
    - Drag coefficient: ±30%
    - Sensor noise level
    """
    rng = np.random.default_rng(seed)
    results = {name: [] for name in gains_dict.keys()}
    
    for i in range(n_samples):
        # Perturb parameters
        perturbed_params = QuadcopterParams(
            mass=quad_params.mass * rng.uniform(0.8, 1.2),
            drag_coef_z=quad_params.drag_coef_z * rng.uniform(0.7, 1.3),
        )
        
        for name, gains in gains_dict.items():
            sim_result = simulate_with_gains(
                gains=gains,
                quad_params=perturbed_params,
                enable_noise=True,
                setpoint=setpoint
            )
            
            perf = get_performance_summary(sim_result, setpoint)
            results[name].append(perf)
    
    return results


def main():
    """Main optimization and analysis pipeline."""
    print("=" * 70)
    print("  Multi-Objective Neuro-Evolutionary PID Tuning")
    print("  for High-Dynamic Systems (Quadcopter Altitude Control)")
    print("=" * 70)
    print()
    
    # Configuration
    SETPOINT = 5.0  # meters
    SIMULATION_DURATION = 5.0  # seconds
    DT = 0.001  # simulation time step
    CONTROL_LIMITS = (-15.0, 15.0)  # thrust delta limits
    
    # NSGA-II parameters
    POPULATION_SIZE = 100
    N_GENERATIONS = 50
    
    # Output directory
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize quadcopter parameters
    quad_params = QuadcopterParams()
    print(f"Quadcopter mass: {quad_params.mass} kg")
    print(f"Hover thrust: {quad_params.mass * quad_params.g:.2f} N")
    print(f"Setpoint altitude: {SETPOINT} m")
    print()
    
    # Create objective function
    objective_fn = create_objective_function(
        quad_params=quad_params,
        dt=DT,
        simulation_duration=SIMULATION_DURATION,
        setpoint=SETPOINT,
        control_limits=CONTROL_LIMITS
    )
    
    # Define search bounds
    bounds = OptimizationBounds(
        Kp_min=1.0, Kp_max=40.0,
        Ki_min=0.0, Ki_max=15.0,
        Kd_min=0.0, Kd_max=8.0
    )
    
    # Create optimizer
    optimizer = NSGA2Optimizer(
        objective_function=objective_fn,
        bounds=bounds,
        population_size=POPULATION_SIZE,
        n_generations=N_GENERATIONS,
        crossover_prob=0.9,
        mutation_prob=0.15,
        seed=42,
        verbose=True
    )
    
    # Run optimization
    print("\nStarting NSGA-II Optimization...")
    start_time = time.time()
    
    pareto_front, history = optimizer.run()
    
    elapsed = time.time() - start_time
    print(f"\nOptimization completed in {elapsed:.1f} seconds")
    print(f"Pareto front size: {len(pareto_front)} solutions")
    
    # Get special solutions
    knee_point = optimizer.get_knee_point(pareto_front)
    aggressive, smooth, efficient = optimizer.get_extreme_solutions(pareto_front)
    
    print("\n" + "=" * 70)
    print("  OPTIMIZATION RESULTS")
    print("=" * 70)
    
    solutions = {
        'Balanced (Knee)': knee_point,
        'Aggressive (min ITAE)': aggressive,
        'Smooth (min TV)': smooth,
        'Efficient (min ICS)': efficient
    }
    
    for name, sol in solutions.items():
        print(f"\n{name}:")
        print(f"  Gains: Kp={sol.genes[0]:.4f}, Ki={sol.genes[1]:.4f}, Kd={sol.genes[2]:.4f}")
        print(f"  ITAE={sol.objectives[0]:.4f}, ICS={sol.objectives[1]:.2f}, TV={sol.objectives[2]:.2f}")
    
    # Run time-domain simulations for each solution
    print("\n" + "-" * 70)
    print("Running time-domain simulations...")
    
    gains_dict = {
        'aggressive': PIDGains.from_array(aggressive.genes),
        'smooth': PIDGains.from_array(smooth.genes),
        'balanced': PIDGains.from_array(knee_point.genes),
        'efficient': PIDGains.from_array(efficient.genes)
    }
    
    time_domain_results = {}
    for name, gains in gains_dict.items():
        time_domain_results[name] = simulate_with_gains(
            gains=gains,
            quad_params=quad_params,
            dt=DT,
            duration=SIMULATION_DURATION,
            setpoint=SETPOINT,
            enable_noise=False
        )
        
        # Print performance metrics
        perf = get_performance_summary(time_domain_results[name], SETPOINT)
        print(f"\n{name.title()} Tune Performance:")
        print(f"  Settling time: {perf['settling_time']:.3f} s")
        print(f"  Overshoot: {perf['overshoot']:.2f}%")
        print(f"  Rise time: {perf['rise_time']:.3f} s")
        print(f"  Steady-state error: {perf['steady_state_error']:.4f} m")
    
    # Monte Carlo Robustness Testing
    print("\n" + "-" * 70)
    print("Running Monte Carlo robustness testing (50 samples)...")
    
    mc_results = run_monte_carlo_robustness(
        gains_dict=gains_dict,
        quad_params=quad_params,
        n_samples=50,
        setpoint=SETPOINT
    )
    
    print("Robustness testing complete!")
    
    # Generate all visualizations
    print("\n" + "-" * 70)
    print("Generating publication-quality figures...")
    
    viz = ParetoVisualizer(output_dir=str(output_dir))
    
    # 1. 3D Pareto front
    print("  - 3D Pareto front...")
    viz.plot_3d_pareto_front(
        pareto_front=pareto_front,
        knee_point=knee_point,
        extreme_points=(aggressive, smooth, efficient),
        save_name="pareto_front_3d.png"
    )
    
    # 2. Pareto projections
    print("  - 2D Pareto projections...")
    viz.plot_pareto_projections(
        pareto_front=pareto_front,
        knee_point=knee_point,
        save_name="pareto_projections.png"
    )
    
    # 3. Time domain comparison
    print("  - Time domain comparison...")
    viz.plot_time_domain_comparison(
        results=time_domain_results,
        setpoint=SETPOINT,
        save_name="time_domain_comparison.png"
    )
    
    # 4. Convergence plot
    print("  - Convergence plot...")
    viz.plot_convergence(
        history=history,
        save_name="convergence.png"
    )
    
    # 5. Monte Carlo robustness
    print("  - Monte Carlo robustness...")
    for metric in ['settling_time', 'overshoot']:
        viz.plot_monte_carlo_robustness(
            monte_carlo_results=mc_results,
            metric=metric,
            save_name=f"monte_carlo_{metric}.png"
        )
    
    # 6. Differential equations
    print("  - Differential equations diagram...")
    viz.plot_differential_equations(save_name="differential_equations.png")
    
    # 7. System block diagram
    print("  - System block diagram...")
    viz.plot_system_block_diagram(save_name="system_block_diagram.png")
    
    # 8. PID gains distribution
    print("  - PID gains distribution...")
    viz.plot_pid_gains_distribution(
        pareto_front=pareto_front,
        knee_point=knee_point,
        save_name="pid_gains_distribution.png"
    )
    
    # 9. Summary figure
    print("  - Comprehensive summary...")
    viz.create_summary_figure(
        pareto_front=pareto_front,
        time_domain_results=time_domain_results,
        history=history,
        knee_point=knee_point,
        extreme_points=(aggressive, smooth, efficient),
        setpoint=SETPOINT,
        save_name="optimization_summary.png"
    )
    
    # Save results to JSON
    results_data = {
        'configuration': {
            'setpoint': SETPOINT,
            'simulation_duration': SIMULATION_DURATION,
            'dt': DT,
            'population_size': POPULATION_SIZE,
            'generations': N_GENERATIONS
        },
        'solutions': {
            name: {
                'gains': {'Kp': float(sol.genes[0]), 'Ki': float(sol.genes[1]), 'Kd': float(sol.genes[2])},
                'objectives': {'ITAE': float(sol.objectives[0]), 'ICS': float(sol.objectives[1]), 'TV': float(sol.objectives[2])}
            }
            for name, sol in [('balanced', knee_point), ('aggressive', aggressive), 
                             ('smooth', smooth), ('efficient', efficient)]
        },
        'pareto_front_size': len(pareto_front),
        'optimization_time_seconds': elapsed
    }
    
    with open(output_dir / 'optimization_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print("\n" + "=" * 70)
    print("  ALL FIGURES SAVED TO: ./figures/")
    print("=" * 70)
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")
    print(f"  - optimization_results.json")
    
    print("\n✓ Pipeline complete!")
    
    return pareto_front, history, time_domain_results, mc_results


if __name__ == "__main__":
    main()
