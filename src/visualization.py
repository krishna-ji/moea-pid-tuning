"""
Visualization Module
====================

Professional visualizations for multi-objective PID optimization:
- 3D Pareto front surface
- Time-domain response comparisons
- Convergence plots
- Monte Carlo robustness analysis
- Differential equations animation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple, Optional
from pathlib import Path


# Set professional style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


class ParetoVisualizer:
    """Visualization tools for multi-objective optimization results."""
    
    def __init__(self, output_dir: str = "figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Color scheme
        self.colors = {
            'aggressive': '#e74c3c',  # Red
            'smooth': '#3498db',      # Blue
            'balanced': '#2ecc71',    # Green
            'efficient': '#f39c12',   # Orange
            'pareto': '#9b59b6',      # Purple
            'reference': '#95a5a6',   # Gray
        }
    
    def plot_3d_pareto_front(
        self,
        pareto_front: List,
        knee_point: Optional[object] = None,
        extreme_points: Optional[Tuple] = None,
        title: str = "3D Pareto Front - Tri-Objective Optimization",
        save_name: str = "pareto_front_3d.png",
        elevation: float = 25,
        azimuth: float = 45
    ) -> plt.Figure:
        """
        Create stunning 3D Pareto front visualization.
        
        Args:
            pareto_front: List of Individual objects
            knee_point: The knee point solution
            extreme_points: (aggressive, smooth, efficient) solutions
            title: Plot title
            save_name: Output filename
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract objectives
        objectives = np.array([ind.objectives for ind in pareto_front])
        itae = objectives[:, 0]
        ics = objectives[:, 1]
        tv = objectives[:, 2]
        
        # Normalize for coloring
        itae_norm = (itae - itae.min()) / (itae.max() - itae.min() + 1e-10)
        
        # Plot Pareto surface points
        scatter = ax.scatter(
            itae, ics, tv,
            c=itae_norm,
            cmap='viridis',
            s=60,
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('ITAE (normalized)', fontsize=11)
        
        # Plot extreme points
        if extreme_points:
            aggressive, smooth, efficient = extreme_points
            
            ax.scatter(*aggressive.objectives, c=self.colors['aggressive'],
                      s=200, marker='^', edgecolors='black', linewidth=2,
                      label='Aggressive (min ITAE)', zorder=5)
            
            ax.scatter(*smooth.objectives, c=self.colors['smooth'],
                      s=200, marker='s', edgecolors='black', linewidth=2,
                      label='Smooth (min TV)', zorder=5)
            
            ax.scatter(*efficient.objectives, c=self.colors['efficient'],
                      s=200, marker='D', edgecolors='black', linewidth=2,
                      label='Efficient (min ICS)', zorder=5)
        
        # Plot knee point
        if knee_point:
            ax.scatter(*knee_point.objectives, c=self.colors['balanced'],
                      s=300, marker='*', edgecolors='black', linewidth=2,
                      label='Balanced (Knee Point)', zorder=6)
        
        # Axis labels with LaTeX
        ax.set_xlabel(r'$J_1$: ITAE (Tracking)', fontsize=12, labelpad=10)
        ax.set_ylabel(r'$J_2$: ICS (Energy)', fontsize=12, labelpad=10)
        ax.set_zlabel(r'$J_3$: TV (Smoothness)', fontsize=12, labelpad=10)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Set viewing angle
        ax.view_init(elev=elevation, azim=azimuth)
        
        # Legend
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
        
        # Save figure
        fig.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        return fig
    
    def plot_pareto_projections(
        self,
        pareto_front: List,
        knee_point: Optional[object] = None,
        save_name: str = "pareto_projections.png"
    ) -> plt.Figure:
        """Create 2D projections of the 3D Pareto front."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        objectives = np.array([ind.objectives for ind in pareto_front])
        itae, ics, tv = objectives[:, 0], objectives[:, 1], objectives[:, 2]
        
        # Common styling
        scatter_style = dict(c=self.colors['pareto'], s=40, alpha=0.6, edgecolors='white')
        knee_style = dict(c=self.colors['balanced'], s=200, marker='*', 
                         edgecolors='black', linewidth=2, zorder=5)
        
        # ITAE vs ICS
        axes[0].scatter(itae, ics, **scatter_style)
        if knee_point:
            axes[0].scatter(knee_point.objectives[0], knee_point.objectives[1], 
                          **knee_style, label='Knee Point')
        axes[0].set_xlabel(r'$J_1$: ITAE (Tracking Accuracy)')
        axes[0].set_ylabel(r'$J_2$: ICS (Energy)')
        axes[0].set_title('ITAE vs ICS Trade-off')
        axes[0].legend()
        
        # ITAE vs TV
        axes[1].scatter(itae, tv, **scatter_style)
        if knee_point:
            axes[1].scatter(knee_point.objectives[0], knee_point.objectives[2], 
                          **knee_style, label='Knee Point')
        axes[1].set_xlabel(r'$J_1$: ITAE (Tracking Accuracy)')
        axes[1].set_ylabel(r'$J_3$: TV (Smoothness)')
        axes[1].set_title('ITAE vs TV Trade-off')
        axes[1].legend()
        
        # ICS vs TV
        axes[2].scatter(ics, tv, **scatter_style)
        if knee_point:
            axes[2].scatter(knee_point.objectives[1], knee_point.objectives[2], 
                          **knee_style, label='Knee Point')
        axes[2].set_xlabel(r'$J_2$: ICS (Energy)')
        axes[2].set_ylabel(r'$J_3$: TV (Smoothness)')
        axes[2].set_title('ICS vs TV Trade-off')
        axes[2].legend()
        
        plt.tight_layout()
        fig.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_time_domain_comparison(
        self,
        results: Dict[str, dict],
        setpoint: float,
        save_name: str = "time_domain_comparison.png"
    ) -> plt.Figure:
        """
        Compare time-domain responses of different PID tunings.
        
        Args:
            results: Dict with keys 'aggressive', 'smooth', 'balanced', 'efficient'
                    Each value is a simulation result dictionary
            setpoint: Target setpoint value
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Colors for each tuning
        colors = {
            'aggressive': self.colors['aggressive'],
            'smooth': self.colors['smooth'],
            'balanced': self.colors['balanced'],
            'efficient': self.colors['efficient']
        }
        
        labels = {
            'aggressive': 'Aggressive (min ITAE)',
            'smooth': 'Smooth (min TV)',
            'balanced': 'Balanced (Knee)',
            'efficient': 'Efficient (min ICS)'
        }
        
        # Position response
        ax = axes[0, 0]
        for name, result in results.items():
            ax.plot(result['time'], result['true_position'], 
                   color=colors.get(name, 'gray'), 
                   label=labels.get(name, name),
                   linewidth=2)
        ax.axhline(y=setpoint, color='black', linestyle='--', 
                  linewidth=1.5, label='Setpoint')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Altitude (m)')
        ax.set_title('Position Response')
        ax.legend(loc='lower right')
        ax.set_xlim(left=0)
        
        # Error response
        ax = axes[0, 1]
        for name, result in results.items():
            ax.plot(result['time'], result['error'],
                   color=colors.get(name, 'gray'),
                   label=labels.get(name, name),
                   linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (m)')
        ax.set_title('Error Response')
        ax.legend(loc='upper right')
        ax.set_xlim(left=0)
        
        # Control signal
        ax = axes[1, 0]
        for name, result in results.items():
            ax.plot(result['time'], result['control'],
                   color=colors.get(name, 'gray'),
                   label=labels.get(name, name),
                   linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Thrust (N)')
        ax.set_title('Control Signal')
        ax.legend(loc='upper right')
        ax.set_xlim(left=0)
        
        # Control signal derivative (smoothness indicator)
        ax = axes[1, 1]
        for name, result in results.items():
            control_diff = np.diff(result['control']) / result['dt']
            ax.plot(result['time'][1:], control_diff,
                   color=colors.get(name, 'gray'),
                   label=labels.get(name, name),
                   linewidth=1.5, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('dThrust/dt (N/s)')
        ax.set_title('Control Rate (Smoothness Indicator)')
        ax.legend(loc='upper right')
        ax.set_xlim(left=0)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_convergence(
        self,
        history: dict,
        save_name: str = "convergence.png"
    ) -> plt.Figure:
        """Plot optimization convergence metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        generations = history['generations']
        
        # Hypervolume
        ax = axes[0]
        ax.plot(generations, history['hypervolume'], 
               color=self.colors['pareto'], linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Hypervolume')
        ax.set_title('Hypervolume Indicator Convergence')
        ax.fill_between(generations, history['hypervolume'], 
                       alpha=0.3, color=self.colors['pareto'])
        
        # Pareto front size
        ax = axes[1]
        ax.plot(generations, history['n_pareto'],
               color=self.colors['balanced'], linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Number of Solutions')
        ax.set_title('Pareto Front Size Evolution')
        ax.fill_between(generations, history['n_pareto'],
                       alpha=0.3, color=self.colors['balanced'])
        
        plt.tight_layout()
        fig.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_monte_carlo_robustness(
        self,
        monte_carlo_results: Dict[str, List[dict]],
        metric: str = 'settling_time',
        save_name: str = "monte_carlo_robustness.png"
    ) -> plt.Figure:
        """
        Visualize Monte Carlo robustness testing results.
        
        Args:
            monte_carlo_results: Dict with keys being tuning names,
                               values being lists of performance dictionaries
            metric: Which metric to visualize
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = list(monte_carlo_results.keys())
        data = []
        
        for name in names:
            values = [r[metric] for r in monte_carlo_results[name] 
                     if np.isfinite(r[metric])]
            data.append(values)
        
        # Box plot
        bp = ax.boxplot(data, labels=names, patch_artist=True)
        
        # Color the boxes
        colors_list = [self.colors.get(n, 'gray') for n in names]
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Monte Carlo Robustness Test: {metric.replace("_", " ").title()}')
        
        # Add scatter of actual points
        for i, (name, values) in enumerate(zip(names, data)):
            x = np.random.normal(i + 1, 0.04, size=len(values))
            ax.scatter(x, values, alpha=0.4, s=20, color=colors_list[i])
        
        plt.tight_layout()
        fig.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_differential_equations(
        self,
        save_name: str = "differential_equations.png"
    ) -> plt.Figure:
        """Create a figure showing the system differential equations."""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Quadcopter Altitude Control: Mathematical Model',
               fontsize=18, fontweight='bold', ha='center', va='top',
               transform=ax.transAxes)
        
        # State space equations
        equations = [
            ('State Variables:', 0.92, True),
            (r'$\mathbf{x} = [z, \dot{z}]^T$ (altitude and velocity)', 0.88, False),
            ('', 0.84, False),
            ('Dynamics:', 0.80, True),
            (r'$\dot{z} = v_z$', 0.76, False),
            (r'$\dot{v}_z = \frac{T}{m}\cos(\phi)\cos(\theta) - g - \frac{D_z}{m}|v_z|v_z$', 0.72, False),
            ('', 0.68, False),
            ('PID Control Law:', 0.64, True),
            (r'$u(t) = K_p e(t) + K_i \int_0^t e(\tau)d\tau + K_d \frac{de(t)}{dt}$', 0.60, False),
            ('', 0.56, False),
            ('Tri-Objective Functions:', 0.52, True),
            (r'$J_1 = \int_0^T t|e(t)|dt$  (ITAE - Tracking Accuracy)', 0.48, False),
            (r'$J_2 = \int_0^T u(t)^2 dt$  (ICS - Energy Efficiency)', 0.44, False),
            (r'$J_3 = \sum_{k=1}^{N}|u_k - u_{k-1}|$  (TV - Control Smoothness)', 0.40, False),
            ('', 0.36, False),
            ('Constraints:', 0.32, True),
            (r'$T_{min} \leq T \leq T_{max}$ (Actuator saturation)', 0.28, False),
            (r'$\tau_m \frac{dT_{actual}}{dt} = T_{cmd} - T_{actual}$ (Motor dynamics)', 0.24, False),
        ]
        
        for item in equations:
            eq, y_pos = item[0], item[1]
            is_bold = item[2] if len(item) > 2 else False
            weight = 'bold' if is_bold else 'normal'
            ax.text(0.1, y_pos, eq, fontsize=14, ha='left', va='top',
                   transform=ax.transAxes, fontweight=weight)
        
        # Add decorative box
        rect = plt.Rectangle((0.05, 0.18), 0.9, 0.78, fill=False,
                            edgecolor='#2c3e50', linewidth=2,
                            transform=ax.transAxes)
        ax.add_patch(rect)
        
        # Parameter table
        ax.text(0.5, 0.12, 'System Parameters', fontsize=14,
               fontweight='bold', ha='center', transform=ax.transAxes)
        
        params_text = (r'$m = 1.5$ kg  |  $g = 9.81$ m/s²  |  '
                      r'$D_z = 0.5$ N/(m/s)²  |  $\tau_m = 0.02$ s')
        ax.text(0.5, 0.06, params_text, fontsize=12, ha='center',
               transform=ax.transAxes)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight',
                   facecolor='white')
        
        return fig
    
    def plot_pid_gains_distribution(
        self,
        pareto_front: List,
        knee_point: Optional[object] = None,
        save_name: str = "pid_gains_distribution.png"
    ) -> plt.Figure:
        """Visualize distribution of PID gains across the Pareto front."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        gains = np.array([ind.genes for ind in pareto_front])
        objectives = np.array([ind.objectives for ind in pareto_front])
        
        # Normalize ITAE for coloring
        itae_norm = (objectives[:, 0] - objectives[:, 0].min()) / \
                   (objectives[:, 0].max() - objectives[:, 0].min() + 1e-10)
        
        titles = [r'$K_p$ Distribution', r'$K_i$ Distribution', r'$K_d$ Distribution']
        
        for i, (ax, title) in enumerate(zip(axes, titles)):
            scatter = ax.scatter(gains[:, i], objectives[:, 0], 
                               c=itae_norm, cmap='viridis', s=50, alpha=0.7)
            if knee_point:
                ax.scatter(knee_point.genes[i], knee_point.objectives[0],
                          c=self.colors['balanced'], s=200, marker='*',
                          edgecolors='black', linewidth=2, label='Knee Point')
            ax.set_xlabel(f'${["K_p", "K_i", "K_d"][i]}$')
            ax.set_ylabel(r'$J_1$: ITAE')
            ax.set_title(title)
            ax.legend()
        
        plt.colorbar(scatter, ax=axes, label='ITAE (normalized)', shrink=0.8)
        plt.tight_layout()
        fig.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_summary_figure(
        self,
        pareto_front: List,
        time_domain_results: Dict[str, dict],
        history: dict,
        knee_point: Optional[object] = None,
        extreme_points: Optional[Tuple] = None,
        setpoint: float = 5.0,
        save_name: str = "optimization_summary.png"
    ) -> plt.Figure:
        """Create a comprehensive summary figure."""
        fig = plt.figure(figsize=(20, 15))
        
        # 3D Pareto front
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        objectives = np.array([ind.objectives for ind in pareto_front])
        itae_norm = (objectives[:, 0] - objectives[:, 0].min()) / \
                   (objectives[:, 0].max() - objectives[:, 0].min() + 1e-10)
        
        ax1.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                   c=itae_norm, cmap='viridis', s=40, alpha=0.7)
        if knee_point:
            ax1.scatter(*knee_point.objectives, c=self.colors['balanced'],
                       s=200, marker='*', edgecolors='black')
        ax1.set_xlabel(r'$J_1$: ITAE')
        ax1.set_ylabel(r'$J_2$: ICS')
        ax1.set_zlabel(r'$J_3$: TV')
        ax1.set_title('3D Pareto Front')
        ax1.view_init(elev=20, azim=45)
        
        # Position response
        ax2 = fig.add_subplot(2, 3, 2)
        for name, result in time_domain_results.items():
            color = self.colors.get(name, 'gray')
            ax2.plot(result['time'], result['true_position'], 
                    color=color, linewidth=2, label=name.title())
        ax2.axhline(y=setpoint, color='black', linestyle='--', label='Setpoint')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Altitude (m)')
        ax2.set_title('Step Response Comparison')
        ax2.legend()
        
        # Control signal
        ax3 = fig.add_subplot(2, 3, 3)
        for name, result in time_domain_results.items():
            color = self.colors.get(name, 'gray')
            ax3.plot(result['time'], result['control'],
                    color=color, linewidth=2, label=name.title())
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Thrust (N)')
        ax3.set_title('Control Effort Comparison')
        ax3.legend()
        
        # Convergence
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(history['generations'], history['hypervolume'],
                color=self.colors['pareto'], linewidth=2)
        ax4.fill_between(history['generations'], history['hypervolume'],
                        alpha=0.3, color=self.colors['pareto'])
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Hypervolume')
        ax4.set_title('Optimization Convergence')
        
        # 2D projections
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.scatter(objectives[:, 0], objectives[:, 1],
                   c=self.colors['pareto'], s=40, alpha=0.6)
        if knee_point:
            ax5.scatter(knee_point.objectives[0], knee_point.objectives[1],
                       c=self.colors['balanced'], s=200, marker='*',
                       edgecolors='black', label='Knee')
        ax5.set_xlabel(r'$J_1$: ITAE')
        ax5.set_ylabel(r'$J_2$: ICS')
        ax5.set_title('ITAE vs ICS Trade-off')
        ax5.legend()
        
        # Gains distribution
        ax6 = fig.add_subplot(2, 3, 6)
        gains = np.array([ind.genes for ind in pareto_front])
        bp = ax6.boxplot([gains[:, 0], gains[:, 1], gains[:, 2]],
                        labels=[r'$K_p$', r'$K_i$', r'$K_d$'],
                        patch_artist=True)
        colors_box = [self.colors['aggressive'], self.colors['balanced'], 
                     self.colors['smooth']]
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax6.set_ylabel('Gain Value')
        ax6.set_title('PID Gains Distribution (Pareto Front)')
        
        plt.suptitle('Multi-Objective PID Optimization Summary', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_system_block_diagram(
        self,
        save_name: str = "system_block_diagram.png"
    ) -> plt.Figure:
        """Create a control system block diagram."""
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # Block style
        block_style = dict(boxstyle='round,pad=0.3', facecolor='lightblue',
                          edgecolor='navy', linewidth=2)
        sum_style = dict(boxstyle='circle,pad=0.3', facecolor='lightyellow',
                        edgecolor='orange', linewidth=2)
        
        # Title
        ax.text(8, 7.5, 'Closed-Loop PID Control System', fontsize=16,
               fontweight='bold', ha='center')
        
        # Setpoint
        ax.annotate('', xy=(1.5, 4), xytext=(0.5, 4),
                   arrowprops=dict(arrowstyle='->', lw=2))
        ax.text(0.3, 4.3, r'$r(t)$', fontsize=12)
        ax.text(0.3, 3.6, 'Setpoint', fontsize=10, style='italic')
        
        # Summing junction
        circle = plt.Circle((2, 4), 0.3, fill=True, facecolor='lightyellow',
                           edgecolor='orange', linewidth=2)
        ax.add_patch(circle)
        ax.text(2, 4, '+', fontsize=14, ha='center', va='center')
        ax.text(2.1, 3.5, '−', fontsize=14, ha='center', va='center')
        
        # Error arrow
        ax.annotate('', xy=(3.5, 4), xytext=(2.3, 4),
                   arrowprops=dict(arrowstyle='->', lw=2))
        ax.text(2.8, 4.3, r'$e(t)$', fontsize=12)
        
        # PID Controller block
        pid_box = FancyBboxPatch((3.5, 3.2), 2.5, 1.6, **block_style)
        ax.add_patch(pid_box)
        ax.text(4.75, 4.3, 'PID Controller', fontsize=11, ha='center', fontweight='bold')
        ax.text(4.75, 3.7, r'$K_p + K_i\int + K_d\frac{d}{dt}$', fontsize=10, ha='center')
        
        # Control signal arrow
        ax.annotate('', xy=(7, 4), xytext=(6, 4),
                   arrowprops=dict(arrowstyle='->', lw=2))
        ax.text(6.4, 4.3, r'$u(t)$', fontsize=12)
        
        # Actuator block
        act_box = FancyBboxPatch((7, 3.3), 2, 1.4, boxstyle='round,pad=0.3',
                                facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
        ax.add_patch(act_box)
        ax.text(8, 4.2, 'Actuator', fontsize=11, ha='center', fontweight='bold')
        ax.text(8, 3.7, r'$\tau_m \dot{T} = u - T$', fontsize=9, ha='center')
        
        # Thrust arrow
        ax.annotate('', xy=(10, 4), xytext=(9, 4),
                   arrowprops=dict(arrowstyle='->', lw=2))
        ax.text(9.4, 4.3, r'$T$', fontsize=12)
        
        # Plant block
        plant_box = FancyBboxPatch((10, 3.2), 2.5, 1.6, boxstyle='round,pad=0.3',
                                  facecolor='lightsalmon', edgecolor='darkred', linewidth=2)
        ax.add_patch(plant_box)
        ax.text(11.25, 4.3, 'Quadcopter', fontsize=11, ha='center', fontweight='bold')
        ax.text(11.25, 3.7, r'$m\ddot{z} = T - mg - D$', fontsize=9, ha='center')
        
        # Output arrow
        ax.annotate('', xy=(14, 4), xytext=(12.5, 4),
                   arrowprops=dict(arrowstyle='->', lw=2))
        ax.text(13.5, 4.3, r'$y(t)$', fontsize=12)
        ax.text(14.2, 4, 'Altitude', fontsize=10, style='italic')
        
        # Feedback path
        ax.plot([13.5, 13.5], [4, 1.5], 'k-', lw=2)
        ax.plot([13.5, 2], [1.5, 1.5], 'k-', lw=2)
        ax.annotate('', xy=(2, 3.7), xytext=(2, 1.5),
                   arrowprops=dict(arrowstyle='->', lw=2))
        
        # Sensor block
        sensor_box = FancyBboxPatch((6.5, 0.8), 2.5, 1.2, boxstyle='round,pad=0.3',
                                   facecolor='plum', edgecolor='purple', linewidth=2)
        ax.add_patch(sensor_box)
        ax.text(7.75, 1.6, 'Sensor + Noise', fontsize=10, ha='center', fontweight='bold')
        ax.text(7.75, 1.1, 'O-U Process', fontsize=9, ha='center', style='italic')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight',
                   facecolor='white')
        
        return fig
