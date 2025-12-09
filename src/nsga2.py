"""
NSGA-II Multi-Objective Optimization
====================================

Non-dominated Sorting Genetic Algorithm II implementation for
Tri-Objective PID tuning.

Features:
- Latin Hypercube Sampling (LHS) for population initialization
- Simulated Binary Crossover (SBX)
- Polynomial Mutation
- Non-dominated sorting
- Crowding distance assignment
- Hypervolume indicator for convergence tracking
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings


@dataclass
class Individual:
    """An individual in the population."""
    genes: np.ndarray  # [Kp, Ki, Kd]
    objectives: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rank: int = 0
    crowding_distance: float = 0.0
    
    def dominates(self, other: 'Individual') -> bool:
        """Check if this individual dominates another."""
        better_in_any = False
        for i in range(len(self.objectives)):
            if self.objectives[i] > other.objectives[i]:
                return False
            if self.objectives[i] < other.objectives[i]:
                better_in_any = True
        return better_in_any
    
    def __lt__(self, other: 'Individual') -> bool:
        """Comparison for sorting (used in selection)."""
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.crowding_distance > other.crowding_distance


@dataclass
class OptimizationBounds:
    """Search space bounds for PID gains."""
    Kp_min: float = 0.1
    Kp_max: float = 50.0
    Ki_min: float = 0.0
    Ki_max: float = 20.0
    Kd_min: float = 0.0
    Kd_max: float = 10.0
    
    @property
    def lower(self) -> np.ndarray:
        return np.array([self.Kp_min, self.Ki_min, self.Kd_min])
    
    @property
    def upper(self) -> np.ndarray:
        return np.array([self.Kp_max, self.Ki_max, self.Kd_max])


class NSGA2Optimizer:
    """
    NSGA-II optimizer for multi-objective PID tuning.
    
    Minimizes three objectives:
        - J1: ITAE (tracking accuracy)
        - J2: ICS (energy efficiency)
        - J3: TV (control smoothness)
    """
    
    def __init__(
        self,
        objective_function: Callable[[np.ndarray], np.ndarray],
        bounds: OptimizationBounds = OptimizationBounds(),
        population_size: int = 100,
        n_generations: int = 50,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        crossover_eta: float = 15.0,  # SBX distribution index
        mutation_eta: float = 20.0,   # Polynomial mutation distribution index
        seed: Optional[int] = None,
        n_workers: int = 1,
        verbose: bool = True
    ):
        self.objective_fn = objective_function
        self.bounds = bounds
        self.pop_size = population_size
        self.n_gen = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.crossover_eta = crossover_eta
        self.mutation_eta = mutation_eta
        self.n_workers = n_workers
        self.verbose = verbose
        
        self.rng = np.random.default_rng(seed)
        
        # History tracking
        self.history = {
            'hypervolume': [],
            'n_pareto': [],
            'best_objectives': [],
            'generations': []
        }
        
    def initialize_population_lhs(self) -> List[Individual]:
        """
        Initialize population using Latin Hypercube Sampling.
        
        LHS ensures better coverage of the search space compared
        to random initialization.
        """
        n_vars = 3  # Kp, Ki, Kd
        lower = self.bounds.lower
        upper = self.bounds.upper
        
        # Create LHS samples
        samples = np.zeros((self.pop_size, n_vars))
        for j in range(n_vars):
            # Divide into equal intervals
            intervals = np.linspace(0, 1, self.pop_size + 1)
            # Sample within each interval
            for i in range(self.pop_size):
                samples[i, j] = self.rng.uniform(intervals[i], intervals[i + 1])
            # Shuffle
            self.rng.shuffle(samples[:, j])
        
        # Scale to bounds
        samples = lower + samples * (upper - lower)
        
        # Create individuals
        population = [Individual(genes=samples[i]) for i in range(self.pop_size)]
        
        return population
    
    def evaluate_population(self, population: List[Individual]):
        """Evaluate objectives for all individuals."""
        if self.n_workers > 1:
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                genes_list = [ind.genes for ind in population]
                results = list(executor.map(self.objective_fn, genes_list))
                for ind, obj in zip(population, results):
                    ind.objectives = np.array(obj)
        else:
            for ind in population:
                ind.objectives = np.array(self.objective_fn(ind.genes))
    
    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[int]]:
        """
        Fast non-dominated sorting (O(MN²)).
        
        Returns list of fronts, where each front is a list of indices.
        """
        n = len(population)
        domination_count = np.zeros(n, dtype=int)
        dominated_by = [[] for _ in range(n)]
        fronts = [[]]
        
        # Compare all pairs
        for i in range(n):
            for j in range(i + 1, n):
                if population[i].dominates(population[j]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif population[j].dominates(population[i]):
                    dominated_by[j].append(i)
                    domination_count[i] += 1
        
        # First front: individuals not dominated by anyone
        for i in range(n):
            if domination_count[i] == 0:
                population[i].rank = 0
                fronts[0].append(i)
        
        # Build subsequent fronts
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        population[j].rank = k + 1
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
                
        return fronts
    
    def crowding_distance_assignment(self, population: List[Individual], front: List[int]):
        """
        Assign crowding distance to individuals in a front.
        
        Measures how close an individual is to its neighbors in objective space.
        """
        if len(front) <= 2:
            for i in front:
                population[i].crowding_distance = np.inf
            return
        
        n_obj = len(population[0].objectives)
        
        # Reset distances
        for i in front:
            population[i].crowding_distance = 0.0
        
        # For each objective
        for m in range(n_obj):
            # Sort front by this objective
            sorted_front = sorted(front, key=lambda i: population[i].objectives[m])
            
            # Boundary points get infinite distance
            population[sorted_front[0]].crowding_distance = np.inf
            population[sorted_front[-1]].crowding_distance = np.inf
            
            # Objective range
            obj_max = population[sorted_front[-1]].objectives[m]
            obj_min = population[sorted_front[0]].objectives[m]
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            # Interior points
            for i in range(1, len(sorted_front) - 1):
                idx = sorted_front[i]
                prev_idx = sorted_front[i - 1]
                next_idx = sorted_front[i + 1]
                
                population[idx].crowding_distance += (
                    population[next_idx].objectives[m] - 
                    population[prev_idx].objectives[m]
                ) / obj_range
    
    def tournament_selection(self, population: List[Individual], k: int = 2) -> Individual:
        """
        Binary tournament selection based on rank and crowding distance.
        """
        candidates = self.rng.choice(len(population), size=k, replace=False)
        best = population[candidates[0]]
        for i in candidates[1:]:
            if population[i] < best:  # Uses __lt__ comparison
                best = population[i]
        return Individual(genes=best.genes.copy())
    
    def sbx_crossover(
        self, 
        parent1: np.ndarray, 
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulated Binary Crossover (SBX).
        
        Creates two offspring from two parents using a polynomial
        probability distribution.
        """
        if self.rng.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        eta = self.crossover_eta
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        for i in range(len(parent1)):
            if self.rng.random() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]
                    
                    yl = self.bounds.lower[i]
                    yu = self.bounds.upper[i]
                    
                    rand = self.rng.random()
                    
                    # Beta calculation
                    beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta + 1))
                    
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                    
                    c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    
                    beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta + 1))
                    
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                    
                    c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                    
                    child1[i] = np.clip(c1, yl, yu)
                    child2[i] = np.clip(c2, yl, yu)
                else:
                    child1[i] = parent1[i]
                    child2[i] = parent2[i]
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
        
        return child1, child2
    
    def polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        """
        Polynomial mutation.
        
        Applies small perturbations based on polynomial distribution.
        """
        mutant = individual.copy()
        eta = self.mutation_eta
        
        for i in range(len(mutant)):
            if self.rng.random() < self.mutation_prob:
                y = mutant[i]
                yl = self.bounds.lower[i]
                yu = self.bounds.upper[i]
                
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                
                rand = self.rng.random()
                mut_pow = 1.0 / (eta + 1)
                
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1))
                    deltaq = 1.0 - val ** mut_pow
                
                mutant[i] = np.clip(y + deltaq * (yu - yl), yl, yu)
        
        return mutant
    
    def compute_hypervolume(
        self, 
        population: List[Individual], 
        reference_point: np.ndarray
    ) -> float:
        """
        Compute hypervolume indicator (2D approximation for speed).
        
        For exact 3D hypervolume, use WFG algorithm or DEAP library.
        """
        # Get Pareto front
        pareto = [ind for ind in population if ind.rank == 0]
        if not pareto:
            return 0.0
        
        # Simple 2D projection for quick estimation
        # For production, use proper 3D hypervolume calculation
        points = np.array([ind.objectives[:2] for ind in pareto])
        
        # Sort by first objective
        sorted_idx = np.argsort(points[:, 0])
        points = points[sorted_idx]
        
        # Compute hypervolume
        hv = 0.0
        prev_y = reference_point[1]
        
        for i in range(len(points)):
            if points[i, 0] < reference_point[0] and points[i, 1] < reference_point[1]:
                hv += (reference_point[0] - points[i, 0]) * (prev_y - points[i, 1])
                prev_y = points[i, 1]
        
        return hv
    
    def create_offspring(self, population: List[Individual]) -> List[Individual]:
        """Create offspring population through selection, crossover, and mutation."""
        offspring = []
        
        while len(offspring) < self.pop_size:
            # Selection
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            
            # Crossover
            child1_genes, child2_genes = self.sbx_crossover(parent1.genes, parent2.genes)
            
            # Mutation
            child1_genes = self.polynomial_mutation(child1_genes)
            child2_genes = self.polynomial_mutation(child2_genes)
            
            offspring.append(Individual(genes=child1_genes))
            if len(offspring) < self.pop_size:
                offspring.append(Individual(genes=child2_genes))
        
        return offspring
    
    def run(self) -> Tuple[List[Individual], dict]:
        """
        Run NSGA-II optimization.
        
        Returns:
            (pareto_front, history) - List of Pareto-optimal individuals and history
        """
        if self.verbose:
            print("=" * 60)
            print("NSGA-II Multi-Objective PID Optimization")
            print("=" * 60)
            print(f"Population size: {self.pop_size}")
            print(f"Generations: {self.n_gen}")
            print(f"Search space: Kp=[{self.bounds.Kp_min}, {self.bounds.Kp_max}], "
                  f"Ki=[{self.bounds.Ki_min}, {self.bounds.Ki_max}], "
                  f"Kd=[{self.bounds.Kd_min}, {self.bounds.Kd_max}]")
            print("-" * 60)
        
        # Initialize population with LHS
        population = self.initialize_population_lhs()
        
        # Evaluate initial population
        self.evaluate_population(population)
        
        # Reference point for hypervolume (worst case)
        ref_point = np.array([1000.0, 100000.0, 10000.0])
        
        # Main loop
        for gen in range(self.n_gen):
            # Create offspring
            offspring = self.create_offspring(population)
            
            # Evaluate offspring
            self.evaluate_population(offspring)
            
            # Combine parent and offspring
            combined = population + offspring
            
            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort(combined)
            
            # Select next generation
            new_population = []
            front_idx = 0
            
            while len(new_population) + len(fronts[front_idx]) <= self.pop_size:
                # Add entire front
                self.crowding_distance_assignment(combined, fronts[front_idx])
                for i in fronts[front_idx]:
                    new_population.append(combined[i])
                front_idx += 1
                if front_idx >= len(fronts):
                    break
            
            # Fill remaining with crowding distance selection
            if len(new_population) < self.pop_size and front_idx < len(fronts):
                self.crowding_distance_assignment(combined, fronts[front_idx])
                sorted_front = sorted(
                    fronts[front_idx], 
                    key=lambda i: combined[i].crowding_distance,
                    reverse=True
                )
                for i in sorted_front:
                    if len(new_population) < self.pop_size:
                        new_population.append(combined[i])
            
            population = new_population
            
            # Compute metrics
            pareto_front = [ind for ind in population if ind.rank == 0]
            hv = self.compute_hypervolume(population, ref_point)
            
            # Store history
            self.history['hypervolume'].append(hv)
            self.history['n_pareto'].append(len(pareto_front))
            self.history['generations'].append(gen)
            
            if pareto_front:
                best_objs = np.array([ind.objectives for ind in pareto_front])
                self.history['best_objectives'].append(best_objs)
            
            if self.verbose and (gen + 1) % 5 == 0:
                print(f"Gen {gen + 1:3d} | Pareto size: {len(pareto_front):3d} | "
                      f"Hypervolume: {hv:.2f}")
        
        # Final Pareto front
        fronts = self.fast_non_dominated_sort(population)
        pareto_front = [population[i] for i in fronts[0]]
        
        if self.verbose:
            print("-" * 60)
            print(f"Optimization complete! Final Pareto front size: {len(pareto_front)}")
        
        return pareto_front, self.history
    
    def get_knee_point(self, pareto_front: List[Individual]) -> Individual:
        """
        Find the knee point of the Pareto front.
        
        The knee point is the solution with maximum distance from
        the line connecting the extreme points.
        """
        if len(pareto_front) <= 2:
            return pareto_front[0]
        
        objectives = np.array([ind.objectives for ind in pareto_front])
        
        # Normalize objectives
        obj_min = objectives.min(axis=0)
        obj_max = objectives.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1.0
        normalized = (objectives - obj_min) / obj_range
        
        # Find extreme points
        ideal = normalized.min(axis=0)
        nadir = normalized.max(axis=0)
        
        # Line from ideal to nadir
        line_vec = nadir - ideal
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return pareto_front[0]
        
        line_unit = line_vec / line_len
        
        # Find point with maximum distance from line
        max_dist = -1
        knee_idx = 0
        
        for i, point in enumerate(normalized):
            vec_to_point = point - ideal
            proj_length = np.dot(vec_to_point, line_unit)
            proj = ideal + proj_length * line_unit
            dist = np.linalg.norm(point - proj)
            
            if dist > max_dist:
                max_dist = dist
                knee_idx = i
        
        return pareto_front[knee_idx]
    
    def get_extreme_solutions(
        self, 
        pareto_front: List[Individual]
    ) -> Tuple[Individual, Individual, Individual]:
        """
        Get extreme solutions for each objective.
        
        Returns:
            (aggressive, smooth, efficient) - Extreme solutions
        """
        objectives = np.array([ind.objectives for ind in pareto_front])
        
        # Best ITAE (aggressive tracking)
        aggressive_idx = np.argmin(objectives[:, 0])
        
        # Best ICS (energy efficient)
        efficient_idx = np.argmin(objectives[:, 1])
        
        # Best TV (smooth control)
        smooth_idx = np.argmin(objectives[:, 2])
        
        return (
            pareto_front[aggressive_idx],
            pareto_front[smooth_idx],
            pareto_front[efficient_idx]
        )
