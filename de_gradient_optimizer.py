"""
DEGradientOptimizer: A hybrid Differential Evolution / Gradient Descent optimizer.

This module implements a population-based global optimizer that combines the
exploratory power of Differential Evolution (DE) with the local convergence
speed of Gradient Descent (GD). Each generation, a fraction of the population
is refined with gradient steps (when a gradient - analytic or finite-difference
- is available), while the rest of the population continues to explore the
search space via DE's mutation/crossover/selection cycle.

The design and theoretical guarantees are described in THEORY.md.

Dependencies: numpy only.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List


ArrayFn = Callable[[np.ndarray], float]
GradFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class OptimizeResult:
    """Container for the outcome of an optimization run."""

    x: np.ndarray
    fun: float
    nit: int
    nfev: int
    history: List[float] = field(default_factory=list)
    success: bool = True
    message: str = "Optimization terminated successfully."


def _finite_difference_gradient(f: ArrayFn, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference gradient approximation.

    grad_i f(x) ~= (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)
    """
    grad = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        e = np.zeros_like(x, dtype=float)
        e[i] = eps
        grad[i] = (f(x + e) - f(x - e)) / (2.0 * eps)
    return grad


class DEGradientOptimizer:
    """Hybrid Differential Evolution + Gradient Descent optimizer.

    Parameters
    ----------
    func : callable
        Objective function to minimize, ``f(x) -> float`` where ``x`` is a
        1-D numpy array.
    bounds : sequence of (low, high)
        Box constraints for each dimension.
    grad : callable, optional
        Gradient of ``func``. If not supplied, a central finite-difference
        approximation is used whenever a gradient step is requested.
    pop_size : int
        Number of individuals in the population.
    F : float
        Differential weight (mutation scale factor), typically in (0, 2].
    CR : float
        Crossover probability, in [0, 1].
    grad_lr : float
        Learning rate (step size) used for the gradient-descent refinement
        step.
    grad_fraction : float
        Fraction of the population (best individuals) refined with a
        gradient step each generation, in [0, 1].
    grad_steps : int
        Number of gradient-descent steps applied per refined individual per
        generation.
    seed : int, optional
        Seed for the random number generator (reproducibility).
    """

    def __init__(
        self,
        func: ArrayFn,
        bounds,
        grad: Optional[GradFn] = None,
        pop_size: int = 30,
        F: float = 0.6,
        CR: float = 0.9,
        grad_lr: float = 0.01,
        grad_fraction: float = 0.3,
        grad_steps: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        self.func = func
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must be a sequence of (low, high) pairs")
        self.dim = self.bounds.shape[0]
        self.lower = self.bounds[:, 0]
        self.upper = self.bounds[:, 1]

        self.grad = grad
        self.pop_size = max(4, int(pop_size))
        self.F = float(F)
        self.CR = float(CR)
        self.grad_lr = float(grad_lr)
        self.grad_fraction = float(np.clip(grad_fraction, 0.0, 1.0))
        self.grad_steps = max(0, int(grad_steps))

        self.rng = np.random.default_rng(seed)
        self.nfev = 0

    # ------------------------------------------------------------------
    # Core building blocks
    # ------------------------------------------------------------------
    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lower, self.upper)

    def _evaluate(self, x: np.ndarray) -> float:
        self.nfev += 1
        return float(self.func(x))

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        if self.grad is not None:
            return np.asarray(self.grad(x), dtype=float)
        return _finite_difference_gradient(self.func, x)

    def _init_population(self) -> np.ndarray:
        return self.rng.uniform(self.lower, self.upper, size=(self.pop_size, self.dim))

    def _mutate(self, pop: np.ndarray, idx: int, best: np.ndarray) -> np.ndarray:
        """DE/current-to-best/1 mutation strategy."""
        candidates = [i for i in range(self.pop_size) if i != idx]
        r1, r2 = self.rng.choice(candidates, size=2, replace=False)
        x = pop[idx]
        mutant = x + self.F * (best - x) + self.F * (pop[r1] - pop[r2])
        return self._clip(mutant)

    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        cross_mask = self.rng.random(self.dim) < self.CR
        # Ensure at least one dimension is taken from the mutant.
        j_rand = self.rng.integers(0, self.dim)
        cross_mask[j_rand] = True
        return np.where(cross_mask, mutant, target)

    def _gradient_refine(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        for _ in range(self.grad_steps):
            g = self._gradient(y)
            y = self._clip(y - self.grad_lr * g)
        return y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def optimize(self, max_generations: int = 200, tol: float = 1e-10) -> OptimizeResult:
        """Run the hybrid DE + gradient-descent optimization loop.

        Parameters
        ----------
        max_generations : int
            Maximum number of generations to run.
        tol : float
            Stop early if the population's best fitness improves by less
            than ``tol`` for a full generation.

        Returns
        -------
        OptimizeResult
        """
        pop = self._init_population()
        fitness = np.array([self._evaluate(ind) for ind in pop])

        best_idx = int(np.argmin(fitness))
        best = pop[best_idx].copy()
        best_fit = float(fitness[best_idx])
        history = [best_fit]

        n_refine = int(round(self.grad_fraction * self.pop_size))

        for gen in range(max_generations):
            prev_best_fit = best_fit

            # --- Differential Evolution step (mutation + crossover + selection) ---
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            for i in range(self.pop_size):
                mutant = self._mutate(pop, i, best)
                trial = self._crossover(pop[i], mutant)
                trial_fit = self._evaluate(trial)
                if trial_fit <= fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = trial_fit
            pop, fitness = new_pop, new_fitness

            # --- Gradient Descent refinement step (elitist local search) ---
            if n_refine > 0 and self.grad_steps > 0:
                order = np.argsort(fitness)
                for rank in range(n_refine):
                    i = order[rank]
                    refined = self._gradient_refine(pop[i])
                    refined_fit = self._evaluate(refined)
                    if refined_fit <= fitness[i]:
                        pop[i] = refined
                        fitness[i] = refined_fit

            best_idx = int(np.argmin(fitness))
            if fitness[best_idx] < best_fit:
                best_fit = float(fitness[best_idx])
                best = pop[best_idx].copy()
            history.append(best_fit)

            if abs(prev_best_fit - best_fit) < tol:
                return OptimizeResult(
                    x=best,
                    fun=best_fit,
                    nit=gen + 1,
                    nfev=self.nfev,
                    history=history,
                    success=True,
                    message="Converged: improvement below tolerance.",
                )

        return OptimizeResult(
            x=best,
            fun=best_fit,
            nit=max_generations,
            nfev=self.nfev,
            history=history,
            success=True,
            message="Maximum generations reached.",
        )
