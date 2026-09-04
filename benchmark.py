"""
Benchmark suite for DEGradientOptimizer.

Compares the hybrid DE+GD optimizer against:
  - Pure Differential Evolution (grad_fraction=0)
  - Pure Gradient Descent from a random start (via grad_steps only, no DE)

on standard non-convex test functions: Sphere, Rosenbrock, Rastrigin, Ackley.

Run:
    python benchmark.py
"""

from __future__ import annotations

import time
import numpy as np

from de_gradient_optimizer import DEGradientOptimizer


# ---------------------------------------------------------------------------
# Benchmark functions and analytic gradients
# ---------------------------------------------------------------------------
def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def sphere_grad(x: np.ndarray) -> np.ndarray:
    return 2.0 * x


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    g = np.zeros_like(x)
    g[:-1] += -400 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2 * (1 - x[:-1])
    g[1:] += 200 * (x[1:] - x[:-1] ** 2)
    return g


def rastrigin(x: np.ndarray) -> float:
    A = 10.0
    return float(A * x.size + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))


def rastrigin_grad(x: np.ndarray) -> np.ndarray:
    A = 10.0
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)


def ackley(x: np.ndarray) -> float:
    a, b, c = 20.0, 0.2, 2 * np.pi
    d = x.size
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))
    return float(
        -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e
    )


def ackley_grad(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # Analytic gradient (careful near x=0 where sqrt term is non-smooth).
    a, b, c = 20.0, 0.2, 2 * np.pi
    d = x.size
    sum1 = np.sum(x ** 2)
    norm = np.sqrt(sum1 / d) + eps
    term1 = a * b * np.exp(-b * norm) * x / (d * norm)
    term2 = (c / d) * np.sin(c * x) * np.exp(np.sum(np.cos(c * x)) / d)
    return term1 + term2


BENCHMARKS = {
    "Sphere": (sphere, sphere_grad, (-5.12, 5.12)),
    "Rosenbrock": (rosenbrock, rosenbrock_grad, (-2.048, 2.048)),
    "Rastrigin": (rastrigin, rastrigin_grad, (-5.12, 5.12)),
    "Ackley": (ackley, ackley_grad, (-32.768, 32.768)),
}


def run_variant(name, func, grad, bounds, dim, seed, **kwargs):
    b = [bounds] * dim
    opt = DEGradientOptimizer(func, b, grad=grad, seed=seed, **kwargs)
    t0 = time.perf_counter()
    result = opt.optimize(max_generations=300, tol=1e-14)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def main(dim: int = 5, n_trials: int = 5):
    variants = {
        "Hybrid DE+GD": dict(grad_fraction=0.3, grad_steps=3, grad_lr=0.01, pop_size=40),
        "Pure DE": dict(grad_fraction=0.0, grad_steps=0, pop_size=40),
        "GD-heavy (grad_fraction=1)": dict(grad_fraction=1.0, grad_steps=3, grad_lr=0.01, pop_size=40),
    }

    print(f"{'Function':<12}{'Variant':<28}{'Best f(x)':>14}{'NFev':>10}{'Time(s)':>10}")
    print("-" * 76)
    for fname, (func, grad, bounds) in BENCHMARKS.items():
        for vname, kwargs in variants.items():
            best_vals = []
            nfevs = []
            times = []
            for trial in range(n_trials):
                result, elapsed = run_variant(
                    fname, func, grad, bounds, dim, seed=trial, **kwargs
                )
                best_vals.append(result.fun)
                nfevs.append(result.nfev)
                times.append(elapsed)
            print(
                f"{fname:<12}{vname:<28}{np.mean(best_vals):>14.6e}"
                f"{np.mean(nfevs):>10.0f}{np.mean(times):>10.3f}"
            )
        print()


if __name__ == "__main__":
    main()
