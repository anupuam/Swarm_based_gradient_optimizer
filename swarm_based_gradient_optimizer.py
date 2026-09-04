"""A small, dependency-free swarm-based gradient optimizer."""

from dataclasses import dataclass
import math
import random
from typing import Callable, List, Optional, Sequence, Tuple


Number = float
Objective = Callable[[Sequence[Number]], Number]
Gradient = Callable[[Sequence[Number]], Sequence[Number]]


@dataclass(frozen=True)
class OptimizationResult:
    """The result returned by :meth:`SwarmBasedGradientOptimizer.optimize`."""

    best_position: Tuple[Number, ...]
    best_value: Number
    history: Tuple[Number, ...]


class SwarmBasedGradientOptimizer:
    """Minimize an objective using particle-swarm and gradient updates.

    Each particle is moved by the usual PSO velocity update and then nudged
    down the objective gradient.  A gradient can be supplied by the caller;
    otherwise a central finite-difference approximation is used.
    """

    def __init__(
        self,
        objective: Objective,
        bounds: Sequence[Tuple[Number, Number]],
        *,
        population_size: int = 30,
        iterations: int = 100,
        learning_rate: Number = 0.01,
        inertia: Number = 0.7,
        cognitive: Number = 1.4,
        social: Number = 1.4,
        gradient: Optional[Gradient] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not bounds:
            raise ValueError("bounds must contain at least one dimension")
        if population_size < 1 or iterations < 1:
            raise ValueError("population_size and iterations must be positive")
        if learning_rate < 0 or inertia < 0 or cognitive < 0 or social < 0:
            raise ValueError("optimizer coefficients must be non-negative")

        parsed_bounds = []
        for lower, upper in bounds:
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError("each bound must be finite and lower < upper")
            parsed_bounds.append((float(lower), float(upper)))

        self.objective = objective
        self.bounds = tuple(parsed_bounds)
        self.population_size = population_size
        self.iterations = iterations
        self.learning_rate = float(learning_rate)
        self.inertia = float(inertia)
        self.cognitive = float(cognitive)
        self.social = float(social)
        self.gradient = gradient
        self._random = random.Random(seed)

    def optimize(self) -> OptimizationResult:
        """Run the optimization and return the best point found."""
        dimension = len(self.bounds)
        positions = [
            [self._random.uniform(lower, upper) for lower, upper in self.bounds]
            for _ in range(self.population_size)
        ]
        velocities = [
            [
                self._random.uniform(-(upper - lower), upper - lower)
                for lower, upper in self.bounds
            ]
            for _ in range(self.population_size)
        ]
        values = [self._evaluate(position) for position in positions]
        personal_positions = [position[:] for position in positions]
        personal_values = values[:]
        best_index = min(range(self.population_size), key=personal_values.__getitem__)
        global_position = personal_positions[best_index][:]
        global_value = personal_values[best_index]
        history: List[Number] = [global_value]

        for _ in range(self.iterations):
            for index, position in enumerate(positions):
                for coordinate, (lower, upper) in enumerate(self.bounds):
                    r1 = self._random.random()
                    r2 = self._random.random()
                    velocities[index][coordinate] = (
                        self.inertia * velocities[index][coordinate]
                        + self.cognitive
                        * r1
                        * (personal_positions[index][coordinate] - position[coordinate])
                        + self.social
                        * r2
                        * (global_position[coordinate] - position[coordinate])
                    )
                    position[coordinate] = min(
                        upper,
                        max(
                            lower,
                            position[coordinate]
                            + velocities[index][coordinate]
                            - self.learning_rate
                            * self._gradient(position, coordinate, dimension)[coordinate],
                        ),
                    )

                value = self._evaluate(position)
                if value < personal_values[index]:
                    personal_values[index] = value
                    personal_positions[index] = position[:]
                    if value < global_value:
                        global_value = value
                        global_position = position[:]
            history.append(global_value)

        return OptimizationResult(tuple(global_position), global_value, tuple(history))

    def _evaluate(self, position: Sequence[Number]) -> Number:
        value = float(self.objective(tuple(position)))
        if not math.isfinite(value):
            raise ValueError("objective must return a finite number")
        return value

    def _gradient(self, position: Sequence[Number], coordinate: int, dimension: int) -> Sequence[Number]:
        if self.gradient is not None:
            result = tuple(float(value) for value in self.gradient(tuple(position)))
            if len(result) != dimension or not all(math.isfinite(value) for value in result):
                raise ValueError("gradient must return one finite value per dimension")
            return result

        point = list(position)
        lower, upper = self.bounds[coordinate]
        step = 1e-8 * max(1.0, abs(point[coordinate]))
        plus = min(upper, point[coordinate] + step)
        minus = max(lower, point[coordinate] - step)
        if plus == minus:
            return [0.0] * dimension
        point[coordinate] = plus
        plus_value = self._evaluate(point)
        point[coordinate] = minus
        minus_value = self._evaluate(point)
        value = (plus_value - minus_value) / (plus - minus)
        result = [0.0] * dimension
        result[coordinate] = value
        return result


__all__ = ["OptimizationResult", "SwarmBasedGradientOptimizer"]
