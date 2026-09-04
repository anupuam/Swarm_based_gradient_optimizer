# Swarm-based gradient optimizer

This repository provides a dependency-free optimizer that combines particle
swarm exploration with a gradient-descent step. It minimizes a callable over
finite box constraints and accepts either an analytic gradient or a central
finite-difference approximation.

```python
from swarm_based_gradient_optimizer import SwarmBasedGradientOptimizer

optimizer = SwarmBasedGradientOptimizer(
    lambda x: sum(value * value for value in x),
    [(-5, 5), (-5, 5)],
    seed=42,
)
result = optimizer.optimize()
print(result.best_position, result.best_value)
```

`OptimizationResult.history` contains the best objective value after
initialization and after each iteration. The optimizer is deterministic when
`seed` is provided.