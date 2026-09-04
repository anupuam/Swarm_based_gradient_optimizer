import unittest

from swarm_based_gradient_optimizer import SwarmBasedGradientOptimizer


class SwarmBasedGradientOptimizerTests(unittest.TestCase):
    def test_gradient_optimizer_improves_quadratic(self):
        objective = lambda point: sum((value - 2.0) ** 2 for value in point)
        gradient = lambda point: [2.0 * (value - 2.0) for value in point]
        optimizer = SwarmBasedGradientOptimizer(
            objective,
            [(-5, 5), (-5, 5)],
            population_size=12,
            iterations=40,
            learning_rate=0.08,
            gradient=gradient,
            seed=7,
        )

        result = optimizer.optimize()

        self.assertLess(result.best_value, 1e-4)
        self.assertEqual(len(result.history), 41)
        self.assertTrue(all(result.history[i] <= result.history[i - 1] for i in range(1, len(result.history))))

    def test_finite_difference_gradient_is_supported(self):
        optimizer = SwarmBasedGradientOptimizer(
            lambda point: (point[0] - 1.5) ** 2,
            [(-4, 4)],
            population_size=8,
            iterations=25,
            learning_rate=0.1,
            seed=3,
        )

        self.assertLess(optimizer.optimize().best_value, 1e-3)

    def test_invalid_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            SwarmBasedGradientOptimizer(lambda point: 0, [(1, 1)])


if __name__ == "__main__":
    unittest.main()
