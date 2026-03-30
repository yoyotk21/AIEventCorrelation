import math

import numpy as np

from models import OptimizationResult, PipelineConfig


class WeightOptimizer:
    """
    Hill climbing with simulated annealing and random restarts.

    Finds the weight vector [w1, ..., w7] that minimizes MSE between the
    weighted combination of feature matrices and the ground truth co-resolution
    matrix, evaluated only over valid pairs (controlled by the mask).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.rng = np.random.default_rng(seed=config.random_seed)

    def compute_mse(
        self,
        weights: np.ndarray,
        feature_matrices: list[np.ndarray],
        ground_truth: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Vectorized MSE over valid pairs. Called ~250K times — must be fast."""
        predicted = sum(w * F for w, F in zip(weights, feature_matrices))
        diff = (predicted - ground_truth) ** 2
        valid = diff[mask]
        return float(valid.mean()) if len(valid) > 0 else float("inf")

    def optimize(
        self,
        feature_matrices: list[np.ndarray],
        ground_truth: np.ndarray,
        mask: np.ndarray,
        feature_names: list[str],
    ) -> OptimizationResult:
        """
        Run hill climbing with random restarts.
        Returns the best weight vector found across all restarts.
        """
        n_features = len(feature_matrices)
        best_weights = np.full(n_features, 1.0 / n_features)
        best_mse = self.compute_mse(best_weights, feature_matrices, ground_truth, mask)
        best_restart = 0

        for restart in range(self.config.num_restarts):
            # Random starting point (first restart uses equal weights)
            if restart == 0:
                initial = np.full(n_features, 1.0 / n_features)
            else:
                initial = self.rng.random(n_features)
                initial /= initial.sum()

            weights, mse = self._single_run(
                initial, feature_matrices, ground_truth, mask
            )

            if mse < best_mse:
                best_mse = mse
                best_weights = weights.copy()
                best_restart = restart

            if (restart + 1) % 10 == 0:
                print(
                    f"  Restart {restart + 1}/{self.config.num_restarts} | "
                    f"best MSE so far: {best_mse:.6f}"
                )

        return OptimizationResult(
            weights=best_weights.tolist(),
            feature_names=feature_names,
            train_mse=best_mse,
            best_restart=best_restart,
        )

    def _single_run(
        self,
        initial_weights: np.ndarray,
        feature_matrices: list[np.ndarray],
        ground_truth: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        One hill-climbing run with simulated annealing from a starting point.
        Returns (best_weights, best_mse).
        """
        weights = initial_weights.copy()
        current_mse = self.compute_mse(weights, feature_matrices, ground_truth, mask)
        temperature = self.config.initial_temperature

        for _ in range(self.config.max_iterations):
            candidate = self._generate_neighbor(weights)
            candidate_mse = self.compute_mse(
                candidate, feature_matrices, ground_truth, mask
            )

            delta = candidate_mse - current_mse
            if delta < 0:
                # Strictly better — always accept
                weights = candidate
                current_mse = candidate_mse
            elif temperature > 1e-10:
                # Worse move: accept with probability exp(-delta / T)
                if self.rng.random() < math.exp(-delta / temperature):
                    weights = candidate
                    current_mse = candidate_mse

            temperature *= self.config.decay_rate

        return weights, current_mse

    def _generate_neighbor(self, weights: np.ndarray) -> np.ndarray:
        """
        Perturb one randomly chosen weight by a small random delta,
        clamp all weights to [0, 1], then re-normalize to sum to 1.
        """
        candidate = weights.copy()
        k = int(self.rng.integers(0, len(weights)))
        delta = self.rng.uniform(
            -self.config.perturbation_size, self.config.perturbation_size
        )
        candidate[k] = np.clip(candidate[k] + delta, 0.0, 1.0)

        total = candidate.sum()
        if total > 0:
            candidate /= total
        else:
            candidate = np.full(len(weights), 1.0 / len(weights))

        return candidate
