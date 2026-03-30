from datetime import datetime, timezone

import numpy as np
from scipy.stats import pearsonr

from data_loader import DataBundle
from models import EvaluationResult, Market, PipelineConfig


class Evaluator:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def split_by_time(self, markets: list[Market]) -> tuple[list[int], list[int]]:
        """
        Split market indices into train/test by resolution date.
        Markets resolved before train_cutoff -> train; after -> test.
        """
        cutoff = datetime.fromisoformat(self.config.train_cutoff).replace(
            tzinfo=timezone.utc
        )
        train_idx, test_idx = [], []
        for i, market in enumerate(markets):
            resolved = market.resolved_at()
            if resolved is None:
                train_idx.append(i)  # no date: put in train
                continue
            resolved_utc = resolved if resolved.tzinfo else resolved.replace(tzinfo=timezone.utc)
            if resolved_utc < cutoff:
                train_idx.append(i)
            else:
                test_idx.append(i)
        return train_idx, test_idx

    def build_ground_truth(
        self, markets: list[Market], indices: list[int]
    ) -> np.ndarray:
        """
        Build an N×N binary co-resolution matrix for the given market subset.
        ground_truth[i][j] = 1.0 if both markets resolved Yes, or both resolved No.
        0.0 if they resolved differently or resolution is unknown.
        """
        sub_markets = [markets[i] for i in indices]
        n = len(sub_markets)
        resolutions = [m.resolved_yes() for m in sub_markets]

        matrix = np.zeros((n, n))
        for i in range(n):
            if resolutions[i] is None:
                continue
            for j in range(i + 1, n):
                if resolutions[j] is None:
                    continue
                if resolutions[i] == resolutions[j]:
                    matrix[i][j] = 1.0
                    matrix[j][i] = 1.0

        np.fill_diagonal(matrix, 0.0)
        return matrix

    def build_valid_pair_mask(
        self, overlap_days_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Boolean N×N mask. True where:
        - Pair has at least min_overlap_days days of overlapping activity
        - Not on the diagonal
        """
        mask = overlap_days_matrix >= self.config.min_overlap_days
        np.fill_diagonal(mask, False)
        return mask

    def compute_mse(
        self,
        weights: np.ndarray,
        feature_matrices: list[np.ndarray],
        ground_truth: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Vectorized MSE for a given weight vector over valid pairs only."""
        predicted = sum(w * F for w, F in zip(weights, feature_matrices))
        diff = (predicted - ground_truth) ** 2
        valid = diff[mask]
        return float(valid.mean()) if len(valid) > 0 else float("inf")

    def run_baselines(
        self,
        feature_matrices: list[np.ndarray],
        ground_truth: np.ndarray,
        mask: np.ndarray,
        optimized_weights: np.ndarray,
    ) -> dict[str, float]:
        """Compute MSE for random, equal, price-only, linreg, and optimized weights."""
        rng = np.random.default_rng(seed=0)
        n_features = len(feature_matrices)

        # Random weights (average of 10 random draws for stability)
        random_mses = []
        for _ in range(10):
            w = rng.random(n_features)
            w /= w.sum()
            random_mses.append(self.compute_mse(w, feature_matrices, ground_truth, mask))
        random_mse = float(np.mean(random_mses))

        # Equal weights
        equal_w = np.full(n_features, 1.0 / n_features)
        equal_mse = self.compute_mse(equal_w, feature_matrices, ground_truth, mask)

        # Price-only (F1 only, index 0)
        price_only_w = np.zeros(n_features)
        price_only_w[0] = 1.0
        price_only_mse = self.compute_mse(
            price_only_w, feature_matrices, ground_truth, mask
        )

        # Linear regression (closed-form optimal)
        linreg_mse = self._linreg_mse(feature_matrices, ground_truth, mask)

        return {
            "random": random_mse,
            "equal": equal_mse,
            "price_only": price_only_mse,
            "linreg": linreg_mse,
            "optimized": self.compute_mse(
                optimized_weights, feature_matrices, ground_truth, mask
            ),
        }

    def _linreg_mse(
        self,
        feature_matrices: list[np.ndarray],
        ground_truth: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Closed-form OLS solution: w = (F^T F)^-1 F^T y."""
        # Build design matrix: (num_valid_pairs, n_features)
        F = np.column_stack([m[mask] for m in feature_matrices])
        y = ground_truth[mask]
        try:
            w, _, _, _ = np.linalg.lstsq(F, y, rcond=None)
            predicted = F @ w
            return float(np.mean((predicted - y) ** 2))
        except np.linalg.LinAlgError:
            return float("inf")

    def evaluate(
        self,
        feature_matrices: list[np.ndarray],
        ground_truth: np.ndarray,
        mask: np.ndarray,
        weights: np.ndarray,
        feature_names: list[str],
    ) -> EvaluationResult:
        """Full evaluation: MSE, Pearson r, and all baselines."""
        predicted = sum(w * F for w, F in zip(weights, feature_matrices))

        pred_valid = predicted[mask]
        truth_valid = ground_truth[mask]

        test_mse = float(np.mean((pred_valid - truth_valid) ** 2))

        if len(pred_valid) > 1 and np.std(pred_valid) > 0 and np.std(truth_valid) > 0:
            r, p = pearsonr(pred_valid, truth_valid)
        else:
            r, p = 0.0, 1.0

        baselines = self.run_baselines(
            feature_matrices, ground_truth, mask, weights
        )

        return EvaluationResult(
            test_mse=test_mse,
            test_pearson_r=float(r),
            test_pearson_p=float(p),
            baseline_mse_random=baselines["random"],
            baseline_mse_equal=baselines["equal"],
            baseline_mse_price_only=baselines["price_only"],
            baseline_mse_linreg=baselines["linreg"],
            weights=weights.tolist(),
            feature_names=feature_names,
        )
