from abc import ABC, abstractmethod

import numpy as np

from data_loader import DataBundle


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Min-max normalize a matrix to [0, 1]. Returns zeros if range is zero."""
    mn, mx = matrix.min(), matrix.max()
    if mx - mn == 0:
        return np.zeros_like(matrix)
    return (matrix - mn) / (mx - mn)


class BaseFeature(ABC):
    """
    Abstract base for all 7 features. Each subclass computes a full N×N matrix
    where cell [i][j] scores the relationship from market i to market j.

    compute() returns raw values — normalization is applied by the pipeline
    after all features have run.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable feature name."""
        ...

    @property
    @abstractmethod
    def is_symmetric(self) -> bool:
        """
        True if matrix[i][j] == matrix[j][i] for all pairs.
        Asymmetric features: F2 (Granger causality), F3 (volume spike).
        """
        ...

    @abstractmethod
    def compute(self, data: DataBundle) -> np.ndarray:
        """
        Compute the raw N×N feature matrix.

        Args:
            data: DataBundle containing markets, prices_df, trades_df.

        Returns:
            np.ndarray of shape (N, N) where N = len(data.markets).
            Diagonal must be 0. Values are raw (not yet normalized to [0,1]).
        """
        ...
