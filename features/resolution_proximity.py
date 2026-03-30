import numpy as np

from data_loader import DataBundle
from features.base import BaseFeature


class ResolutionProximityFeature(BaseFeature):
    """
    F6: Exponential decay based on how many days apart two markets resolved.
    score = exp(-|days_between| / decay_days). Already in [0, 1].
    """

    def __init__(self, decay_days: float = 30.0):
        self.decay_days = decay_days

    @property
    def name(self) -> str:
        return "resolution_proximity"

    @property
    def is_symmetric(self) -> bool:
        return True

    def compute(self, data: DataBundle) -> np.ndarray:
        # Convert resolution datetimes to days-since-epoch (float), NaN if missing
        timestamps = np.array([
            m.resolved_at().timestamp() / 86400.0 if m.resolved_at() is not None else np.nan
            for m in data.markets
        ])

        diff = np.abs(timestamps[:, None] - timestamps[None, :])
        matrix = np.exp(-diff / self.decay_days)
        matrix = np.nan_to_num(matrix, nan=0.0)
        np.fill_diagonal(matrix, 0.0)
        return matrix
