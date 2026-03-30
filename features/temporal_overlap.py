import numpy as np

from data_loader import DataBundle
from features.base import BaseFeature


class TemporalOverlapFeature(BaseFeature):
    """
    F7: Fraction of the shorter market's duration that overlapped with the other.
    overlap_days / min(duration_i, duration_j). Already in [0, 1].

    Also used by the pipeline to build the valid-pair mask (min overlap filter).
    The raw overlap_days matrix is available via compute_overlap_days().
    """

    @property
    def name(self) -> str:
        return "temporal_overlap"

    @property
    def is_symmetric(self) -> bool:
        return True

    def compute(self, data: DataBundle) -> np.ndarray:
        overlap_days, min_durations = self._compute_overlap_and_durations(data)
        safe_min = np.maximum(min_durations, 1.0)
        matrix = overlap_days / safe_min
        np.fill_diagonal(matrix, 0.0)
        return matrix

    def compute_overlap_days(self, data: DataBundle) -> np.ndarray:
        """Returns the raw N×N overlap-in-days matrix (used for pair masking)."""
        overlap_days, _ = self._compute_overlap_and_durations(data)
        np.fill_diagonal(overlap_days, 0.0)
        return overlap_days

    def _compute_overlap_and_durations(
        self, data: DataBundle
    ) -> tuple[np.ndarray, np.ndarray]:
        starts = np.array([
            m.start_date.timestamp() / 86400.0 if m.start_date is not None else np.nan
            for m in data.markets
        ])
        ends = np.array([
            m.resolved_at().timestamp() / 86400.0 if m.resolved_at() is not None else np.nan
            for m in data.markets
        ])
        durations = ends - starts

        overlap_start = np.maximum(starts[:, None], starts[None, :])
        overlap_end = np.minimum(ends[:, None], ends[None, :])
        overlap_days = np.maximum(0.0, overlap_end - overlap_start)
        overlap_days = np.nan_to_num(overlap_days, nan=0.0)

        min_durations = np.minimum(durations[:, None], durations[None, :])
        min_durations = np.nan_to_num(min_durations, nan=0.0)

        return overlap_days, min_durations
