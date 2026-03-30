import numpy as np
import pandas as pd

from data_loader import DataBundle
from features.base import BaseFeature


class VolumeSpikeFeature(BaseFeature):
    """
    F3: Asymmetric spike co-occurrence.

    A "spike day" for market i is any day where daily_volume > mean + sigma*std.
    matrix[i][j] = |spike_days_i ∩ spike_days_j| / |spike_days_i|

    Measures: when market i has a busy day, does market j also tend to be busy?
    Asymmetric because the denominator is always relative to market i.
    """

    def __init__(self, sigma_threshold: float = 2.0):
        self.sigma_threshold = sigma_threshold

    @property
    def name(self) -> str:
        return "volume_spike"

    @property
    def is_symmetric(self) -> bool:
        return False

    def compute(self, data: DataBundle) -> np.ndarray:
        spike_days = self._compute_spike_days(data)

        n = len(data.markets)
        matrix = np.zeros((n, n))

        for i in range(n):
            mid_i = data.market_ids[i]
            spikes_i = spike_days.get(mid_i, set())
            if not spikes_i:
                continue
            for j in range(n):
                if i == j:
                    continue
                mid_j = data.market_ids[j]
                spikes_j = spike_days.get(mid_j, set())
                if not spikes_j:
                    continue
                matrix[i][j] = len(spikes_i & spikes_j) / len(spikes_i)

        return matrix

    def _compute_spike_days(self, data: DataBundle) -> dict[str, set[str]]:
        """
        For each market, find the set of date strings where volume spiked.
        Returns dict: market_id -> set of date_utc strings.
        """
        spike_days: dict[str, set[str]] = {}

        for mid, group in data.trades_df.groupby("market_id"):
            volumes = group["daily_volume"].values.astype(float)
            if len(volumes) < 3:
                continue
            mean = volumes.mean()
            std = volumes.std()
            if std == 0:
                continue
            threshold = mean + self.sigma_threshold * std
            spike_mask = volumes > threshold
            spike_dates = set(group["date_utc"].values[spike_mask])
            if spike_dates:
                spike_days[str(mid)] = spike_dates

        return spike_days
