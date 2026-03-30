import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from data_loader import DataBundle
from features.base import BaseFeature
from features.price_correlation import PriceCorrelationFeature


class GrangerCausalityFeature(BaseFeature):
    """
    F2: Granger causality from market i to market j.

    Tests whether knowing market i's past prices improves prediction of
    market j's future prices beyond what j's own history provides.
    Asymmetric: matrix[i][j] != matrix[j][i] in general.

    Uses the best (highest) F-statistic across all tested lags.
    Pairs with insufficient overlapping data are scored 0.
    """

    def __init__(self, max_lag: int = 3):
        self.max_lag = max_lag

    @property
    def name(self) -> str:
        return "granger_causality"

    @property
    def is_symmetric(self) -> bool:
        return False

    def compute(self, data: DataBundle) -> np.ndarray:
        # Reuse the price series builder from PriceCorrelationFeature
        price_series = PriceCorrelationFeature()._build_price_series(data)

        n = len(data.markets)
        matrix = np.zeros((n, n))
        min_obs = self.max_lag + 2

        for i in range(n):
            mid_i = data.market_ids[i]
            if mid_i not in price_series:
                continue
            for j in range(n):
                if i == j:
                    continue
                mid_j = data.market_ids[j]
                if mid_j not in price_series:
                    continue

                score = self._compute_pair(
                    price_series[mid_i], price_series[mid_j], min_obs
                )
                matrix[i][j] = score

        return matrix

    def _compute_pair(
        self, s_i: pd.Series, s_j: pd.Series, min_obs: int
    ) -> float:
        """
        Run Granger causality test for i -> j direction.
        Returns the best F-statistic across all lags, or 0 on failure.
        """
        overlap = s_i.index.intersection(s_j.index)
        if len(overlap) < min_obs:
            return 0.0

        a = s_i[overlap].values.astype(float)
        b = s_j[overlap].values.astype(float)

        # Drop NaN pairs
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        if len(a) < min_obs:
            return 0.0

        # Constant series causes degenerate regression
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0

        # grangercausalitytests expects a 2-column array: [dependent, cause]
        # i.e., [j, i] to test "does i Granger-cause j?"
        data_array = np.column_stack([b, a])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = grangercausalitytests(
                    data_array, maxlag=self.max_lag, verbose=False
                )
            # Extract best (max) F-statistic across all lags
            f_stats = [
                results[lag][0]["ssr_ftest"][0]
                for lag in range(1, self.max_lag + 1)
                if lag in results
            ]
            if not f_stats:
                return 0.0
            return float(max(f_stats))
        except Exception:
            return 0.0
