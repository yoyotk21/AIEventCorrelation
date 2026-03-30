import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from data_loader import DataBundle
from features.base import BaseFeature


class PriceCorrelationFeature(BaseFeature):
    """
    F1: Absolute Pearson correlation of daily prices during the first 40% of
    the two markets' overlapping trading period.

    Uses only the early window to avoid leaking information about what we're
    trying to predict (whether prices co-move toward resolution).
    """

    def __init__(self, early_window_fraction: float = 0.4):
        self.early_window_fraction = early_window_fraction

    @property
    def name(self) -> str:
        return "price_correlation"

    @property
    def is_symmetric(self) -> bool:
        return True

    def compute(self, data: DataBundle) -> np.ndarray:
        # Build a price series per market: keep only the "Yes" outcome token
        # (or the first token if outcomes aren't labeled), one price per day.
        price_series = self._build_price_series(data)

        n = len(data.markets)
        matrix = np.zeros((n, n))

        for i in range(n):
            mid_i = data.market_ids[i]
            if mid_i not in price_series:
                continue
            for j in range(i + 1, n):
                mid_j = data.market_ids[j]
                if mid_j not in price_series:
                    continue

                score = self._compute_pair(price_series[mid_i], price_series[mid_j])
                matrix[i][j] = score
                matrix[j][i] = score

        return matrix

    def _build_price_series(self, data: DataBundle) -> dict[str, pd.Series]:
        """
        For each market, extract a date-indexed price Series for the primary
        outcome token (Yes token, or first token if no Yes outcome exists).
        Returns a dict: market_id -> pd.Series(index=date_str, values=price).
        """
        df = data.prices_df.copy()

        # Prefer the "Yes" outcome; fall back to the first available token per market
        yes_rows = df[df["outcome"] == "Yes"]
        other_rows = df[df["outcome"] != "Yes"]

        # Markets that have a Yes token
        has_yes = set(yes_rows["market_id"].unique())

        # For markets without a Yes token, take the first token alphabetically
        first_token = (
            other_rows[~other_rows["market_id"].isin(has_yes)]
            .sort_values("token_id")
            .groupby("market_id")["token_id"]
            .first()
        )

        price_series: dict[str, pd.Series] = {}

        # Process Yes-token markets
        for mid, group in yes_rows.groupby("market_id"):
            series = group.set_index("date_utc")["price"].sort_index()
            price_series[str(mid)] = series

        # Process fallback markets
        for mid, token_id in first_token.items():
            rows = other_rows[
                (other_rows["market_id"] == mid) & (other_rows["token_id"] == token_id)
            ]
            series = rows.set_index("date_utc")["price"].sort_index()
            price_series[str(mid)] = series

        return price_series

    def _compute_pair(self, s_i: pd.Series, s_j: pd.Series) -> float:
        """
        Compute absolute Pearson r for the early window of the overlapping period.
        Returns 0.0 if there are fewer than 3 overlapping points.
        """
        overlap_dates = s_i.index.intersection(s_j.index)
        if len(overlap_dates) < 3:
            return 0.0

        # Take only the first early_window_fraction of the overlap
        split = max(2, int(len(overlap_dates) * self.early_window_fraction))
        early_dates = overlap_dates[:split]

        a = s_i[early_dates].values.astype(float)
        b = s_j[early_dates].values.astype(float)

        # Drop any NaN pairs
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        if len(a) < 2:
            return 0.0

        # Degenerate case: constant series
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0

        r, _ = pearsonr(a, b)
        return float(abs(r))
