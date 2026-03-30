import numpy as np
import pandas as pd

from data_loader import DataBundle
from models import Market


class MispricingAnalyzer:
    """
    Identifies and quantifies mispriced parlays.

    A parlay combining markets A and B is priced at P(A) * P(B) if the market
    treats them as independent. If our predictivity matrix says they're
    correlated, the true joint probability is higher — the parlay is underpriced.
    """

    def compute_mispricing_table(
        self,
        data: DataBundle,
        predictivity_matrix: np.ndarray,
        test_indices: list[int],
        ground_truth: np.ndarray,
        mask: np.ndarray,
    ) -> pd.DataFrame:
        """
        For each valid test pair, compute:
        - predicted_score: M[i][j] from the predictivity matrix
        - naive_parlay: P(A) * P(B) using pre-resolution market prices
        - actual_coresolution: ground_truth[i][j]
        - mispricing: actual_coresolution - naive_parlay

        Returns a DataFrame sorted by predicted_score descending.
        """
        test_markets = [data.markets[i] for i in test_indices]
        n = len(test_markets)

        rows = []
        for i in range(n):
            market_i = test_markets[i]
            p_i = self._get_yes_price(market_i)

            for j in range(n):
                if i == j or not mask[i][j]:
                    continue

                market_j = test_markets[j]
                p_j = self._get_yes_price(market_j)

                naive_parlay = (p_i * p_j) if (p_i is not None and p_j is not None) else None
                actual = ground_truth[i][j]
                predicted = float(predictivity_matrix[i][j])

                rows.append({
                    "market_i_id": market_i.id,
                    "market_i_question": market_i.question[:60],
                    "market_j_id": market_j.id,
                    "market_j_question": market_j.question[:60],
                    "predicted_score": predicted,
                    "naive_parlay": naive_parlay,
                    "actual_coresolution": float(actual),
                    "mispricing": (
                        float(actual) - naive_parlay
                        if naive_parlay is not None
                        else None
                    ),
                })

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df.sort_values("predicted_score", ascending=False).reset_index(drop=True)

    def stratified_analysis(
        self,
        mispricing_df: pd.DataFrame,
        quantiles: list[float] = [0.10, 0.25, 0.50, 1.0],
    ) -> pd.DataFrame:
        """
        Group pairs by predicted score quantile and compute average naive parlay
        price, actual co-resolution rate, and gap (mispricing) per group.

        This is the key result table: does higher predicted correlation
        correspond to a larger gap between actual outcomes and naive pricing?
        """
        df = mispricing_df.dropna(subset=["naive_parlay"]).copy()
        if df.empty:
            return pd.DataFrame()

        df = df.sort_values("predicted_score", ascending=False).reset_index(drop=True)
        n = len(df)

        rows = []
        prev_end = 0
        for q in quantiles:
            end = int(np.ceil(n * q))
            subset = df.iloc[prev_end:end]
            if subset.empty:
                prev_end = end
                continue
            rows.append({
                "quantile_top": f"Top {int(q * 100)}%",
                "n_pairs": len(subset),
                "avg_predicted_score": subset["predicted_score"].mean(),
                "avg_naive_parlay": subset["naive_parlay"].mean(),
                "actual_coresolution_rate": subset["actual_coresolution"].mean(),
                "avg_mispricing": subset["mispricing"].mean(),
            })
            prev_end = end

        return pd.DataFrame(rows)

    def _get_yes_price(self, market: Market) -> float | None:
        """
        Returns the implied probability (Yes outcome price) for a market,
        taken from the market's outcome_prices at the time of data collection.
        Returns None if unavailable.
        """
        if not market.outcomes or not market.outcome_prices:
            return None
        try:
            yes_idx = market.outcomes.index("Yes")
            price = market.outcome_prices[yes_idx]
            # At resolution, prices snap to 0 or 1 — try to get a mid-life price.
            # For now we use the stored price; a future improvement would be to
            # pull a price from prices_daily.csv at the midpoint of the market's life.
            return float(price) if 0.0 < price < 1.0 else None
        except (ValueError, IndexError):
            return None
