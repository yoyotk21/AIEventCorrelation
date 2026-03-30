import numpy as np

from data_loader import DataBundle
from features.base import BaseFeature


class SameCategoryFeature(BaseFeature):
    """
    F4: Binary flag — 1.0 if both markets share the same Polymarket category,
    0.0 otherwise. Already in [0,1], normalization is a no-op.
    """

    @property
    def name(self) -> str:
        return "same_category"

    @property
    def is_symmetric(self) -> bool:
        return True

    def compute(self, data: DataBundle) -> np.ndarray:
        categories = np.array(
            [m.category if m.category is not None else "" for m in data.markets],
            dtype=object,
        )
        matrix = (categories[:, None] == categories[None, :]).astype(float)
        # Markets with no category should not match each other
        no_cat = categories == ""
        matrix[no_cat[:, None] & no_cat[None, :]] = 0.0
        np.fill_diagonal(matrix, 0.0)
        return matrix
