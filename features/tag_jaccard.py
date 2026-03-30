import numpy as np

from data_loader import DataBundle
from features.base import BaseFeature


class TagJaccardFeature(BaseFeature):
    """
    F5: Jaccard similarity of the two markets' tag sets.
    |A ∩ B| / |A ∪ B|. Already in [0, 1].
    """

    @property
    def name(self) -> str:
        return "tag_jaccard"

    @property
    def is_symmetric(self) -> bool:
        return True

    def compute(self, data: DataBundle) -> np.ndarray:
        tag_sets = [set(m.tags) for m in data.markets]
        n = len(tag_sets)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                union_size = len(tag_sets[i] | tag_sets[j])
                if union_size == 0:
                    continue
                score = len(tag_sets[i] & tag_sets[j]) / union_size
                matrix[i][j] = score
                matrix[j][i] = score

        return matrix
