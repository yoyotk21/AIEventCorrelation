from features.base import BaseFeature, normalize
from features.granger_causality import GrangerCausalityFeature
from features.price_correlation import PriceCorrelationFeature
from features.resolution_proximity import ResolutionProximityFeature
from features.same_category import SameCategoryFeature
from features.tag_jaccard import TagJaccardFeature
from features.temporal_overlap import TemporalOverlapFeature
from features.volume_spike import VolumeSpikeFeature
from models import PipelineConfig


def get_all_features(config: PipelineConfig) -> list[BaseFeature]:
    """Instantiate all 7 features in canonical order (F1-F7)."""
    return [
        PriceCorrelationFeature(early_window_fraction=config.early_window_fraction),
        GrangerCausalityFeature(max_lag=config.granger_max_lag),
        VolumeSpikeFeature(sigma_threshold=config.spike_threshold_sigma),
        SameCategoryFeature(),
        TagJaccardFeature(),
        ResolutionProximityFeature(decay_days=config.resolution_decay_days),
        TemporalOverlapFeature(),
    ]


__all__ = [
    "BaseFeature",
    "normalize",
    "get_all_features",
    "PriceCorrelationFeature",
    "GrangerCausalityFeature",
    "VolumeSpikeFeature",
    "SameCategoryFeature",
    "TagJaccardFeature",
    "ResolutionProximityFeature",
    "TemporalOverlapFeature",
]
