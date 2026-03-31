import json
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Market(BaseModel):
    id: str
    condition_id: str
    slug: str
    question: str
    category: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    closed_time: Optional[datetime] = None
    volume: float = 0.0
    liquidity: float = 0.0
    tags: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)
    outcome_prices: list[float] = Field(default_factory=list)
    clob_token_ids: list[str] = Field(default_factory=list)

    @field_validator("tags", "outcomes", "clob_token_ids", mode="before")
    @classmethod
    def parse_json_list(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            try:
                parsed = json.loads(s)
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                return []
        return []

    @field_validator("outcome_prices", mode="before")
    @classmethod
    def parse_outcome_prices(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [float(x) for x in v]
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [float(x) for x in parsed]
            except (json.JSONDecodeError, ValueError):
                return []
        return []

    @field_validator("start_date", "end_date", "closed_time", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        if v is None or (isinstance(v, float) and v != v):  # NaN check
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    def resolved_at(self) -> Optional[datetime]:
        """Returns the actual resolution time, falling back to end_date."""
        return self.closed_time or self.end_date

    def duration_days(self) -> int:
        """Number of days the market was active."""
        start = self.start_date
        end = self.resolved_at()
        if start is None or end is None:
            return 0
        return max(0, (end - start).days)

    def resolved_yes(self) -> Optional[bool]:
        """
        Returns True if the market resolved Yes, False if No, None if unknown.
        Checks the first outcome price: 1.0 = Yes resolved, 0.0 = No resolved.
        """
        if not self.outcome_prices or not self.outcomes:
            return None
        try:
            yes_idx = self.outcomes.index("Yes")
            price = self.outcome_prices[yes_idx]
            if price >= 0.99:
                return True
            if price <= 0.01:
                return False
        except (ValueError, IndexError):
            pass
        return None


class PipelineConfig(BaseModel):
    data_dir: str = "."
    train_cutoff: str = "2025-06-01"
    min_overlap_days: int = 7
    early_window_fraction: float = 0.4
    spike_threshold_sigma: float = 2.0
    granger_max_lag: int = 3
    resolution_decay_days: float = 30.0
    # Hill climbing
    num_restarts: int = 50
    max_iterations: int = 5000
    initial_temperature: float = 1.0
    decay_rate: float = 0.995
    perturbation_size: float = 0.05
    random_seed: Optional[int] = 42


class OptimizationResult(BaseModel):
    weights: list[float]
    feature_names: list[str]
    train_mse: float
    best_restart: int

    model_config = {"arbitrary_types_allowed": True}


class EvaluationResult(BaseModel):
    test_mse: float
    test_pearson_r: float
    test_pearson_p: float
    baseline_mse_random: float
    baseline_mse_equal: float
    baseline_mse_price_only: float
    baseline_mse_linreg: float
    weights: list[float]
    feature_names: list[str]
