"""Data types for the decision framework."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class LambdaContext:
    """Lambda values for a fixture."""
    lambda_home: float
    lambda_away: float
    
    @property
    def lambda_ratio(self) -> float:
        """Ratio of away/home lambda (higher = away team stronger)."""
        return self.lambda_away / self.lambda_home if self.lambda_home > 0 else 0.0
    
    @property
    def lambda_sum(self) -> float:
        """Sum of home + away lambda (higher = more goals expected)."""
        return self.lambda_home + self.lambda_away


@dataclass(frozen=True)
class MarketScore:
    """Score for a single market."""
    market_code: str
    selection: str
    win_rate: float
    confidence: Literal["high", "medium", "low"]
    league_factor: float  # 1.0 = good league, 0.5 = neutral, 0.0 = avoid
    pattern_match: str  # Which pattern matched
    score: float  # Overall score (0-1)


@dataclass(frozen=True)
class MarketRecommendation:
    """Final recommendation for a market."""
    market_code: str
    selection: str
    score: float
    win_rate: float
    confidence: str
    reason: str
    league_compatible: bool


@dataclass(frozen=True)
class FixtureDecision:
    """Complete decision output for a fixture."""
    fixture_id: int
    home_team: str
    away_team: str
    league_code: str
    league_name: str
    match_datetime_utc: datetime | None
    lambda_ctx: LambdaContext
    recommendations: tuple[MarketRecommendation, ...]
    top_pick: MarketRecommendation | None
    avoid_markets: tuple[str, ...]
