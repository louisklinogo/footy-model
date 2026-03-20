"""Decision framework for market recommendations."""

from .types import LambdaContext, MarketScore, MarketRecommendation, FixtureDecision
from .patterns import (
    MarketPattern,
    ALL_PATTERNS,
    PATTERNS_BY_CATEGORY,
    find_pattern_match,
    find_best_pattern_by_market,
    get_patterns_for_market,
    pattern_to_dict,
)

# Framework module may not exist yet
try:
    from .framework import DecisionFramework
    _has_framework = True
except ImportError:
    _has_framework = False

__all__ = [
    "LambdaContext",
    "MarketScore",
    "MarketRecommendation",
    "FixtureDecision",
    # Patterns
    "MarketPattern",
    "ALL_PATTERNS",
    "PATTERNS_BY_CATEGORY",
    "find_pattern_match",
    "find_best_pattern_by_market",
    "get_patterns_for_market",
    "pattern_to_dict",
]

# Conditionally export DecisionFramework
if _has_framework:
    __all__.append("DecisionFramework")
