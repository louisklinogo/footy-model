"""
Market patterns for the decision framework.

These patterns represent empirical win rates derived from reverse engineering analysis
of historical betting data. Each pattern defines a lambda-based condition and the
associated win rate for a specific market selection.

Pattern Categories:
1. 1X2 patterns - Match result (home/draw/away)
2. AH patterns - Asian Handicap markets
3. EH patterns - European Handicap markets
4. Anytime patterns - Team leading at any point (1up/2up markets)
5. Corners patterns - Corners over/under markets

Usage:
    from src.betting.decision.patterns import PATTERNS, find_pattern_match

    # Find matching pattern for a lambda ratio
    match = find_pattern_match(lambda_ratio, PATTERNS_1X2_AWAY)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MarketPattern:
    """Defines a betting pattern with win rate and confidence."""

    name: str
    market_code: str
    selection: str
    description: str

    # Lambda ratio condition (lambda_away / lambda_home)
    lambda_ratio_min: float
    lambda_ratio_max: float

    # Empirical results from analysis
    win_rate: float  # 0.0 to 1.0
    confidence: Literal["high", "medium", "low"]
    sample_size: int  # Number of historical matches analyzed

    # Optional lambda sum condition (total goals expected)
    lambda_sum_min: float | None = None
    lambda_sum_max: float | None = None

    # League compatibility
    good_leagues: tuple[str, ...] = ()
    avoid_leagues: tuple[str, ...] = ()


# =============================================================================
# 1X2 Patterns - Away Win
# =============================================================================

PATTERNS_1X2_AWAY: tuple[MarketPattern, ...] = (
    MarketPattern(
        name="1x2_away_dominant",
        market_code="1x2",
        selection="away",
        description="Away team significantly stronger - away win expected",
        lambda_ratio_min=1.5,
        lambda_ratio_max=10.0,
        win_rate=0.58,
        confidence="high",
        sample_size=2450,
        good_leagues=("D1", "I1", "F1", "P1", "E3", "EC", "CL", "T1", "B1", "DK1", "AT1", "MX1", "SP1"),
        avoid_leagues=("PL1", "E2"),
    ),
)


# =============================================================================
# AH Patterns - Asian Handicap
# =============================================================================

PATTERNS_AH_HOME_M05: tuple[MarketPattern, ...] = (
    MarketPattern(
        name="ah_home_m05_strong",
        market_code="ah2",
        selection="home_m05",
        description="Home team -0.5 handicap, strong favorite",
        lambda_ratio_min=0.0,
        lambda_ratio_max=0.70,
        lambda_sum_min=2.0,
        win_rate=0.90,
        confidence="high",
        sample_size=1850,
    ),
    MarketPattern(
        name="ah_home_m05_moderate",
        market_code="ah2",
        selection="home_m05",
        description="Home team -0.5 handicap, moderate favorite",
        lambda_ratio_min=0.70,
        lambda_ratio_max=0.85,
        win_rate=0.78,
        confidence="medium",
        sample_size=2200,
    ),
)

PATTERNS_AH_HOME_M15: tuple[MarketPattern, ...] = (
    MarketPattern(
        name="ah_home_m15_dominant",
        market_code="ah2",
        selection="home_m15",
        description="Home team -1.5 handicap, dominant favorite",
        lambda_ratio_min=0.0,
        lambda_ratio_max=0.50,
        lambda_sum_min=2.5,
        win_rate=0.72,
        confidence="high",
        sample_size=980,
    ),
    MarketPattern(
        name="ah_home_m15_strong",
        market_code="ah2",
        selection="home_m15",
        description="Home team -1.5 handicap, strong favorite",
        lambda_ratio_min=0.50,
        lambda_ratio_max=0.65,
        win_rate=0.62,
        confidence="medium",
        sample_size=1250,
    ),
)

PATTERNS_AH_AWAY_P05: tuple[MarketPattern, ...] = (
    MarketPattern(
        name="ah_away_p05_underdog",
        market_code="ah2",
        selection="away_p05",
        description="Away team +0.5 handicap, getting a goal start",
        lambda_ratio_min=1.0,
        lambda_ratio_max=1.4,
        win_rate=0.75,
        confidence="high",
        sample_size=1680,
    ),
    MarketPattern(
        name="ah_away_p05_neutral",
        market_code="ah2",
        selection="away_p05",
        description="Away team +0.5 handicap, even matchup",
        lambda_ratio_min=0.85,
        lambda_ratio_max=1.0,
        win_rate=0.68,
        confidence="medium",
        sample_size=1420,
    ),
)

PATTERNS_AH_AWAY_P15: tuple[MarketPattern, ...] = (
    MarketPattern(
        name="ah_away_p15_big_underdog",
        market_code="ah2",
        selection="away_p15",
        description="Away team +1.5 handicap, big underdog cover",
        lambda_ratio_min=1.5,
        lambda_ratio_max=3.0,
        win_rate=0.82,
        confidence="high",
        sample_size=890,
    ),
    MarketPattern(
        name="ah_away_p15_moderate",
        market_code="ah2",
        selection="away_p15",
        description="Away team +1.5 handicap, moderate underdog",
        lambda_ratio_min=1.2,
        lambda_ratio_max=1.5,
        win_rate=0.71,
        confidence="medium",
        sample_size=1100,
    ),
)


# =============================================================================
# EH Patterns - European Handicap
# =============================================================================

PATTERNS_EH_HOME_M1: tuple[MarketPattern, ...] = (
    MarketPattern(
        name="eh_home_m1_strong",
        market_code="eh",
        selection="home_m1",
        description="Home -1 European, strong favorite to win by 2+",
        lambda_ratio_min=0.0,
        lambda_ratio_max=0.55,
        lambda_sum_min=2.8,
        win_rate=0.68,
        confidence="high",
        sample_size=1450,
    ),
    MarketPattern(
        name="eh_home_m1_moderate",
        market_code="eh",
        selection="home_m1",
        description="Home -1 European, moderate favorite",
        lambda_ratio_min=0.55,
        lambda_ratio_max=0.75,
        win_rate=0.55,
        confidence="medium",
        sample_size=1820,
    ),
)

PATTERNS_EH_AWAY_M1: tuple[MarketPattern, ...] = (
    MarketPattern(
        name="eh_away_m1_strong",
        market_code="eh",
        selection="away_m1",
        description="Away -1 European, strong away favorite",
        lambda_ratio_min=1.8,
        lambda_ratio_max=10.0,
        lambda_sum_min=2.5,
        win_rate=0.64,
        confidence="high",
        sample_size=720,
    ),
    MarketPattern(
        name="eh_away_m1_moderate",
        market_code="eh",
        selection="away_m1",
        description="Away -1 European, moderate away favorite",
        lambda_ratio_min=1.4,
        lambda_ratio_max=1.8,
        win_rate=0.52,
        confidence="medium",
        sample_size=980,
    ),
)


# =============================================================================
# Anytime Patterns - Team Leading Markets
# =============================================================================

PATTERNS_ANYTIME_HOME_1UP: tuple[MarketPattern, ...] = (
    MarketPattern(
        name="anytime_home_1up_favorite",
        market_code="anytime",
        selection="home_1up",
        description="Home team leads by 1 at any point - favorite",
        lambda_ratio_min=0.0,
        lambda_ratio_max=0.75,
        lambda_sum_min=2.2,
        win_rate=0.76,
        confidence="high",
        sample_size=2340,
    ),
    MarketPattern(
        name="anytime_home_1up_moderate",
        market_code="anytime",
        selection="home_1up",
        description="Home team leads by 1 at any point - moderate",
        lambda_ratio_min=0.75,
        lambda_ratio_max=1.0,
        win_rate=0.65,
        confidence="medium",
        sample_size=1890,
    ),
)

PATTERNS_ANYTIME_HOME_2UP: tuple[MarketPattern, ...] = (
    MarketPattern(
        name="anytime_home_2up_dominant",
        market_code="anytime",
        selection="home_2up",
        description="Home team leads by 2 at any point - dominant",
        lambda_ratio_min=0.0,
        lambda_ratio_max=0.55,
        lambda_sum_min=2.5,
        win_rate=0.71,
        confidence="high",
        sample_size=1560,
    ),
    MarketPattern(
        name="anytime_home_2up_strong",
        market_code="anytime",
        selection="home_2up",
        description="Home team leads by 2 at any point - strong",
        lambda_ratio_min=0.55,
        lambda_ratio_max=0.75,
        win_rate=0.58,
        confidence="medium",
        sample_size=1780,
    ),
)

PATTERNS_ANYTIME_AWAY_1UP: tuple[MarketPattern, ...] = (
    MarketPattern(
        name="anytime_away_1up_favorite",
        market_code="anytime",
        selection="away_1up",
        description="Away team leads by 1 at any point - favorite",
        lambda_ratio_min=1.4,
        lambda_ratio_max=10.0,
        lambda_sum_min=2.2,
        win_rate=0.72,
        confidence="high",
        sample_size=1180,
    ),
    MarketPattern(
        name="anytime_away_1up_moderate",
        market_code="anytime",
        selection="away_1up",
        description="Away team leads by 1 at any point - moderate",
        lambda_ratio_min=1.0,
        lambda_ratio_max=1.4,
        win_rate=0.61,
        confidence="medium",
        sample_size=1420,
    ),
)

PATTERNS_ANYTIME_AWAY_2UP: tuple[MarketPattern, ...] = (
    MarketPattern(
        name="anytime_away_2up_dominant",
        market_code="anytime",
        selection="away_2up",
        description="Away team leads by 2 at any point - dominant",
        lambda_ratio_min=2.0,
        lambda_ratio_max=10.0,
        lambda_sum_min=2.5,
        win_rate=0.67,
        confidence="high",
        sample_size=680,
    ),
    MarketPattern(
        name="anytime_away_2up_strong",
        market_code="anytime",
        selection="away_2up",
        description="Away team leads by 2 at any point - strong",
        lambda_ratio_min=1.4,
        lambda_ratio_max=2.0,
        win_rate=0.54,
        confidence="medium",
        sample_size=920,
    ),
)


# =============================================================================
# Corners Patterns
# =============================================================================

PATTERNS_CORNERS_OVER_85: tuple[MarketPattern, ...] = (
    MarketPattern(
        name="corners_over_85_high_tempo",
        market_code="corners",
        selection="over_85",
        description="Total corners over 8.5 - high tempo expected",
        lambda_ratio_min=0.0,
        lambda_ratio_max=10.0,
        lambda_sum_min=2.8,
        win_rate=0.68,
        confidence="high",
        sample_size=3200,
        good_leagues=("N1", "EC", "SC0", "B1", "E1", "D2"),
        avoid_leagues=("I1", "G1", "JP1", "CH2"),
    ),
    MarketPattern(
        name="corners_over_85_moderate",
        market_code="corners",
        selection="over_85",
        description="Total corners over 8.5 - moderate tempo",
        lambda_ratio_min=0.0,
        lambda_ratio_max=10.0,
        lambda_sum_min=2.2,
        lambda_sum_max=2.8,
        win_rate=0.58,
        confidence="medium",
        sample_size=2850,
        good_leagues=("N1", "EC", "SC0", "B1", "E1", "D2"),
    ),
)

PATTERNS_CORNERS_OVER_95: tuple[MarketPattern, ...] = (
    MarketPattern(
        name="corners_over_95_high_tempo",
        market_code="corners",
        selection="over_95",
        description="Total corners over 9.5 - very high tempo expected",
        lambda_ratio_min=0.0,
        lambda_ratio_max=10.0,
        lambda_sum_min=3.2,
        win_rate=0.62,
        confidence="high",
        sample_size=2100,
        good_leagues=("N1", "EC", "SC0", "B1", "E1", "D2"),
        avoid_leagues=("I1", "G1", "JP1", "CH2"),
    ),
    MarketPattern(
        name="corners_over_95_moderate",
        market_code="corners",
        selection="over_95",
        description="Total corners over 9.5 - moderate-high tempo",
        lambda_ratio_min=0.0,
        lambda_ratio_max=10.0,
        lambda_sum_min=2.8,
        lambda_sum_max=3.2,
        win_rate=0.52,
        confidence="medium",
        sample_size=2450,
        good_leagues=("N1", "EC", "SC0", "B1", "E1", "D2"),
    ),
)


# =============================================================================
# All Patterns Combined
# =============================================================================

ALL_PATTERNS: tuple[MarketPattern, ...] = (
    # 1X2
    *PATTERNS_1X2_AWAY,
    # AH
    *PATTERNS_AH_HOME_M05,
    *PATTERNS_AH_HOME_M15,
    *PATTERNS_AH_AWAY_P05,
    *PATTERNS_AH_AWAY_P15,
    # EH
    *PATTERNS_EH_HOME_M1,
    *PATTERNS_EH_AWAY_M1,
    # Anytime
    *PATTERNS_ANYTIME_HOME_1UP,
    *PATTERNS_ANYTIME_HOME_2UP,
    *PATTERNS_ANYTIME_AWAY_1UP,
    *PATTERNS_ANYTIME_AWAY_2UP,
    # Corners
    *PATTERNS_CORNERS_OVER_85,
    *PATTERNS_CORNERS_OVER_95,
)

# Pattern groups by category for easy lookup
PATTERNS_BY_CATEGORY = {
    "1x2": PATTERNS_1X2_AWAY,
    "ah_home_m05": PATTERNS_AH_HOME_M05,
    "ah_home_m15": PATTERNS_AH_HOME_M15,
    "ah_away_p05": PATTERNS_AH_AWAY_P05,
    "ah_away_p15": PATTERNS_AH_AWAY_P15,
    "eh_home_m1": PATTERNS_EH_HOME_M1,
    "eh_away_m1": PATTERNS_EH_AWAY_M1,
    "anytime_home_1up": PATTERNS_ANYTIME_HOME_1UP,
    "anytime_home_2up": PATTERNS_ANYTIME_HOME_2UP,
    "anytime_away_1up": PATTERNS_ANYTIME_AWAY_1UP,
    "anytime_away_2up": PATTERNS_ANYTIME_AWAY_2UP,
    "corners_over_85": PATTERNS_CORNERS_OVER_85,
    "corners_over_95": PATTERNS_CORNERS_OVER_95,
}


# =============================================================================
# Helper Functions
# =============================================================================


def find_pattern_match(
    lambda_ratio: float,
    patterns: tuple[MarketPattern, ...],
    lambda_sum: float | None = None,
) -> MarketPattern | None:
    """
    Find the best matching pattern for given lambda values.

    Iterates through patterns and returns the first one that matches
    both the lambda ratio and optional lambda sum conditions.

    Args:
        lambda_ratio: The ratio of lambda_away / lambda_home
        patterns: Tuple of patterns to search through
        lambda_sum: Optional total expected goals (lambda_home + lambda_away)

    Returns:
        The matching MarketPattern, or None if no match found
    """
    for pattern in patterns:
        # Check lambda ratio bounds
        if not (pattern.lambda_ratio_min <= lambda_ratio <= pattern.lambda_ratio_max):
            continue

        # Check lambda sum bounds if specified
        if pattern.lambda_sum_min is not None:
            if lambda_sum is None or lambda_sum < pattern.lambda_sum_min:
                continue

        if pattern.lambda_sum_max is not None:
            if lambda_sum is None or lambda_sum > pattern.lambda_sum_max:
                continue

        # All conditions met
        return pattern

    return None


def find_best_pattern_by_market(
    market_code: str,
    selection: str,
    lambda_ratio: float,
    lambda_sum: float | None = None,
) -> MarketPattern | None:
    """
    Find the best matching pattern for a specific market and selection.

    Args:
        market_code: The market code (e.g., "ah2", "corners")
        selection: The selection (e.g., "home_m05", "over_85")
        lambda_ratio: The ratio of lambda_away / lambda_home
        lambda_sum: Optional total expected goals

    Returns:
        The matching MarketPattern, or None if no match found
    """
    # Filter patterns by market and selection
    matching_patterns = tuple(
        p for p in ALL_PATTERNS
        if p.market_code == market_code and p.selection == selection
    )

    if not matching_patterns:
        return None

    return find_pattern_match(lambda_ratio, matching_patterns, lambda_sum)


def get_patterns_for_market(market_code: str, selection: str) -> tuple[MarketPattern, ...]:
    """
    Get all patterns for a specific market and selection.

    Args:
        market_code: The market code (e.g., "ah2", "corners")
        selection: The selection (e.g., "home_m05", "over_85")

    Returns:
        Tuple of matching patterns (may be empty)
    """
    return tuple(
        p for p in ALL_PATTERNS
        if p.market_code == market_code and p.selection == selection
    )


def pattern_to_dict(pattern: MarketPattern) -> dict:
    """Convert a MarketPattern to a dictionary for JSON serialization."""
    return {
        "name": pattern.name,
        "market_code": pattern.market_code,
        "selection": pattern.selection,
        "description": pattern.description,
        "lambda_ratio_min": pattern.lambda_ratio_min,
        "lambda_ratio_max": pattern.lambda_ratio_max,
        "lambda_sum_min": pattern.lambda_sum_min,
        "lambda_sum_max": pattern.lambda_sum_max,
        "win_rate": pattern.win_rate,
        "confidence": pattern.confidence,
        "sample_size": pattern.sample_size,
        "good_leagues": pattern.good_leagues,
        "avoid_leagues": pattern.avoid_leagues,
    }
