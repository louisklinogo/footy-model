"""
Market scorer functions for the decision framework.

Each scorer function evaluates a specific market using lambda-based patterns
and league compatibility factors. Returns a MarketScore if a pattern matches,
or None if no suitable pattern is found.

Score calculation:
    score = win_rate * confidence_weight * league_factor

    where:
        - win_rate: Empirical win rate from pattern analysis
        - confidence_weight: high=1.0, medium=0.7, low=0.4
        - league_factor: 1.0 (good), 0.5 (neutral), 0.0 (avoid)
"""

from __future__ import annotations

from .league_filters import (
    get_league_factor_1x2_away,
    get_league_factor_corners,
)
from .patterns import (
    MarketPattern,
    PATTERNS_1X2_AWAY,
    PATTERNS_AH_HOME_M05,
    PATTERNS_AH_HOME_M15,
    PATTERNS_AH_AWAY_P05,
    PATTERNS_AH_AWAY_P15,
    PATTERNS_EH_HOME_M1,
    PATTERNS_EH_AWAY_M1,
    PATTERNS_ANYTIME_HOME_1UP,
    PATTERNS_ANYTIME_HOME_2UP,
    PATTERNS_ANYTIME_AWAY_1UP,
    PATTERNS_ANYTIME_AWAY_2UP,
    PATTERNS_CORNERS_OVER_85,
    PATTERNS_CORNERS_OVER_95,
    find_pattern_match,
)
from .types import LambdaContext, MarketScore


# =============================================================================
# Confidence Weight Mapping
# =============================================================================

CONFIDENCE_WEIGHTS: dict[str, float] = {
    "high": 1.0,
    "medium": 0.7,
    "low": 0.4,
}


# =============================================================================
# Helper Functions
# =============================================================================


def _calculate_score(
    win_rate: float,
    confidence: str,
    league_factor: float,
) -> float:
    """
    Calculate the overall score for a market recommendation.

    Args:
        win_rate: Empirical win rate from pattern (0.0 to 1.0)
        confidence: Confidence level ("high", "medium", "low")
        league_factor: League compatibility factor (0.0 to 1.0)

    Returns:
        Combined score (0.0 to 1.0)
    """
    confidence_weight = CONFIDENCE_WEIGHTS.get(confidence, 0.4)
    return win_rate * confidence_weight * league_factor


def _build_market_score(
    pattern: MarketPattern,
    league_factor: float,
) -> MarketScore:
    """
    Build a MarketScore from a matched pattern.

    Args:
        pattern: The matched MarketPattern
        league_factor: The league compatibility factor

    Returns:
        MarketScore instance
    """
    score = _calculate_score(
        win_rate=pattern.win_rate,
        confidence=pattern.confidence,
        league_factor=league_factor,
    )

    return MarketScore(
        market_code=pattern.market_code,
        selection=pattern.selection,
        win_rate=pattern.win_rate,
        confidence=pattern.confidence,
        league_factor=league_factor,
        pattern_match=pattern.name,
        score=score,
    )


# =============================================================================
# 1X2 Market Scorers
# =============================================================================


def score_1x2_away(ctx: LambdaContext, league_code: str) -> MarketScore | None:
    """
    Score the 1X2 Away market.

    Evaluates away win potential based on lambda ratio (away team strength).
    Higher lambda_ratio indicates stronger away team.

    Args:
        ctx: LambdaContext with home/away lambda values
        league_code: League identifier (e.g., "E1", "D1")

    Returns:
        MarketScore if pattern matches, None otherwise
    """
    # Get league factor for 1X2 away markets
    league_factor = get_league_factor_1x2_away(league_code)

    # Skip if league is not compatible
    if league_factor == 0.0:
        return None

    # Find matching pattern
    pattern = find_pattern_match(
        lambda_ratio=ctx.lambda_ratio,
        patterns=PATTERNS_1X2_AWAY,
        lambda_sum=ctx.lambda_sum,
    )

    if pattern is None:
        return None

    return _build_market_score(pattern, league_factor)


# =============================================================================
# Asian Handicap Market Scorers
# =============================================================================


def score_ah_home_m05(ctx: LambdaContext, league_code: str) -> MarketScore | None:
    """
    Score the Asian Handicap Home -0.5 market.

    Home team must win by 1+ goals. Suitable when home team is a strong favorite.

    Args:
        ctx: LambdaContext with home/away lambda values
        league_code: League identifier

    Returns:
        MarketScore if pattern matches, None otherwise
    """
    # AH markets are compatible with all leagues (neutral factor)
    league_factor = 1.0

    pattern = find_pattern_match(
        lambda_ratio=ctx.lambda_ratio,
        patterns=PATTERNS_AH_HOME_M05,
        lambda_sum=ctx.lambda_sum,
    )

    if pattern is None:
        return None

    return _build_market_score(pattern, league_factor)


def score_ah_home_m15(ctx: LambdaContext, league_code: str) -> MarketScore | None:
    """
    Score the Asian Handicap Home -1.5 market.

    Home team must win by 2+ goals. Suitable for dominant home favorites.

    Args:
        ctx: LambdaContext with home/away lambda values
        league_code: League identifier

    Returns:
        MarketScore if pattern matches, None otherwise
    """
    league_factor = 1.0

    pattern = find_pattern_match(
        lambda_ratio=ctx.lambda_ratio,
        patterns=PATTERNS_AH_HOME_M15,
        lambda_sum=ctx.lambda_sum,
    )

    if pattern is None:
        return None

    return _build_market_score(pattern, league_factor)


def score_ah_away_p05(ctx: LambdaContext, league_code: str) -> MarketScore | None:
    """
    Score the Asian Handicap Away +0.5 market.

    Away team starts with 0.5 goal advantage. Bet wins if away team wins or draws.

    Args:
        ctx: LambdaContext with home/away lambda values
        league_code: League identifier

    Returns:
        MarketScore if pattern matches, None otherwise
    """
    league_factor = 1.0

    pattern = find_pattern_match(
        lambda_ratio=ctx.lambda_ratio,
        patterns=PATTERNS_AH_AWAY_P05,
        lambda_sum=ctx.lambda_sum,
    )

    if pattern is None:
        return None

    return _build_market_score(pattern, league_factor)


def score_ah_away_p15(ctx: LambdaContext, league_code: str) -> MarketScore | None:
    """
    Score the Asian Handicap Away +1.5 market.

    Away team starts with 1.5 goal advantage. Bet wins if away team loses by 1,
    draws, or wins.

    Args:
        ctx: LambdaContext with home/away lambda values
        league_code: League identifier

    Returns:
        MarketScore if pattern matches, None otherwise
    """
    league_factor = 1.0

    pattern = find_pattern_match(
        lambda_ratio=ctx.lambda_ratio,
        patterns=PATTERNS_AH_AWAY_P15,
        lambda_sum=ctx.lambda_sum,
    )

    if pattern is None:
        return None

    return _build_market_score(pattern, league_factor)


# =============================================================================
# European Handicap Market Scorers
# =============================================================================


def score_eh_home_m1(ctx: LambdaContext, league_code: str) -> MarketScore | None:
    """
    Score the European Handicap Home -1 market.

    Home team must win by exactly 2+ goals for the bet to win.
    A 1-goal win results in a push (draw on handicap).

    Args:
        ctx: LambdaContext with home/away lambda values
        league_code: League identifier

    Returns:
        MarketScore if pattern matches, None otherwise
    """
    league_factor = 1.0

    pattern = find_pattern_match(
        lambda_ratio=ctx.lambda_ratio,
        patterns=PATTERNS_EH_HOME_M1,
        lambda_sum=ctx.lambda_sum,
    )

    if pattern is None:
        return None

    return _build_market_score(pattern, league_factor)


def score_eh_away_m1(ctx: LambdaContext, league_code: str) -> MarketScore | None:
    """
    Score the European Handicap Away -1 market.

    Away team must win by exactly 2+ goals for the bet to win.
    A 1-goal win results in a push (draw on handicap).

    Args:
        ctx: LambdaContext with home/away lambda values
        league_code: League identifier

    Returns:
        MarketScore if pattern matches, None otherwise
    """
    league_factor = 1.0

    pattern = find_pattern_match(
        lambda_ratio=ctx.lambda_ratio,
        patterns=PATTERNS_EH_AWAY_M1,
        lambda_sum=ctx.lambda_sum,
    )

    if pattern is None:
        return None

    return _build_market_score(pattern, league_factor)


# =============================================================================
# Anytime Leading Market Scorers
# =============================================================================


def score_anytime_home_1up(ctx: LambdaContext, league_code: str) -> MarketScore | None:
    """
    Score the Anytime Home 1up market.

    Bet wins if home team leads by 1+ goals at any point during the match.

    Args:
        ctx: LambdaContext with home/away lambda values
        league_code: League identifier

    Returns:
        MarketScore if pattern matches, None otherwise
    """
    league_factor = 1.0

    pattern = find_pattern_match(
        lambda_ratio=ctx.lambda_ratio,
        patterns=PATTERNS_ANYTIME_HOME_1UP,
        lambda_sum=ctx.lambda_sum,
    )

    if pattern is None:
        return None

    return _build_market_score(pattern, league_factor)


def score_anytime_home_2up(ctx: LambdaContext, league_code: str) -> MarketScore | None:
    """
    Score the Anytime Home 2up market.

    Bet wins if home team leads by 2+ goals at any point during the match.

    Args:
        ctx: LambdaContext with home/away lambda values
        league_code: League identifier

    Returns:
        MarketScore if pattern matches, None otherwise
    """
    league_factor = 1.0

    pattern = find_pattern_match(
        lambda_ratio=ctx.lambda_ratio,
        patterns=PATTERNS_ANYTIME_HOME_2UP,
        lambda_sum=ctx.lambda_sum,
    )

    if pattern is None:
        return None

    return _build_market_score(pattern, league_factor)


def score_anytime_away_1up(ctx: LambdaContext, league_code: str) -> MarketScore | None:
    """
    Score the Anytime Away 1up market.

    Bet wins if away team leads by 1+ goals at any point during the match.

    Args:
        ctx: LambdaContext with home/away lambda values
        league_code: League identifier

    Returns:
        MarketScore if pattern matches, None otherwise
    """
    league_factor = 1.0

    pattern = find_pattern_match(
        lambda_ratio=ctx.lambda_ratio,
        patterns=PATTERNS_ANYTIME_AWAY_1UP,
        lambda_sum=ctx.lambda_sum,
    )

    if pattern is None:
        return None

    return _build_market_score(pattern, league_factor)


def score_anytime_away_2up(ctx: LambdaContext, league_code: str) -> MarketScore | None:
    """
    Score the Anytime Away 2up market.

    Bet wins if away team leads by 2+ goals at any point during the match.

    Args:
        ctx: LambdaContext with home/away lambda values
        league_code: League identifier

    Returns:
        MarketScore if pattern matches, None otherwise
    """
    league_factor = 1.0

    pattern = find_pattern_match(
        lambda_ratio=ctx.lambda_ratio,
        patterns=PATTERNS_ANYTIME_AWAY_2UP,
        lambda_sum=ctx.lambda_sum,
    )

    if pattern is None:
        return None

    return _build_market_score(pattern, league_factor)


# =============================================================================
# Corners Market Scorers
# =============================================================================


def score_corners_over_85(ctx: LambdaContext, league_code: str) -> MarketScore | None:
    """
    Score the Corners Over 8.5 market.

    Bet wins if total corners in the match exceeds 8.5.
    Higher lambda_sum indicates more attacking play and expected corners.

    Args:
        ctx: LambdaContext with home/away lambda values
        league_code: League identifier

    Returns:
        MarketScore if pattern matches, None otherwise
    """
    # Get league factor for corners markets
    league_factor = get_league_factor_corners(league_code)

    # Skip if league is not compatible
    if league_factor == 0.0:
        return None

    pattern = find_pattern_match(
        lambda_ratio=ctx.lambda_ratio,
        patterns=PATTERNS_CORNERS_OVER_85,
        lambda_sum=ctx.lambda_sum,
    )

    if pattern is None:
        return None

    return _build_market_score(pattern, league_factor)


def score_corners_over_95(ctx: LambdaContext, league_code: str) -> MarketScore | None:
    """
    Score the Corners Over 9.5 market.

    Bet wins if total corners in the match exceeds 9.5.
    Requires very high tempo match with significant corner expectation.

    Args:
        ctx: LambdaContext with home/away lambda values
        league_code: League identifier

    Returns:
        MarketScore if pattern matches, None otherwise
    """
    # Get league factor for corners markets
    league_factor = get_league_factor_corners(league_code)

    # Skip if league is not compatible
    if league_factor == 0.0:
        return None

    pattern = find_pattern_match(
        lambda_ratio=ctx.lambda_ratio,
        patterns=PATTERNS_CORNERS_OVER_95,
        lambda_sum=ctx.lambda_sum,
    )

    if pattern is None:
        return None

    return _build_market_score(pattern, league_factor)


# =============================================================================
# Scorers Registry
# =============================================================================

SCORERS: dict[str, callable] = {
    "1x2_away": score_1x2_away,
    "ah_home_m05": score_ah_home_m05,
    "ah_home_m15": score_ah_home_m15,
    "ah_away_p05": score_ah_away_p05,
    "ah_away_p15": score_ah_away_p15,
    "eh_home_m1": score_eh_home_m1,
    "eh_away_m1": score_eh_away_m1,
    "anytime_home_1up": score_anytime_home_1up,
    "anytime_home_2up": score_anytime_home_2up,
    "anytime_away_1up": score_anytime_away_1up,
    "anytime_away_2up": score_anytime_away_2up,
    "corners_over_85": score_corners_over_85,
    "corners_over_95": score_corners_over_95,
}
