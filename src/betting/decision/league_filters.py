"""
League filters for market recommendations.

Defines league quality tiers for different betting markets and provides
functions to evaluate league compatibility and confidence factors.
"""

from __future__ import annotations

# =============================================================================
# League Sets for 1X2 Away Markets
# =============================================================================

GOOD_LEAGUES_1X2_AWAY: set[str] = {
    "D1",   # Bundesliga (Germany)
    "I1",   # Serie A (Italy)
    "F1",   # Ligue 1 (France)
    "P1",   # Primeira Liga (Portugal)
    "E3",   # League One (England)
    "EC",   # Championship (England)
    "CL",   # Champions League
    "T1",   # Super Lig (Turkey)
    "B1",   # First Division A (Belgium)
    "DK1",  # Superliga (Denmark)
    "AT1",  # Bundesliga (Austria)
    "MX1",  # Liga MX (Mexico)
    "SP1",  # La Liga (Spain)
}

BAD_LEAGUES_1X2_AWAY: set[str] = {
    "PL1",  # Ekstraklasa (Poland)
    "E2",   # Championship (England) - lower confidence
}

# =============================================================================
# League Sets for Corners Markets
# =============================================================================

HIGH_CORNERS_LEAGUES: set[str] = {
    "N1",   # Eredivisie (Netherlands)
    "EC",   # Championship (England)
    "SC0",  # Premiership (Scotland)
    "B1",   # First Division A (Belgium)
    "E1",   # Premier League (England)
    "D2",   # 2. Bundesliga (Germany)
}

LOW_CORNERS_LEAGUES: set[str] = {
    "I1",   # Serie A (Italy)
    "G1",   # Super League (Greece)
    "JP1",  # J1 League (Japan)
    "CH2",  # Challenge League (Switzerland)
}

# =============================================================================
# Helper Functions
# =============================================================================


def get_league_factor_1x2_away(league_code: str) -> float:
    """
    Get confidence factor for 1X2 away bets in a given league.

    Args:
        league_code: The league code (e.g., "E1", "D1")

    Returns:
        1.0 for good leagues (full confidence)
        0.5 for neutral leagues (reduced confidence)
        0.0 for bad leagues (no bets)
    """
    if league_code in GOOD_LEAGUES_1X2_AWAY:
        return 1.0
    if league_code in BAD_LEAGUES_1X2_AWAY:
        return 0.0
    return 0.5


def get_league_factor_corners(league_code: str) -> float:
    """
    Get confidence factor for corners bets in a given league.

    Args:
        league_code: The league code (e.g., "E1", "D1")

    Returns:
        1.0 for high-corners leagues (full confidence)
        0.5 for neutral leagues (reduced confidence)
        0.0 for low-corners leagues (no bets)
    """
    if league_code in HIGH_CORNERS_LEAGUES:
        return 1.0
    if league_code in LOW_CORNERS_LEAGUES:
        return 0.0
    return 0.5


def is_league_compatible(market_code: str, league_code: str) -> bool:
    """
    Check if a league is compatible with a given market type.

    A league is incompatible if the factor is 0.0 (no bets allowed).

    Args:
        market_code: The market code (e.g., "1x2_away", "corners_over")
        league_code: The league code (e.g., "E1", "D1")

    Returns:
        True if the league is compatible with the market, False otherwise
    """
    # Normalize market code for comparison
    market_lower = market_code.lower()

    # Check 1X2 away markets
    if "1x2" in market_lower and "away" in market_lower:
        return get_league_factor_1x2_away(league_code) > 0.0

    # Check corners markets
    if "corner" in market_lower:
        return get_league_factor_corners(league_code) > 0.0

    # Default: all other markets are compatible with all leagues
    return True
