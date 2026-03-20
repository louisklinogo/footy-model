"""
Decision framework for market recommendations.

The DecisionFramework is the main decision engine that scores all markets
for a fixture and returns ranked recommendations.
"""

from __future__ import annotations

from typing import Any

from src.db.db_utils import connect_db
from .league_filters import is_league_compatible
from .market_scorers import SCORERS
from .types import (
    FixtureDecision,
    LambdaContext,
    MarketRecommendation,
    MarketScore,
)


class DecisionFramework:
    """Main decision engine that scores all markets and returns recommendations."""

    def __init__(self) -> None:
        """Initialize the framework with market scorers."""
        self.scorers = SCORERS

    def analyze(self, fixture_id: int) -> FixtureDecision:
        """
        Analyze a fixture and return market recommendations.

        Steps:
        1. Get fixture info (league, teams) from DB
        2. Get lambda values from predictions table (model_name='lambda_xgb')
        3. Build LambdaContext
        4. Score all 14 markets using scorers
        5. Filter out None scores
        6. Sort by score descending
        7. Convert to MarketRecommendation objects
        8. Return FixtureDecision with top 5 recommendations

        Args:
            fixture_id: The fixture ID to analyze

        Returns:
            FixtureDecision with ranked market recommendations

        Raises:
            ValueError: If fixture not found or lambda values missing
        """
        conn = connect_db()
        try:
            # 1. Get fixture info from database
            fixture_info = self._fetch_fixture_info(conn, fixture_id)
            if fixture_info is None:
                raise ValueError(f"Fixture {fixture_id} not found")

            league_code = fixture_info["league_code"]
            home_team = fixture_info["home_team"]
            away_team = fixture_info["away_team"]

            # 2. Get lambda values from predictions table
            lambda_values = self._fetch_lambda_values(conn, fixture_id)
            if "home" not in lambda_values or "away" not in lambda_values:
                raise ValueError(
                    f"Lambda values not found for fixture {fixture_id}. "
                    "Run lambda prediction first."
                )

            # 3. Build LambdaContext
            lambda_ctx = LambdaContext(
                lambda_home=lambda_values["home"],
                lambda_away=lambda_values["away"],
            )

            # 4. Score all markets
            scores: list[MarketScore] = []
            for market_key, scorer in self.scorers.items():
                score = scorer(lambda_ctx, league_code)
                if score is not None:
                    scores.append(score)

            # 5. Sort by score descending
            scores.sort(key=lambda s: s.score, reverse=True)

            # 6. Convert to MarketRecommendation (top 5)
            recommendations = []
            for i, score in enumerate(scores[:5]):
                recommendation = self._score_to_recommendation(
                    score, league_code, rank=i + 1
                )
                recommendations.append(recommendation)

            # 7. Determine avoid markets (leagues with 0.0 factor)
            avoid_markets = self._get_avoid_markets(league_code)

            # 8. Build and return FixtureDecision
            return FixtureDecision(
                fixture_id=fixture_id,
                home_team=home_team,
                away_team=away_team,
                league_code=league_code,
                lambda_ctx=lambda_ctx,
                recommendations=tuple(recommendations),
                top_pick=recommendations[0] if recommendations else None,
                avoid_markets=avoid_markets,
            )
        finally:
            conn.close()

    def analyze_batch(self, fixture_ids: list[int]) -> list[FixtureDecision]:
        """
        Analyze multiple fixtures.

        Args:
            fixture_ids: List of fixture IDs to analyze

        Returns:
            List of FixtureDecision objects (one per fixture)
        """
        return [self.analyze(fid) for fid in fixture_ids]

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    def _fetch_fixture_info(
        self, conn, fixture_id: int
    ) -> dict[str, Any] | None:
        """Fetch fixture info from database."""
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    f.fixture_id,
                    f.league_code,
                    ht.team_name as home_team,
                    at.team_name as away_team
                FROM fixtures f
                JOIN teams ht ON ht.team_id = f.home_team_id
                JOIN teams at ON at.team_id = f.away_team_id
                WHERE f.fixture_id = %s
                """,
                (fixture_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row))

    def _fetch_lambda_values(self, conn, fixture_id: int) -> dict[str, float]:
        """
        Fetch lambda values from predictions table.

        Lambda values are stored with model_name='lambda_xgb' and model_version='v1'.
        The actual lambda value is in metadata_json->>'lambda'.
        """
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    market_code,
                    metadata_json->>'lambda' as lambda_value
                FROM predictions
                WHERE fixture_id = %s
                  AND model_name = 'lambda_xgb'
                  AND model_version = 'v1'
                  AND market_code IN ('lambda_home', 'lambda_away')
                """,
                (fixture_id,),
            )
            result = {}
            for row in cur.fetchall():
                market_code = row[0]
                lambda_str = row[1]
                if market_code and lambda_str:
                    # Extract 'home' or 'away' from 'lambda_home' / 'lambda_away'
                    key = market_code.replace("lambda_", "")
                    try:
                        result[key] = float(lambda_str)
                    except (ValueError, TypeError):
                        pass
            return result

    def _score_to_recommendation(
        self,
        score: MarketScore,
        league_code: str,
        rank: int,
    ) -> MarketRecommendation:
        """Convert a MarketScore to a MarketRecommendation."""
        # Build a human-readable reason
        reason = self._build_reason(score, rank)

        # Check league compatibility
        league_compatible = is_league_compatible(
            score.market_code, league_code
        )

        return MarketRecommendation(
            market_code=score.market_code,
            selection=score.selection,
            score=score.score,
            win_rate=score.win_rate,
            confidence=score.confidence,
            reason=reason,
            league_compatible=league_compatible,
        )

    def _build_reason(self, score: MarketScore, rank: int) -> str:
        """Build a human-readable reason for the recommendation."""
        confidence_str = score.confidence.upper()
        league_str = "good league" if score.league_factor == 1.0 else "neutral league"
        if score.league_factor == 0.0:
            league_str = "avoid league"

        return (
            f"#{rank} pick: {score.pattern_match} "
            f"({confidence_str} confidence, {league_str}). "
            f"Win rate: {score.win_rate:.0%}"
        )

    def _get_avoid_markets(self, league_code: str) -> tuple[str, ...]:
        """Get markets to avoid for this league based on league filters."""
        avoid = []
        for market_key in self.scorers.keys():
            if not is_league_compatible(market_key, league_code):
                avoid.append(market_key)
        return tuple(avoid)
