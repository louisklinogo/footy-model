"""
Command-line interface for the betting decision framework.

Usage:
    # Analyze a single fixture
    python -m src.betting.decision --fixture 12345

    # Analyze all fixtures for a date
    python -m src.betting.decision --date 2024-01-15

    # Control output
    python -m src.betting.decision --fixture 12345 --top 5 --min-score 0.4 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from typing import Any

from .framework import DecisionFramework
from .types import FixtureDecision, MarketRecommendation


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="decision",
        description="Betting decision framework - analyze fixtures and get market recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Analyze a single fixture:
    python -m src.betting.decision --fixture 12345

  Analyze all fixtures for a date:
    python -m src.betting.decision --date 2024-01-15

  Get top 5 recommendations with minimum score 0.4:
    python -m src.betting.decision --fixture 12345 --top 5 --min-score 0.4

  Output as JSON:
    python -m src.betting.decision --fixture 12345 --json
        """,
    )

    # Mutually exclusive group for fixture vs date
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--fixture",
        type=int,
        help="Single fixture ID to analyze",
    )
    input_group.add_argument(
        "--date",
        type=str,
        help="Date for batch processing (YYYY-MM-DD format)",
    )

    # Output options
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Number of recommendations to show (default: 3)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.3,
        help="Minimum score threshold for recommendations (default: 0.3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    return parser


def recommendation_to_dict(rec: MarketRecommendation) -> dict[str, Any]:
    """Convert a MarketRecommendation to a dictionary for JSON output."""
    return {
        "market_code": rec.market_code,
        "selection": rec.selection,
        "score": round(rec.score, 3),
        "win_rate": round(rec.win_rate, 3),
        "confidence": rec.confidence,
        "reason": rec.reason,
        "league_compatible": rec.league_compatible,
    }


def fixture_to_dict(decision: FixtureDecision, top: int, min_score: float) -> dict[str, Any]:
    """Convert a FixtureDecision to a dictionary for JSON output."""
    # Filter recommendations by min_score and limit to top N
    filtered_recs = [
        r for r in decision.recommendations
        if r.score >= min_score
    ][:top]

    return {
        "fixture_id": decision.fixture_id,
        "home_team": decision.home_team,
        "away_team": decision.away_team,
        "league_code": decision.league_code,
        "lambda_home": decision.lambda_ctx.lambda_home,
        "lambda_away": decision.lambda_ctx.lambda_away,
        "recommendations": [recommendation_to_dict(r) for r in filtered_recs],
        "top_pick": recommendation_to_dict(decision.top_pick) if decision.top_pick else None,
        "avoid_markets": list(decision.avoid_markets),
    }


def print_fixture_human(decision: FixtureDecision, top: int, min_score: float) -> None:
    """Print a FixtureDecision in human-readable format."""
    print(f"\n{'='*60}")
    print(f"Fixture: {decision.home_team} vs {decision.away_team}")
    print(f"League: {decision.league_code} | ID: {decision.fixture_id}")
    print(f"Lambda: home={decision.lambda_ctx.lambda_home:.2f}, away={decision.lambda_ctx.lambda_away:.2f}")
    print(f"{'='*60}")

    # Filter recommendations by min_score and limit to top N
    filtered_recs = [
        r for r in decision.recommendations
        if r.score >= min_score
    ][:top]

    if not filtered_recs:
        print("\nNo recommendations meet the criteria.")
        return

    print(f"\nTop {len(filtered_recs)} Recommendations (min score: {min_score}):")
    print("-" * 60)

    for i, rec in enumerate(filtered_recs, 1):
        print(f"\n{i}. {rec.market_code.upper()} - {rec.selection}")
        print(f"   Score: {rec.score:.3f} | Win Rate: {rec.win_rate:.0%} | {rec.confidence.upper()}")
        print(f"   {rec.reason}")
        if not rec.league_compatible:
            print(f"   ⚠️  League compatibility issue")

    if decision.avoid_markets:
        print(f"\nMarkets to avoid for this league: {', '.join(decision.avoid_markets)}")


def analyze_single_fixture(
    framework: DecisionFramework,
    fixture_id: int,
    top: int,
    min_score: float,
    as_json: bool,
) -> int:
    """Analyze a single fixture and print results."""
    try:
        decision = framework.analyze(fixture_id)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if as_json:
        output = fixture_to_dict(decision, top, min_score)
        print(json.dumps(output, indent=2))
    else:
        print_fixture_human(decision, top, min_score)

    return 0


def analyze_date(
    framework: DecisionFramework,
    date_str: str,
    top: int,
    min_score: float,
    as_json: bool,
) -> int:
    """Analyze all fixtures for a date and print results."""
    # Parse date
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"Error: Invalid date format '{date_str}'. Use YYYY-MM-DD.", file=sys.stderr)
        return 1

    # Get fixture IDs for the date from database
    from src.db.db_utils import connect_db

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT fixture_id
                FROM fixtures
                WHERE DATE(match_datetime_utc) = %s
                ORDER BY fixture_id
                """,
                (date,),
            )
            fixture_ids = [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

    if not fixture_ids:
        print(f"No fixtures found for date {date_str}", file=sys.stderr)
        return 1

    # Analyze all fixtures
    results = []
    for fixture_id in fixture_ids:
        try:
            decision = framework.analyze(fixture_id)
            results.append(decision)
        except ValueError as e:
            print(f"Warning: Skipping fixture {fixture_id}: {e}", file=sys.stderr)

    if as_json:
        output = [fixture_to_dict(d, top, min_score) for d in results]
        print(json.dumps(output, indent=2))
    else:
        print(f"\nAnalyzing {len(results)} fixtures for {date_str}")
        for decision in results:
            print_fixture_human(decision, top, min_score)

    return 0


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Initialize framework
    framework = DecisionFramework()

    # Dispatch based on input type
    if args.fixture is not None:
        return analyze_single_fixture(
            framework,
            args.fixture,
            args.top,
            args.min_score,
            args.json,
        )
    else:
        return analyze_date(
            framework,
            args.date,
            args.top,
            args.min_score,
            args.json,
        )


if __name__ == "__main__":
    sys.exit(main())
