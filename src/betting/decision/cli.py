"""
Command-line interface for the betting decision framework.

Usage:
    # Analyze all fixtures for today (default)
    python -m src.betting.decision

    # Analyze all fixtures for a specific date
    python -m src.betting.decision --date 2026-03-20

    # Analyze fixtures for a date range
    python -m src.betting.decision --date-range "2026-03-20:2026-03-27"

    # Analyze a single fixture
    python -m src.betting.decision --fixture 12345

    # Show detailed output for each fixture
    python -m src.betting.decision --date 2026-03-20 --detailed

    # Output as JSON
    python -m src.betting.decision --date 2026-03-20 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
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
  Analyze all fixtures for today (default):
    python -m src.betting.decision

  Analyze all fixtures for a specific date:
    python -m src.betting.decision --date 2026-03-20

  Analyze fixtures for a date range:
    python -m src.betting.decision --date-range "2026-03-20:2026-03-27"

  Analyze a single fixture:
    python -m src.betting.decision --fixture 12345

  Show detailed output:
    python -m src.betting.decision --date 2026-03-20 --detailed

  Output as JSON:
    python -m src.betting.decision --date 2026-03-20 --json
        """,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--fixture",
        type=int,
        help="Single fixture ID to analyze",
    )
    input_group.add_argument(
        "--date",
        type=str,
        help="Single date to analyze (YYYY-MM-DD format)",
    )
    input_group.add_argument(
        "--date-range",
        type=str,
        help="Date range to analyze (format: YYYY-MM-DD:YYYY-MM-DD)",
    )

    # Output format options
    output_group = parser.add_mutually_exclusive_group(required=False)
    output_group.add_argument(
        "--summary",
        action="store_true",
        default=True,
        help="Show summary table (default)",
    )
    output_group.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed output for each fixture",
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    # Filtering options
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

    return parser


def parse_date(date_str: str) -> date:
    """Parse a date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Invalid date format '{date_str}'. Use YYYY-MM-DD.")


def parse_date_range(date_range: str) -> tuple[date, date]:
    """Parse a date range string in YYYY-MM-DD:YYYY-MM-DD format."""
    parts = date_range.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid date range format '{date_range}'. Use YYYY-MM-DD:YYYY-MM-DD."
        )
    start_date = parse_date(parts[0])
    end_date = parse_date(parts[1])
    if start_date > end_date:
        raise ValueError(f"Start date {start_date} must be before end date {end_date}")
    return start_date, end_date


def format_datetime(dt: datetime | None) -> str:
    """Format a datetime for display.
    
    If time is 00:00 (midnight), show as TBD since the actual time is likely not set.
    """
    if dt is None:
        return "N/A"
    # If time is midnight (00:00), it's likely TBD
    if dt.hour == 0 and dt.minute == 0:
        return f"{dt.strftime('%Y-%m-%d')} TBD"
    return dt.strftime("%Y-%m-%d %H:%M")


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
    filtered_recs = [
        r for r in decision.recommendations
        if r.score >= min_score
    ][:top]

    return {
        "fixture_id": decision.fixture_id,
        "match_datetime_utc": decision.match_datetime_utc.isoformat() if decision.match_datetime_utc else None,
        "league_code": decision.league_code,
        "league_name": decision.league_name,
        "home_team": decision.home_team,
        "away_team": decision.away_team,
        "lambda_home": decision.lambda_ctx.lambda_home,
        "lambda_away": decision.lambda_ctx.lambda_away,
        "lambda_ratio": round(decision.lambda_ctx.lambda_ratio, 3),
        "recommendations": [recommendation_to_dict(r) for r in filtered_recs],
        "top_pick": recommendation_to_dict(decision.top_pick) if decision.top_pick else None,
        "avoid_markets": list(decision.avoid_markets),
    }


def print_summary_table(
    decisions: list[FixtureDecision],
    top: int,
    min_score: float,
) -> None:
    """Print a summary table of all decisions."""
    if not decisions:
        print("No fixtures found.")
        return

    # Define column widths
    col_widths = {
        "id": 8,
        "datetime": 17,
        "league": 30,
        "teams": 35,
        "ratio": 8,
        "rec": 25,
        "score": 6,
        "win_rate": 8,
    }

    # Header
    header = (
        f"{'ID':<{col_widths['id']}} "
        f"{'Date/Time':<{col_widths['datetime']}} "
        f"{'League':<{col_widths['league']}} "
        f"{'Home vs Away':<{col_widths['teams']}} "
        f"{'L-Ratio':>{col_widths['ratio']}} "
        f"{'Top Recommendation':<{col_widths['rec']}} "
        f"{'Score':>{col_widths['score']}} "
        f"{'Win Rate':>{col_widths['win_rate']}}"
    )
    print("\n" + header)
    print("-" * len(header))

    for d in decisions:
        # Get top recommendation that meets min_score
        top_recs = [r for r in d.recommendations if r.score >= min_score][:top]
        top_rec = top_recs[0] if top_recs else None

        # Format the row
        fixture_id = str(d.fixture_id)[:col_widths['id']]
        dt_str = format_datetime(d.match_datetime_utc)[:col_widths['datetime']]
        league = d.league_name[:col_widths['league'] - 1] if len(d.league_name) >= col_widths['league'] else d.league_name
        teams = f"{d.home_team} vs {d.away_team}"[:col_widths['teams'] - 1]
        ratio = f"{d.lambda_ctx.lambda_ratio:.2f}"
        
        if top_rec:
            rec_str = f"{top_rec.market_code.upper()} {top_rec.selection}"[:col_widths['rec'] - 1]
            score = f"{top_rec.score:.3f}"
            win_rate = f"{top_rec.win_rate:.0%}"
        else:
            rec_str = "No rec"
            score = "N/A"
            win_rate = "N/A"

        print(
            f"{fixture_id:<{col_widths['id']}} "
            f"{dt_str:<{col_widths['datetime']}} "
            f"{league:<{col_widths['league']}} "
            f"{teams:<{col_widths['teams']}} "
            f"{ratio:>{col_widths['ratio']}} "
            f"{rec_str:<{col_widths['rec']}} "
            f"{score:>{col_widths['score']}} "
            f"{win_rate:>{col_widths['win_rate']}}"
        )

    print("-" * len(header))
    print(f"\nTotal: {len(decisions)} fixture(s)")


def print_detailed_fixture(
    decision: FixtureDecision,
    top: int,
    min_score: float,
) -> None:
    """Print detailed output for a single fixture."""
    print(f"\n{'='*70}")
    print(f"Fixture #{decision.fixture_id}: {decision.home_team} vs {decision.away_team}")
    print(f"League: {decision.league_name} ({decision.league_code})")
    print(f"Date/Time: {format_datetime(decision.match_datetime_utc)}")
    print(f"Lambda: home={decision.lambda_ctx.lambda_home:.2f}, away={decision.lambda_ctx.lambda_away:.2f}, ratio={decision.lambda_ctx.lambda_ratio:.2f}")
    print(f"{'='*70}")

    # Filter recommendations by min_score and limit to top N
    filtered_recs = [
        r for r in decision.recommendations
        if r.score >= min_score
    ][:top]

    if not filtered_recs:
        print("\nNo recommendations meet the criteria.")
        return

    print(f"\nTop {len(filtered_recs)} Recommendations (min score: {min_score}):")
    print("-" * 70)

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
    detailed: bool,
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
    elif detailed:
        print_detailed_fixture(decision, top, min_score)
    else:
        print_summary_table([decision], top, min_score)

    return 0


def analyze_fixtures(
    framework: DecisionFramework,
    fixture_ids: list[int],
    top: int,
    min_score: float,
    as_json: bool,
    detailed: bool,
) -> int:
    """Analyze multiple fixtures and print results."""
    if not fixture_ids:
        print("No fixtures found for the specified criteria.", file=sys.stderr)
        return 1

    # Analyze all fixtures
    decisions = []
    for fixture_id in fixture_ids:
        try:
            decision = framework.analyze(fixture_id)
            decisions.append(decision)
        except ValueError as e:
            print(f"Warning: Skipping fixture {fixture_id}: {e}", file=sys.stderr)

    if not decisions:
        print("No valid fixtures to analyze.", file=sys.stderr)
        return 1

    # Sort by datetime (None values go to the end)
    decisions.sort(key=lambda d: (d.match_datetime_utc is None, d.match_datetime_utc))

    if as_json:
        output = [fixture_to_dict(d, top, min_score) for d in decisions]
        print(json.dumps(output, indent=2))
    elif detailed:
        for decision in decisions:
            print_detailed_fixture(decision, top, min_score)
    else:
        print_summary_table(decisions, top, min_score)

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

    # Determine what to analyze
    if args.fixture is not None:
        return analyze_single_fixture(
            framework,
            args.fixture,
            args.top,
            args.min_score,
            args.json,
            args.detailed,
        )

    # Get fixture IDs based on date parameters
    try:
        if args.date_range:
            start_date, end_date = parse_date_range(args.date_range)
            fixture_ids = framework.get_fixture_ids_by_date_range(start_date, end_date)
            date_desc = f"{start_date} to {end_date}"
        elif args.date:
            target_date = parse_date(args.date)
            fixture_ids = framework.get_fixture_ids_by_date(target_date)
            date_desc = str(target_date)
        else:
            # Default to today
            today = date.today()
            fixture_ids = framework.get_fixture_ids_by_date(today)
            date_desc = f"today ({today})"
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Print what we're analyzing
    if not args.json:
        print(f"\nAnalyzing {len(fixture_ids)} fixture(s) for {date_desc}")

    return analyze_fixtures(
        framework,
        fixture_ids,
        args.top,
        args.min_score,
        args.json,
        args.detailed,
    )


if __name__ == "__main__":
    sys.exit(main())
