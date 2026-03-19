"""
Walk-forward backtest for match profile pipeline.

Tests the full pipeline: profile -> story -> opportunities -> accas -> settlement.

Usage:
    python src/betting/backtest_match_profiles.py --since-days 365 --strategy balanced --bankroll 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.betting.acca_builder import AccaStrategy, build_accas, Acca, AccaLeg
from src.betting.match_profile import build_match_profile, MatchProfile
from src.betting.opportunity_finder import find_opportunities, MatchOpportunities
from src.betting.settlement import settle_market
from src.betting.story_generator import generate_story, MatchStory
from src.db.db_utils import connect_db, get_database_url


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class SettledLeg:
    """A leg that has been settled with actual results."""
    fixture_id: int
    market_code: str
    selection: str
    odds: float
    model_prob: float
    story_signals: tuple[str, ...]
    result: str  # "win", "loss", "push"
    return_factor: float


@dataclass
class SettledAcca:
    """An accumulator that has been settled with actual results."""
    acca_id: str
    date: str
    stake: float
    legs: list[SettledLeg]
    combined_odds: float
    combined_prob: float
    result_return_factor: float
    pnl: float
    hit: bool
    strategy: str


@dataclass
class BacktestMetrics:
    """Aggregated metrics from the backtest."""
    total_accas: int = 0
    total_stake: float = 0.0
    total_pnl: float = 0.0
    total_return: float = 0.0
    hits: int = 0
    starting_bankroll: float = 0.0
    ending_bankroll: float = 0.0
    max_drawdown: float = 0.0
    peak_bankroll: float = 0.0

    # By story type
    by_story_type: dict[str, dict[str, float]] = field(default_factory=dict)

    # By market
    by_market: dict[str, dict[str, float]] = field(default_factory=dict)

    # By strategy
    by_strategy: dict[str, dict[str, float]] = field(default_factory=dict)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest for match profile pipeline."
    )
    parser.add_argument(
        "--since-days",
        type=int,
        default=365,
        help="Number of days to look back (default: 365)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=[s.value for s in AccaStrategy],
        default=AccaStrategy.BALANCED.value,
        help="Accumulator strategy (default: balanced)",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Initial bankroll (default: 1000)",
    )
    parser.add_argument(
        "--stake-fraction",
        type=float,
        default=0.01,
        help="Stake fraction per acca (default: 0.01)",
    )
    parser.add_argument(
        "--max-legs",
        type=int,
        default=3,
        help="Maximum legs per acca (default: 3)",
    )
    parser.add_argument(
        "--min-legs",
        type=int,
        default=2,
        help="Minimum legs per acca (default: 2)",
    )
    parser.add_argument(
        "--max-accas-per-day",
        type=int,
        default=1,
        help="Maximum accas per day (default: 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of fixtures to process (for testing)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/reports/backtests"),
        help="Output directory for reports",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress details",
    )
    return parser.parse_args()


# =============================================================================
# Data Fetching
# =============================================================================


def fetch_finished_fixtures(since_days: int, limit: int | None = None) -> list[dict[str, Any]]:
    """Fetch finished fixtures with results from the last N days."""
    query = """
    SELECT DISTINCT
        f.fixture_id,
        f.match_datetime_utc,
        f.league_code,
        f.home_team_id,
        f.away_team_id,
        ht.team_name as home_team_name,
        at.team_name as away_team_name,
        fr.home_goals,
        fr.away_goals,
        fs.h_corners,
        fs.a_corners,
        ils.home_led_by_1_any,
        ils.away_led_by_1_any,
        ils.home_led_by_2_any,
        ils.away_led_by_2_any
    FROM fixtures f
    JOIN teams ht ON ht.team_id = f.home_team_id
    JOIN teams at ON at.team_id = f.away_team_id
    LEFT JOIN fixture_results fr ON fr.fixture_id = f.fixture_id
    LEFT JOIN fixture_stats_premium fs ON fs.fixture_id = f.fixture_id
    LEFT JOIN fixture_incident_lead_states ils ON ils.fixture_id = f.fixture_id
    WHERE f.status = 'ft'
      AND f.match_datetime_utc IS NOT NULL
      AND f.match_datetime_utc >= NOW() - (%s || ' days')::interval
    ORDER BY f.match_datetime_utc ASC
    """
    if limit:
        query += f" LIMIT {limit}"

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, (since_days,))
            rows = cur.fetchall()
            cols = [desc[0] for desc in (cur.description or [])]
        return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()


# =============================================================================
# Settlement
# =============================================================================


def settle_leg(
    leg: AccaLeg,
    fixture_result: dict[str, Any],
) -> SettledLeg:
    """Settle a single leg against actual results."""
    home_goals = fixture_result.get("home_goals")
    away_goals = fixture_result.get("away_goals")
    home_corners = fixture_result.get("h_corners")
    away_corners = fixture_result.get("a_corners")

    incident_states = {
        "home_led_by_1_any": bool(fixture_result.get("home_led_by_1_any")),
        "away_led_by_1_any": bool(fixture_result.get("away_led_by_1_any")),
        "home_led_by_2_any": bool(fixture_result.get("home_led_by_2_any")),
        "away_led_by_2_any": bool(fixture_result.get("away_led_by_2_any")),
    }

    settlement = settle_market(
        leg.market_code,
        odds=leg.odds,
        home_goals=home_goals,
        away_goals=away_goals,
        home_corners=home_corners,
        away_corners=away_corners,
        incident_lead_states=incident_states,
    )

    result = "push" if settlement.outcome == "push" else ("win" if settlement.return_factor > 1.0 else "loss")

    return SettledLeg(
        fixture_id=leg.fixture_id,
        market_code=leg.market_code,
        selection=leg.selection,
        odds=leg.odds,
        model_prob=leg.model_prob,
        story_signals=leg.story_signals,
        result=result,
        return_factor=settlement.return_factor,
    )


def settle_acca(
    acca: Acca,
    fixture_results: dict[int, dict[str, Any]],
    stake: float,
    acca_id: str,
    date: str,
) -> SettledAcca:
    """Settle an accumulator against actual results."""
    settled_legs = []
    return_factor = 1.0

    for leg in acca.legs:
        fixture_result = fixture_results.get(leg.fixture_id, {})
        settled_leg = settle_leg(leg, fixture_result)
        settled_legs.append(settled_leg)
        return_factor *= settled_leg.return_factor

    pnl = stake * (return_factor - 1.0)
    hit = return_factor > 1.0

    return SettledAcca(
        acca_id=acca_id,
        date=date,
        stake=stake,
        legs=settled_legs,
        combined_odds=acca.combined_odds,
        combined_prob=acca.combined_prob,
        result_return_factor=return_factor,
        pnl=pnl,
        hit=hit,
        strategy=acca.strategy,
    )


# =============================================================================
# Pipeline Execution
# =============================================================================


def process_fixture(
    fixture_id: int,
    verbose: bool = False,
) -> MatchOpportunities | None:
    """Run the full pipeline for a single fixture."""
    try:
        if verbose:
            print(f"  Building profile for fixture {fixture_id}...")
        profile = build_match_profile(fixture_id)

        if verbose:
            print(f"  Generating story...")
        story = generate_story(profile)

        if verbose:
            print(f"  Finding opportunities...")
        opps = find_opportunities(profile, story)

        return opps
    except Exception as e:
        if verbose:
            print(f"  Error processing fixture {fixture_id}: {e}")
        return None


def run_backtest(
    fixtures: list[dict[str, Any]],
    strategy: AccaStrategy,
    initial_bankroll: float,
    stake_fraction: float,
    max_legs: int,
    min_legs: int,
    max_accas_per_day: int,
    verbose: bool = False,
) -> tuple[list[SettledAcca], BacktestMetrics]:
    """Run walk-forward backtest on fixtures."""
    # Group fixtures by date
    fixtures_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fixture in fixtures:
        dt = fixture.get("match_datetime_utc")
        if dt:
            if isinstance(dt, datetime):
                date_str = dt.strftime("%Y-%m-%d")
            else:
                date_str = str(dt)[:10]
            fixtures_by_date[date_str].append(fixture)

    # Build fixture results lookup
    fixture_results: dict[int, dict[str, Any]] = {}
    for fixture in fixtures:
        fixture_results[fixture["fixture_id"]] = fixture

    bankroll = initial_bankroll
    peak = bankroll
    max_drawdown = 0.0
    all_acca_ids: set[str] = set()
    settled_accas: list[SettledAcca] = []

    metrics = BacktestMetrics(starting_bankroll=initial_bankroll)

    # Process each day chronologically
    for date in sorted(fixtures_by_date.keys()):
        day_fixtures = fixtures_by_date[date]

        if verbose:
            print(f"\n{date}: Processing {len(day_fixtures)} fixtures...")

        # Process all fixtures for this day
        day_opps: list[MatchOpportunities] = []
        for fixture in day_fixtures:
            fixture_id = fixture["fixture_id"]
            opps = process_fixture(fixture_id, verbose)
            if opps:
                day_opps.append(opps)

        if not day_opps:
            continue

        # Build accas for this day
        if verbose:
            print(f"  Building accas with {strategy.value} strategy...")

        accas = build_accas(
            match_opps=day_opps,
            strategy=strategy,
            max_legs=max_legs,
            min_legs=min_legs,
            min_odds=1.3,
            max_accas=max_accas_per_day * 3,  # Get more options, then pick best
        )

        if not accas:
            continue

        # Select top accas for this day (up to max_accas_per_day)
        selected_accas = accas[:max_accas_per_day]

        for i, acca in enumerate(selected_accas):
            if bankroll <= 0:
                break

            stake = bankroll * stake_fraction
            if stake <= 0:
                break

            acca_id = f"{date}-{i+1:03d}"
            if acca_id in all_acca_ids:
                continue
            all_acca_ids.add(acca_id)

            # Settle the acca
            settled = settle_acca(acca, fixture_results, stake, acca_id, date)

            # Update bankroll
            bankroll += settled.pnl
            peak = max(peak, bankroll)
            if peak > 0:
                drawdown = (peak - bankroll) / peak
                max_drawdown = max(max_drawdown, drawdown)

            settled_accas.append(settled)

            if verbose:
                result_str = "WIN" if settled.hit else "LOSS"
                print(f"    {acca_id}: {result_str} | PnL: {settled.pnl:+.2f} | "
                      f"Odds: {settled.combined_odds:.2f} | Bankroll: {bankroll:.2f}")

            # Update metrics
            metrics.total_accas += 1
            metrics.total_stake += settled.stake
            metrics.total_pnl += settled.pnl
            metrics.total_return += settled.result_return_factor * settled.stake
            if settled.hit:
                metrics.hits += 1

            # Track by story signals
            for leg in settled.legs:
                for signal in leg.story_signals:
                    if signal not in metrics.by_story_type:
                        metrics.by_story_type[signal] = {"count": 0, "wins": 0, "pnl": 0.0}
                    metrics.by_story_type[signal]["count"] += 1
                    if leg.result == "win":
                        metrics.by_story_type[signal]["wins"] += 1
                    metrics.by_story_type[signal]["pnl"] += settled.pnl / len(settled.legs)

            # Track by market
            for leg in settled.legs:
                market = leg.market_code
                if market not in metrics.by_market:
                    metrics.by_market[market] = {"count": 0, "wins": 0, "pnl": 0.0}
                metrics.by_market[market]["count"] += 1
                if leg.result == "win":
                    metrics.by_market[market]["wins"] += 1
                metrics.by_market[market]["pnl"] += settled.pnl / len(settled.legs)

    metrics.ending_bankroll = bankroll
    metrics.max_drawdown = max_drawdown
    metrics.peak_bankroll = peak

    return settled_accas, metrics


# =============================================================================
# Output
# =============================================================================


def calculate_summary_metrics(metrics: BacktestMetrics) -> dict[str, Any]:
    """Calculate summary statistics from metrics."""
    roi = (metrics.total_pnl / metrics.total_stake) if metrics.total_stake > 0 else 0.0
    win_rate = (metrics.hits / metrics.total_accas) if metrics.total_accas > 0 else 0.0

    return {
        "total_accas": metrics.total_accas,
        "total_stake": round(metrics.total_stake, 2),
        "total_pnl": round(metrics.total_pnl, 2),
        "total_return": round(metrics.total_return, 2),
        "roi": round(roi, 4),
        "win_rate": round(win_rate, 4),
        "starting_bankroll": round(metrics.starting_bankroll, 2),
        "ending_bankroll": round(metrics.ending_bankroll, 2),
        "max_drawdown": round(metrics.max_drawdown, 4),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_md(path: Path, data: dict[str, Any]) -> None:
    """Write summary as markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = data.get("summary", {})
    by_story = data.get("by_story_type", {})
    by_market = data.get("by_market", {})

    lines = [
        "# Match Profile Backtest",
        "",
        f"- Generated: {data.get('generated_at_utc')}",
        f"- Since Days: {data.get('since_days')}",
        f"- Strategy: {data.get('strategy')}",
        "",
        "## Summary",
        "",
        f"- Total Accas: {summary.get('total_accas')}",
        f"- Total Stake: {summary.get('total_stake')}",
        f"- Total PnL: {summary.get('total_pnl')}",
        f"- ROI: {summary.get('roi'):.2%}",
        f"- Win Rate: {summary.get('win_rate'):.2%}",
        f"- Starting Bankroll: {summary.get('starting_bankroll')}",
        f"- Ending Bankroll: {summary.get('ending_bankroll')}",
        f"- Max Drawdown: {summary.get('max_drawdown'):.2%}",
        "",
        "## By Story Type",
        "",
        "| Story Signal | Count | Wins | Win Rate | PnL |",
        "|-------------|-------|------|----------|-----|",
    ]

    for signal, stats in sorted(by_story.items(), key=lambda x: x[1].get("pnl", 0), reverse=True):
        count = stats.get("count", 0)
        wins = stats.get("wins", 0)
        pnl = stats.get("pnl", 0)
        wr = wins / count if count > 0 else 0
        lines.append(f"| {signal} | {count} | {wins} | {wr:.1%} | {pnl:+.2f} |")

    lines.extend([
        "",
        "## By Market",
        "",
        "| Market | Count | Wins | Win Rate | PnL |",
        "|--------|-------|------|----------|-----|",
    ])

    for market, stats in sorted(by_market.items(), key=lambda x: x[1].get("pnl", 0), reverse=True):
        count = stats.get("count", 0)
        wins = stats.get("wins", 0)
        pnl = stats.get("pnl", 0)
        wr = wins / count if count > 0 else 0
        lines.append(f"| {market} | {count} | {wins} | {wr:.1%} | {pnl:+.2f} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def settled_acca_to_dict(acca: SettledAcca) -> dict[str, Any]:
    """Convert SettledAcca to dictionary."""
    return {
        "acca_id": acca.acca_id,
        "date": acca.date,
        "stake": acca.stake,
        "combined_odds": acca.combined_odds,
        "combined_prob": acca.combined_prob,
        "result_return_factor": acca.result_return_factor,
        "pnl": acca.pnl,
        "hit": acca.hit,
        "strategy": acca.strategy,
        "num_legs": len(acca.legs),
        "legs": [
            {
                "fixture_id": leg.fixture_id,
                "market_code": leg.market_code,
                "selection": leg.selection,
                "odds": leg.odds,
                "model_prob": leg.model_prob,
                "story_signals": list(leg.story_signals),
                "result": leg.result,
                "return_factor": leg.return_factor,
            }
            for leg in acca.legs
        ],
    }


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    args = parse_args()

    # Verify database connection
    try:
        _ = get_database_url()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Fetch fixtures
    print(f"Fetching fixtures from last {args.since_days} days...")
    fixtures = fetch_finished_fixtures(args.since_days, limit=args.limit)
    print(f"Found {len(fixtures)} finished fixtures")

    if not fixtures:
        print("No fixtures to process")
        return 0

    # Run backtest
    strategy = AccaStrategy(args.strategy)
    print(f"\nRunning backtest with {strategy.value} strategy...")
    print(f"  Initial bankroll: {args.bankroll}")
    print(f"  Stake fraction: {args.stake_fraction}")
    print(f"  Max legs: {args.max_legs}")
    print(f"  Min legs: {args.min_legs}")

    settled_accas, metrics = run_backtest(
        fixtures=fixtures,
        strategy=strategy,
        initial_bankroll=args.bankroll,
        stake_fraction=args.stake_fraction,
        max_legs=args.max_legs,
        min_legs=args.min_legs,
        max_accas_per_day=args.max_accas_per_day,
        verbose=args.verbose,
    )

    # Calculate summary
    summary = calculate_summary_metrics(metrics)

    # Print summary
    print(f"\n{'=' * 60}")
    print("BACKTEST RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total Accas: {summary['total_accas']}")
    print(f"  Total Stake: {summary['total_stake']:.2f}")
    print(f"  Total PnL: {summary['total_pnl']:+.2f}")
    print(f"  ROI: {summary['roi']:.2%}")
    print(f"  Win Rate: {summary['win_rate']:.2%}")
    print(f"  Starting Bankroll: {summary['starting_bankroll']:.2f}")
    print(f"  Ending Bankroll: {summary['ending_bankroll']:.2f}")
    print(f"  Max Drawdown: {summary['max_drawdown']:.2%}")

    # Write outputs
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    stem = f"match_profile_backtest_{ts}"
    out_dir = args.out_dir

    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"

    # Prepare output data
    output_data = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "since_days": args.since_days,
        "strategy": args.strategy,
        "bankroll": args.bankroll,
        "stake_fraction": args.stake_fraction,
        "max_legs": args.max_legs,
        "min_legs": args.min_legs,
        "summary": summary,
        "by_story_type": metrics.by_story_type,
        "by_market": metrics.by_market,
        "settled_accas": [settled_acca_to_dict(a) for a in settled_accas],
    }

    # CSV rows (one per acca)
    csv_rows = [
        {
            "acca_id": a.acca_id,
            "date": a.date,
            "stake": a.stake,
            "combined_odds": a.combined_odds,
            "result_return_factor": a.result_return_factor,
            "pnl": a.pnl,
            "hit": a.hit,
            "num_legs": len(a.legs),
        }
        for a in settled_accas
    ]

    write_csv(csv_path, csv_rows)
    write_json(json_path, output_data)
    write_md(md_path, output_data)

    print(f"\nWrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote MD: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
