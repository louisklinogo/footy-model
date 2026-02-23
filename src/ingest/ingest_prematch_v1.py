"""Ingest prematch JSON files into database.

Reads prematch JSON files (odds + injuries/availability) and ingests to:
- fixture_availability table (injuries, lineups)
- fixture_odds_snapshots table (prematch odds)

Usage:
    python src/ingest/ingest_prematch_v1.py --league E0
    python src/ingest/ingest_prematch_v1.py --all
    python src/ingest/ingest_prematch_v1.py --dry-run --league E0
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from psycopg2.extras import Json, execute_values

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db.db_utils import connect_db

PREMATCH_ROOT = ROOT / "data" / "v1" / "prematch_json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest prematch JSON files to database"
    )
    parser.add_argument("--league", type=str, help="Single league code to process")
    parser.add_argument("--all", action="store_true", help="Process all leagues")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )
    parser.add_argument("--skip-odds", action="store_true", help="Skip odds ingestion")
    return parser.parse_args()


def get_fixture_id_by_flashscore(cur, flashscore_id: str) -> int | None:
    """Look up fixture_id by flashscore_id."""
    cur.execute(
        "SELECT fixture_id FROM fixtures WHERE flashscore_id = %s",
        (flashscore_id,),
    )
    row = cur.fetchone()
    return row[0] if row else None


def ingest_availability(
    cur,
    fixture_id: int,
    availability: dict,
    scraped_at: datetime | None,
    dry_run: bool = False,
) -> bool:
    """Upsert availability data into fixture_availability table."""
    home = availability.get("home", {})
    away = availability.get("away", {})

    if dry_run:
        return True

    cur.execute(
        """
        INSERT INTO fixture_availability (
            fixture_id,
            home_missing, home_questionable, home_lineup,
            away_missing, away_questionable, away_lineup,
            scraped_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (fixture_id) DO UPDATE SET
            home_missing = EXCLUDED.home_missing,
            home_questionable = EXCLUDED.home_questionable,
            home_lineup = EXCLUDED.home_lineup,
            away_missing = EXCLUDED.away_missing,
            away_questionable = EXCLUDED.away_questionable,
            away_lineup = EXCLUDED.away_lineup,
            scraped_at = EXCLUDED.scraped_at,
            ingested_at = NOW()
        """,
        (
            fixture_id,
            Json(home.get("missing", [])),
            Json(home.get("questionable", [])),
            Json(home.get("lineup", [])),
            Json(away.get("missing", [])),
            Json(away.get("questionable", [])),
            Json(away.get("lineup", [])),
            scraped_at,
        ),
    )
    return True


def ingest_odds(
    cur,
    fixture_id: int,
    odds: dict,
    scraped_at: datetime | None,
    dry_run: bool = False,
) -> bool:
    """Upsert prematch odds into fixture_odds_snapshots table using DB function."""
    if dry_run:
        return True

    ou_json = odds.get("ou", {})
    ah_json = odds.get("ah", {})
    one_x_two_json = odds.get("1x2", {})

    # Use the DB function we created
    cur.execute(
        "SELECT upsert_prematch_odds(%s, %s, %s, %s, %s)",
        (fixture_id, Json(ou_json), Json(ah_json), Json(one_x_two_json), scraped_at),
    )
    return True


def process_prematch_file(
    cur,
    json_path: Path,
    skip_odds: bool = False,
    dry_run: bool = False,
) -> tuple[str, str]:
    """Process a single prematch JSON file. Returns (status, message)."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return ("error", f"Invalid JSON: {e}")

    flashscore_id = data.get("id") or json_path.stem
    availability = data.get("availability", {})
    odds = data.get("odds", {})
    scraped_at_str = data.get("scraped_at")

    # Parse scraped_at
    scraped_at = None
    if scraped_at_str:
        try:
            scraped_at = datetime.fromisoformat(scraped_at_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass

    # Look up fixture_id
    fixture_id = get_fixture_id_by_flashscore(cur, flashscore_id)
    if not fixture_id:
        return ("skipped", f"Fixture not found in DB: {flashscore_id}")

    # Ingest availability
    try:
        ingest_availability(cur, fixture_id, availability, scraped_at, dry_run)
    except Exception as e:
        return ("error", f"Availability ingest failed: {e}")

    # Ingest odds
    if not skip_odds and odds:
        try:
            ingest_odds(cur, fixture_id, odds, scraped_at, dry_run)
        except Exception as e:
            return ("error", f"Odds ingest failed: {e}")

    # Count injuries for reporting
    home_injuries = len(availability.get("home", {}).get("missing", []))
    away_injuries = len(availability.get("away", {}).get("missing", []))

    return ("success", f"{flashscore_id}: {home_injuries}h/{away_injuries}a injuries")


def process_league(
    league: str,
    skip_odds: bool = False,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Process all prematch JSON files for a league. Returns (success, skipped, errors)."""
    league_dir = PREMATCH_ROOT / league
    if not league_dir.exists():
        print(f"  No prematch directory for {league}")
        return (0, 0, 0)

    json_files = list(league_dir.glob("*.json"))
    if not json_files:
        print(f"  No JSON files found for {league}")
        return (0, 0, 0)

    print(f"  Processing {len(json_files)} files for {league}...")

    conn = connect_db()
    success, skipped, errors = 0, 0, 0

    try:
        with conn:
            with conn.cursor() as cur:
                for json_path in json_files:
                    status, msg = process_prematch_file(
                        cur, json_path, skip_odds, dry_run
                    )
                    if status == "success":
                        success += 1
                        if success <= 5 or success % 20 == 0:
                            print(f"    [OK] {msg}")
                    elif status == "skipped":
                        skipped += 1
                    else:
                        errors += 1
                        print(f"    [ERR] {msg}")

        if not dry_run:
            conn.commit()
    finally:
        conn.close()

    return (success, skipped, errors)


def discover_leagues() -> list[str]:
    """Discover all league directories in prematch_json."""
    if not PREMATCH_ROOT.exists():
        return []
    return sorted([d.name for d in PREMATCH_ROOT.iterdir() if d.is_dir()])


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("Prematch Data Ingestion")
    print("=" * 60)

    if args.dry_run:
        print("DRY RUN MODE - no changes will be made")
        print()

    # Determine which leagues to process
    if args.all:
        leagues = discover_leagues()
        if not leagues:
            print("No prematch JSON directories found.")
            return 1
        print(f"Processing all leagues: {', '.join(leagues)}")
    elif args.league:
        leagues = [args.league]
    else:
        print("Specify --league CODE or --all")
        return 1

    print()

    total_success, total_skipped, total_errors = 0, 0, 0

    for league in leagues:
        s, sk, e = process_league(league, args.skip_odds, args.dry_run)
        total_success += s
        total_skipped += sk
        total_errors += e

    print()
    print("Summary:")
    print(f"  Success: {total_success}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Errors:  {total_errors}")

    if args.dry_run:
        print()
        print("DRY RUN - no changes were made to the database")

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
