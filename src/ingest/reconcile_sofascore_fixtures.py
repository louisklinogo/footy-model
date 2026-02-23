"""
Reconcile fixtures: find and update SofaScore match IDs.

This script:
1. Gets fixtures from database that don't have sofascore_id
2. Queries SofaScore API by date to find matching matches
3. Updates fixtures with sofascore_id

Usage:
    python src/ingest/reconcile_sofascore_fixtures.py --league E0 --days 7
    python src/ingest/reconcile_sofascore_fixtures.py --league E0 --all-unmatched
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import psycopg2

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import get_database_url

def get_league_sofascore_info(conn, league_code: str) -> dict | None:
    """Get SofaScore league/season IDs from database."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT sofascore_league_id, sofascore_season_id
            FROM leagues
            WHERE league_code = %s
            AND sofascore_league_id IS NOT NULL
        """, (league_code,))
        result = cur.fetchone()
        if result:
            return {"league_id": result[0], "season_id": result[1]}
        return None


async def get_matches_by_date(api, date_str: str) -> list[dict[str, Any]]:
    """Get all matches for a specific date from SofaScore."""
    from sofascore_wrapper.match import Match

    match = Match(api, 0)  # Dummy ID, we only use the class method
    try:
        data = await match.games_by_date("football", date_str)
        return data.get("events", [])
    except Exception as e:
        print(f"Error fetching matches for {date_str}: {e}")
        return []


def get_team_sofascore_id(conn, team_name: str) -> int | None:
    """Get SofaScore team ID from team_aliases."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ta.team_id
            FROM team_aliases ta
            WHERE ta.provider = 'sofascore' 
            AND LOWER(ta.alias_name) = LOWER(%s)
        """,
            (team_name,),
        )
        result = cur.fetchone()
        return result[0] if result else None


def get_unmatched_fixtures(
    conn, league_code: str, days: int | None = None
) -> list[dict]:
    """Get fixtures without sofascore_id."""
    with conn.cursor() as cur:
        sql = """
            SELECT 
                f.fixture_id,
                f.flashscore_id,
                f.league_code,
                ht.team_name AS home_team,
                at.team_name AS away_team,
                f.match_datetime_utc
            FROM fixtures f
            JOIN teams ht ON f.home_team_id = ht.team_id
            JOIN teams at ON f.away_team_id = at.team_id
            WHERE f.league_code = %s
            AND f.sofascore_id IS NULL
        """
        params = [league_code]

        if days:
            sql += " AND f.match_datetime_utc >= NOW() - INTERVAL '%s days'"
            params.append(days)

        sql += " ORDER BY f.match_datetime_utc"

        cur.execute(sql, params)
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def match_fixture_to_sofascore(
    fixture: dict, sofascore_matches: list[dict], team_aliases: dict[str, str]
) -> str | None:
    """Find matching SofaScore match ID for a fixture.
    
    team_aliases: dict mapping DB team name (lowercase) -> SofaScore team name (lowercase)
    """
    fixture_date = fixture["match_datetime_utc"]
    if not fixture_date:
        return None

    home_team = fixture["home_team"]
    away_team = fixture["away_team"]

    # Get SofaScore team names from aliases
    home_sofa_name = team_aliases.get(home_team.lower())
    away_sofa_name = team_aliases.get(away_team.lower())

    for match in sofascore_matches:
        match_home = match.get("homeTeam", {})
        match_away = match.get("awayTeam", {})

        # Try matching by SofaScore team name
        if home_sofa_name and away_sofa_name:
            match_home_name = match_home.get("name", "").lower()
            match_away_name = match_away.get("name", "").lower()
            if home_sofa_name in match_home_name or match_home_name in home_sofa_name:
                if away_sofa_name in match_away_name or match_away_name in away_sofa_name:
                    return str(match.get("id"))


def update_fixture_sofascore_id(conn, fixture_id: int, sofascore_id: str) -> bool:
    """Update fixture with sofascore_id."""
    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                UPDATE fixtures 
                SET sofascore_id = %s, updated_at = NOW()
                WHERE fixture_id = %s
            """,
                (sofascore_id, fixture_id),
            )
            return True
        except Exception as e:
            print(f"  Error updating fixture: {e}")
            return False


def load_team_aliases(conn) -> dict[str, str]:
    """Load mapping from DB team name to SofaScore team name."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT LOWER(t.team_name), LOWER(ta.alias_name)
            FROM teams t
            JOIN team_aliases ta ON t.team_id = ta.team_id
            WHERE ta.provider = 'sofascore'
        """)
        return {row[0]: row[1] for row in cur.fetchall()}


async def reconcile_fixtures(
    league_code: str, days: int | None = None, dry_run: bool = False
) -> None:
    """Main function to reconcile fixtures with SofaScore IDs."""
    from sofascore_wrapper.api import SofascoreAPI

    # Connect to database
    database_url = get_database_url()
    conn = psycopg2.connect(database_url)

    # Get league info from DB
    league_info = get_league_sofascore_info(conn, league_code)
    if not league_info:
        print(f"Error: League {league_code} not configured or missing SofaScore IDs.")
        conn.close()
        return

    try:
        # Load team aliases
        team_aliases = load_team_aliases(conn)
        print(f"Loaded {len(team_aliases)} team aliases")

        # Get unmatched fixtures
        fixtures = get_unmatched_fixtures(conn, league_code, days)
        print(f"Found {len(fixtures)} fixtures without sofascore_id")

        if not fixtures:
            print("Nothing to do.")
            return

        # Group fixtures by date
        fixtures_by_date: dict[str, list[dict]] = {}
        for fixture in fixtures:
            if fixture["match_datetime_utc"]:
                date_str = fixture["match_datetime_utc"].strftime("%Y-%m-%d")
                if date_str not in fixtures_by_date:
                    fixtures_by_date[date_str] = []
                fixtures_by_date[date_str].append(fixture)

        # Process each date
        api = SofascoreAPI()
        try:
            matched = 0
            unmatched = []

            for date_str, date_fixtures in sorted(fixtures_by_date.items()):
                print(f"\nProcessing {date_str} ({len(date_fixtures)} fixtures)...")

                # Fetch SofaScore matches for this date
                sofascore_matches = await get_matches_by_date(api, date_str)
                print(f"  Found {len(sofascore_matches)} SofaScore matches")

                for fixture in date_fixtures:
                    sofascore_id = match_fixture_to_sofascore(
                        fixture, sofascore_matches, team_aliases
                    )

                    if sofascore_id:
                        print(
                            f"  ✓ {fixture['home_team']} vs {fixture['away_team']} -> {sofascore_id}"
                        )
                        if not dry_run:
                            update_fixture_sofascore_id(
                                conn, fixture["fixture_id"], sofascore_id
                            )
                        matched += 1
                    else:
                        unmatched.append(fixture)
                        print(
                            f"  ✗ {fixture['home_team']} vs {fixture['away_team']} -> NO MATCH"
                        )

            if not dry_run:
                conn.commit()
                print(f"\nCommitted {matched} sofascore_ids to database")
            else:
                print(f"\nDry run: would update {matched} fixtures")

            if unmatched:
                print(f"\nUnmatched fixtures ({len(unmatched)}):")
                for f in unmatched:
                    print(
                        f"  - {f['home_team']} vs {f['away_team']} ({f['match_datetime_utc']})"
                    )

        finally:
            await api.close()

    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile fixtures with SofaScore IDs"
    )
    parser.add_argument("--league", required=True, help="League code (e.g., E0)")
    parser.add_argument(
        "--days", type=int, help="Only process fixtures from last N days"
    )
    parser.add_argument(
        "--all-unmatched", action="store_true", help="Process all unmatched fixtures"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show matches without updating"
    )
    args = parser.parse_args()

    days = None if args.all_unmatched else (args.days or 7)
    asyncio.run(reconcile_fixtures(args.league, days, args.dry_run))


if __name__ == "__main__":
    main()
