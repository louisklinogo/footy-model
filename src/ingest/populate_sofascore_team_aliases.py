"""
Populate team_aliases with SofaScore team mappings.

This script:
1. Gets all leagues from our DB
2. Gets all teams from our DB grouped by league
3. Fetches teams from SofaScore for each league (by mapping league_code to sofascore league_id)
4. Does fuzzy matching to map SofaScore teams to our teams
5. Inserts into team_aliases table

Usage:
    python src/ingest/populate_sofascore_team_aliases.py --dry-run
    python src/ingest/populate_sofascore_team_aliases.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import io
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


def normalize_name(name: str) -> str:
    """Normalize team name for comparison."""
    return (
        name.lower()
        .strip()
        .replace("&", "and")
        .replace("fc", "")
        .replace("afc", "")
        .replace("  ", " ")
        .strip()
    )


def get_db_leagues(conn) -> list[dict]:
    """Get all leagues from database."""
    with conn.cursor() as cur:
        cur.execute("SELECT league_code, league_name FROM leagues ORDER BY league_code")
        return [
            {"league_code": row[0], "league_name": row[1]} for row in cur.fetchall()
        ]


def get_db_teams(conn) -> dict[str, dict[str, int]]:
    """Get all teams from database grouped by league_code."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT t.team_id, t.team_name, t.league_code 
            FROM teams t
            WHERE t.league_code IS NOT NULL
            ORDER BY t.league_code, t.team_name
        """)
        teams_by_league: dict[str, dict[str, int]] = {}
        for row in cur.fetchall():
            league_code = row[2]
            if league_code not in teams_by_league:
                teams_by_league[league_code] = {}
            teams_by_league[league_code][row[1]] = row[0]
        return teams_by_league


async def find_sofascore_league(api, league_name: str) -> dict | None:
    """Find SofaScore league by name."""
    from sofascore_wrapper.search import Search

    search = Search(api, league_name)
    results = await search.search_leagues(sport="football")

    for item in results.get("results", []):
        if item.get("type") == "uniqueTournament":
            entity = item.get("entity", {})
            return {
                "sofascore_id": entity.get("id"),
                "sofascore_name": entity.get("name"),
            }
    return None





async def get_sofascore_teams_for_league(
    api, sofascore_league_id: int, season_id: int | None = None
) -> list[dict]:
    """Get all teams from SofaScore for a league."""
    from sofascore_wrapper.league import League

    league = League(api, sofascore_league_id)

    # Get current season if not provided
    if not season_id:
        try:
            seasons = await league.get_seasons()
            if seasons:
                season_id = seasons[0].get("id")
        except Exception as e:
            print(f"  Error getting seasons: {e}")





    if not season_id:
        return []

    try:
        standings = await league.standings(season_id)
        teams = []
        for row in standings.get("standings", [{}])[0].get("rows", []):
            team = row.get("team", {})
            teams.append(
                {
                    "sofascore_id": team.get("id"),
                    "sofascore_name": team.get("name"),
                    "short_name": team.get("shortName"),
                }
            )
        return teams
    except Exception as e:
        print(f"  Error fetching teams: {e}")
        return []


def fuzzy_match_team(
    sofascore_name: str, db_teams: dict[str, int]
) -> tuple[int | None, str | None]:
    """Find best matching team using fuzzy logic."""
    sofa_normalized = normalize_name(sofascore_name)

    # 1. Try exact match (normalized)
    for team_name, team_id in db_teams.items():
        if normalize_name(team_name) == sofa_normalized:
            return team_id, team_name

    # 2. Try substring match (either contains the other)
    for team_name, team_id in db_teams.items():
        db_normalized = normalize_name(team_name)
        if db_normalized in sofa_normalized or sofa_normalized in db_normalized:
            return team_id, team_name

    # 3. Try matching without common suffixes
    sofa_stripped = sofa_normalized
    for suffix in [
        "united",
        "city",
        "fc",
        "town",
        "albion",
        "rovers",
        "wanderers",
        "athletic",
    ]:
        sofa_stripped = sofa_stripped.replace(suffix, "").strip()

    for team_name, team_id in db_teams.items():
        db_stripped = normalize_name(team_name)
        for suffix in [
            "united",
            "city",
            "fc",
            "town",
            "albion",
            "rovers",
            "wanderers",
            "athletic",
        ]:
            db_stripped = db_stripped.replace(suffix, "").strip()
        if (
            db_stripped
            and sofa_stripped
            and (
                db_stripped == sofa_stripped
                or db_stripped in sofa_stripped
                or sofa_stripped in db_stripped
            )
        ):
            return team_id, team_name

    return None, None


def insert_team_alias(conn, team_id: int, sofascore_name: str) -> bool:
    """Insert a team alias mapping."""
    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                INSERT INTO team_aliases (provider, alias_name, team_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (provider, alias_name) DO UPDATE
                SET team_id = EXCLUDED.team_id
            """,
                ("sofascore", sofascore_name, team_id),
            )
            return True
        except Exception as e:
            print(f"  Error inserting alias: {e}")
            return False


async def populate_aliases(dry_run: bool = False) -> None:
    """Main function to populate team aliases dynamically."""
    from sofascore_wrapper.api import SofascoreAPI

    database_url = get_database_url()
    conn = psycopg2.connect(database_url)

    try:
        # Get all leagues and teams from DB
        db_leagues = get_db_leagues(conn)
        db_teams_by_league = get_db_teams(conn)

        print(f"Found {len(db_leagues)} leagues in database")
        print(
            f"Found {sum(len(t) for t in db_teams_by_league.values())} teams in database across {len(db_teams_by_league)} leagues"
        )

        api = SofascoreAPI()
        try:
            total_matched = 0
            total_unmatched = 0

            for league in db_leagues:
                league_code = league["league_code"]
                league_name = league["league_name"]

                if league_code not in db_teams_by_league:
                    continue

                db_teams = db_teams_by_league[league_code]
                print(f"\n[{league_code}] {league_name} ({len(db_teams)} teams in DB)")

                # Find SofaScore league
                sofascore_league = await find_sofascore_league(api, league_name)
                if not sofascore_league or not sofascore_league.get("sofascore_id"):
                    print(f"  Could not find SofaScore league for {league_name}")
                    continue

                print(
                    f"  Found SofaScore league: {sofascore_league['sofascore_name']} (ID: {sofascore_league['sofascore_id']})"
                )

                # Get teams from SofaScore
                sofascore_teams = await get_sofascore_teams_for_league(
                    api, sofascore_league["sofascore_id"]
                )

                if not sofascore_teams:
                    print(f"  No teams found on SofaScore")
                    continue

                print(f"  Found {len(sofascore_teams)} teams on SofaScore")

                # Match teams
                for sofa_team in sofascore_teams:
                    team_id, matched_name = fuzzy_match_team(
                        sofa_team["sofascore_name"], db_teams
                    )

                    if team_id:
                        print(f"  [OK] {sofa_team['sofascore_name']} -> {matched_name}")
                        if not dry_run:
                            insert_team_alias(
                                conn, team_id, sofa_team["sofascore_name"]
                            )
                        total_matched += 1
                    else:
                        print(f"  [NO MATCH] {sofa_team['sofascore_name']}")
                        total_unmatched += 1

            if not dry_run:
                conn.commit()
                print(f"\nCommitted {total_matched} team aliases to database")
            else:
                print(f"\nDry run: would insert {total_matched} team aliases")

            if total_unmatched:
                print(f"Unmatched teams: {total_unmatched}")

        finally:
            await api.close()

    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate SofaScore team aliases dynamically"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show matches without inserting"
    )
    args = parser.parse_args()

    asyncio.run(populate_aliases(args.dry_run))


if __name__ == "__main__":
    main()
