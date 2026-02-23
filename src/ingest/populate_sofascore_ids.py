"""
Populate SofaScore IDs for leagues and teams.

This script:
1. Searches SofaScore for each league in our DB
2. Updates leagues with sofascore_league_id and sofascore_season_id
3. Gets teams from standings for each league
4. Updates teams with sofascore_id

Usage:
    python src/ingest/populate_sofascore_ids.py --leagues E0,E1
    python src/ingest/populate_sofascore_ids.py --all
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import io
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import psycopg2

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import get_database_url


async def search_league(api, league_name: str) -> dict | None:
    """Search for a league on SofaScore."""
    from sofascore_wrapper.search import Search

    search = Search(api, league_name)
    data = await search.search_leagues(sport="football")

    # search_leagues returns {"results": [...]}
    results = data.get("results", [])
    if results:
        # Return first match's entity (the league info)
        return results[0].get("entity")
    return None


async def get_league_seasons(api, league_id: int) -> list[dict]:
    """Get available seasons for a league."""
    from sofascore_wrapper.league import League

    league = League(api, league_id)
    seasons = await league.get_seasons()
    return seasons


async def get_league_teams(api, league_id: int, season_id: int) -> list[dict]:
    """Get teams from league standings."""
    from sofascore_wrapper.league import League

    league = League(api, league_id)
    try:
        data = await league.standings(season_id)
        if data:
            # Structure: {"standings": [{"rows": [{"team": {...}}]}]}
            teams = []
            for table in data.get("standings", []):
                for row in table.get("rows", []):
                    team = row.get("team")
                    if team:
                        teams.append(
                            {
                                "id": team.get("id"),
                                "name": team.get("name"),
                            }
                        )
            return teams
    except Exception as e:
        print(f"  Error getting standings: {e}")
    return []


def get_db_leagues(conn, league_codes: list[str] | None = None) -> list[dict]:
    """Get leagues from database."""
    with conn.cursor() as cur:
        if league_codes:
            cur.execute(
                """
                SELECT league_id, league_code, league_name, country
                FROM leagues
                WHERE league_code = ANY(%s)
                """,
                (league_codes,),
            )
        else:
            cur.execute("""
                SELECT league_id, league_code, league_name, country
                FROM leagues
            """)
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def update_league_sofascore_ids(
    conn, league_id: int, sofascore_league_id: int, sofascore_season_id: int
) -> bool:
    """Update league with SofaScore IDs."""
    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                UPDATE leagues
                SET sofascore_league_id = %s,
                    sofascore_season_id = %s,
                    updated_at = NOW()
                WHERE league_id = %s
                """,
                (sofascore_league_id, sofascore_season_id, league_id),
            )
            return True
        except Exception as e:
            print(f"  Error updating league: {e}")
            return False


def get_team_aliases_map(conn) -> dict[str, tuple[int, str]]:
    """Get mapping from lowercase alias name to (team_id, db_team_name)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT LOWER(ta.alias_name), t.team_id, t.team_name
            FROM team_aliases ta
            JOIN teams t ON ta.team_id = t.team_id
            WHERE ta.provider = 'sofascore'
        """)
        return {row[0]: (row[1], row[2]) for row in cur.fetchall()}


def update_team_sofascore_id(conn, team_id: int, sofascore_id: int) -> bool:
    """Update team with SofaScore ID."""
    with conn.cursor() as cur:
        try:
            cur.execute(
                """
                UPDATE teams
                SET sofascore_id = %s, updated_at = NOW()
                WHERE team_id = %s
                """,
                (sofascore_id, team_id),
            )
            return True
        except Exception as e:
            print(f"    Error updating team: {e}")
            return False


async def populate_ids(
    league_codes: list[str] | None = None, dry_run: bool = False
) -> None:
    """Main function to populate SofaScore IDs."""
    from sofascore_wrapper.api import SofascoreAPI

    database_url = get_database_url()
    conn = psycopg2.connect(database_url)

    try:
        # Get leagues from DB
        leagues = get_db_leagues(conn, league_codes)
        print(f"Found {len(leagues)} leagues to process")

        # Load team aliases
        team_aliases = get_team_aliases_map(conn)
        print(f"Loaded {len(team_aliases)} team aliases")

        api = SofascoreAPI()
        try:
            for league in leagues:
                print(
                    f"\nProcessing {league['league_code']} - {league['league_name']}..."
                )

                # Search for league on SofaScore
                search_name = league["league_name"]
                if league["country"]:
                    search_name = f"{league['country']} {league['league_name']}"

                print(f"  Searching for: {search_name}")
                sofa_league = await search_league(api, search_name)

                if not sofa_league:
                    print(f"  ✗ League not found on SofaScore")
                    continue

                sofa_league_id = sofa_league.get("id")
                sofa_league_name = sofa_league.get("name", "Unknown")
                print(f"  ✓ Found: {sofa_league_name} (ID: {sofa_league_id})")

                # Get seasons
                seasons = await get_league_seasons(api, sofa_league_id)
                if not seasons:
                    print(f"  ✗ No seasons found")
                    continue

                # Use the latest season (first in list, usually current)
                current_season = seasons[0]
                season_id = current_season.get("id")
                year = current_season.get("year", "unknown")
                print(f"  ✓ Season: {year} (ID: {season_id})")

                # Update league
                if not dry_run:
                    update_league_sofascore_ids(
                        conn, league["league_id"], sofa_league_id, season_id
                    )
                    print(f"  ✓ Updated league IDs")

                # Get teams from standings
                teams = await get_league_teams(api, sofa_league_id, season_id)
                print(f"  Found {len(teams)} teams in standings")

                matched = 0
                for team in teams:
                    sofa_team_name = team["name"].lower()
                    sofa_team_id = team["id"]

                    # Try to match by alias
                    if sofa_team_name in team_aliases:
                        team_id, db_team_name = team_aliases[sofa_team_name]
                        if not dry_run:
                            update_team_sofascore_id(conn, team_id, sofa_team_id)
                        print(
                            f"    ✓ {db_team_name} -> {team['name']} (ID: {sofa_team_id})"
                        )
                        matched += 1
                    else:
                        # Try fuzzy match
                        found = False
                        for alias_name, (team_id, db_team_name) in team_aliases.items():
                            if (
                                alias_name in sofa_team_name
                                or sofa_team_name in alias_name
                            ):
                                if not dry_run:
                                    update_team_sofascore_id(
                                        conn, team_id, sofa_team_id
                                    )
                                print(
                                    f"    ✓ {db_team_name} -> {team['name']} (ID: {sofa_team_id}) [fuzzy]"
                                )
                                matched += 1
                                found = True
                                break
                        if not found:
                            print(f"    ✗ No match for: {team['name']}")

                print(f"  Matched {matched}/{len(teams)} teams")

            if not dry_run:
                conn.commit()
                print("\nCommitted changes to database")
            else:
                print("\nDry run - no changes made")

        finally:
            await api.close()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate SofaScore IDs")
    parser.add_argument("--leagues", help="Comma-separated league codes (e.g., E0,E1)")
    parser.add_argument("--all", action="store_true", help="Process all leagues")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show changes without updating"
    )
    args = parser.parse_args()

    league_codes = None
    if args.leagues:
        league_codes = [c.strip() for c in args.leagues.split(",")]
    elif not args.all:
        print("Specify --leagues or --all")
        sys.exit(1)

    asyncio.run(populate_ids(league_codes, args.dry_run))


if __name__ == "__main__":
    main()
