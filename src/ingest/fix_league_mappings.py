"""
Fix incorrect SofaScore league mappings.

This script:
1. Searches for correct league IDs for leagues with missing/wrong mappings
2. Updates the leagues table with correct sofascore_league_id and sofascore_season_id

Usage:
    python src/ingest/fix_league_mappings.py --dry-run
    python src/ingest/fix_league_mappings.py
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


# Manual overrides for leagues that need specific search terms
LEAGUE_SEARCH_OVERRIDES = {
    "I2": "Italy Serie B",  # Serie B - needs exact name
    "B1": "Belgium First Division A",  # Jupiler Pro League
    "RO1": "Romania SuperLiga",        # Liga I - use SuperLiga name
    "RU1": "Russia Premier League",
}


async def search_league(api, search_term: str) -> list[dict]:
    """Search for a league on SofaScore and return all matches."""
    from sofascore_wrapper.search import Search

    search = Search(api, search_term)
    data = await search.search_leagues(sport="football")

    results = data.get("results", [])
    leagues = []
    for item in results:
        if item.get("type") == "uniqueTournament":
            entity = item.get("entity", {})
            leagues.append(
                {
                    "id": entity.get("id"),
                    "name": entity.get("name"),
                    "slug": entity.get("slug"),
                }
            )
    return leagues


async def get_league_seasons(api, league_id: int) -> list[dict]:
    """Get available seasons for a league."""
    from sofascore_wrapper.league import League

    league = League(api, league_id)
    seasons = await league.get_seasons()
    return seasons


async def fix_league_mappings(dry_run: bool = False) -> None:
    """Fix incorrect league mappings."""
    from sofascore_wrapper.api import SofascoreAPI

    database_url = get_database_url()
    conn = psycopg2.connect(database_url)

    try:
        api = SofascoreAPI()
        try:
            # Get leagues that need fixing
            leagues_to_fix = []

            # Check I2 - currently mapped to Serie A (wrong)
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT league_id, league_code, league_name, country, 
                           sofascore_league_id, sofascore_season_id
                    FROM leagues 
                    WHERE league_code = 'I2'
                """)
                row = cur.fetchone()
                if row:
                    leagues_to_fix.append(
                        {
                            "league_id": row[0],
                            "league_code": row[1],
                            "league_name": row[2],
                            "country": row[3],
                            "current_sofa_id": row[4],
                            "current_season_id": row[5],
                        }
                    )

            # Check B1 - no mapping
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT league_id, league_code, league_name, country, 
                           sofascore_league_id, sofascore_season_id
                    FROM leagues 
                    WHERE league_code = 'B1'
                """)
                row = cur.fetchone()
                if row:
                    leagues_to_fix.append(
                        {
                            "league_id": row[0],
                            "league_code": row[1],
                            "league_name": row[2],
                            "country": row[3],
                            "current_sofa_id": row[4],
                            "current_season_id": row[5],
                        }
                    )

            # Check RO1 - might be wrong
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT league_id, league_code, league_name, country, 
                           sofascore_league_id, sofascore_season_id
                    FROM leagues 
                    WHERE league_code = 'RO1'
                """)
                row = cur.fetchone()
                if row:
                    leagues_to_fix.append(
                        {
                            "league_id": row[0],
                            "league_code": row[1],
                            "league_name": row[2],
                            "country": row[3],
                            "current_sofa_id": row[4],
                            "current_season_id": row[5],
                        }
                    )

            print(f"Checking {len(leagues_to_fix)} leagues for correct mappings...\n")

            for league in leagues_to_fix:
                print(
                    f"[{league['league_code']}] {league['league_name']} ({league['country']})"
                )
                print(
                    f"  Current: league_id={league['current_sofa_id']}, season_id={league['current_season_id']}"
                )

                # Search with override term if available
                search_term = LEAGUE_SEARCH_OVERRIDES.get(
                    league["league_code"],
                    f"{league['country']} {league['league_name']}"
                    if league["country"]
                    else league["league_name"],
                )
                print(f"  Searching for: '{search_term}'")

                matches = await search_league(api, search_term)

                if not matches:
                    # Try alternative searches
                    alt_terms = [
                        league["league_name"],
                        league["league_name"].replace("Liga 1", "Liga I"),
                        league["league_name"].replace("Jupiler", "Pro League"),
                    ]
                    for alt_term in alt_terms:
                        matches = await search_league(api, alt_term)
                        if matches:
                            print(f"  Found with alternative: '{alt_term}'")
                            break

                if matches:
                    print(f"  Found {len(matches)} matching leagues:")
                    for i, m in enumerate(matches[:5]):
                        print(f"    {i + 1}. {m['name']} (ID: {m['id']})")

                    # Pick the best match
                    best_match = matches[0]

                    # For Serie B, we need to be careful - Serie A is ID 23
                    if league["league_code"] == "I2":
                        # Find Serie B (should be different from Serie A)
                        for m in matches:
                            if m["id"] != 23:  # Not Serie A
                                best_match = m
                                print(
                                    f"  -> Selecting Serie B: {m['name']} (ID: {m['id']})"
                                )
                                break

                    # Get seasons
                    seasons = await get_league_seasons(api, best_match["id"])
                    if seasons:
                        current_season = seasons[0]
                        print(
                            f"  Latest season: {current_season.get('year')} (ID: {current_season.get('id')})"
                        )

                        if not dry_run:
                            with conn.cursor() as cur:
                                cur.execute(
                                    """
                                    UPDATE leagues
                                    SET sofascore_league_id = %s,
                                        sofascore_season_id = %s,
                                        updated_at = NOW()
                                    WHERE league_id = %s
                                """,
                                    (
                                        best_match["id"],
                                        current_season.get("id"),
                                        league["league_id"],
                                    ),
                                )
                            print(f"  ✓ Updated!")
                        else:
                            print(
                                f"  [DRY RUN] Would update to league_id={best_match['id']}, season_id={current_season.get('id')}"
                            )
                    else:
                        print(f"  ✗ No seasons found")
                else:
                    print(f"  ✗ No matches found on SofaScore")

                print()

            if not dry_run:
                conn.commit()
                print("Committed changes to database")
            else:
                print("Dry run - no changes made")

        finally:
            await api.close()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix incorrect league mappings")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show changes without updating"
    )
    args = parser.parse_args()

    asyncio.run(fix_league_mappings(args.dry_run))


if __name__ == "__main__":
    main()
