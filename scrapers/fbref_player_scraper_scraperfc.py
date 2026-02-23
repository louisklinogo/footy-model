#!/usr/bin/env python3
"""
FBref Player Stats Scraper using ScraperFC

Scrapes player stats from FBref for leagues using ScraperFC's FBref module.
Handles Cloudflare automatically.

Usage:
    python scrapers/fbref_player_scraper_scraperfc.py [league_code]

Examples:
    python scrapers/fbref_player_scraper_scraperfc.py E0    # Premier League
    python scrapers/fbref_player_scraper_scraperfc.py        # All leagues
"""

import ScraperFC as sfc
import pandas as pd
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from db.db_utils import connect_db

# Load league mapping from existing file
MAPPING_PATH = PROJECT_ROOT / "data" / "v1" / "fbref_league_mapping.json"
with open(MAPPING_PATH) as f:
    LEAGUE_MAPPING = json.load(f)

# Current season for FBref
# Use 2025-2026 if data is available, otherwise fallback to 2024-2025
CURRENT_SEASON = "2025-2026"

# Map fbref_league_mapping.json names to ScraperFC names
LEAGUE_NAME_MAP = {
    "Premier League": "England Premier League",
    "La Liga": "Spain La Liga",
    "Bundesliga": "Germany Bundesliga",
    "Serie A": "Italy Serie A",
    "Ligue 1": "France Ligue 1",
    "Eredivisie": "Netherlands Eredivisie",
    "Primeira Liga": "Portugal Primeira Liga",
}


def get_player_stats(
    league_code: str, season: str = CURRENT_SEASON
) -> pd.DataFrame | None:
    """Scrape player stats for a league."""
    fb = sfc.FBref(wait_time=6)

    # Get FBref competition name from mapping
    league_info = LEAGUE_MAPPING.get(league_code)
    if not league_info:
        print(f"  [WARN] Unknown league code: {league_code}")
        return None

    fb_name_raw = league_info["fbref_name"]
    fb_name = LEAGUE_NAME_MAP.get(fb_name_raw, fb_name_raw)

    try:
        print(f"  Scraping {league_code} ({fb_name}) - {season}")

        # Try scraping all stats categories to see what's available
        print(f"  Scraping all stats for {season}...")
        all_stats = fb.scrape_all_stats(season, fb_name)
        
        print(f"  Available categories: {list(all_stats.keys())}")
        
        # Find first category with player data
        result = None
        for cat, data_tuple in all_stats.items():
            if data_tuple and len(data_tuple) >= 3:
                player_data = data_tuple[2]  # player_stats is 3rd element in tuple
                if player_data is not None and not player_data.empty:
                    print(f"  Found player data in category: {cat} ({len(player_data)} players)")
                    result = {cat: player_data}
                    break
        
        if result is None:
            print(f"  [WARN] No player data found in any category")
            return None

        player_stats = result.get("player_stats")

        if player_stats is None or player_stats.empty:
            print(f"  [WARN] No player stats for {league_code}")
            return None

        # Add league info
        player_stats["league_code"] = league_code
        player_stats["league_name"] = fb_name
        player_stats["season"] = season

        print(f"  [OK] Found {len(player_stats)} players")
        return player_stats

    except Exception as e:
        print(f"  [ERROR] {league_code}: {e}")
        return None


def scrape_to_db(league_code: str = None):
    """Scrape and save to DB."""
    leagues_to_scrape = []

    if league_code:
        # Single league
        if league_code not in LEAGUE_MAPPING:
            print(f"[ERROR] Unknown league code: {league_code}")
            return
        leagues_to_scrape = [league_code]
    else:
        # All leagues
        leagues_to_scrape = list(LEAGUE_MAPPING.keys())

    all_players = []

    for lc in leagues_to_scrape:
        print(f"\n[{lc}] Scraping")
        df = get_player_stats(lc, CURRENT_SEASON)

        if df is not None and not df.empty:
            all_players.append(df)

    if not all_players:
        print("No data scraped")
        return

    # Combine all
    combined = pd.concat(all_players, ignore_index=True)
    print(f"\nTotal players scraped: {len(combined)}")

    # Save to DB
    save_to_db(combined)

    return combined


def save_to_db(df: pd.DataFrame):
    """Save player stats to database."""
    conn = connect_db()
    cur = conn.cursor()

    # Create table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS player_season_stats (
            id SERIAL PRIMARY KEY,
            player_name TEXT,
            team TEXT,
            position TEXT,
            age TEXT,
            nationality TEXT,
            games INTEGER,
            minutes INTEGER,
            goals INTEGER,
            assists INTEGER,
            xg REAL,
            xa REAL,
            cards_yellow INTEGER,
            cards_red INTEGER,
            passes_completed INTEGER,
            passes_attempted REAL,
            pass_accuracy REAL,
            tackles INTEGER,
            interceptions INTEGER,
            blocks INTEGER,
            shots_on_target INTEGER,
            league TEXT,
            season TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (player_name, team, season)
        )
    """)

    # Insert rows
    records = df.to_dict("records")

    for record in records:
        try:
            cur.execute(
                """
                INSERT INTO player_season_stats (
                    player_name, team, position, age, nationality,
                    games, minutes, goals, assists, xg, xa,
                    cards_yellow, cards_red, passes_completed, passes_attempted,
                    pass_accuracy, tackles, interceptions, blocks, shots_on_target,
                    league, season
                )
                VALUES (
                    %(Player)s, %(Squad)s, %(Pos)s, %(Age)s, %(Nation)s,
                    %(Games)s, %(Minutes)s, %(Goals)s, %(Assists)s, %(xG)s, %(xA)s,
                    %(Cards Yellow)s, %(Cards Red)s, %(Passes Completed)s, %(Passes Attempted)s,
                    %(Pass Accuracy)s, %(Tackles)s, %(Interceptions)s, %(Blocks)s, %(Shots on Target)s,
                    %(league)s, %(season)s
                )
                ON CONFLICT (player_name, team, season) DO UPDATE SET
                    games = EXCLUDED.games,
                    minutes = EXCLUDED.minutes,
                    goals = EXCLUDED.goals,
                    assists = EXCLUDED.assists,
                    xg = EXCLUDED.xg,
                    xa = EXCLUDED.xa
            """,
                record,
            )
        except Exception as e:
            print(f"  [WARN] Insert error: {record.get('Player', 'unknown')}: {e}")

    conn.commit()
    cur.close()
    conn.close()

    print(f"Saved {len(records)} players to DB")


def main():
    league = sys.argv[1] if len(sys.argv) > 1 else None

    print(f"FBref Player Stats Scraper (via ScraperFC)")
    print(f"   Season: {CURRENT_SEASON}")
    print(f"   League: {league or 'all'}")

    result = scrape_to_db(league)

    if result is not None:
        print(f"\nDone! Scraped {len(result)} players")
    else:
        print(f"\nNo data scraped")


if __name__ == "__main__":
    main()
