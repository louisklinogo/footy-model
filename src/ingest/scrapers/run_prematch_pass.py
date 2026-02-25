import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db

def get_upcoming_fixtures(hours_ahead: int = 48, start_time_str: str = None) -> dict[str, list[str]]:
    """Fetch upcoming flashscore IDs grouped by league."""
    conn = connect_db()
    
    time_filter = "AND match_datetime_utc > NOW()"
    if start_time_str:
        time_filter = f"AND match_datetime_utc >= '{start_time_str}'"
        
    query = f"""
        SELECT league_code, flashscore_id 
        FROM fixtures 
        WHERE status != 'ft' 
          {time_filter}
          AND match_datetime_utc < NOW() + interval '{hours_ahead} hours'
    """
    leagues = {}
    with conn.cursor() as cur:
        cur.execute(query)
        for league_code, fs_id in cur.fetchall():
            if not fs_id: continue
            if league_code not in leagues:
                leagues[league_code] = []
            leagues[league_code].append(fs_id)
    conn.close()
    return leagues

def run_pass(start_time_str: str = None):
    print(f"[{datetime.now().isoformat()}] Starting Pre-Match Odds Enrichment Pass...")
    if start_time_str:
        print(f"Filtering for matches starting after: {start_time_str}")
    
    # 1. Get targets
    leagues = get_upcoming_fixtures(start_time_str=start_time_str)
    if not leagues:
        print("No upcoming fixtures found.")
        return

    total_matches = sum(len(ids) for ids in leagues.values())
    print(f"Found {total_matches} matches across {len(leagues)} leagues.")

    # 2. Trigger Safe Scraper for each league
    scraper_path = ROOT_DIR / 'scrapers' / 'safe_prematch_enricher.js'
    
    for league, ids in leagues.items():
        print(f"--- Processing {league} ({len(ids)} matches) ---")
        
        # Pass IDs as a JSON string to avoid shell argument length limits
        ids_json = json.dumps(ids)
        
        try:
            subprocess.run(
                ["node", str(scraper_path), league, ids_json],
                check=True,
                cwd=str(ROOT_DIR)
            )
        except subprocess.CalledProcessError as e:
            print(f"Error scraping {league}: {e}")

    print(f"\n[{datetime.now().isoformat()}] Pass completed. JSONs stored in data/v1/prematch_json/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-time", type=str, help="ISO format min kickoff time (UTC)")
    args = parser.parse_args()
    
    run_pass(start_time_str=args.start_time)
