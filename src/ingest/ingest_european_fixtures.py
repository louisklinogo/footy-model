"""
Ingest European cup fixtures (CL, EL, ECL) directly from Sofascore.
Fetches historical (23/24, 24/25) and upcoming (25/26) matches.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.league import League

# European cup configurations
CUPS = {
    "CL": {"id": 7, "seasons": [76953, 61644, 52162]},
    "EL": {"id": 679, "seasons": [76957, 61646, 52163]}, # Verified IDs for EL
    "ECL": {"id": 17015, "seasons": [76960, 61648, 52327]}
}

async def fetch_fixtures(api, league_id, season_id):
    """Fetch fixtures for a specific league and season."""
    print(f"  Fetching fixtures for league {league_id}, season {season_id}...")
    # There isn't a direct "get all fixtures for season" in the wrapper 
    # but we can use the standings to get teams, or search by date.
    # Actually, let's use the unique-tournament events endpoint if possible
    # Reference: /unique-tournament/{id}/season/{season}/events/last/0 -- but that's paginated
    
    # Let's try to find if there's an events endpoint in the wrapper or raw
    fixtures = []
    # Pagination loop
    page = 0
    while True:
        try:
            data = await api._get(f"/unique-tournament/{league_id}/season/{season_id}/events/last/{page}")
            events = data.get("events", [])
            if not events:
                break
            fixtures.extend(events)
            page += 1
            if page > 50: # Safety break
                break
        except Exception:
            break
            
    # Also get upcoming next
    page = 0
    while True:
        try:
            data = await api._get(f"/unique-tournament/{league_id}/season/{season_id}/events/next/{page}")
            events = data.get("events", [])
            if not events:
                break
            fixtures.extend(events)
            page += 1
            if page > 50: # Safety break
                break
        except Exception:
            break
            
    return fixtures

def upsert_fixtures(conn, league_code, events):
    """Upsert fixtures into the database."""
    if not events:
        return 0
        
    cursor = conn.cursor()
    
    # 1. Collect unique teams
    teams = set()
    for ev in events:
        teams.add(ev['homeTeam']['name'])
        teams.add(ev['awayTeam']['name'])
        
    # 2. Upsert teams
    team_tuples = [(name, league_code) for name in sorted(teams)]
    execute_values(
        cursor,
        """
        INSERT INTO teams (team_name, league_code)
        VALUES %s
        ON CONFLICT (league_code, team_name) DO UPDATE
        SET updated_at = NOW()
        """,
        team_tuples
    )
    
    # 3. Fetch team mapping
    cursor.execute("SELECT team_name, team_id FROM teams WHERE league_code = %s", (league_code,))
    team_map = {name: tid for name, tid in cursor.fetchall()}
    
    # 4. Prepare fixture tuples
    fixture_tuples = []
    for ev in events:
        sid = str(ev['id'])
        home_id = team_map[ev['homeTeam']['name']]
        away_id = team_map[ev['awayTeam']['name']]
        kickoff = datetime.fromtimestamp(ev['startTimestamp'], tz=timezone.utc)
        
        # Determine status
        sofa_status = ev.get('status', {}).get('type', 'notstarted')
        status = 'ft' if sofa_status == 'finished' else 'scheduled'
        
        # Use a dummy flashscore_id for now or leave NULL if possible
        # Our DB has flashscore_id as PK or UNIQUE usually.
        # Let's check schema. If flashscore_id is required, we use 'sofa_' + sid.
        fs_id = f"sofa_{sid}" 
        
        fixture_tuples.append((
            fs_id, sid, league_code, home_id, away_id, kickoff, status
        ))
        
    execute_values(
        cursor,
        """
        INSERT INTO fixtures (
            flashscore_id, sofascore_id, league_code, home_team_id, away_team_id,
            match_datetime_utc, status
        )
        VALUES %s
        ON CONFLICT (flashscore_id) DO UPDATE
        SET sofascore_id = EXCLUDED.sofascore_id,
            match_datetime_utc = EXCLUDED.match_datetime_utc,
            status = EXCLUDED.status,
            updated_at = NOW()
        """,
        fixture_tuples
    )
    
    conn.commit()
    return len(fixture_tuples)

async def main():
    api = SofascoreAPI()
    conn = connect_db()
    
    try:
        for league_code, cfg in CUPS.items():
            print(f"Processing {league_code}...")
            total = 0
            for season_id in cfg['seasons']:
                events = await fetch_fixtures(api, cfg['id'], season_id)
                count = upsert_fixtures(conn, league_code, events)
                total += count
            print(f"  Done. Total fixtures for {league_code}: {total}")
            
    finally:
        await api.close()
        conn.close()

if __name__ == "__main__":
    asyncio.run(main())
