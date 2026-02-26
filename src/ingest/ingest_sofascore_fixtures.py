import asyncio
import argparse
import sys
import logging
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import execute_values
from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.league import League

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_URL = "postgresql://neondb_owner:npg_csxeyQ6fNXF7@ep-young-tree-ab8emfpl-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require"

LEAGUES_TO_INGEST = {
    "CL": 7,
    "EL": 679,
    "ECL": 17015
}

SEASONS_TO_KEEP = ["25/26", "24/25", "23/24"]

def _safe_fromtimestamp(ts):
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
    except (OverflowError, OSError, ValueError):
        # Fallback for negative timestamps on Windows or malformed data
        return datetime(1970, 1, 1) if ts < 0 else datetime(2099, 12, 31)

async def ingest_league_fixtures(api, league_code, unique_tournament_id, conn, dry_run=False):
    logger.info(f"Starting ingestion for {league_code} (ID: {unique_tournament_id}) {'[DRY RUN]' if dry_run else ''}")
    league = League(api, unique_tournament_id)
    
    try:
        seasons = await league.get_seasons()
    except Exception as e:
        logger.error(f"Failed to fetch seasons for {league_code}: {e}")
        return

    for season in seasons:
        if season['year'] not in SEASONS_TO_KEEP:
            continue
        
        logger.info(f"Processing season: {season['name']} (ID: {season['id']})")
        
        try:
            rounds_data = await league.rounds(season['id'])
            rounds = rounds_data.get('rounds', [])
            if not rounds:
                logger.warning(f"  No rounds found for season {season['name']}")
                continue
        except Exception as e:
            logger.error(f"  Failed to fetch rounds for {season['name']}: {e}")
            continue

        for round_info in rounds:
            round_id = round_info['round']
            round_slug = round_info.get('slug')
            round_name = round_info.get('name', f"Round {round_id}")
            
            try:
                # 1. Try standard numeric endpoint
                try:
                    data = await league.league_fixtures_per_round(season['id'], round_id)
                except Exception as e:
                    if "404" in str(e) and round_slug:
                        # 2. Try slug fallback if numeric fails
                        logger.info(f"  {round_name}: Numeric endpoint 404, trying slug '{round_slug}'")
                        data = await api._get(f"/unique-tournament/{unique_tournament_id}/season/{season['id']}/events/round/{round_id}/slug/{round_slug}")
                    else:
                        raise e

                events = data.get('events', []) if isinstance(data, dict) else []
                
                if not events:
                    continue
                
                logger.info(f"  {round_name}: Found {len(events)} events")
                upsert_fixtures(conn, events, league_code, season['year'], dry_run=dry_run)
                
            except Exception as e:
                logger.warning(f"  {round_name} failed: {e}")
                conn.rollback()
                continue

def upsert_fixtures(conn, events, league_code, season_label, dry_run=False):
    try:
        with conn.cursor() as cur:
            for event in events:
                sofascore_id = str(event['id'])
                match_datetime = _safe_fromtimestamp(event['startTimestamp'])
                status_code = event.get('status', {}).get('type')
                
                db_status = 'scheduled'
                settlement_status = 'pending'
                
                if status_code == 'finished':
                    db_status = 'ft'
                    settlement_status = 'due'
                elif status_code == 'inprogress':
                    db_status = 'live'
                elif status_code == 'canceled':
                    db_status = 'cancelled'
                elif status_code == 'postponed':
                    db_status = 'postponed'

                home_team_sofa_id = event['homeTeam']['id']
                away_team_sofa_id = event['awayTeam']['id']
                
                home_team_id = lookup_team_id(cur, home_team_sofa_id, event['homeTeam']['name'], league_code, dry_run=dry_run)
                away_team_id = lookup_team_id(cur, away_team_sofa_id, event['awayTeam']['name'], league_code, dry_run=dry_run)

                if dry_run:
                    logger.info(f"    [DRY RUN] Would upsert fixture: {event['homeTeam']['name']} vs {event['awayTeam']['name']} ({match_datetime}) -> Status: {db_status}, Settlement: {settlement_status}")
                    continue

                cur.execute("""
                    INSERT INTO fixtures (
                        sofascore_id, league_code, home_team_id, away_team_id, 
                        match_datetime_utc, status, season, settlement_status
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s
                    ) ON CONFLICT (sofascore_id) WHERE sofascore_id IS NOT NULL DO UPDATE SET
                        status = EXCLUDED.status,
                        match_datetime_utc = EXCLUDED.match_datetime_utc,
                        settlement_status = EXCLUDED.settlement_status,
                        updated_at = NOW()
                    RETURNING fixture_id;
                """, (
                    sofascore_id, league_code, home_team_id, away_team_id,
                    match_datetime, db_status, season_label, settlement_status
                ))
                
                # Fetch the returned fixture_id
                fixture_id = cur.fetchone()[0]
                
                # If match is finished, natively ingest the final scorelines
                if db_status == 'ft':
                    home_goals = event.get('homeScore', {}).get('display')
                    away_goals = event.get('awayScore', {}).get('display')
                    
                    if home_goals is not None and away_goals is not None:
                        cur.execute("""
                            INSERT INTO fixture_results (
                                fixture_id, home_goals, away_goals, 
                                result_status, settled_at
                            ) VALUES (
                                %s, %s, %s, 'ft', NOW()
                            ) ON CONFLICT (fixture_id) DO UPDATE SET
                                home_goals = EXCLUDED.home_goals,
                                away_goals = EXCLUDED.away_goals,
                                settled_at = NOW();
                        """, (fixture_id, home_goals, away_goals))
                
        if not dry_run:
            conn.commit()
        else:
            conn.rollback()
    except Exception as e:
        conn.rollback()
        raise e

def lookup_team_id(cur, sofa_id, team_name, league_code, dry_run=False):
    # 1. Try by sofascore_id
    cur.execute("SELECT team_id FROM teams WHERE sofascore_id = %s", (sofa_id,))
    res = cur.fetchone()
    if res:
        return res[0]
    
    # 2. Try by name and league
    cur.execute("SELECT team_id FROM teams WHERE league_code = %s AND team_name = %s", (league_code, team_name))
    res = cur.fetchone()
    if res:
        if not dry_run:
            cur.execute("UPDATE teams SET sofascore_id = %s WHERE team_id = %s", (sofa_id, res[0]))
        else:
            logger.info(f"    [DRY RUN] Would update team {res[0]} (Name: {team_name}) with sofascore_id {sofa_id}")
        return res[0]
    
    # 3. Create if not exists (with robust conflict handling)
    if dry_run:
        logger.info(f"    [DRY RUN] Would create new team: {team_name} (SofaID: {sofa_id})")
        return -1
        
    cur.execute("""
        INSERT INTO teams (team_name, league_code, sofascore_id)
        VALUES (%s, %s, %s)
        ON CONFLICT (league_code, team_name) DO UPDATE SET
            sofascore_id = EXCLUDED.sofascore_id
        RETURNING team_id;
    """, (team_name, league_code, sofa_id))
    return cur.fetchone()[0]

async def main():
    args = parse_args()
    api = SofascoreAPI()
    conn = psycopg2.connect(DB_URL)
    
    try:
        for league_code, unique_id in LEAGUES_TO_INGEST.items():
            await ingest_league_fixtures(api, league_code, unique_id, conn, dry_run=args.dry_run)
    finally:
        await api.close()
        conn.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Ingest European cup fixtures from Sofascore.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without committing changes.")
    return parser.parse_args()

if __name__ == "__main__":
    asyncio.run(main())
