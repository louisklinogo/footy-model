"""
Ingest player metadata and match-level performance stats from SofaScore.
"""

import sys
import asyncio
import argparse
from pathlib import Path
import datetime
from typing import List, Dict, Any

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportUnreachable=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false, reportImplicitStringConcatenation=false, reportMissingTypeStubs=false

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Add vendor to sys.path
VENDOR_DIR = ROOT_DIR / "vendor" / "sofascore-wrapper"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from src.db.db_utils import connect_db
from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.match import Match

def parse_args():
    parser = argparse.ArgumentParser(description="Ingest player stats from SofaScore")
    parser.add_argument("--league", type=str, default=None, help="Filter by league code")
    parser.add_argument("--limit", type=int, default=10, help="Max fixtures to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't commit to DB")
    return parser.parse_args()

def _unix_to_datetime(raw: float) -> datetime.datetime | None:
    try:
        ts = float(raw)
    except (TypeError, ValueError):
        return None
    # Treat large values as milliseconds.
    if ts > 1e11:
        ts = ts / 1000.0
    try:
        return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None

def fetch_target_fixtures(league_code: str | None, limit: int) -> List[Dict[str, Any]]:
    query = """
    SELECT f.fixture_id, f.sofascore_id, f.home_team_id, f.away_team_id, f.league_code
    FROM fixtures f
    LEFT JOIN (
        SELECT fixture_id, COUNT(*) as player_count 
        FROM fixture_player_stats 
        GROUP BY fixture_id
    ) fps ON fps.fixture_id = f.fixture_id
    WHERE f.status = 'ft' 
      AND f.sofascore_id IS NOT NULL
      AND (fps.player_count IS NULL OR fps.player_count < 10)
    """
    params = []
    if league_code:
        query += " AND f.league_code = %s"
        params.append(league_code)
    
    query += " ORDER BY f.match_datetime_utc DESC LIMIT %s"
    params.append(limit)
    
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()

async def ingest_fixture_players(api: SofascoreAPI, fixture: Dict[str, Any], dry_run: bool):
    ss_id = fixture["sofascore_id"]
    f_id = fixture["fixture_id"]
    
    print(f"Processing Fixture {f_id} (SS: {ss_id}, League: {fixture['league_code']})...")
    
    match = Match(api, int(ss_id))
    try:
        home_data = await match.lineups_home()
        away_data = await match.lineups_away()
    except Exception as e:
        print(f"  Error fetching lineups for {ss_id}: {e}")
        return

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            # Process Home
            await _upsert_lineup(cur, f_id, fixture["home_team_id"], home_data, True, dry_run)
            # Process Away
            await _upsert_lineup(cur, f_id, fixture["away_team_id"], away_data, False, dry_run)
            
            if not dry_run:
                conn.commit()
                print(f"  Successfully ingested players for {f_id}")
    finally:
        conn.close()

async def _upsert_lineup(cur, fixture_id, team_id, lineup_data, is_home, dry_run):
    players_list = lineup_data.get("starters", []) + lineup_data.get("substitutes", [])
    
    for p_entry in players_list:
        p_info = p_entry.get("player", {})
        stats = p_entry.get("statistics", {})
        if not p_info or not p_info.get("id"):
            continue
            
        ss_player_id = str(p_info["id"])
        
        # 1. Upsert Player Metadata
        dob = None
        if p_info.get("dateOfBirthTimestamp"):
            dt = _unix_to_datetime(p_info["dateOfBirthTimestamp"])
            if dt:
                dob = dt.date()
            
        mv = p_info.get("proposedMarketValueRaw", {}).get("value")
        
        cur.execute("""
            INSERT INTO players (
                sofascore_id, name, slug, short_name, position, 
                user_count, market_value_euro, nationality_code, 
                country_name, date_of_birth, height, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (sofascore_id) DO UPDATE SET
                name = EXCLUDED.name,
                user_count = EXCLUDED.user_count,
                market_value_euro = EXCLUDED.market_value_euro,
                updated_at = NOW()
            RETURNING player_id
        """, (
            ss_player_id, 
            p_info.get("name"), 
            p_info.get("slug"),
            p_info.get("shortName"), 
            p_info.get("position"),
            p_info.get("userCount"), 
            mv,
            p_info.get("country", {}).get("alpha2"),
            p_info.get("country", {}).get("name"),
            dob,
            p_info.get("height")
        ))
        db_player_id = cur.fetchone()[0]
        
        # 2. Upsert Fixture Stats
        aerial_total = (stats.get("aerialWon", 0) or 0) + (stats.get("aerialLost", 0) or 0)
        
        # Ground Duels
        ground_won = stats.get("duelWon", 0) or 0
        ground_lost = stats.get("duelLost", 0) or 0
        ground_total = ground_won + ground_lost
        
        cur.execute("""
            INSERT INTO fixture_player_stats (
                fixture_id, player_id, team_id, is_home,
                position_played, captain,
                minutes_played, rating, substituted_in, substituted_out,
                goals, assists, expected_goals, expected_goals_ot, expected_assists,
                shots_on_target, shots_off_target, shots_blocked,
                key_passes, penalty_scored, penalty_missed,
                touches, passes_total, passes_accurate,
                crosses_total, crosses_accurate,
                tackles_total, interceptions, clearances, blocked_shots,
                ground_duels_total, ground_duels_won,
                aerial_duels_total, aerial_duels_won,
                possessions_lost, fouls, fouled_count,
                yellow_card, red_card,
                saves, punches, runs_out, high_claims
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s
            ) ON CONFLICT (fixture_id, player_id) DO UPDATE SET
                rating = EXCLUDED.rating,
                minutes_played = EXCLUDED.minutes_played,
                goals = EXCLUDED.goals,
                assists = EXCLUDED.assists,
                expected_goals = EXCLUDED.expected_goals,
                expected_goals_ot = EXCLUDED.expected_goals_ot,
                expected_assists = EXCLUDED.expected_assists,
                touches = EXCLUDED.touches,
                passes_total = EXCLUDED.passes_total,
                passes_accurate = EXCLUDED.passes_accurate,
                ground_duels_total = EXCLUDED.ground_duels_total,
                ground_duels_won = EXCLUDED.ground_duels_won,
                aerial_duels_total = EXCLUDED.aerial_duels_total,
                aerial_duels_won = EXCLUDED.aerial_duels_won,
                yellow_card = EXCLUDED.yellow_card,
                red_card = EXCLUDED.red_card,
                created_at = NOW()
        """, (
            fixture_id, db_player_id, team_id, is_home,
            p_entry.get("position"), p_entry.get("captain", False),
            stats.get("minutesPlayed"), stats.get("rating"), 
            p_entry.get("substitute", False) and stats.get("minutesPlayed", 0) > 0,
            not p_entry.get("substitute", False) and stats.get("minutesPlayed", 1) < 90,
            stats.get("goals", 0), stats.get("goalAssist", 0), stats.get("expectedGoals"), stats.get("expectedGoalsOnTarget"), stats.get("expectedAssists"),
            stats.get("onTargetScoringAttempt", 0), stats.get("shotOffTarget", 0), stats.get("blockedScoringAttempt", 0),
            stats.get("keyPass", 0), stats.get("penaltyWon", 0), stats.get("penaltyMiss", 0), # penaltyWon maps to goals usually in SS, but we'll check
            stats.get("touches", 0), stats.get("totalPass", 0), stats.get("accuratePass", 0),
            stats.get("totalCross", 0), stats.get("accurateCross", 0),
            stats.get("totalTackle", 0), stats.get("interceptionWon", 0), stats.get("totalClearance", 0), stats.get("outfielderBlock", 0),
            ground_total, ground_won,
            aerial_total, stats.get("aerialWon", 0),
            stats.get("possessionLostCtrl", 0), stats.get("fouls", 0), stats.get("wasFouled", 0),
            stats.get("yellowCard", 0), stats.get("redCard", 0),
            stats.get("saves", 0), stats.get("punches", 0), stats.get("runsOut", 0), stats.get("goodHighClaim", 0)
        ))

async def main():
    args = parse_args()
    fixtures = fetch_target_fixtures(args.league, args.limit)
    
    if not fixtures:
        print("No fixtures found needing player stats.")
        return
        
    print(f"Found {len(fixtures)} fixtures to process.")
    
    api = SofascoreAPI()
    try:
        for f in fixtures:
            try:
                await ingest_fixture_players(api, f, args.dry_run)
            except Exception as e:
                print(f"  Failed for fixture {f['fixture_id']}: {e}")
    finally:
        await api.close()

if __name__ == "__main__":
    asyncio.run(main())
