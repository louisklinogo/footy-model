
import os
import json
import psycopg2
import sys
import logging
from pathlib import Path
from datetime import datetime


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db_utils import connect_db

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path("data/scraper")
PREMIUM_DIR = BASE_DIR / "premium_v4"
MAPPING_FILE = Path("data/team_mappings.json")

def load_json(path):
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except:
            return None

def get_match_metadata():
    metadata = {}
    for f in BASE_DIR.glob("match_ids_*.json"):
        data = load_json(f)
        if data:
            for m in data:
                metadata[m['id']] = m
    return metadata

def parse_fs_date(date_str):
    try:
        parts = date_str.split('.')
        day = int(parts[0])
        month = int(parts[1])
        # Assume 2025 for July-Dec, 2026 for Jan-June
        year = 2025 if month >= 7 else 2026
        return datetime(year, month, day).date()
    except:
        return None

def parse_val(val_str):
    if val_str is None: return None
    if isinstance(val_str, (int, float)): return float(val_str)
    # Handle percentages "43%"
    if '%' in val_str:
        val_str = val_str.split('%')[0]
    # Handle fractions "75% (12/16)" -> 75
    if '(' in val_str:
        val_str = val_str.split('(')[0].strip()
    try:
        return float(val_str)
    except:
        return None

def ingest():
    conn = connect_db()
    cur = conn.cursor()
    
    # 1. Re-create table with full schema if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS matches_premium (
            match_id BIGINT PRIMARY KEY,
            flashscore_id TEXT,
            h_xg FLOAT, a_xg FLOAT,
            h_xgot FLOAT, a_xgot FLOAT,
            h_xa FLOAT, a_xa FLOAT,
            h_big_chances INT, a_big_chances INT,
            h_possession FLOAT, a_possession FLOAT,
            h_box_touches INT, a_box_touches INT,
            h_crosses INT, a_crosses INT,
            h_blocked_shots INT, a_blocked_shots INT,
            h_through_passes INT, a_through_passes INT,
            h_sot INT, a_sot INT,
            h_shots_inside_box INT, a_shots_inside_box INT,
            h_corners INT, a_corners INT,
            h_goals_prevented FLOAT, a_goals_prevented FLOAT,
            h_tackles_pct FLOAT, a_tackles_pct FLOAT,
            h_interceptions INT, a_interceptions INT,
            h_errors_lead_to_shot INT, a_errors_lead_to_shot INT,
            odds_o15 FLOAT, odds_u15 FLOAT,
            odds_o25 FLOAT, odds_u25 FLOAT,
            odds_o35 FLOAT, odds_u35 FLOAT,
            ah_line FLOAT, ah_home FLOAT, ah_away FLOAT,
            fidelity_score FLOAT
        )
    """)
    conn.commit()

    metadata = get_match_metadata()
    mappings = load_json(MAPPING_FILE) or {}
    
    ingested = 0
    matched = 0
    total_processed = 0
    
    # Pre-fetch all 25/26 matches to speed up lookup
    cur.execute("SELECT id, date, home_team, away_team FROM matches WHERE season = '2526'")
    db_matches = cur.fetchall()
    
    # Create a lookup dict for speed
    # Key: (date, home, away)
    lookup = {}
    for mid, mdate, h, a in db_matches:
        lookup[(mdate, h, a)] = mid

    for league_dir in PREMIUM_DIR.iterdir():
        if not league_dir.is_dir(): continue
        logger.info(f"Ingesting league: {league_dir.name}")
        
        for json_file in league_dir.glob("*.json"):
            total_processed += 1
            if total_processed % 100 == 0:
                logger.info(f"Scanning file {total_processed}: {json_file.name}")
            fs_id = json_file.stem
            m_meta = metadata.get(fs_id)
            if not m_meta: continue
            
            m_date = parse_fs_date(m_meta['date'])
            if not m_date: continue
            
            home, away = m_meta['home'], m_meta['away']
            h_db, a_db = mappings.get(home, home), mappings.get(away, away)
            
            # Fast lookup
            match_id = lookup.get((m_date, h_db, a_db))
            if not match_id:
                # Try swap (sometimes scrapers flip)
                match_id = lookup.get((m_date, a_db, h_db))
            
            if not match_id:
                continue
                
            matched += 1
            data = load_json(json_file)
            if not data or 'stats' not in data: continue
            
            s = data['stats']
            h, a = s.get('home', {}), s.get('away', {})
            
            # 1. Stats Tuple
            stats_vals = [
                parse_val(h.get('Expected goals (xG)')), parse_val(a.get('Expected goals (xG)')),
                parse_val(h.get('xG on target (xGOT)')), parse_val(a.get('xG on target (xGOT)')),
                parse_val(h.get('Expected assists (xA)')), parse_val(a.get('Expected assists (xA)')),
                parse_val(h.get('Big chances')), parse_val(a.get('Big chances')),
                parse_val(h.get('Ball possession')), parse_val(a.get('Ball possession')),
                parse_val(h.get('Touches in opposition box')), parse_val(a.get('Touches in opposition box')),
                parse_val(h.get('Crosses')), parse_val(a.get('Crosses')),
                parse_val(h.get('Blocked shots')), parse_val(a.get('Blocked shots')),
                parse_val(h.get('Accurate through passes')), parse_val(a.get('Accurate through passes')),
                parse_val(h.get('Shots on target')), parse_val(a.get('Shots on target')),
                parse_val(h.get('Shots inside the box')), parse_val(a.get('Shots inside the box')),
                parse_val(h.get('Corner kicks')), parse_val(a.get('Corner kicks')),
                parse_val(h.get('Goals prevented')), parse_val(a.get('Goals prevented')),
                parse_val(h.get('Tackles')), parse_val(a.get('Tackles')),
                parse_val(h.get('Interceptions')), parse_val(a.get('Interceptions')),
                parse_val(h.get('Errors leading to shot')), parse_val(a.get('Errors leading to shot'))
            ]
            
            # 2. Odds Extraction
            ou = data.get('odds', {}).get('ou', {})
            odds_vals = [
                parse_val(ou.get('1.5', {}).get('over')), parse_val(ou.get('1.5', {}).get('under')),
                parse_val(ou.get('2.5', {}).get('over')), parse_val(ou.get('2.5', {}).get('under')),
                parse_val(ou.get('3.5', {}).get('over')), parse_val(ou.get('3.5', {}).get('under'))
            ]
            
            ah = data.get('odds', {}).get('ah', {})
            ah_line, ah_h, ah_a = None, None, None
            if ah:
                line = list(ah.keys())[0]
                ah_line = parse_val(line)
                ah_h = parse_val(ah[line].get('over'))
                ah_a = parse_val(ah[line].get('under'))
            
            odds_vals += [ah_line, ah_h, ah_a]
            
            # 3. Fidelity
            fidelity = sum(1 for x in stats_vals[:4] if x is not None) / 4.0
            
            # 4. Final Data
            row = [match_id, fs_id] + stats_vals + odds_vals + [fidelity]
            
            cur.execute("""
                INSERT INTO matches_premium (
                    match_id, flashscore_id, h_xg, a_xg, h_xgot, a_xgot, h_xa, a_xa, 
                    h_big_chances, a_big_chances, h_possession, a_possession, 
                    h_box_touches, a_box_touches, h_crosses, a_crosses, 
                    h_blocked_shots, a_blocked_shots, h_through_passes, a_through_passes, 
                    h_sot, a_sot, h_shots_inside_box, a_shots_inside_box, 
                    h_corners, a_corners, h_goals_prevented, a_goals_prevented, 
                    h_tackles_pct, a_tackles_pct, h_interceptions, a_interceptions, 
                    h_errors_lead_to_shot, a_errors_lead_to_shot,
                    odds_o15, odds_u15, odds_o25, odds_u25, odds_o35, odds_u35, 
                    ah_line, ah_home, ah_away, fidelity_score
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (match_id) DO UPDATE SET
                    flashscore_id = EXCLUDED.flashscore_id,
                    fidelity_score = EXCLUDED.fidelity_score
            """, row)
            ingested += 1
            
            if ingested % 500 == 0:
                conn.commit()
                logger.info(f"Checkpoint: {ingested} rows committed...")

    conn.commit()
    conn.close()
    logger.info(f"Ingestion Finished! Processed: {total_processed}, Matched: {matched}, Ingested: {ingested}")

if __name__ == "__main__":
    ingest()
