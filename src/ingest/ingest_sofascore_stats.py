import asyncio
import argparse
import sys
import logging
import json
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sofascore_wrapper.api import SofascoreAPI

from src.db.db_utils import connect_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Map SofaScore names to our column suffixes
NAME_MAP = {
    "Expected goals": "xg",
    "Expected goals on target": "xgot",
    "Expected assists": "xa",
    "Big chances": "big_chances",
    "Ball possession": "possession",
    "Touches in penalty area": "box_touches",
    "Crosses": "crosses",
    "Blocked shots": "blocked_shots",
    "Through balls": "through_passes",
    "Shots on target": "sot",
    "Shots inside box": "shots_inside_box",
    "Corner kicks": "corners",
    "Goals prevented": "goals_prevented",
    "Tackles won": "tackles_pct", # Might need parsing (e.g. "10/15")
    "Interceptions": "interceptions",
    "Error led to shot": "errors_lead_to_shot",
    "Offsides": "offsides",
    "Fouls": "fouls",
    "Dispossessed": "dispossessed",
    "Hit woodwork": "hit_woodwork",
    "Dribbles": "dribbles", # Will split into success/total
    "Recovered balls": "ball_recoveries",
    "Yellow cards": "yellow_cards",
    "Red cards": "red_cards",
    "Duels": "duels",
    "Ground duels": "ground_duels",
    "Aerial duels": "aerial_duels",
    "Final third entries": "final_third_entries",
    "Long balls": "long_balls",
}

def parse_sofa_value(val):
    if val is None: return 0
    if isinstance(val, str):
        if "%" in val:
            return float(val.replace("%", ""))
        if "/" in val:
            # For "10/15" types, we often want the success count or the percentage.
            # Our schema has tactles_pct but others are success/total.
            parts = val.split('/')
            return float(parts[0]) # Default to success count
    return float(val)

def parse_sofa_total(val):
    if isinstance(val, str) and "/" in val:
        return float(val.split('/')[1])
    return 0

async def ingest_match_stats(api, conn, fixture_id, sofascore_id, dry_run=False):
    try:
        try:
            data = await api._get(f"/event/{sofascore_id}/statistics")
        except Exception as api_err:
            if "404" in str(api_err):
                logger.warning(f"  Match stats 404 (Not Available) for {sofascore_id}. Marking as zero-fidelity.")
                if not dry_run:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO fixture_stats_premium (fixture_id, fidelity_score, raw_json)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (fixture_id) DO UPDATE SET
                                fidelity_score = EXCLUDED.fidelity_score,
                                raw_json = EXCLUDED.raw_json,
                                ingested_at = NOW();
                        """, (fixture_id, 0, json.dumps({"error": "SofaScore statistics not available (404)"})))
                        conn.commit()
                return
            else:
                raise api_err

        if not data or 'statistics' not in data:
            logger.warning(f"  No stats object found for {sofascore_id}")
            return

        row_data = {"fixture_id": fixture_id}
        
        for period_data in data['statistics']:
            period = period_data['period']
            suffix = ""
            if period == "1ST": suffix = "_p1"
            elif period == "2ND": suffix = "_p2"
            elif period != "ALL": continue # Skip other periods if any
            
            for group in period_data.get('groups', []):
                for item in group.get('statisticsItems', []):
                    name = item['name']
                    if name in NAME_MAP:
                        col_base = NAME_MAP[name]
                        home_val = parse_sofa_value(item.get('homeValue'))
                        away_val = parse_sofa_value(item.get('awayValue'))
                        
                        # Special handling for specialized columns
                        if col_base == "dribbles":
                            row_data[f"h_dribbles_success{suffix}"] = home_val
                            row_data[f"a_dribbles_success{suffix}"] = away_val
                            row_data[f"h_dribbles_total{suffix}"] = parse_sofa_total(item.get('homeValue'))
                            row_data[f"a_dribbles_total{suffix}"] = parse_sofa_total(item.get('awayValue'))
                        elif col_base == "long_balls":
                            row_data[f"h_accurate_long_balls{suffix}"] = home_val
                            row_data[f"a_accurate_long_balls{suffix}"] = away_val
                            row_data[f"h_total_long_balls{suffix}"] = parse_sofa_total(item.get('homeValue'))
                            row_data[f"a_total_long_balls{suffix}"] = parse_sofa_total(item.get('awayValue'))
                        elif col_base in ["duels", "ground_duels", "aerial_duels"]:
                            # These are only for 'ALL' in our schema currently, but we can adapt
                            if suffix == "":
                                row_data[f"h_{col_base}_won"] = home_val
                                row_data[f"a_{col_base}_won"] = away_val
                        else:
                            row_data[f"h_{col_base}{suffix}"] = home_val
                            row_data[f"a_{col_base}{suffix}"] = away_val

        # Determine Fidelity Score
        # 1.0 = H1/H2 stats present
        # 0.5 = Only Full Match stats present
        has_p1 = any("_p1" in k for k in row_data.keys())
        fidelity = 1.0 if has_p1 else 0.5
        row_data["fidelity_score"] = fidelity

        if dry_run:
            logger.info(f"    [DRY RUN] Would ingest stats for fixture {fixture_id} (Fidelity: {fidelity})")
            return

        # Insert into DB
        columns = list(row_data.keys())
        values = [row_data[c] for c in columns]
        placeholders = ["%s"] * len(columns)
        
        update_cols = [f"{c} = EXCLUDED.{c}" for c in columns if c != 'fixture_id']
        set_parts = update_cols + ["raw_json = EXCLUDED.raw_json", "ingested_at = NOW()"]
        
        query = f"""
            INSERT INTO fixture_stats_premium ({', '.join(columns)}, raw_json)
            VALUES ({', '.join(placeholders)}, %s)
            ON CONFLICT (fixture_id) DO UPDATE SET
                {', '.join(set_parts)};
        """
        
        with conn.cursor() as cur:
            cur.execute(query, values + [json.dumps(data)])
        conn.commit()
        logger.info(f"  Successfully ingested stats for fixture {fixture_id}")

    except Exception as e:
        logger.error(f"  Failed to ingest stats for {sofascore_id}: {e}")
        conn.rollback()

async def main():
    parser = argparse.ArgumentParser(description="Ingest backfill statistics from Sofascore.")
    parser.add_argument("--limit", type=int, default=100, help="Max fixtures to process.")
    parser.add_argument("--league", type=str, help="Filter by league code (e.g., CL, EL, ECL).")
    parser.add_argument(
        "--fixture-ids-file",
        type=str,
        default=None,
        help="Optional CSV/TXT file with fixture_id values (first column).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = SofascoreAPI()
    conn = connect_db()
    
    try:
        with conn.cursor() as cur:
            params = []
            if args.fixture_ids_file:
                fixture_ids: list[int] = []
                for raw_line in Path(args.fixture_ids_file).read_text(encoding="utf-8").splitlines():
                    token = raw_line.split(",")[0].strip()
                    if not token or token.lower() == "fixture_id":
                        continue
                    try:
                        fixture_ids.append(int(token))
                    except ValueError:
                        continue
                query = """
                    SELECT f.fixture_id, f.sofascore_id
                    FROM fixtures f
                    WHERE f.sofascore_id IS NOT NULL
                      AND f.status = 'ft'
                      AND f.fixture_id = ANY(%s)
                """
                params.append(fixture_ids)
                if args.league:
                    query += " AND f.league_code = %s"
                    params.append(args.league)
                query += " ORDER BY f.match_datetime_utc DESC, f.fixture_id DESC LIMIT %s;"
                params.append(args.limit)
            else:
                # Find fixtures needing stats and include rows with missing corners settlement fields.
                # We explicitly skip rows that have a SofaScore error message in the JSON to avoid retrying 404s.
                query = """
                    SELECT f.fixture_id, f.sofascore_id 
                    FROM fixtures f
                    LEFT JOIN fixture_stats_premium s ON f.fixture_id = s.fixture_id
                    WHERE f.sofascore_id IS NOT NULL 
                      AND f.status = 'ft'
                      AND (
                        s.fixture_id IS NULL 
                        OR (s.fidelity_score = 0.0 AND s.raw_json->>'error' IS NULL)
                        OR s.h_corners IS NULL
                        OR s.a_corners IS NULL
                        OR s.h_possession IS NULL
                        OR s.h_xg_p1 IS NULL
                      )
                """
                if args.league:
                    query += " AND f.league_code = %s"
                    params.append(args.league)
                query += " LIMIT %s;"
                params.append(args.limit)

            cur.execute(query, tuple(params))
            fixtures = cur.fetchall()
            
        logger.info(f"Found {len(fixtures)} fixtures needing stats ingestion.")
        
        for fid, sofa_id in fixtures:
            logger.info(f"Processing fixture {fid} (SofaID: {sofa_id})...")
            try:
                # Open a fresh connection per fixture to avoid idle timeouts
                conn_local = connect_db()
                try:
                    await ingest_match_stats(api, conn_local, fid, sofa_id, dry_run=args.dry_run)
                finally:
                    conn_local.close()
            except Exception as e:
                logger.error(f"  Failed for fixture {fid}: {e}")
            await asyncio.sleep(0.5) # Slight delay to be polite

    finally:
        await api.close()
        # the initial conn used to fetch fixtures was closed earlier or can be closed
        try:
            conn.close()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())
