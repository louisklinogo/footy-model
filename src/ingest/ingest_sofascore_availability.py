"""
Ingest player availability (missing players and lineups) from SofaScore.
Supports both scheduled fixtures (pre-match) and historical fixtures (backfill).

Usage:
    python src/ingest/ingest_sofascore_availability.py --limit 50 --status scheduled
    python src/ingest/ingest_sofascore_availability.py --limit 200 --status ft --league E0
    python src/ingest/ingest_sofascore_availability.py --dry-run --limit 5
"""

import sys
import asyncio
import argparse
from pathlib import Path
import datetime
from typing import List, Dict, Any

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false, reportArgumentType=false

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

VENDOR_DIR = ROOT_DIR / "vendor" / "sofascore-wrapper"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from src.db.db_utils import connect_db
from sofascore_wrapper.api import SofascoreAPI


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest player availability from SofaScore")
    parser.add_argument("--league", type=str, default=None, help="Filter by league code")
    parser.add_argument("--limit", type=int, default=50, help="Max fixtures to process")
    parser.add_argument("--status", type=str, default="scheduled",
                        choices=["scheduled", "ft", "all"], help="Fixture status to process")
    parser.add_argument("--dry-run", action="store_true", help="Fetch data but don't write to DB")
    return parser.parse_args()


def fetch_target_fixtures(league_code: str | None, status: str, limit: int) -> List[Dict[str, Any]]:
    """Fetch fixtures that are missing availability data."""
    query = """
    SELECT f.fixture_id, f.sofascore_id, f.home_team_id, f.away_team_id, f.league_code, f.status
    FROM fixtures f
    LEFT JOIN (
        SELECT fixture_id, COUNT(*) AS avail_count
        FROM player_availability
        GROUP BY fixture_id
    ) pa ON pa.fixture_id = f.fixture_id
    WHERE f.sofascore_id IS NOT NULL
      AND (pa.avail_count IS NULL OR pa.avail_count < 10)
    """
    params: list = []
    if status != "all":
        query += " AND f.status = %s"
        params.append(status)

    if league_code:
        query += " AND f.league_code = %s"
        params.append(league_code)

    query += " ORDER BY f.match_datetime_utc ASC LIMIT %s"
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


async def ingest_availability(api: SofascoreAPI, conn, fixture: Dict[str, Any], dry_run: bool):
    ss_id = fixture["sofascore_id"]
    f_id = fixture["fixture_id"]

    print(f"  Processing Fixture {f_id} (SS: {ss_id}, Status: {fixture['status']})...")

    try:
        lineups_raw = await api._get(f"/event/{ss_id}/lineups")
    except Exception as e:
        print(f"    Error fetching lineups for {ss_id}: {e}")
        return

    if not lineups_raw:
        print(f"    No lineup data for {ss_id}")
        return

    home_raw = lineups_raw.get("home", {})
    away_raw = lineups_raw.get("away", {})

    if dry_run:
        h_missing = len(home_raw.get("missingPlayers", []))
        a_missing = len(away_raw.get("missingPlayers", []))
        h_starters = sum(1 for p in home_raw.get("players", []) if not p.get("substitute"))
        a_starters = sum(1 for p in away_raw.get("players", []) if not p.get("substitute"))
        print(f"    [DRY RUN] Home: {h_starters} starters, {h_missing} missing | "
              f"Away: {a_starters} starters, {a_missing} missing")
        return

    with conn.cursor() as cur:
        _upsert_availability_rows(cur, f_id, fixture["home_team_id"], home_raw)
        _upsert_availability_rows(cur, f_id, fixture["away_team_id"], away_raw)
    conn.commit()
    print(f"    OK - committed availability for fixture {f_id}")


def _upsert_availability_rows(cur, fixture_id: int, team_id: int, data: dict):
    """Upsert starter/bench/missing/doubtful rows into player_availability."""

    # --- 1. Lineup players (Starter / Bench) ---
    for p_entry in data.get("players", []):
        p_info = p_entry.get("player", {})
        if not p_info or not p_info.get("id"):
            continue

        status = "bench" if p_entry.get("substitute") else "starter"
        db_p_id = _get_or_create_player(cur, p_info)

        cur.execute("""
            INSERT INTO player_availability (fixture_id, player_id, team_id, status, source)
            VALUES (%s, %s, %s, %s, 'sofascore_lineup')
            ON CONFLICT (fixture_id, player_id) DO UPDATE SET
                status = EXCLUDED.status,
                recorded_at = NOW()
        """, (fixture_id, db_p_id, team_id, status))

    # --- 2. Missing players (Missing / Doubtful) ---
    for m_entry in data.get("missingPlayers", []):
        p_info = m_entry.get("player", {})
        if not p_info or not p_info.get("id"):
            continue

        player_class = m_entry.get("playerClass", "")
        status = "doubtful" if "doubt" in player_class.lower() else "missing"

        description = m_entry.get("description") or m_entry.get("type") or "unknown"
        reason_code = m_entry.get("reason")
        expected_return = m_entry.get("expectedEndDate")

        db_p_id = _get_or_create_player(cur, p_info)

        cur.execute("""
            INSERT INTO player_availability (
                fixture_id, player_id, team_id, status, reason, 
                reason_code, expected_return, source
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'sofascore_missing')
            ON CONFLICT (fixture_id, player_id) DO UPDATE SET
                status = EXCLUDED.status,
                reason = EXCLUDED.reason,
                reason_code = EXCLUDED.reason_code,
                expected_return = EXCLUDED.expected_return,
                recorded_at = NOW()
        """, (fixture_id, db_p_id, team_id, status, description, reason_code, expected_return))


def _get_or_create_player(cur, p_info: dict) -> int:
    """Atomically upsert player metadata and return internal player_id."""
    ss_id = str(p_info["id"])

    dob = None
    if p_info.get("dateOfBirthTimestamp"):
        dob = datetime.date.fromtimestamp(p_info["dateOfBirthTimestamp"])

    mv = (p_info.get("proposedMarketValueRaw") or {}).get("value")

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
        ss_id,
        p_info.get("name"),
        p_info.get("slug"),
        p_info.get("shortName"),
        p_info.get("position"),
        p_info.get("userCount"),
        mv,
        (p_info.get("country") or {}).get("alpha2"),
        (p_info.get("country") or {}).get("name"),
        dob,
        p_info.get("height"),
    ))
    return cur.fetchone()[0]


async def main():
    args = parse_args()
    try:
        api = SofascoreAPI()
        try:
            fixtures = fetch_target_fixtures(args.league, args.status, args.limit)

            if not fixtures:
                print(f"No fixtures found needing availability (Status: {args.status}).")
                return

            print(f"Found {len(fixtures)} fixtures to process.")
            
            for f in fixtures:
                conn = connect_db()
                try:
                    await ingest_availability(api, conn, f, args.dry_run)
                finally:
                    conn.close()
                await asyncio.sleep(1)
        finally:
            await api.close()
    except Exception as e:
        print(f"Main Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
