import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db
from src.common.pipeline_logging import create_pipeline_run, finalize_pipeline_run

from psycopg2.extras import execute_values


DISCOVERY_PATH_CANDIDATES = [
    Path("data/v1/discovery"),
    Path("data/scraper"),
]
REGISTRY_PATH = Path("scrapers/config/league_registry.json")


def load_registry():
    leagues = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {l["league_code"]: l for l in leagues}


def load_discovery_rows(league_code: str):
    file_name = f"discovery_fixtures_{league_code}.json"
    for root in DISCOVERY_PATH_CANDIDATES:
        p = root / file_name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    candidates = ", ".join(str(root / file_name) for root in DISCOVERY_PATH_CANDIDATES)
    raise FileNotFoundError(f"Discovery file not found. Tried: {candidates}")


def upsert_league(cur, league_code: str, league_name: str):
    cur.execute(
        """
        INSERT INTO leagues (league_code, league_name)
        VALUES (%s, %s)
        ON CONFLICT (league_code) DO UPDATE
        SET league_name = EXCLUDED.league_name,
            updated_at = NOW()
        """,
        (league_code, league_name),
    )


def ingest_league(league_code: str, limit: int | None):
    """Batch-ingest a league's discovery fixtures into DB.

    Uses execute_values for bulk upserts (~4 queries per league instead of
    ~4 per fixture), keeping each league in a single short transaction that
    won't time out on Neon's connection pooler.
    """
    registry = load_registry()
    meta = registry.get(league_code)
    if not meta:
        raise ValueError(f"Unknown league_code in registry: {league_code}")

    rows = load_discovery_rows(league_code)
    if limit is not None:
        rows = rows[:limit]

    # --- Pre-process in memory: collect valid rows, unique teams, statuses ---
    now = datetime.now(timezone.utc)
    valid_rows: list[tuple[str, str, str, str, str]] = []
    team_names: set[str] = set()

    for r in rows:
        fs_id = r.get("flashscore_id")
        kickoff = r.get("kickoff_datetime_utc")
        home = r.get("home_team")
        away = r.get("away_team")
        if not (fs_id and kickoff and home and away):
            continue

        # Minimal status heuristic: past -> ft, future -> scheduled.
        status = "scheduled"
        try:
            dt = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
            if dt < now:
                status = "ft"
        except Exception:
            status = "scheduled"

        team_names.add(home)
        team_names.add(away)
        valid_rows.append((fs_id, kickoff, home, away, status))

    skipped = len(rows) - len(valid_rows)

    # --- Single short transaction: 4 queries total ---
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            # 1. Upsert league
            upsert_league(cur, league_code, meta.get("league_name", league_code))

            # 2. Batch upsert teams
            if team_names:
                team_tuples = [(name, league_code) for name in sorted(team_names)]
                execute_values(
                    cur,
                    """
                    INSERT INTO teams (team_name, league_code)
                    VALUES %s
                    ON CONFLICT (league_code, team_name) DO UPDATE
                    SET updated_at = NOW()
                    """,
                    team_tuples,
                )

            # 3. Fetch team_id map
            cur.execute(
                "SELECT team_name, team_id FROM teams WHERE league_code = %s",
                (league_code,),
            )
            team_id_map = {name: tid for name, tid in cur.fetchall()}

            # 4. Batch upsert fixtures
            if valid_rows:
                fixture_tuples = []
                for fs_id, kickoff, home, away, status in valid_rows:
                    home_id = team_id_map[home]
                    away_id = team_id_map[away]
                    url = f"https://www.flashscore.com/match/{fs_id}/"
                    fixture_tuples.append(
                        (fs_id, league_code, home_id, away_id, kickoff, status, url)
                    )
                execute_values(
                    cur,
                    """
                    INSERT INTO fixtures (
                        flashscore_id, league_code, home_team_id, away_team_id,
                        match_datetime_utc, status, flashscore_url
                    )
                    VALUES %s
                    ON CONFLICT (flashscore_id) DO UPDATE
                    SET league_code = EXCLUDED.league_code,
                        home_team_id = EXCLUDED.home_team_id,
                        away_team_id = EXCLUDED.away_team_id,
                        match_datetime_utc = EXCLUDED.match_datetime_utc,
                        status = EXCLUDED.status,
                        flashscore_url = EXCLUDED.flashscore_url,
                        updated_at = NOW()
                    """,
                    fixture_tuples,
                    template="(%s, %s, %s, %s, %s::timestamptz, %s, %s)",
                )

        conn.commit()
    finally:
        conn.close()

    return {"league_code": league_code, "rows": len(rows), "inserted": len(valid_rows), "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description="Ingest discovery_fixtures_<league>.json into leagues/teams/fixtures")
    parser.add_argument("--league", required=True, help="League code, e.g. E0")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows to ingest")
    args = parser.parse_args()

    run_id = create_pipeline_run(
        "ingest_discovered_fixtures_v1",
        details_json={"league_code": args.league, "limit": args.limit},
    )
    try:
        result = ingest_league(args.league, args.limit)
        finalize_pipeline_run(
            run_id,
            "success",
            message=(
                f"Ingested {result['inserted']} fixtures for {args.league}; "
                f"skipped {result['skipped']}"
            ),
            details_json=result,
        )
        print(json.dumps(result, indent=2))
    except Exception as exc:
        finalize_pipeline_run(
            run_id,
            "fail",
            message=str(exc),
            details_json={"league_code": args.league, "limit": args.limit},
        )
        raise


if __name__ == "__main__":
    if not (os.getenv("DATABASE_URL") or os.getenv("DEV_DATABASE_URL")):
        raise SystemExit(
            "Missing DATABASE_URL (or DEV_DATABASE_URL). Set DATABASE_URL before running this script."
        )
    main()
