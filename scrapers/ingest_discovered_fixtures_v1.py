import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db_utils import connect_db
from pipeline_logging import create_pipeline_run, finalize_pipeline_run


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


def upsert_team(cur, league_code: str, team_name: str) -> int:
    cur.execute(
        """
        INSERT INTO teams (team_name, league_code)
        VALUES (%s, %s)
        ON CONFLICT (league_code, team_name) DO UPDATE
        SET updated_at = NOW()
        RETURNING team_id
        """,
        (team_name, league_code),
    )
    return int(cur.fetchone()[0])


def upsert_fixture(
    cur,
    flashscore_id: str,
    league_code: str,
    home_team_id: int,
    away_team_id: int,
    match_datetime_utc: str,
    status: str,
):
    flashscore_url = f"https://www.flashscore.com/match/{flashscore_id}/"
    cur.execute(
        """
        INSERT INTO fixtures (
            flashscore_id,
            league_code,
            home_team_id,
            away_team_id,
            match_datetime_utc,
            status,
            flashscore_url
        )
        VALUES (%s, %s, %s, %s, %s::timestamptz, %s, %s)
        ON CONFLICT (flashscore_id) DO UPDATE
        SET league_code = EXCLUDED.league_code,
            home_team_id = EXCLUDED.home_team_id,
            away_team_id = EXCLUDED.away_team_id,
            match_datetime_utc = EXCLUDED.match_datetime_utc,
            status = EXCLUDED.status,
            flashscore_url = EXCLUDED.flashscore_url,
            updated_at = NOW()
        """,
        (
            flashscore_id,
            league_code,
            home_team_id,
            away_team_id,
            match_datetime_utc,
            status,
            flashscore_url,
        ),
    )


def ingest_league(league_code: str, limit: int | None):
    registry = load_registry()
    meta = registry.get(league_code)
    if not meta:
        raise ValueError(f"Unknown league_code in registry: {league_code}")

    rows = load_discovery_rows(league_code)
    if limit is not None:
        rows = rows[:limit]

    inserted = 0
    skipped = 0

    conn = connect_db()
    try:
        with conn:
            with conn.cursor() as cur:
                upsert_league(cur, league_code, meta.get("league_name", league_code))

                for r in rows:
                    fs_id = r.get("flashscore_id")
                    kickoff = r.get("kickoff_datetime_utc")
                    home = r.get("home_team")
                    away = r.get("away_team")
                    if not (fs_id and kickoff and home and away):
                        skipped += 1
                        continue

                    # Minimal status heuristic for seed: past -> ft, future -> scheduled.
                    # This will be corrected later by settlement/enrichment.
                    status = "scheduled"
                    try:
                        # ISO string in UTC with Z
                        from datetime import datetime, timezone

                        dt = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
                        if dt < datetime.now(timezone.utc):
                            status = "ft"
                    except Exception:
                        status = "scheduled"

                    home_id = upsert_team(cur, league_code, home)
                    away_id = upsert_team(cur, league_code, away)
                    upsert_fixture(cur, fs_id, league_code, home_id, away_id, kickoff, status)
                    inserted += 1
    finally:
        conn.close()

    return {"league_code": league_code, "rows": len(rows), "inserted": inserted, "skipped": skipped}


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
