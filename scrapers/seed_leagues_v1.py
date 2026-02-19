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


REGISTRY_PATH = Path("scrapers/config/league_registry.json")


def load_registry():
    leagues = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(leagues, list):
        raise ValueError("League registry must be a JSON array")
    return leagues


def upsert_league(cur, league):
    cur.execute(
        """
        INSERT INTO leagues (league_code, league_name, country)
        VALUES (%s, %s, %s)
        ON CONFLICT (league_code) DO UPDATE
        SET league_name = EXCLUDED.league_name,
            country = EXCLUDED.country,
            updated_at = NOW()
        """,
        (
            league.get("league_code"),
            league.get("league_name") or league.get("league_code"),
            league.get("flashscore_slug", "").split("/")[0] if league.get("flashscore_slug") else None,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Seed leagues table from league_registry.json")
    parser.add_argument(
        "--enabled-only",
        action="store_true",
        help="Only seed leagues where enabled=true in the registry",
    )
    args = parser.parse_args()

    enabled_only = bool(args.enabled_only)
    run_id = create_pipeline_run("seed_leagues_v1", details_json={"enabled_only": enabled_only})

    conn = None
    inserted = 0
    try:
        leagues = load_registry()
        if enabled_only:
            leagues = [l for l in leagues if l.get("enabled")]

        conn = connect_db()
        with conn:
            with conn.cursor() as cur:
                for league in leagues:
                    if not league.get("league_code"):
                        continue
                    upsert_league(cur, league)
                    inserted += 1
        result = {"seeded": inserted, "enabled_only": enabled_only}
        finalize_pipeline_run(
            run_id,
            "success",
            message=f"Seeded {inserted} leagues",
            details_json=result,
        )
        print(json.dumps(result, indent=2))
    except Exception as exc:
        finalize_pipeline_run(
            run_id,
            "fail",
            message=str(exc),
            details_json={"enabled_only": enabled_only, "seeded": inserted},
        )
        raise
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    if not (os.getenv("DATABASE_URL") or os.getenv("DEV_DATABASE_URL")):
        raise SystemExit(
            "Missing DATABASE_URL (or DEV_DATABASE_URL). Set DATABASE_URL before running this script."
        )
    main()
