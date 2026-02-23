"""Prematch data pipeline orchestrator.

Scrapes prematch data (odds + injuries), ingests to DB, and runs AI research.

Usage:
    python src/pipelines/daily_prematch_pipeline.py --days 3
    python src/pipelines/daily_prematch_pipeline.py --leagues E0,E1 --days 2
    python src/pipelines/daily_prematch_pipeline.py --dry-run

Pipeline steps:
    1. Scrape prematch data (odds + injuries) via prematch_enricher.js
    2. Ingest to DB (fixture_availability + fixture_odds_snapshots)
    3. Run AI research enrichment for fixtures with injury data
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.pipeline_logging import create_pipeline_run, finalize_pipeline_run
from src.db.db_utils import connect_db

REGISTRY_PATH = ROOT / "scrapers" / "config" / "league_registry.json"
PREMATCH_ROOT = ROOT / "data" / "v1" / "prematch_json"
IDS_ROOT = ROOT / "data" / "v1" / "ids" / "prematch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prematch data pipeline")
    parser.add_argument(
        "--leagues",
        default=None,
        help="Comma-separated league codes (defaults to enabled leagues)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Days ahead to scrape (default: 3)",
    )
    parser.add_argument(
        "--skip-research",
        action="store_true",
        help="Skip AI research enrichment step",
    )
    parser.add_argument(
        "--research-limit",
        type=int,
        default=10,
        help="Max fixtures to research per run (default: 10)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Log commands without executing"
    )
    return parser.parse_args()


def parse_league_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    leagues = [token.strip() for token in raw.split(",") if token.strip()]
    if not leagues:
        raise ValueError("--leagues was provided but no valid league codes were found")
    return leagues


def load_enabled_leagues() -> list[str]:
    leagues = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return [
        str(row["league_code"])
        for row in leagues
        if row.get("enabled") and row.get("league_code")
    ]


def run_step(job_name: str, cmd: list[str], dry_run: bool) -> bool:
    print(f"\n{'=' * 60}")
    print(f"STEP: {job_name}")
    print("=" * 60)
    print("COMMAND:", " ".join(cmd))

    run_id = create_pipeline_run(
        job_name=job_name,
        message="Step started",
        details_json={"command": cmd, "dry_run": dry_run},
    )

    if dry_run:
        finalize_pipeline_run(
            run_id,
            "success",
            message="Dry run: command not executed",
            details_json={"command": cmd, "dry_run": True, "executed": False},
        )
        print("DRY RUN: skipped execution")
        return True

    start = datetime.now(UTC)
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")

        duration_sec = (datetime.now(UTC) - start).total_seconds()
        finalize_pipeline_run(
            run_id,
            "success",
            message=f"Step succeeded in {duration_sec:.1f}s",
            details_json={
                "command": cmd,
                "dry_run": False,
                "executed": True,
                "returncode": completed.returncode,
                "duration_sec": duration_sec,
            },
        )
        print(f"SUCCESS: {job_name} ({duration_sec:.1f}s)")
        return True
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, end="")
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="")

        duration_sec = (datetime.now(UTC) - start).total_seconds()
        stderr_message = (exc.stderr or "").strip()
        fail_message = (
            stderr_message or f"Command failed with exit code {exc.returncode}"
        )
        finalize_pipeline_run(
            run_id,
            "fail",
            message=fail_message,
            details_json={
                "command": cmd,
                "dry_run": False,
                "executed": True,
                "returncode": exc.returncode,
                "duration_sec": duration_sec,
            },
        )
        print(f"FAILED: {job_name} (exit {exc.returncode})")
        return False


def get_upcoming_fixtures_for_league(league: str, days: int) -> list[dict]:
    """Query DB for upcoming fixtures needing prematch data."""
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT f.fixture_id, f.flashscore_id, f.match_datetime_utc
                FROM fixtures f
                WHERE f.league_code = %s
                AND f.status = 'scheduled'
                AND f.match_datetime_utc > NOW()
                AND f.match_datetime_utc <= NOW() + (%s || ' days')::interval
                ORDER BY f.match_datetime_utc ASC
                """,
                (league, days),
            )
            rows = cur.fetchall()
        return [{"id": row[1]} for row in rows]
    finally:
        conn.close()


def write_ids_file(league: str, fixtures: list[dict]) -> Path:
    """Write fixture IDs to file for scraper."""
    IDS_ROOT.mkdir(parents=True, exist_ok=True)
    ids_path = IDS_ROOT / f"match_ids_{league}.json"
    ids_path.write_text(json.dumps(fixtures, indent=2), encoding="utf-8")
    return ids_path


def main() -> int:
    args = parse_args()
    leagues_arg = parse_league_list(args.leagues)
    leagues = leagues_arg if leagues_arg is not None else load_enabled_leagues()
    if not leagues:
        raise RuntimeError("No leagues available for prematch pipeline")

    print("=" * 60)
    print("PREMATCH DATA PIPELINE")
    print("=" * 60)
    print(f"Leagues: {', '.join(leagues)}")
    print(f"Days ahead: {args.days}")
    print(f"Dry run: {args.dry_run}")
    print()

    total_scraped = 0
    total_ingested = 0
    failed_leagues: list[str] = []

    # Step 1: Scrape prematch data per league
    for league in leagues:
        # Get fixtures needing prematch data
        fixtures = get_upcoming_fixtures_for_league(league, args.days)
        if not fixtures:
            print(f"No upcoming fixtures for {league}, skipping...")
            continue

        print(f"\n--- {league}: {len(fixtures)} fixtures to scrape ---")

        # Write IDs file
        ids_path = write_ids_file(league, fixtures)

        # Run prematch enricher
        scrape_cmd = [
            "node",
            str(ROOT / "scrapers" / "prematch_enricher.js"),
            league,
            str(ids_path),
        ]

        if not run_step(f"prematch.scrape.{league}", scrape_cmd, args.dry_run):
            failed_leagues.append(league)
            continue

        total_scraped += len(fixtures)

        # Step 2: Ingest prematch data
        ingest_cmd = [
            sys.executable,
            str(ROOT / "src" / "ingest" / "ingest_prematch_v1.py"),
            "--league",
            league,
        ]

        if not run_step(f"prematch.ingest.{league}", ingest_cmd, args.dry_run):
            failed_leagues.append(league)
            continue

        total_ingested += len(fixtures)

    # Step 3: AI Research Enrichment (if not skipped)
    if not args.skip_research:
        research_cmd = [
            sys.executable,
            str(ROOT / "src" / "modeling" / "ai_research_enrich.py"),
            "--days",
            str(args.days),
            "--limit",
            str(args.research_limit),
        ]

        if not run_step("prematch.ai_research", research_cmd, args.dry_run):
            print("WARNING: AI research step failed, but pipeline continues")

    # Summary
    print(f"\n{'=' * 60}")
    print("PREMATCH PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Leagues processed: {len(leagues) - len(failed_leagues)}/{len(leagues)}")
    print(f"Fixtures scraped: {total_scraped}")
    print(f"Fixtures ingested: {total_ingested}")

    if failed_leagues:
        print(f"Failed leagues: {', '.join(failed_leagues)}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        raise SystemExit(1)
