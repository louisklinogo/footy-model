"""Fixtures-first daily pipeline orchestrator."""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pipeline_logging import create_pipeline_run, finalize_pipeline_run


ROOT = Path(".")
REGISTRY_PATH = ROOT / "scrapers" / "config" / "league_registry.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixtures-first daily pipeline")
    parser.add_argument(
        "--mode",
        choices=("incremental", "seed"),
        default="incremental",
        help="Discovery mode for fixture seeding",
    )
    parser.add_argument(
        "--leagues",
        default=None,
        help="Comma-separated league codes (defaults to enabled leagues)",
    )
    parser.add_argument("--days", type=int, default=3, help="Prediction horizon in days")
    parser.add_argument(
        "--since",
        default=None,
        help="Readiness scope lower bound (YYYY-MM-DD). Defaults to today-30d.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Log commands without executing")
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
    return [str(row["league_code"]) for row in leagues if row.get("enabled") and row.get("league_code")]


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
        fail_message = stderr_message or f"Command failed with exit code {exc.returncode}"
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


def main() -> int:
    args = parse_args()
    leagues_arg = parse_league_list(args.leagues)
    leagues = leagues_arg if leagues_arg is not None else load_enabled_leagues()
    if not leagues:
        raise RuntimeError("No leagues available for per-league pipeline steps")

    since = args.since or (datetime.now(UTC).date() - timedelta(days=30)).isoformat()

    discover_cmd = ["node", str(ROOT / "scrapers" / "run_seed_all.js"), "--mode", args.mode]
    ingest_discovery_cmd = [
        sys.executable,
        str(ROOT / "scrapers" / "ingest_discovered_fixtures_all_v1.py"),
    ]
    if leagues_arg is not None:
        league_csv = ",".join(leagues_arg)
        discover_cmd.extend(["--leagues", league_csv])
        ingest_discovery_cmd.extend(["--leagues", league_csv])

    if not run_step("daily.discover_fixtures", discover_cmd, args.dry_run):
        raise RuntimeError("Pipeline aborted at fixture discovery step")

    if not run_step("daily.ingest_discovered_fixtures", ingest_discovery_cmd, args.dry_run):
        raise RuntimeError("Pipeline aborted at discovered fixture ingest step")

    for league in leagues:
        enrich_cmd = [
            "node",
            str(ROOT / "scrapers" / "premium_enricher_v4.js"),
            league,
            "--ids-root",
            "data/v1/ids",
            "--out-root",
            "data/v1/premium",
        ]
        if not run_step(f"daily.premium_enrich.{league}", enrich_cmd, args.dry_run):
            raise RuntimeError(f"Pipeline aborted at premium enrich step for league={league}")

        ingest_premium_cmd = [
            sys.executable,
            str(ROOT / "scrapers" / "ingest_premium_fixtures_v1.py"),
            "--league",
            league,
        ]
        if not run_step(f"daily.premium_ingest.{league}", ingest_premium_cmd, args.dry_run):
            raise RuntimeError(f"Pipeline aborted at premium ingest step for league={league}")

    for league in leagues:
        readiness_cmd = [
            sys.executable,
            str(ROOT / "scrapers" / "check_premium_readiness.py"),
            "--league",
            league,
            "--since",
            since,
        ]
        readiness_ok = run_step(f"daily.premium_readiness_gate.{league}", readiness_cmd, args.dry_run)
        if not readiness_ok:
            raise RuntimeError(
                f"Readiness gate failed for league={league}; aborting snapshots/predict/export/score"
            )

    for league in leagues:
        snapshot_cmd = [
            sys.executable,
            str(ROOT / "scrapers" / "build_team_premium_snapshots_v1.py"),
            "--league",
            league,
        ]
        if not run_step(f"daily.build_snapshots.{league}", snapshot_cmd, args.dry_run):
            raise RuntimeError(f"Pipeline aborted at snapshot build step for league={league}")

    for league in leagues:
        predict_cmd = [
            sys.executable,
            str(ROOT / "models" / "predict_v3_fixtures_first.py"),
            "--league",
            league,
            "--days",
            str(args.days),
        ]
        if not run_step(f"daily.predict.{league}", predict_cmd, args.dry_run):
            raise RuntimeError(f"Pipeline aborted at prediction step for league={league}")

    for league in leagues:
        export_cmd = [
            sys.executable,
            str(ROOT / "models" / "export_predictions_v3_fixtures_first.py"),
            "--league",
            league,
            "--days",
            str(args.days),
        ]
        if not run_step(f"daily.export_predictions_csv.{league}", export_cmd, args.dry_run):
            raise RuntimeError(f"Pipeline aborted at export step for league={league}")

    for league in leagues:
        score_cmd = [
            sys.executable,
            str(ROOT / "models" / "score_predictions_v3_fixtures_first.py"),
            "--league",
            league,
        ]
        if not run_step(f"daily.score_predictions.{league}", score_cmd, args.dry_run):
            raise RuntimeError(f"Pipeline aborted at scoring step for league={league}")

    print(f"\n{'=' * 60}")
    print("DAILY FIXTURES-FIRST PIPELINE COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        raise SystemExit(1)
