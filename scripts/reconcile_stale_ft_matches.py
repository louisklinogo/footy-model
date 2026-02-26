"""
Reconcile stale fixtures and FT-without-results rows via SofaScore event API.

Primary use:
- Backfill stale scheduled fixtures that are already finished.
- Backfill fixtures marked FT but missing rows in fixture_results.

Usage:
  python scripts/reconcile_stale_ft_matches.py --hours-passed 3 --limit 200
  python scripts/reconcile_stale_ft_matches.py --league E0 --limit 100
  python scripts/reconcile_stale_ft_matches.py --fixture-ids 101,102,103 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

VENDOR_DIR = ROOT_DIR / "vendor" / "sofascore-wrapper"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from sofascore_wrapper.api import SofascoreAPI
from src.common.pipeline_logging import create_pipeline_run, finalize_pipeline_run
from src.common.script_logger import ScriptLogger, default_log_path
from src.db.db_utils import connect_db


JOB_NAME = "reconcile_stale_ft_matches"
_LOGGER: ScriptLogger | None = None
_LOG_PATH: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile stale fixtures / missing FT results via SofaScore"
    )
    parser.add_argument(
        "--hours-passed",
        type=int,
        default=3,
        help="Scheduled fixtures older than this (hours) are considered stale",
    )
    parser.add_argument("--limit", type=int, default=200, help="Maximum fixtures to process")
    parser.add_argument("--league", type=str, default=None, help="Optional league code filter")
    parser.add_argument(
        "--fixture-ids",
        type=str,
        default=None,
        help="Optional comma-separated fixture_id list to reconcile directly",
    )
    parser.add_argument("--sleep-sec", type=float, default=0.2, help="Sleep between API calls")
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file path")
    parser.add_argument("--dry-run", action="store_true", help="Do not write DB changes")
    return parser.parse_args()


def _configure_logging(log_file: str | None) -> None:
    global _LOGGER, _LOG_PATH
    path = (
        Path(log_file).expanduser()
        if isinstance(log_file, str) and log_file.strip()
        else default_log_path(ROOT_DIR, JOB_NAME)
    )
    _LOGGER = ScriptLogger(path)
    _LOG_PATH = path
    _LOGGER.info(f"Log file: {path}")


def _log_info(message: str) -> None:
    if _LOGGER is not None:
        _LOGGER.info(message)
    else:
        print(message)


def _log_warn(message: str) -> None:
    if _LOGGER is not None:
        _LOGGER.warn(message)
    else:
        print(message, file=sys.stderr)


def _start_run(details: dict[str, object], dry_run: bool) -> int | None:
    payload = dict(details)
    if _LOG_PATH is not None:
        payload["log_file"] = str(_LOG_PATH)
    try:
        return create_pipeline_run(
            job_name=JOB_NAME,
            message="reconcile started",
            details_json=payload,
        )
    except Exception as exc:
        if dry_run:
            _log_warn(f"WARN: pipeline run logging unavailable: {exc}")
            return None
        raise


def _finish_run(
    run_id: int | None,
    status: str,
    message: str,
    details: dict[str, object],
) -> None:
    if run_id is None:
        return
    payload = dict(details)
    if _LOG_PATH is not None:
        payload["log_file"] = str(_LOG_PATH)
    finalize_pipeline_run(run_id, status, message=message, details_json=payload)


def parse_fixture_ids(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values or None


def fetch_candidates(
    hours_passed: int,
    limit: int,
    league: str | None,
    fixture_ids: list[int] | None,
) -> list[dict[str, Any]]:
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            if fixture_ids:
                cur.execute(
                    """
                    SELECT
                        f.fixture_id,
                        f.sofascore_id,
                        f.flashscore_id,
                        f.league_code,
                        f.status,
                        f.match_datetime_utc
                    FROM fixtures f
                    WHERE f.fixture_id = ANY(%s)
                      AND f.sofascore_id IS NOT NULL
                    ORDER BY f.match_datetime_utc ASC NULLS LAST, f.fixture_id ASC
                    """,
                    (fixture_ids,),
                )
            else:
                query = """
                    WITH ft_missing AS (
                        SELECT f2.fixture_id
                        FROM fixtures f2
                        LEFT JOIN fixture_results fr2 ON fr2.fixture_id = f2.fixture_id
                        WHERE f2.status = 'ft'
                          AND fr2.result_id IS NULL
                    )
                    SELECT
                        f.fixture_id,
                        f.sofascore_id,
                        f.flashscore_id,
                        f.league_code,
                        f.status,
                        f.match_datetime_utc
                    FROM fixtures f
                    LEFT JOIN ft_missing m ON m.fixture_id = f.fixture_id
                    WHERE f.sofascore_id IS NOT NULL
                      AND (
                        (
                          f.status = 'scheduled'
                          AND f.match_datetime_utc IS NOT NULL
                          AND f.match_datetime_utc < NOW() - (%s || ' hours')::interval
                        )
                        OR m.fixture_id IS NOT NULL
                      )
                """
                params: list[Any] = [hours_passed]
                if league:
                    query += " AND f.league_code = %s"
                    params.append(league)
                query += """
                    ORDER BY (f.status = 'ft') DESC, f.match_datetime_utc ASC NULLS LAST, f.fixture_id ASC
                    LIMIT %s
                """
                params.append(limit)
                cur.execute(query, tuple(params))

            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()


def _unwrap_event(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        nested = payload.get("event")
        if isinstance(nested, dict):
            return nested
        return payload
    return {}


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raw = str(value).strip()
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _score_from_event(event: dict[str, Any]) -> tuple[int | None, int | None]:
    home = event.get("homeScore") if isinstance(event.get("homeScore"), dict) else {}
    away = event.get("awayScore") if isinstance(event.get("awayScore"), dict) else {}

    h = _to_int(home.get("display"))
    a = _to_int(away.get("display"))
    if h is not None and a is not None:
        return h, a

    h = _to_int(home.get("current"))
    a = _to_int(away.get("current"))
    return h, a


def _update_fixture_terminal_status(fixture_id: int, status: str, dry_run: bool) -> None:
    if dry_run:
        _log_info(f"  DRY RUN: would set fixture={fixture_id} status={status}")
        return

    conn = connect_db()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE fixtures
                    SET status = %s,
                        settlement_status = CASE
                            WHEN %s IN ('cancelled', 'postponed', 'abandoned') THEN 'failed'
                            ELSE settlement_status
                        END,
                        updated_at = NOW()
                    WHERE fixture_id = %s
                    """,
                    (status, status, fixture_id),
                )
    finally:
        conn.close()


def _settle_fixture(fixture_id: int, home_goals: int, away_goals: int, dry_run: bool) -> None:
    if dry_run:
        _log_info(
            f"  DRY RUN: would settle fixture={fixture_id} result={home_goals}-{away_goals}"
        )
        return

    conn = connect_db()
    try:
        with conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        "SELECT settle_fixture_by_fixture_id(%s, %s, %s)",
                        (fixture_id, home_goals, away_goals),
                    )
                except Exception:
                    # Fallback for environments missing migration 011.
                    cur.execute(
                        """
                        UPDATE fixtures
                        SET status = 'ft',
                            settlement_status = 'settled',
                            updated_at = NOW()
                        WHERE fixture_id = %s
                        """,
                        (fixture_id,),
                    )
                    cur.execute(
                        """
                        INSERT INTO fixture_results (
                            fixture_id, home_goals, away_goals, result_status, settled_at
                        )
                        VALUES (%s, %s, %s, 'ft', NOW())
                        ON CONFLICT (fixture_id) DO UPDATE SET
                            home_goals = EXCLUDED.home_goals,
                            away_goals = EXCLUDED.away_goals,
                            result_status = EXCLUDED.result_status,
                            settled_at = NOW()
                        """,
                        (fixture_id, home_goals, away_goals),
                    )
    finally:
        conn.close()


async def run_reconcile(args: argparse.Namespace) -> tuple[int, dict[str, int]]:
    fixture_ids = parse_fixture_ids(args.fixture_ids)
    targets = fetch_candidates(
        hours_passed=args.hours_passed,
        limit=args.limit,
        league=args.league,
        fixture_ids=fixture_ids,
    )
    counts = {
        "processed": 0,
        "settled": 0,
        "cancelled_or_postponed": 0,
        "unfinished": 0,
        "missing_scores": 0,
        "fetch_errors": 0,
    }
    if not targets:
        _log_info("No reconcile candidates found.")
        return 0, counts

    api = SofascoreAPI()
    try:
        for row in targets:
            fixture_id = int(row["fixture_id"])
            sofa_id = str(row["sofascore_id"])
            kickoff = row.get("match_datetime_utc")
            _log_info(f"Fixture {fixture_id} (Sofa={sofa_id}, kickoff={kickoff})")

            counts["processed"] += 1
            try:
                payload = await api._get(f"/event/{sofa_id}")
            except Exception as exc:
                counts["fetch_errors"] += 1
                _log_warn(f"  Fetch error: {exc}")
                await asyncio.sleep(args.sleep_sec)
                continue

            event = _unwrap_event(payload)
            status_type = str(event.get("status", {}).get("type", "")).lower()

            if status_type == "finished":
                home_goals, away_goals = _score_from_event(event)
                if home_goals is None or away_goals is None:
                    counts["missing_scores"] += 1
                    _log_warn("  Finished event but score missing; skipping.")
                    await asyncio.sleep(args.sleep_sec)
                    continue
                _settle_fixture(
                    fixture_id=fixture_id,
                    home_goals=home_goals,
                    away_goals=away_goals,
                    dry_run=bool(args.dry_run),
                )
                counts["settled"] += 1
                _log_info(f"  Settled to FT: {home_goals}-{away_goals}")
            elif status_type in {"canceled", "postponed"}:
                mapped = "cancelled" if status_type == "canceled" else "postponed"
                _update_fixture_terminal_status(fixture_id, mapped, bool(args.dry_run))
                counts["cancelled_or_postponed"] += 1
                _log_info(f"  Updated fixture status to {mapped}")
            else:
                counts["unfinished"] += 1
                _log_info(f"  Not terminal yet (status.type={status_type or 'unknown'}).")

            await asyncio.sleep(args.sleep_sec)
    finally:
        await api.close()

    _log_info("\nSummary:")
    _log_info(f"  processed: {counts['processed']}")
    _log_info(f"  settled: {counts['settled']}")
    _log_info(f"  cancelled_or_postponed: {counts['cancelled_or_postponed']}")
    _log_info(f"  unfinished: {counts['unfinished']}")
    _log_info(f"  missing_scores: {counts['missing_scores']}")
    _log_info(f"  fetch_errors: {counts['fetch_errors']}")
    return 0, counts


def main() -> int:
    args = parse_args()
    _configure_logging(args.log_file if isinstance(args.log_file, str) else None)
    _log_info("=" * 60)
    _log_info("Sofa Reconcile: Stale/FT Missing Results")
    _log_info("=" * 60)
    _log_info(
        f"started_at={datetime.now(timezone.utc).isoformat()} "
        f"dry_run={args.dry_run} league={args.league} limit={args.limit}"
    )
    details: dict[str, object] = {
        "hours_passed": int(args.hours_passed),
        "limit": int(args.limit),
        "league": args.league,
        "fixture_ids": args.fixture_ids,
        "dry_run": bool(args.dry_run),
        "log_file": str(_LOG_PATH) if _LOG_PATH is not None else None,
    }
    run_id = _start_run(details, bool(args.dry_run))
    try:
        rc, counts = asyncio.run(run_reconcile(args))
        details.update({f"count_{k}": int(v) for k, v in counts.items()})
        _finish_run(run_id, "success", "reconcile complete", details)
        return rc
    except Exception as exc:
        details["error"] = str(exc)
        _finish_run(run_id, "fail", f"reconcile failed: {exc}", details)
        _log_warn(f"Reconcile failed: {exc}")
        return 1
    finally:
        if _LOGGER is not None:
            _LOGGER.close()


if __name__ == "__main__":
    raise SystemExit(main())
