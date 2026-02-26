"""
Poll Sofascore odds for upcoming fixtures and snapshot into fixture_odds_markets.

Cadence:
  - Every 2 hours until T-3h
  - Every 30 minutes in last 3 hours
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

VENDOR_DIR = ROOT_DIR / "vendor" / "sofascore-wrapper"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from src.db.db_utils import connect_db
from src.common.pipeline_logging import create_pipeline_run, finalize_pipeline_run
from src.common.script_logger import ScriptLogger, default_log_path
from src.ingest.backfill_sofascore_odds_markets_v1 import extract_markets
from sofascore_wrapper.api import SofascoreAPI

JOB_NAME = "sofascore_odds_polling"
_LOGGER: ScriptLogger | None = None
_LOG_PATH: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll Sofascore odds for upcoming fixtures")
    parser.add_argument("--days", type=int, default=3, help="Lookahead window in days")
    parser.add_argument("--league", type=str, default=None, help="Optional league code filter")
    parser.add_argument("--provider-id", type=int, default=1, help="Provider ID (default: Bet365 = 1)")
    parser.add_argument("--dry-run", action="store_true", help="Fetch data but do not write to DB")
    parser.add_argument("--sleep-sec", type=float, default=1.0, help="Sleep between API calls")
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file path")
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
            message="odds polling started",
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


def _cadence_for_fixture(match_dt: datetime) -> timedelta:
    now = datetime.now(timezone.utc)
    hours_to_kickoff = (match_dt - now).total_seconds() / 3600.0
    if hours_to_kickoff <= 3:
        return timedelta(minutes=30)
    return timedelta(hours=2)


def _is_not_found_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "404" in msg or "not found" in msg


def fetch_upcoming(args: argparse.Namespace) -> list[dict[str, Any]]:
    query = """
        SELECT f.fixture_id, f.sofascore_id, f.match_datetime_utc,
               (
                 SELECT MAX(fom.snapshot_time_utc)
                 FROM fixture_odds_markets fom
                 WHERE fom.fixture_id = f.fixture_id
                   AND fom.provider = 'sofascore'
                   AND fom.snapshot_type = 'latest_pre_match'
               ) AS last_snapshot_time
        FROM fixtures f
        WHERE f.status = 'scheduled'
          AND f.sofascore_id IS NOT NULL
          AND f.match_datetime_utc BETWEEN NOW() AND NOW() + (%s || ' days')::interval
    """
    params: list[Any] = [args.days]
    if args.league:
        query += " AND f.league_code = %s"
        params.append(args.league)
    query += " ORDER BY f.match_datetime_utc ASC"

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()


async def poll_fixture(api: SofascoreAPI, args: argparse.Namespace, fixture: dict[str, Any]) -> int:
    fixture_id = int(fixture["fixture_id"])
    ss_id = fixture["sofascore_id"]
    match_dt = fixture["match_datetime_utc"]
    last_snapshot = fixture.get("last_snapshot_time")

    if match_dt is None:
        return 0

    cadence = _cadence_for_fixture(match_dt)
    if last_snapshot is not None:
        age = datetime.now(timezone.utc) - last_snapshot
        if age < cadence:
            return 0

    odds_payload = await api._get(f"/event/{ss_id}/odds/1/all")
    markets = extract_markets(odds_payload or {}, args.provider_id)
    if not markets:
        return 0

    if args.dry_run:
        return len(markets)

    rows = []
    for market in markets:
        odds_json = {
            "provider": "sofascore",
            "provider_id": args.provider_id,
            "market_code": market["market_code"],
            "prices_opening": market["prices_opening"],
            "prices_latest": market["prices_latest"],
            "implied_prob": market["implied_prob"],
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "raw": market["raw"],
        }
        rows.append(
            (
                fixture_id,
                "sofascore",
                args.provider_id,
                market["market_code"],
                market["line_num"],
                market["line_text"],
                json.dumps(odds_json),
            )
        )

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO fixture_odds_markets (
                    fixture_id,
                    provider,
                    provider_id,
                    market_code,
                    line_num,
                    line_text,
                    odds_json,
                    snapshot_time_utc,
                    snapshot_type
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, NOW(), 'latest_pre_match')
                ON CONFLICT ON CONSTRAINT uq_fixture_odds_markets DO UPDATE SET
                    odds_json = EXCLUDED.odds_json,
                    created_at = NOW()
                """,
                rows,
            )
        conn.commit()
    finally:
        conn.close()

    return len(rows)


async def run_polling(args: argparse.Namespace) -> tuple[int, dict[str, int]]:
    fixtures = fetch_upcoming(args)
    counts = {
        "fixtures_total": len(fixtures),
        "fixtures_polled": 0,
        "fixtures_with_markets": 0,
        "rows_written": 0,
        "fetch_not_found": 0,
        "fetch_errors": 0,
    }
    if not fixtures:
        _log_info("No fixtures to poll.")
        return 0, counts

    api = SofascoreAPI()
    try:
        _log_info(f"Polling fixtures: {len(fixtures)}")
        for fixture in fixtures:
            counts["fixtures_polled"] += 1
            fixture_id = int(fixture["fixture_id"])
            ss_id = fixture.get("sofascore_id")
            try:
                count = await poll_fixture(api, args, fixture)
            except Exception as exc:
                if _is_not_found_error(exc):
                    counts["fetch_not_found"] += 1
                    _log_warn(
                        f"Skip fixture {fixture_id} (Sofa={ss_id}): odds endpoint returned not found ({exc})"
                    )
                    await asyncio.sleep(args.sleep_sec)
                    continue
                counts["fetch_errors"] += 1
                raise
            if count:
                counts["fixtures_with_markets"] += 1
                counts["rows_written"] += count
            await asyncio.sleep(args.sleep_sec)
        _log_info(f"Inserted {counts['rows_written']} odds snapshot rows.")
    finally:
        await api.close()

    return 0, counts


def main() -> int:
    args = parse_args()
    _configure_logging(args.log_file if isinstance(args.log_file, str) else None)
    details: dict[str, object] = {
        "days": int(args.days),
        "league": args.league,
        "provider_id": int(args.provider_id),
        "sleep_sec": float(args.sleep_sec),
        "dry_run": bool(args.dry_run),
        "log_file": str(_LOG_PATH) if _LOG_PATH is not None else None,
    }
    run_id = _start_run(details, bool(args.dry_run))
    try:
        rc, counts = asyncio.run(run_polling(args))
        details.update({f"count_{k}": int(v) for k, v in counts.items()})
        _finish_run(run_id, "success", "odds polling complete", details)
        return rc
    except Exception as exc:
        details["error"] = str(exc)
        _finish_run(run_id, "fail", f"odds polling failed: {exc}", details)
        _log_warn(f"Odds polling failed: {exc}")
        return 1
    finally:
        if _LOGGER is not None:
            _LOGGER.close()


if __name__ == "__main__":
    raise SystemExit(main())
