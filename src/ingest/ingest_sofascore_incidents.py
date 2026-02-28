"""
Ingest match incidents (goal timeline + cards/subs/period markers) from SofaScore.

Usage:
    python src/ingest/ingest_sofascore_incidents.py --limit 100 --status ft
    python src/ingest/ingest_sofascore_incidents.py --league E0 --limit 50
    python src/ingest/ingest_sofascore_incidents.py --dry-run --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import Json

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false, reportAttributeAccessIssue=false, reportArgumentType=false

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

VENDOR_DIR = ROOT_DIR / "vendor" / "sofascore-wrapper"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from src.db.db_utils import connect_db
from sofascore_wrapper.api import SofascoreAPI


DB_CONNECT_MAX_ATTEMPTS = 5
DB_WRITE_MAX_ATTEMPTS = 5
DB_RETRY_INITIAL_DELAY_SECONDS = 1.0
DB_RETRY_MAX_DELAY_SECONDS = 8.0
TRANSIENT_DB_ERROR_MARKERS = (
    "control plane request failed",
    "connection to server",
    "server closed the connection",
    "could not connect to server",
    "timeout expired",
    "connection not open",
    "ssl syscall error",
)


def _is_transient_db_error(exc: BaseException) -> bool:
    if isinstance(exc, psycopg2.OperationalError):
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in TRANSIENT_DB_ERROR_MARKERS)


def _connect_db_with_retry(
    *,
    context: str,
    max_attempts: int = DB_CONNECT_MAX_ATTEMPTS,
):
    delay = DB_RETRY_INITIAL_DELAY_SECONDS
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return connect_db()
        except Exception as exc:
            last_exc = exc
            if not _is_transient_db_error(exc) or attempt >= max_attempts:
                raise
            print(
                f"  transient DB connect error ({context}) "
                f"(attempt {attempt}/{max_attempts}): {exc}"
            )
            print(f"  retrying DB connect in {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2.0, DB_RETRY_MAX_DELAY_SECONDS)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("DB connect retry loop exited unexpectedly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest SofaScore match incidents.")
    parser.add_argument("--league", type=str, default=None, help="Filter by league code")
    parser.add_argument("--limit", type=int, default=50, help="Max fixtures to process")
    parser.add_argument(
        "--fixture-ids-file",
        type=str,
        default=None,
        help="Optional CSV/TXT file with fixture_id values (first column).",
    )
    parser.add_argument(
        "--status",
        type=str,
        default="ft",
        choices=["scheduled", "ft", "all"],
        help="Fixture status to process (default: ft)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch only, do not write")
    return parser.parse_args()


def _load_fixture_ids(path_str: str | None) -> list[int]:
    if not path_str:
        return []
    fixture_ids: list[int] = []
    for raw_line in Path(path_str).read_text(encoding="utf-8").splitlines():
        token = raw_line.split(",")[0].strip()
        if not token or token.lower() == "fixture_id":
            continue
        try:
            fixture_ids.append(int(token))
        except ValueError:
            continue
    return fixture_ids


def ensure_schema() -> None:
    conn = _connect_db_with_retry(context="ensure_schema")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS fixture_incidents_sofascore (
                    fixture_id BIGINT NOT NULL REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
                    incident_uid TEXT NOT NULL,
                    incident_id BIGINT,
                    sofascore_id TEXT,
                    incident_type TEXT NOT NULL,
                    incident_class TEXT,
                    reason TEXT,
                    incident_text TEXT,
                    is_home BOOLEAN,
                    minute SMALLINT,
                    added_time SMALLINT,
                    time_seconds INTEGER,
                    reversed_period_time SMALLINT,
                    reversed_period_time_seconds INTEGER,
                    home_score SMALLINT,
                    away_score SMALLINT,
                    player_sofascore_id BIGINT,
                    player_name TEXT,
                    assist1_sofascore_id BIGINT,
                    assist1_name TEXT,
                    raw_incident JSONB NOT NULL DEFAULT '{}'::jsonb,
                    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (fixture_id, incident_uid)
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fixture_incidents_sofascore_fixture
                ON fixture_incidents_sofascore (fixture_id);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fixture_incidents_sofascore_type
                ON fixture_incidents_sofascore (incident_type);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fixture_incidents_sofascore_minute
                ON fixture_incidents_sofascore (minute);
                """
            )
        conn.commit()
    finally:
        conn.close()


def fetch_target_fixtures(
    league_code: str | None,
    status: str,
    limit: int,
    fixture_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    fixture_ids = fixture_ids or []
    query = """
        SELECT
            f.fixture_id,
            f.sofascore_id,
            f.league_code,
            f.status,
            f.match_datetime_utc
        FROM fixtures f
    """
    params: list[Any] = []

    if fixture_ids:
        query += " WHERE f.sofascore_id IS NOT NULL AND f.fixture_id = ANY(%s)"
        params.append(fixture_ids)
    else:
        query += """
            LEFT JOIN (
                SELECT fixture_id, COUNT(*) AS incident_count
                FROM fixture_incidents_sofascore
                GROUP BY fixture_id
            ) fi ON fi.fixture_id = f.fixture_id
            WHERE f.sofascore_id IS NOT NULL
              AND (fi.fixture_id IS NULL OR fi.incident_count = 0)
        """

    if status != "all":
        query += " AND f.status = %s"
        params.append(status)

    if league_code:
        query += " AND f.league_code = %s"
        params.append(league_code)

    if status == "scheduled":
        query += " ORDER BY f.match_datetime_utc ASC, f.fixture_id ASC"
    else:
        query += " ORDER BY f.match_datetime_utc DESC, f.fixture_id DESC"
    query += " LIMIT %s"
    params.append(limit)

    conn = _connect_db_with_retry(context="fetch_target_fixtures")
    try:
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _incident_uid(incident: dict[str, Any]) -> str:
    incident_id = _to_int(incident.get("id"))
    if incident_id is not None:
        return f"id:{incident_id}"

    player = incident.get("player") or {}
    key_payload = {
        "incidentType": incident.get("incidentType"),
        "incidentClass": incident.get("incidentClass"),
        "reason": incident.get("reason"),
        "text": incident.get("text"),
        "time": incident.get("time"),
        "addedTime": incident.get("addedTime"),
        "timeSeconds": incident.get("timeSeconds"),
        "reversedPeriodTime": incident.get("reversedPeriodTime"),
        "reversedPeriodTimeSeconds": incident.get("reversedPeriodTimeSeconds"),
        "isHome": incident.get("isHome"),
        "homeScore": incident.get("homeScore"),
        "awayScore": incident.get("awayScore"),
        "playerId": player.get("id"),
    }
    digest = hashlib.sha1(
        json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"hash:{digest}"


def _normalize_incident(
    fixture_id: int,
    sofascore_id: str,
    incident: dict[str, Any],
) -> dict[str, Any]:
    player = incident.get("player") or {}
    assist1 = incident.get("assist1") or {}

    return {
        "fixture_id": fixture_id,
        "incident_uid": _incident_uid(incident),
        "incident_id": _to_int(incident.get("id")),
        "sofascore_id": sofascore_id,
        "incident_type": str(incident.get("incidentType") or "unknown"),
        "incident_class": incident.get("incidentClass"),
        "reason": incident.get("reason"),
        "incident_text": incident.get("text"),
        "is_home": incident.get("isHome"),
        "minute": _to_int(incident.get("time")),
        "added_time": _to_int(incident.get("addedTime")),
        "time_seconds": _to_int(incident.get("timeSeconds")),
        "reversed_period_time": _to_int(incident.get("reversedPeriodTime")),
        "reversed_period_time_seconds": _to_int(incident.get("reversedPeriodTimeSeconds")),
        "home_score": _to_int(incident.get("homeScore")),
        "away_score": _to_int(incident.get("awayScore")),
        "player_sofascore_id": _to_int(player.get("id")),
        "player_name": player.get("name"),
        "assist1_sofascore_id": _to_int(assist1.get("id")),
        "assist1_name": assist1.get("name"),
        "raw_incident": incident,
    }


def _replace_fixture_incidents(
    fixture_id: int,
    records: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> None:
    if dry_run:
        return

    # Safe to retry on transient failures because this operation fully replaces a fixture's incidents.
    delay = DB_RETRY_INITIAL_DELAY_SECONDS
    for attempt in range(1, DB_WRITE_MAX_ATTEMPTS + 1):
        conn = None
        try:
            conn = _connect_db_with_retry(
                context=f"replace_fixture_incidents fixture_id={fixture_id}",
                max_attempts=1,
            )
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM fixture_incidents_sofascore WHERE fixture_id = %s",
                    (fixture_id,),
                )
                for row in records:
                    cur.execute(
                        """
                        INSERT INTO fixture_incidents_sofascore (
                            fixture_id,
                            incident_uid,
                            incident_id,
                            sofascore_id,
                            incident_type,
                            incident_class,
                            reason,
                            incident_text,
                            is_home,
                            minute,
                            added_time,
                            time_seconds,
                            reversed_period_time,
                            reversed_period_time_seconds,
                            home_score,
                            away_score,
                            player_sofascore_id,
                            player_name,
                            assist1_sofascore_id,
                            assist1_name,
                            raw_incident,
                            ingested_at,
                            updated_at
                        )
                        VALUES (
                            %(fixture_id)s,
                            %(incident_uid)s,
                            %(incident_id)s,
                            %(sofascore_id)s,
                            %(incident_type)s,
                            %(incident_class)s,
                            %(reason)s,
                            %(incident_text)s,
                            %(is_home)s,
                            %(minute)s,
                            %(added_time)s,
                            %(time_seconds)s,
                            %(reversed_period_time)s,
                            %(reversed_period_time_seconds)s,
                            %(home_score)s,
                            %(away_score)s,
                            %(player_sofascore_id)s,
                            %(player_name)s,
                            %(assist1_sofascore_id)s,
                            %(assist1_name)s,
                            %(raw_incident)s,
                            NOW(),
                            NOW()
                        )
                        ON CONFLICT (fixture_id, incident_uid) DO UPDATE SET
                            incident_id = EXCLUDED.incident_id,
                            sofascore_id = EXCLUDED.sofascore_id,
                            incident_type = EXCLUDED.incident_type,
                            incident_class = EXCLUDED.incident_class,
                            reason = EXCLUDED.reason,
                            incident_text = EXCLUDED.incident_text,
                            is_home = EXCLUDED.is_home,
                            minute = EXCLUDED.minute,
                            added_time = EXCLUDED.added_time,
                            time_seconds = EXCLUDED.time_seconds,
                            reversed_period_time = EXCLUDED.reversed_period_time,
                            reversed_period_time_seconds = EXCLUDED.reversed_period_time_seconds,
                            home_score = EXCLUDED.home_score,
                            away_score = EXCLUDED.away_score,
                            player_sofascore_id = EXCLUDED.player_sofascore_id,
                            player_name = EXCLUDED.player_name,
                            assist1_sofascore_id = EXCLUDED.assist1_sofascore_id,
                            assist1_name = EXCLUDED.assist1_name,
                            raw_incident = EXCLUDED.raw_incident,
                            updated_at = NOW()
                        """,
                        {
                            **row,
                            "raw_incident": Json(row["raw_incident"]),
                        },
                    )
            conn.commit()
            return
        except Exception as exc:
            if conn is not None:
                try:
                    conn.rollback()
                except Exception:
                    pass
            if _is_transient_db_error(exc) and attempt < DB_WRITE_MAX_ATTEMPTS:
                print(
                    "  transient DB write error for fixture "
                    f"{fixture_id} (attempt {attempt}/{DB_WRITE_MAX_ATTEMPTS}): {exc}"
                )
                print(f"  retrying fixture write in {delay:.1f}s")
                time.sleep(delay)
                delay = min(delay * 2.0, DB_RETRY_MAX_DELAY_SECONDS)
                continue
            raise
        finally:
            if conn is not None:
                conn.close()


async def ingest_fixture_incidents(
    api: SofascoreAPI,
    fixture: dict[str, Any],
    *,
    dry_run: bool,
) -> str:
    fixture_id = int(fixture["fixture_id"])
    sofascore_id = str(fixture["sofascore_id"])
    status = str(fixture.get("status") or "")
    print(
        f"Processing fixture {fixture_id} (SS: {sofascore_id}, "
        f"status={status}, league={fixture.get('league_code')})..."
    )

    try:
        payload = await api._get(f"/event/{sofascore_id}/incidents")
    except Exception as exc:
        msg = str(exc)
        if "404" in msg:
            print("  incidents 404: not available yet")
            return "not_found"
        print(f"  incidents fetch error: {exc}")
        return "error"

    if not isinstance(payload, dict):
        print("  invalid incidents payload (not a dict)")
        return "error"

    incidents = payload.get("incidents", [])
    if not isinstance(incidents, list):
        print("  invalid incidents payload ('incidents' is not a list)")
        return "error"

    records = [
        _normalize_incident(fixture_id=fixture_id, sofascore_id=sofascore_id, incident=incident)
        for incident in incidents
        if isinstance(incident, dict)
    ]

    goal_count = sum(1 for row in records if row["incident_type"] == "goal")
    print(f"  incidents={len(records)} goals={goal_count}")

    if not records:
        print("  no incidents returned; skipping write")
        return "empty"

    _replace_fixture_incidents(fixture_id, records, dry_run=dry_run)
    if dry_run:
        print("  [DRY RUN] no DB write")
    else:
        print("  committed incidents")
    return "success"


async def main() -> None:
    args = parse_args()
    ensure_schema()
    fixture_ids = _load_fixture_ids(args.fixture_ids_file)

    fixtures = fetch_target_fixtures(
        league_code=args.league,
        status=args.status,
        limit=args.limit,
        fixture_ids=fixture_ids,
    )
    if not fixtures:
        print("No fixtures found needing incidents ingestion.")
        return

    print(f"Found {len(fixtures)} fixtures needing incidents ingestion.")
    api = SofascoreAPI()
    summary = {"success": 0, "empty": 0, "not_found": 0, "error": 0}
    try:
        for fixture in fixtures:
            try:
                outcome = await ingest_fixture_incidents(api, fixture, dry_run=args.dry_run)
            except Exception as exc:
                print(f"  fixture {fixture.get('fixture_id')} failed with unexpected error: {exc}")
                outcome = "error"
            summary[outcome] = summary.get(outcome, 0) + 1
            await asyncio.sleep(0.4)
    finally:
        await api.close()

    print(
        "Done: "
        f"success={summary['success']} "
        f"empty={summary['empty']} "
        f"not_found={summary['not_found']} "
        f"error={summary['error']}"
    )


if __name__ == "__main__":
    asyncio.run(main())
