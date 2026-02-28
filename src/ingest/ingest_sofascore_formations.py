"""
Ingest starting formation (e.g. '4-3-3') for home and away sides from SofaScore.

Usage:
    python src/ingest/ingest_sofascore_formations.py --limit 100 --status ft
    python src/ingest/ingest_sofascore_formations.py --league E1 --limit 50
    python src/ingest/ingest_sofascore_formations.py --dry-run --limit 5
    python src/ingest/ingest_sofascore_formations.py --fixture-ids-file artifacts/missing.csv
"""

from __future__ import annotations

import argparse
import asyncio
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
    parser = argparse.ArgumentParser(description="Ingest SofaScore match formations.")
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
                CREATE TABLE IF NOT EXISTS fixture_formations (
                    fixture_id       BIGINT PRIMARY KEY REFERENCES fixtures(fixture_id) ON DELETE CASCADE,
                    home_formation   TEXT,
                    away_formation   TEXT,
                    raw_home_json    JSONB NOT NULL DEFAULT '{}'::jsonb,
                    raw_away_json    JSONB NOT NULL DEFAULT '{}'::jsonb,
                    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fixture_formations_home
                ON fixture_formations (home_formation);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fixture_formations_away
                ON fixture_formations (away_formation);
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
            LEFT JOIN fixture_formations ff ON ff.fixture_id = f.fixture_id
            WHERE f.sofascore_id IS NOT NULL
              AND ff.fixture_id IS NULL
        """

    if status != "all":
        query += " AND f.status = %s"
        params.append(status)

    if league_code:
        query += " AND f.league_code = %s"
        params.append(league_code)

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


def _upsert_formation(
    fixture_id: int,
    home_formation: str | None,
    away_formation: str | None,
    raw_home: dict[str, Any],
    raw_away: dict[str, Any],
    *,
    dry_run: bool,
) -> None:
    if dry_run:
        return

    delay = DB_RETRY_INITIAL_DELAY_SECONDS
    for attempt in range(1, DB_WRITE_MAX_ATTEMPTS + 1):
        conn = None
        try:
            conn = _connect_db_with_retry(
                context=f"upsert_formation fixture_id={fixture_id}",
                max_attempts=1,
            )
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO fixture_formations (
                        fixture_id,
                        home_formation,
                        away_formation,
                        raw_home_json,
                        raw_away_json,
                        ingested_at,
                        updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (fixture_id) DO UPDATE SET
                        home_formation = EXCLUDED.home_formation,
                        away_formation = EXCLUDED.away_formation,
                        raw_home_json  = EXCLUDED.raw_home_json,
                        raw_away_json  = EXCLUDED.raw_away_json,
                        updated_at     = NOW()
                    """,
                    (
                        fixture_id,
                        home_formation,
                        away_formation,
                        Json(raw_home),
                        Json(raw_away),
                    ),
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
                    f"  transient DB write error for fixture "
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


async def ingest_fixture_formation(
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
        payload = await api._get(f"/event/{sofascore_id}/lineups")
    except Exception as exc:
        msg = str(exc)
        if "404" in msg:
            print("  lineups 404: not available")
            return "not_found"
        print(f"  lineups fetch error: {exc}")
        return "error"

    if not isinstance(payload, dict):
        print("  invalid lineups payload (not a dict)")
        return "error"

    home_block: dict[str, Any] = payload.get("home") or {}
    away_block: dict[str, Any] = payload.get("away") or {}

    home_formation: str | None = home_block.get("formation") or None
    away_formation: str | None = away_block.get("formation") or None

    print(f"  home_formation={home_formation!r}  away_formation={away_formation!r}")

    if home_formation is None and away_formation is None:
        print("  no formation data in payload; writing nulls to prevent re-fetch")

    _upsert_formation(
        fixture_id=fixture_id,
        home_formation=home_formation,
        away_formation=away_formation,
        raw_home=home_block,
        raw_away=away_block,
        dry_run=dry_run,
    )

    if dry_run:
        print("  [DRY RUN] no DB write")
    else:
        print("  committed formation")

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
        print("No fixtures found needing formation ingestion.")
        return

    print(f"Found {len(fixtures)} fixtures needing formation ingestion.")
    api = SofascoreAPI()
    summary: dict[str, int] = {"success": 0, "not_found": 0, "error": 0}
    try:
        for fixture in fixtures:
            try:
                outcome = await ingest_fixture_formation(api, fixture, dry_run=args.dry_run)
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
        f"not_found={summary['not_found']} "
        f"error={summary['error']}"
    )


if __name__ == "__main__":
    asyncio.run(main())
