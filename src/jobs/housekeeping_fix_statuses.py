"""Housekeeping job: Report stale fixture statuses.

Finds fixtures that are still marked as 'scheduled' but are past match time.
Reports them for review - does NOT modify status.
The tick job handles settlement retries via fixture_job_state table.
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db.db_utils import connect_db


@dataclass(frozen=True)
class Options:
    hours_passed: int
    limit: int


def parse_args() -> Options:
    parser = argparse.ArgumentParser(
        description="Report stale fixture statuses (scheduled but past match time)"
    )
    _ = parser.add_argument(
        "--hours-passed",
        type=int,
        default=3,
        help="Hours after match time to consider fixture stale (default: 3)",
    )
    _ = parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum fixtures to report (default: 500)",
    )
    ns = parser.parse_args()
    return Options(
        hours_passed=int(ns.hours_passed),
        limit=int(ns.limit),
    )


def get_stale_fixtures(
    hours_passed: int, limit: int
) -> list[tuple[int, str | None, str | None, str, str]]:
    """Get stale fixtures with Sofa/Flash IDs using v2 DB function (legacy fallback)."""
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            rows: list[tuple[Any, ...]]
            try:
                cur.execute(
                    """
                    SELECT fixture_id, sofascore_id, flashscore_id, league_code, match_datetime_utc
                    FROM get_stale_fixtures_v2(%s)
                    LIMIT %s
                    """,
                    (hours_passed, limit),
                )
                rows = cur.fetchall()
            except Exception:
                cur.execute(
                    "SELECT fixture_id, flashscore_id, league_code, match_datetime_utc FROM get_stale_fixtures(%s) LIMIT %s",
                    (hours_passed, limit),
                )
                legacy_rows = cur.fetchall()
                rows = [
                    (row[0], None, row[1], row[2], row[3]) for row in legacy_rows
                ]
    finally:
        conn.close()

    results: list[tuple[int, str | None, str | None, str, str]] = []
    for row in rows:
        if not isinstance(row[0], int) or not isinstance(row[3], str):
            continue
        sofa_id = row[1].strip() if isinstance(row[1], str) and row[1].strip() else None
        flash_id = row[2].strip() if isinstance(row[2], str) and row[2].strip() else None
        dt_str = row[4].isoformat() if row[4] else "unknown"
        results.append((row[0], sofa_id, flash_id, row[3], dt_str))
    return results


def main() -> int:
    options = parse_args()

    print("=" * 60)
    print("Housekeeping: Report Stale Fixture Statuses")
    print("=" * 60)

    print(
        f"Finding fixtures scheduled but past match time by {options.hours_passed}h..."
    )

    stale = get_stale_fixtures(options.hours_passed, options.limit)
    print(f"Found {len(stale)} stale fixtures")

    if not stale:
        print("No stale fixtures found. Done.")
        return 0

    # Group by league for summary
    by_league: dict[str, int] = defaultdict(int)
    for fixture_id, sofa_id, fs_id, league, dt in stale:
        by_league[league] += 1

    print("\nBy league:")
    for league, count in sorted(by_league.items(), key=lambda x: -x[1])[:15]:
        print(f"  {league}: {count}")

    print("\nSample fixtures (first 10):")
    for fixture_id, sofa_id, fs_id, league, dt in stale[:10]:
        id_parts = []
        if sofa_id:
            id_parts.append(f"sofa={sofa_id}")
        if fs_id:
            id_parts.append(f"flash={fs_id}")
        if not id_parts:
            id_parts.append("no_external_id")
        print(f"  {league}: fixture={fixture_id} ({', '.join(id_parts)}) (match_time: {dt})")

    print(f"\nTotal: {len(stale)} stale fixtures (tick job will retry settlement)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
