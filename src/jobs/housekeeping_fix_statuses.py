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
) -> list[tuple[int, str, str, str]]:
    """Get fixtures that are scheduled but past match time using DB function."""
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT fixture_id, flashscore_id, league_code, match_datetime_utc FROM get_stale_fixtures(%s) LIMIT %s",
                (hours_passed, limit),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    results: list[tuple[int, str, str, str]] = []
    for row in rows:
        if (
            isinstance(row[0], int)
            and isinstance(row[1], str)
            and isinstance(row[2], str)
        ):
            dt_str = row[3].isoformat() if row[3] else "unknown"
            results.append((row[0], row[1], row[2], dt_str))
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
    for fixture_id, fs_id, league, dt in stale:
        by_league[league] += 1

    print("\nBy league:")
    for league, count in sorted(by_league.items(), key=lambda x: -x[1])[:15]:
        print(f"  {league}: {count}")

    print("\nSample fixtures (first 10):")
    for fixture_id, fs_id, league, dt in stale[:10]:
        print(f"  {league}: {fs_id} (match_time: {dt})")

    print(f"\nTotal: {len(stale)} stale fixtures (tick job will retry settlement)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
