"""Housekeeping job: Settle FT fixtures missing results.

Finds fixtures marked as 'ft' (full-time) but without result records
and attempts to settle them from available premium data.
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db.db_utils import connect_db

PREMIUM_ROOT = ROOT / "data" / "v1" / "premium"


@dataclass(frozen=True)
class Options:
    dry_run: bool
    limit: int


def parse_args() -> Options:
    parser = argparse.ArgumentParser(
        description="Settle FT fixtures missing result records"
    )
    _ = parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum fixtures to process (default: 100)",
    )
    _ = parser.add_argument(
        "--dry-run", action="store_true", help="Show what would change"
    )
    ns = parser.parse_args()
    return Options(
        dry_run=bool(ns.dry_run),
        limit=int(ns.limit),
    )


def get_ft_without_results(
    limit: int,
) -> list[tuple[int, str | None, str | None, str, str]]:
    """Get FT fixtures without result records using v2 DB function (legacy fallback)."""
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            rows: list[tuple[Any, ...]]
            try:
                cur.execute(
                    """
                    SELECT fixture_id, sofascore_id, flashscore_id, league_code, match_datetime_utc
                    FROM get_ft_without_results_v2()
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
            except Exception:
                cur.execute(
                    "SELECT fixture_id, flashscore_id, league_code, match_datetime_utc FROM get_ft_without_results() LIMIT %s",
                    (limit,),
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


def find_premium_json(league: str, flashscore_id: str | None) -> Path | None:
    """Find the premium JSON file for a fixture."""
    if not flashscore_id:
        return None
    json_path = PREMIUM_ROOT / league / f"{flashscore_id}.json"
    if json_path.exists():
        return json_path
    return None


def extract_result_from_premium(json_path: Path) -> tuple[int, int] | None:
    """Extract home_goals, away_goals from premium JSON."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        # Premium JSON structure varies - check common locations
        if "result" in data:
            result = data["result"]
            if isinstance(result, dict):
                home = result.get("home_goals") or result.get("home")
                away = result.get("away_goals") or result.get("away")
                if isinstance(home, int) and isinstance(away, int):
                    return (home, away)
        # Check for score field
        if "score" in data:
            score = data["score"]
            if isinstance(score, dict):
                home = score.get("home") or score.get("fulltime", {}).get("home")
                away = score.get("away") or score.get("fulltime", {}).get("away")
                if isinstance(home, int) and isinstance(away, int):
                    return (home, away)
        # Check for final_score
        if "final_score" in data:
            score = data["final_score"]
            if isinstance(score, dict):
                home = score.get("home")
                away = score.get("away")
                if isinstance(home, int) and isinstance(away, int):
                    return (home, away)
        return None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def settle_fixture_record(
    fixture_id: int,
    flashscore_id: str | None,
    home_goals: int,
    away_goals: int,
    dry_run: bool,
) -> bool:
    """Settle a fixture using fixture_id helper (legacy flashscore fallback)."""
    if dry_run:
        ident = f"fixture={fixture_id}"
        if flashscore_id:
            ident += f", flash={flashscore_id}"
        print(f"  DRY RUN: Would settle {ident} ({home_goals}-{away_goals})")
        return True

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
                    if not flashscore_id:
                        raise
                    cur.execute(
                        "SELECT settle_fixture(%s, %s, %s)",
                        (flashscore_id, home_goals, away_goals),
                    )
                # This job settles from Flashscore premium JSON snapshots.
                try:
                    cur.execute(
                        """
                        UPDATE fixture_results
                        SET result_source = 'flashscore',
                            settled_at = NOW()
                        WHERE fixture_id = %s
                        """,
                        (fixture_id,),
                    )
                except Exception:
                    # Backward compatibility: older schemas may not have result_source.
                    pass
        return True
    except Exception as e:
        print(f"  Error settling fixture={fixture_id}: {e}")
        return False
    finally:
        conn.close()


def main() -> int:
    options = parse_args()

    print("=" * 60)
    print("Housekeeping: Settle FT Fixtures Missing Results")
    print("=" * 60)

    print("Finding FT fixtures without result records...")

    missing_results = get_ft_without_results(options.limit)
    print(f"Found {len(missing_results)} FT fixtures without results")

    if not missing_results:
        print("All FT fixtures have result records. Done.")
        return 0

    settled = 0
    not_found = 0
    no_score = 0

    for fixture_id, sofa_id, flashscore_id, league, dt in missing_results:
        json_path = find_premium_json(league, flashscore_id)
        if json_path is None:
            not_found += 1
            continue

        result = extract_result_from_premium(json_path)
        if result is None:
            no_score += 1
            print(f"  No score found in: {json_path}")
            continue

        home_goals, away_goals = result
        if settle_fixture_record(
            fixture_id,
            flashscore_id,
            home_goals,
            away_goals,
            options.dry_run,
        ):
            settled += 1

    print(f"\nSummary:")
    print(f"  Total FT without results: {len(missing_results)}")
    print(f"  Premium JSON not found: {not_found}")
    print(f"  No score in JSON: {no_score}")
    print(f"  {'Would settle' if options.dry_run else 'Settled'}: {settled}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
