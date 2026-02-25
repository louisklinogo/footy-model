from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportUnusedCallResult=false, reportImplicitStringConcatenation=false


REPORT_PATH = Path(".sisyphus/evidence/task-7-leakage-audit.json")
MAX_EXAMPLES = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit potential Layer 2 upstream data leakage vectors."
    )
    parser.add_argument("--league", type=str, default=None, help="Optional league code")
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Scheduled-fixture horizon in days (default: 14)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max fixtures to inspect per mode (default: 500)",
    )
    args = parser.parse_args()
    if args.days <= 0:
        raise ValueError("--days must be > 0")
    if args.limit <= 0:
        raise ValueError("--limit must be > 0")
    return args


def _table_exists(cur, table_name: str) -> bool:
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
        )
        """,
        (table_name,),
    )
    row = cur.fetchone()
    return bool(row[0]) if row else False


def _column_exists(cur, table_name: str, column_name: str) -> bool:
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
              AND column_name = %s
        )
        """,
        (table_name, column_name),
    )
    row = cur.fetchone()
    return bool(row[0]) if row else False


def _fetch_all_dicts(cur) -> list[dict[str, Any]]:
    cols = [desc[0] for desc in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def run_availability_check(
    cur, days: int, limit: int, league: str | None
) -> dict[str, Any]:
    check: dict[str, Any] = {
        "status": "pass",
        "skipped": False,
        "reason": None,
        "violation_count": 0,
        "fixture_sample_count": 0,
        "examples": [],
    }

    if not _table_exists(cur, "player_availability"):
        check["status"] = "skipped"
        check["skipped"] = True
        check["reason"] = "table player_availability missing"
        return check

    if not _column_exists(cur, "player_availability", "recorded_at"):
        check["status"] = "failed"
        check["reason"] = "column player_availability.recorded_at missing"
        return check

    params: list[Any] = [days]
    league_sql = ""
    if league:
        league_sql = " AND f.league_code = %s"
        params.append(league)
    params.append(limit)

    cur.execute(
        f"""
        WITH target_fixtures AS (
            SELECT f.fixture_id, f.match_datetime_utc
            FROM fixtures f
            WHERE f.status = 'scheduled'
              AND f.match_datetime_utc <= NOW() + make_interval(days => %s)
              AND f.match_datetime_utc > NOW() - interval '6 hours'
              {league_sql}
            ORDER BY f.match_datetime_utc ASC
            LIMIT %s
        )
        SELECT COUNT(*)
        FROM player_availability pa
        JOIN target_fixtures tf ON tf.fixture_id = pa.fixture_id
        WHERE pa.recorded_at > tf.match_datetime_utc
        """,
        tuple(params),
    )
    count_row = cur.fetchone()
    violation_count = int(count_row[0]) if count_row else 0
    check["violation_count"] = violation_count

    cur.execute(
        f"""
        WITH target_fixtures AS (
            SELECT f.fixture_id, f.match_datetime_utc
            FROM fixtures f
            WHERE f.status = 'scheduled'
              AND f.match_datetime_utc <= NOW() + make_interval(days => %s)
              AND f.match_datetime_utc > NOW() - interval '6 hours'
              {league_sql}
            ORDER BY f.match_datetime_utc ASC
            LIMIT %s
        )
        SELECT COUNT(*)
        FROM target_fixtures
        """,
        tuple(params),
    )
    fixture_count_row = cur.fetchone()
    check["fixture_sample_count"] = (
        int(fixture_count_row[0]) if fixture_count_row else 0
    )

    if violation_count > 0:
        check["status"] = "failed"
        cur.execute(
            f"""
            WITH target_fixtures AS (
                SELECT f.fixture_id, f.match_datetime_utc
                FROM fixtures f
                WHERE f.status = 'scheduled'
                  AND f.match_datetime_utc <= NOW() + make_interval(days => %s)
                  AND f.match_datetime_utc > NOW() - interval '6 hours'
                  {league_sql}
                ORDER BY f.match_datetime_utc ASC
                LIMIT %s
            )
            SELECT
                pa.fixture_id,
                pa.player_id,
                pa.team_id,
                pa.status,
                pa.recorded_at,
                tf.match_datetime_utc
            FROM player_availability pa
            JOIN target_fixtures tf ON tf.fixture_id = pa.fixture_id
            WHERE pa.recorded_at > tf.match_datetime_utc
            ORDER BY pa.recorded_at DESC
            LIMIT %s
            """,
            tuple([*params, MAX_EXAMPLES]),
        )
        check["examples"] = _fetch_all_dicts(cur)

    return check


def run_snapshot_sample_size_check(
    cur,
    limit: int,
    league: str | None,
) -> tuple[dict[str, Any], list[int]]:
    check: dict[str, Any] = {
        "status": "pass",
        "skipped": False,
        "reason": None,
        "fixture_sample_count": 0,
        "team_rows_with_snapshot": 0,
        "missing_snapshot_rows": 0,
        "violation_count": 0,
        "examples": [],
    }

    if not _table_exists(cur, "team_premium_snapshots"):
        check["status"] = "failed"
        check["reason"] = "table team_premium_snapshots missing"
        return check, []

    if not _column_exists(cur, "team_premium_snapshots", "sample_size"):
        check["status"] = "failed"
        check["reason"] = "column team_premium_snapshots.sample_size missing"
        return check, []

    fixture_params: list[Any] = []
    league_sql = ""
    if league:
        league_sql = " AND f.league_code = %s"
        fixture_params.append(league)
    fixture_params.append(limit)

    cur.execute(
        f"""
        SELECT f.fixture_id
        FROM fixtures f
        WHERE f.status = 'ft'
          AND f.match_datetime_utc IS NOT NULL
          {league_sql}
        ORDER BY f.match_datetime_utc DESC
        LIMIT %s
        """,
        tuple(fixture_params),
    )
    sampled_fixtures = [int(row[0]) for row in cur.fetchall()]
    check["fixture_sample_count"] = len(sampled_fixtures)

    if not sampled_fixtures:
        return check, []

    cur.execute(
        """
        WITH sampled AS (
            SELECT f.fixture_id, f.league_code, f.match_datetime_utc, f.home_team_id, f.away_team_id
            FROM fixtures f
            WHERE f.fixture_id = ANY(%s)
        ),
        team_rows AS (
            SELECT s.fixture_id, s.league_code, s.match_datetime_utc, t.team_id, t.is_home
            FROM sampled s
            CROSS JOIN LATERAL (
                VALUES (s.home_team_id, TRUE), (s.away_team_id, FALSE)
            ) AS t(team_id, is_home)
        ),
        enriched AS (
            SELECT
                tr.fixture_id,
                tr.team_id,
                tr.is_home,
                tr.match_datetime_utc,
                tps.sample_size,
                (
                    SELECT COUNT(*)
                    FROM fixtures pf
                    JOIN fixture_stats_premium fsp ON fsp.fixture_id = pf.fixture_id
                    WHERE pf.status = 'ft'
                      AND pf.league_code = tr.league_code
                      AND pf.match_datetime_utc < tr.match_datetime_utc
                      AND (pf.home_team_id = tr.team_id OR pf.away_team_id = tr.team_id)
                ) AS prior_premium_games
            FROM team_rows tr
            LEFT JOIN team_premium_snapshots tps
              ON tps.fixture_id = tr.fixture_id
             AND tps.team_id = tr.team_id
             AND tps.is_home = tr.is_home
        )
        SELECT
            COUNT(*) FILTER (WHERE sample_size IS NOT NULL),
            COUNT(*) FILTER (WHERE sample_size IS NULL),
            COUNT(*) FILTER (WHERE sample_size IS NOT NULL AND sample_size > prior_premium_games)
        FROM enriched
        """,
        (sampled_fixtures,),
    )
    agg_row = cur.fetchone()
    if agg_row:
        check["team_rows_with_snapshot"] = int(agg_row[0])
        check["missing_snapshot_rows"] = int(agg_row[1])
        check["violation_count"] = int(agg_row[2])

    if check["violation_count"] > 0:
        check["status"] = "failed"
        cur.execute(
            """
            WITH sampled AS (
                SELECT f.fixture_id, f.league_code, f.match_datetime_utc, f.home_team_id, f.away_team_id
                FROM fixtures f
                WHERE f.fixture_id = ANY(%s)
            ),
            team_rows AS (
                SELECT s.fixture_id, s.league_code, s.match_datetime_utc, t.team_id, t.is_home
                FROM sampled s
                CROSS JOIN LATERAL (
                    VALUES (s.home_team_id, TRUE), (s.away_team_id, FALSE)
                ) AS t(team_id, is_home)
            ),
            enriched AS (
                SELECT
                    tr.fixture_id,
                    tr.team_id,
                    tr.match_datetime_utc,
                    tps.sample_size,
                    (
                        SELECT COUNT(*)
                        FROM fixtures pf
                        JOIN fixture_stats_premium fsp ON fsp.fixture_id = pf.fixture_id
                        WHERE pf.status = 'ft'
                          AND pf.league_code = tr.league_code
                          AND pf.match_datetime_utc < tr.match_datetime_utc
                          AND (pf.home_team_id = tr.team_id OR pf.away_team_id = tr.team_id)
                    ) AS prior_premium_games
                FROM team_rows tr
                LEFT JOIN team_premium_snapshots tps
                  ON tps.fixture_id = tr.fixture_id
                 AND tps.team_id = tr.team_id
                 AND tps.is_home = tr.is_home
            )
            SELECT
                fixture_id,
                team_id,
                sample_size,
                prior_premium_games,
                match_datetime_utc AS kickoff
            FROM enriched
            WHERE sample_size IS NOT NULL
              AND sample_size > prior_premium_games
            ORDER BY kickoff DESC
            LIMIT %s
            """,
            (sampled_fixtures, MAX_EXAMPLES),
        )
        check["examples"] = _fetch_all_dicts(cur)

    return check, sampled_fixtures


def run_lambda_timing_check(cur, sampled_fixtures: list[int]) -> dict[str, Any]:
    check: dict[str, Any] = {
        "status": "info",
        "skipped": False,
        "reason": None,
        "violation_count": 0,
        "examples": [],
    }

    if not sampled_fixtures:
        check["skipped"] = True
        check["reason"] = "no sampled FT fixtures"
        return check

    cur.execute(
        """
        SELECT COUNT(*)
        FROM predictions p
        JOIN fixtures f ON f.fixture_id = p.fixture_id
        WHERE p.model_name = 'lambda_xgb'
          AND p.fixture_id = ANY(%s)
          AND p.created_at > f.match_datetime_utc
        """,
        (sampled_fixtures,),
    )
    row = cur.fetchone()
    check["violation_count"] = int(row[0]) if row else 0

    if check["violation_count"] > 0:
        cur.execute(
            """
            SELECT
                p.fixture_id,
                p.prediction_id,
                p.market_code,
                p.model_name,
                p.created_at,
                f.match_datetime_utc AS kickoff
            FROM predictions p
            JOIN fixtures f ON f.fixture_id = p.fixture_id
            WHERE p.model_name = 'lambda_xgb'
              AND p.fixture_id = ANY(%s)
              AND p.created_at > f.match_datetime_utc
            ORDER BY p.created_at DESC
            LIMIT %s
            """,
            (sampled_fixtures, MAX_EXAMPLES),
        )
        check["examples"] = _fetch_all_dicts(cur)

    return check


def write_report(report: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")


def print_summary(report: dict[str, Any]) -> None:
    availability = report["availability_check"]
    snapshots = report["snapshot_sample_size_check"]
    lambda_info = report["lambda_timing_check"]

    print("Layer2 Leakage Audit")
    print(f"report_path={REPORT_PATH}")
    print(
        "availability_check="
        f"{availability['status']} "
        f"violations={availability.get('violation_count', 0)} "
        f"skipped={availability.get('skipped', False)}"
    )
    print(
        "snapshot_sample_size_check="
        f"{snapshots['status']} "
        f"violations={snapshots.get('violation_count', 0)}"
    )
    print(
        "lambda_timing_check="
        f"{lambda_info['status']} "
        f"late_rows={lambda_info.get('violation_count', 0)}"
    )


def main() -> int:
    args = parse_args()
    started_at = datetime.now(timezone.utc).isoformat()

    report: dict[str, Any] = {
        "started_at": started_at,
        "args": {
            "league": args.league,
            "days": args.days,
            "limit": args.limit,
        },
    }

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            availability_check = run_availability_check(
                cur=cur,
                days=args.days,
                limit=args.limit,
                league=args.league,
            )
            snapshot_check, sampled_fixtures = run_snapshot_sample_size_check(
                cur=cur,
                limit=args.limit,
                league=args.league,
            )
            lambda_check = run_lambda_timing_check(
                cur=cur, sampled_fixtures=sampled_fixtures
            )
    finally:
        conn.close()

    report["availability_check"] = availability_check
    report["snapshot_sample_size_check"] = snapshot_check
    report["lambda_timing_check"] = lambda_check

    critical_failed = False
    for key in ("availability_check", "snapshot_sample_size_check"):
        status = report[key].get("status")
        skipped = bool(report[key].get("skipped"))
        if status == "failed":
            critical_failed = True
        if status == "skipped" and not skipped:
            critical_failed = True

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["overall_status"] = "failed" if critical_failed else "pass"

    write_report(report)
    print_summary(report)

    return 1 if critical_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
