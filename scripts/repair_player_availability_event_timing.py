from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair player_availability.event_recorded_at for PIT-safe historical timing."
    )
    parser.add_argument(
        "--status",
        type=str,
        default="ft",
        choices=["ft", "scheduled", "all"],
        help="Fixture status scope to repair (default: ft).",
    )
    parser.add_argument(
        "--anchor-minutes",
        type=int,
        default=60,
        help="Clamp target to kickoff minus this many minutes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report impact without writing updates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/layer2_reconciliation"),
        help="Directory for repair report artifacts.",
    )
    args = parser.parse_args()
    if args.anchor_minutes < 0:
        raise ValueError("--anchor-minutes must be >= 0")
    return args


def _table_has_column(cur, table_name: str, column_name: str) -> bool:
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
    return bool(row and row[0])


def _status_clause(status: str) -> tuple[str, tuple[object, ...]]:
    if status == "all":
        return "", ()
    return "AND f.status = %s", (status,)


def collect_stats(conn, *, status: str) -> dict[str, int]:
    status_sql, params = _status_clause(status)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                COUNT(*) AS rows_total,
                COUNT(*) FILTER (WHERE pa.event_recorded_at IS NULL) AS rows_null_timing,
                COUNT(*) FILTER (WHERE pa.event_recorded_at > f.match_datetime_utc) AS rows_after_kickoff,
                COUNT(DISTINCT pa.fixture_id) AS fixtures_with_rows,
                COUNT(DISTINCT pa.fixture_id) FILTER (
                    WHERE pa.event_recorded_at IS NULL OR pa.event_recorded_at > f.match_datetime_utc
                ) AS fixtures_with_bad_timing
            FROM player_availability pa
            JOIN fixtures f ON f.fixture_id = pa.fixture_id
            WHERE f.match_datetime_utc IS NOT NULL
              {status_sql}
            """,
            params,
        )
        row = cur.fetchone()
    return {
        "rows_total": int(row[0] or 0),
        "rows_null_timing": int(row[1] or 0),
        "rows_after_kickoff": int(row[2] or 0),
        "fixtures_with_rows": int(row[3] or 0),
        "fixtures_with_bad_timing": int(row[4] or 0),
    }


def run_repair(
    conn,
    *,
    status: str,
    anchor_minutes: int,
) -> int:
    status_sql, params = _status_clause(status)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE player_availability pa
            SET event_recorded_at = f.match_datetime_utc - make_interval(mins => %s::int)
            FROM fixtures f
            WHERE pa.fixture_id = f.fixture_id
              AND f.match_datetime_utc IS NOT NULL
              {status_sql}
              AND (pa.event_recorded_at IS NULL OR pa.event_recorded_at > f.match_datetime_utc)
            """,
            (anchor_minutes, *params),
        )
        updated = int(cur.rowcount or 0)
    conn.commit()
    return updated


def main() -> None:
    args = parse_args()
    started_at = datetime.now(UTC)

    conn = connect_db()
    try:
        with conn.cursor() as cur:
            if not _table_has_column(cur, "player_availability", "event_recorded_at"):
                raise RuntimeError(
                    "player_availability.event_recorded_at is missing. Apply lineage migration first."
                )

        before = collect_stats(conn, status=args.status)
        updated_rows = 0
        if not args.dry_run:
            updated_rows = run_repair(
                conn,
                status=args.status,
                anchor_minutes=args.anchor_minutes,
            )
        after = collect_stats(conn, status=args.status)
    finally:
        conn.close()

    generated_at = datetime.now(UTC)
    report = {
        "started_at": started_at.isoformat(),
        "generated_at": generated_at.isoformat(),
        "scope": {
            "status": args.status,
            "anchor_minutes": args.anchor_minutes,
            "dry_run": bool(args.dry_run),
        },
        "before": before,
        "repair": {
            "updated_rows": updated_rows,
        },
        "after": after,
    }

    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"player_availability_event_timing_repair_{stamp}.json"
    md_path = args.output_dir / f"player_availability_event_timing_repair_{stamp}.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Player Availability Event Timing Repair")
    lines.append("")
    lines.append(f"Generated: {generated_at.isoformat()}")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- status: {args.status}")
    lines.append(f"- anchor_minutes: {args.anchor_minutes}")
    lines.append(f"- dry_run: {str(args.dry_run).lower()}")
    lines.append("")
    lines.append("| Metric | Before | After |")
    lines.append("| --- | ---: | ---: |")
    for key in (
        "rows_total",
        "rows_null_timing",
        "rows_after_kickoff",
        "fixtures_with_rows",
        "fixtures_with_bad_timing",
    ):
        lines.append(f"| {key} | {before.get(key, 0)} | {after.get(key, 0)} |")
    lines.append("")
    lines.append(f"- updated_rows: {updated_rows}")
    lines.append(f"- json: `{json_path}`")
    lines.append(f"- markdown: `{md_path}`")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()

