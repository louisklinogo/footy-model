from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


def main() -> None:
    parser = argparse.ArgumentParser(description="Report player availability coverage by league")
    parser.add_argument("--season-start", type=int, required=True, help="Start year (YYYY)")
    parser.add_argument("--season-end", type=int, required=True, help="End year (YYYY)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/coverage"),
        help="Output directory for coverage report",
    )
    args = parser.parse_args()

    conn = connect_db()
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH base AS (
                SELECT fixture_id, league_code
                FROM fixtures
                WHERE status = 'ft'
                  AND EXTRACT(YEAR FROM match_datetime_utc) BETWEEN %s AND %s
            ),
            availability AS (
                SELECT DISTINCT fixture_id FROM player_availability
            ),
            player_stats AS (
                SELECT DISTINCT fixture_id FROM fixture_player_stats
            )
            SELECT
                b.league_code,
                COUNT(*) AS fixtures_total,
                COUNT(a.fixture_id) AS fixtures_with_availability,
                COUNT(ps.fixture_id) AS fixtures_with_player_stats
            FROM base b
            LEFT JOIN availability a ON a.fixture_id = b.fixture_id
            LEFT JOIN player_stats ps ON ps.fixture_id = b.fixture_id
            GROUP BY b.league_code
            ORDER BY b.league_code
            """,
            (args.season_start, args.season_end),
        )
        rows = cur.fetchall()

    conn.close()

    report_rows = []
    for league_code, total, with_avail, with_stats in rows:
        total = int(total)
        with_avail = int(with_avail)
        with_stats = int(with_stats)
        report_rows.append(
            {
                "league_code": league_code,
                "fixtures_total": total,
                "fixtures_with_availability": with_avail,
                "availability_pct": (with_avail / total) if total else 0.0,
                "fixtures_with_player_stats": with_stats,
                "player_stats_pct": (with_stats / total) if total else 0.0,
            }
        )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "season_start": args.season_start,
        "season_end": args.season_end,
        "leagues": report_rows,
        "playbook": [
            "python src/ingest/ingest_sofascore_availability.py --status ft --league <CODE> --limit <N>",
            "python src/db/audit_layer2_situational_leakage.py --limit 50",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "player_availability_coverage.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
