from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


def get_db_metrics() -> dict[str, object]:
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    current_database(),
                    current_user,
                    COALESCE(inet_server_addr()::text, 'local_socket'),
                    inet_server_port()
                """
            )
            db_name, db_user, db_host, db_port = cur.fetchone()

            target_sql = """
                SELECT fixture_id
                FROM fixtures
                WHERE status = 'ft' AND sofascore_id IS NOT NULL
            """

            cur.execute(f"SELECT COUNT(*) FROM ({target_sql}) t")
            total_target = cur.fetchone()[0]

            cur.execute(
                f"""
                SELECT
                    COUNT(s.fixture_id) as presence,
                    COUNT(s.fixture_id) FILTER (WHERE s.fidelity_score > 0) as attempted,
                    COUNT(s.fixture_id) FILTER (WHERE s.fidelity_score >= 1.0) as yielded
                FROM ({target_sql}) t
                LEFT JOIN fixture_stats_premium s ON t.fixture_id = s.fixture_id
                """
            )
            s_row = cur.fetchone()
            stats_presence = s_row[0]
            stats_attempted = s_row[1]
            stats_yielded = s_row[2]

            cur.execute(
                f"""
                SELECT COUNT(DISTINCT s.fixture_id)
                FROM ({target_sql}) t
                JOIN fixture_player_stats s ON t.fixture_id = s.fixture_id
                """
            )
            players_done = cur.fetchone()[0]

            cur.execute(
                f"""
                SELECT COUNT(DISTINCT a.fixture_id)
                FROM ({target_sql}) t
                JOIN player_availability a ON t.fixture_id = a.fixture_id
                """
            )
            availability_done = cur.fetchone()[0]

            cur.execute(
                f"""
                SELECT COUNT(DISTINCT m.fixture_id)
                FROM ({target_sql}) t
                JOIN fixture_odds_markets m ON t.fixture_id = m.fixture_id
                """
            )
            odds_done = cur.fetchone()[0]

            cur.execute("SELECT to_regclass('public.fixture_incidents_sofascore')")
            incidents_table = cur.fetchone()[0]
            if incidents_table:
                cur.execute(
                    f"""
                    SELECT COUNT(DISTINCT i.fixture_id)
                    FROM ({target_sql}) t
                    JOIN fixture_incidents_sofascore i ON t.fixture_id = i.fixture_id
                    """
                )
                incidents_done = cur.fetchone()[0]
            else:
                incidents_done = 0

            cur.execute(
                """
                SELECT
                    status,
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE sofascore_id IS NOT NULL) as linked
                FROM fixtures
                GROUP BY status
                """
            )
            dist_rows = cur.fetchall()
            dist_cols = [desc[0] for desc in cur.description]
            distribution = [dict(zip(dist_cols, row)) for row in dist_rows]

            return {
                "db_name": db_name,
                "db_user": db_user,
                "db_host": db_host,
                "db_port": db_port,
                "total_target": total_target,
                "stats_presence": stats_presence,
                "stats_attempted": stats_attempted,
                "stats_yielded": stats_yielded,
                "players_done": players_done,
                "availability_done": availability_done,
                "odds_done": odds_done,
                "incidents_done": incidents_done,
                "distribution": distribution,
            }
    finally:
        conn.close()


def generate_report() -> str:
    metrics = get_db_metrics()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def row(label: str, done: int, total: int) -> str:
        rem = max(0, total - done)
        pct = (done / total) * 100 if total > 0 else 0
        status = "done" if done >= total else "in_progress"
        return f"| **{label}** | {done} | {rem} | {total} | {pct:.1f}% | {status} |"

    dist_table = "| Status | Total | Linked | Unlinked | Linked % |\n| :--- | :--- | :--- | :--- | :--- |\n"
    total_db = 0
    for dist in metrics["distribution"]:
        total = int(dist["total"])
        linked = int(dist["linked"])
        unlinked = total - linked
        pct = (linked / total) * 100 if total > 0 else 0
        dist_table += f"| {dist['status']} | {total} | {linked} | {unlinked} | {pct:.1f}% |\n"
        total_db += total

    report = f"""# Data Backfill Progress Report
Generated: {now}
Database Target: **{metrics['db_name']}** (`{metrics['db_host']}:{metrics['db_port']}` as `{metrics['db_user']}`)

Target Universe: **{metrics['total_target']}** (Finished matches with Sofascore IDs)

## The Big Picture (Database Mapping)
We have **{total_db}** total fixtures in the database.

{dist_table}

## Coverage Summary (Target Universe)

| Category | Done | Remaining | Total | Progress | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
{row('Match Stats (Presence)', int(metrics['stats_presence']), int(metrics['total_target']))}
{row('H1/H2 Update (Attempted)', int(metrics['stats_attempted']), int(metrics['total_target']))}
{row('H1/H2 Data (Yielded)', int(metrics['stats_yielded']), int(metrics['total_target']))}
{row('Player Stats', int(metrics['players_done']), int(metrics['total_target']))}
{row('Availability (Injuries)', int(metrics['availability_done']), int(metrics['total_target']))}
{row('Incidents (Timeline)', int(metrics['incidents_done']), int(metrics['total_target']))}
{row('Match Odds (Backfill)', int(metrics['odds_done']), int(metrics['total_target']))}

## Active Backfill Scripts

To continue the backfill, use these commands in the project root:

- **Match Stats (Upgrade/Basic)**:
  `python scripts/backfill_batch_sofascore.py --type stats --total 1000 --limit 100`

- **Player Stats**:
  `python scripts/backfill_batch_sofascore.py --type players --total 1000 --limit 100`

- **Availability / Injuries**:
  `python scripts/backfill_batch_sofascore.py --type availability --total 1000 --limit 100 --status ft`

- **Incidents (Timeline)**:
  `python scripts/backfill_batch_sofascore.py --type incidents --total 1000 --limit 100 --status ft`

- **Match Odds**:
  `python src/ingest/backfill_sofascore_odds_markets_v1.py --limit 100`
"""
    return report


if __name__ == "__main__":
    print("Fetching DB metrics...")
    report_md = generate_report()
    print("\n" + report_md)

    docs_path = Path("docs/backfill_report.md")
    docs_path.parent.mkdir(exist_ok=True)
    with docs_path.open("w", encoding="utf-8") as handle:
        handle.write(report_md)
    print(f"\nReport saved to {docs_path.absolute()}")
