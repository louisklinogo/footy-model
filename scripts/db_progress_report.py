import psycopg2
import sys
from pathlib import Path
from datetime import datetime

DB_URL = "postgresql://neondb_owner:npg_csxeyQ6fNXF7@ep-young-tree-ab8emfpl-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require"

def get_db_metrics():
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            # Consistent target definition
            target_sql = """
                SELECT fixture_id 
                FROM fixtures 
                WHERE status = 'ft' AND sofascore_id IS NOT NULL
            """
            
            # 1. Total Target
            cur.execute(f"SELECT COUNT(*) FROM ({target_sql}) t")
            total_target = cur.fetchone()[0]

            # 2. Stats Progress (Strict)
            cur.execute(f"""
                SELECT 
                    COUNT(s.fixture_id) as presence,
                    COUNT(s.fixture_id) FILTER (WHERE s.fidelity_score > 0) as attempted,
                    COUNT(s.fixture_id) FILTER (WHERE s.fidelity_score >= 1.0) as yielded
                FROM ({target_sql}) t
                LEFT JOIN fixture_stats_premium s ON t.fixture_id = s.fixture_id
            """)
            s_row = cur.fetchone()
            stats_presence = s_row[0]
            stats_attempted = s_row[1]
            stats_yielded = s_row[2]

            # 3. Player Stats Progress (Strict)
            cur.execute(f"""
                SELECT COUNT(DISTINCT s.fixture_id) 
                FROM ({target_sql}) t
                JOIN fixture_player_stats s ON t.fixture_id = s.fixture_id
            """)
            players_done = cur.fetchone()[0]

            # 4. Availability Progress (Strict)
            cur.execute(f"""
                SELECT COUNT(DISTINCT a.fixture_id) 
                FROM ({target_sql}) t
                JOIN player_availability a ON t.fixture_id = a.fixture_id
            """)
            availability_done = cur.fetchone()[0]

            # 5. Odds Data (Backfill)
            cur.execute(f"""
                SELECT COUNT(DISTINCT m.fixture_id) 
                FROM ({target_sql}) t
                JOIN fixture_odds_markets m ON t.fixture_id = m.fixture_id
            """)
            odds_done = cur.fetchone()[0]

            # 6. Global Distribution Metrics
            cur.execute("""
                SELECT 
                    status,
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE sofascore_id IS NOT NULL) as linked
                FROM fixtures
                GROUP BY status
            """)
            dist_rows = cur.fetchall()
            dist_cols = [desc[0] for desc in cur.description]
            distribution = [dict(zip(dist_cols, r)) for r in dist_rows]

            return {
                "total_target": total_target,
                "stats_presence": stats_presence,
                "stats_attempted": stats_attempted,
                "stats_yielded": stats_yielded,
                "players_done": players_done,
                "availability_done": availability_done,
                "odds_done": odds_done,
                "distribution": distribution
            }
    finally:
        conn.close()

def generate_report():
    m = get_db_metrics()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def row(label, done, total):
        rem = max(0, total - done)
        pct = (done / total) * 100 if total > 0 else 0
        status = '✅' if done >= total else '🔄'
        return f"| **{label}** | {done} | {rem} | {total} | {pct:.1f}% | {status} |"

    dist_table = "| Status | Total | Linked | Unlinked | Linked % |\n| :--- | :--- | :--- | :--- | :--- |\n"
    total_db = 0
    for d in m['distribution']:
        unlinked = d['total'] - d['linked']
        pct = (d['linked'] / d['total']) * 100 if d['total'] > 0 else 0
        dist_table += f"| {d['status']} | {d['total']} | {d['linked']} | {unlinked} | {pct:.1f}% |\n"
        total_db += d['total']

    report = f"""# Data Backfill Progress Report
Generated: {now}

Target Universe: **{m['total_target']}** (Finished matches with Sofascore IDs)

## 🌎 The Big Picture (Database Mapping)
We have **{total_db}** total fixtures in the database.

{dist_table}

## 📊 Coverage Summary (Target Universe)

| Category | Done | Remaining | Total | Progress | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
{row('Match Stats (Presence)', m['stats_presence'], m['total_target'])}
{row('H1/H2 Update (Attempted)', m['stats_attempted'], m['total_target'])}
{row('H1/H2 Data (Yielded)', m['stats_yielded'], m['total_target'])}
{row('Player Stats', m['players_done'], m['total_target'])}
{row('Availability (Injuries)', m['availability_done'], m['total_target'])}
{row('Match Odds (Backfill)', m['odds_done'], m['total_target'])}

## 🛠️ Active Backfill Scripts

To continue the backfill, use these commands in the project root:

- **Match Stats (Upgrade/Basic)**:
  `python scripts/backfill_batch_sofascore.py --type stats --total 1000 --limit 100`

- **Player Stats**:
  `python scripts/backfill_batch_sofascore.py --type players --total 1000 --limit 100`

- **Availability / Injuries**:
  `python scripts/backfill_batch_sofascore.py --type availability --total 1000 --limit 100 --status ft`

- **Match Odds**:
  `python src/ingest/backfill_sofascore_odds_markets_v1.py --limit 100`
"""
    return report

if __name__ == "__main__":
    print("Fetching DB metrics...")
    report_md = generate_report()
    
    # Print to console
    print("\n" + report_md)
    
    # Save to file
    docs_path = Path("docs/backfill_report.md")
    docs_path.parent.mkdir(exist_ok=True)
    with open(docs_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\nReport saved to {docs_path.absolute()}")
