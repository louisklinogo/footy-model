"""
Reconcile fixtures for all leagues with SofaScore IDs.
"""

import asyncio
import sys
import io
import psycopg2
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import get_database_url
from src.ingest.reconcile_sofascore_fixtures import reconcile_fixtures


async def main():
    database_url = get_database_url()
    conn = psycopg2.connect(database_url)

    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT league_code
            FROM leagues
            WHERE sofascore_league_id IS NOT NULL
            AND league_code NOT IN ('I2', 'RO1')
            ORDER BY league_code
        """)
        leagues = [r[0] for r in cur.fetchall()]
        conn.close()

        print(f"Reconciling {len(leagues)} leagues...")

        for league_code in leagues:
            print(f"\n{'=' * 50}")
            print(f"Processing {league_code}...")
            print("=" * 50)
            await reconcile_fixtures(league_code, days=None, dry_run=False)

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
