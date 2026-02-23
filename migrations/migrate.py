#!/usr/bin/env python3
"""
Simple migration runner for footy-model.

Usage:
    python migrations/migrate.py [--dry-run]

Features:
    - Tracks applied migrations in `schema_migrations` table
    - Runs .sql files in order by filename (001, 002, etc.)
    - Skips already-applied migrations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg2
from psycopg2 import sql

# Add parent dir to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import get_database_url


MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_name TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def get_applied_migrations(cur) -> set[str]:
    """Get set of already-applied migration names."""
    cur.execute("SELECT migration_name FROM schema_migrations")
    return {row[0] for row in cur.fetchall()}


def get_pending_migrations(migrations_dir: Path) -> list[Path]:
    """Get list of .sql migration files sorted by name."""
    files = sorted(migrations_dir.glob("*.sql"))
    # Exclude this script if it somehow gets picked up
    return [f for f in files if f.name != "migrate.py"]


def apply_migration(cur, migration_path: Path, dry_run: bool = False) -> bool:
    """Apply a single migration file. Returns True if applied."""
    migration_name = migration_path.name

    # Read the SQL
    migration_sql = migration_path.read_text(encoding="utf-8")

    if dry_run:
        print(f"  [DRY-RUN] Would apply: {migration_name}")
        return False

    try:
        # Execute the migration
        cur.execute(migration_sql)

        # Record the migration
        cur.execute(
            "INSERT INTO schema_migrations (migration_name) VALUES (%s) ON CONFLICT DO NOTHING",
            (migration_name,),
        )

        print(f"  [OK] Applied: {migration_name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {migration_name}")
        print(f"    Error: {e}")
        raise


def run_migrations(dry_run: bool = False) -> None:
    """Run all pending migrations."""
    migrations_dir = Path(__file__).parent

    if not migrations_dir.exists():
        print(f"Migrations directory not found: {migrations_dir}")
        return

    conn = psycopg2.connect(get_database_url())

    try:
        with conn.cursor() as cur:
            # Ensure migrations table exists (ALWAYS commit this)
            cur.execute(MIGRATIONS_TABLE)
            conn.commit()

            applied = get_applied_migrations(cur)

            # Get pending migrations
            pending = get_pending_migrations(migrations_dir)

            if not pending:
                print("No migration files found.")
                return

            print(f"Found {len(pending)} migration file(s).")
            print(f"Already applied: {len(applied)}")

            # Apply each pending migration
            applied_count = 0
            for migration_path in pending:
                if migration_path.name in applied:
                    print(f"  - Skipped (already applied): {migration_path.name}")
                    continue

                if apply_migration(cur, migration_path, dry_run):
                    applied_count += 1
                    if not dry_run:
                        conn.commit()

            if dry_run:
                print(f"\n[DRY-RUN] Would apply {applied_count} migration(s).")
                conn.rollback()
            else:
                print(f"\nApplied {applied_count} new migration(s).")

    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be applied without making changes",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Footy-Model Migration Runner")
    print("=" * 60)

    run_migrations(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
