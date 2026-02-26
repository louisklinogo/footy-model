# pyright: reportUnknownParameterType=false, reportMissingParameterType=false

from __future__ import annotations

from conftest import TestDbCase


def test_fixture_odds_markets_schema(db_case: TestDbCase) -> None:
    conn = db_case.conn
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_name = 'fixture_odds_markets'
            """
        )
        assert cur.fetchone() is not None, "fixture_odds_markets table missing"

        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'fixture_odds_markets'
            """
        )
        cols = {row[0] for row in cur.fetchall()}
        expected = {
            "odds_id",
            "fixture_id",
            "provider",
            "provider_id",
            "market_code",
            "line_num",
            "line_text",
            "odds_json",
            "snapshot_time_utc",
            "snapshot_type",
            "created_at",
        }
        missing = expected - cols
        assert not missing, f"fixture_odds_markets missing columns: {sorted(missing)}"

        cur.execute(
            """
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_name = 'fixture_odds_markets'
              AND constraint_type = 'UNIQUE'
            """
        )
        unique_constraints = {row[0] for row in cur.fetchall()}
        assert (
            "uq_fixture_odds_markets" in unique_constraints
        ), "uq_fixture_odds_markets constraint missing"
