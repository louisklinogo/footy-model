from conftest import run_script


def test_external_context_tables_exist_after_bootstrap(db_case) -> None:
    result = run_script(["src/ingest/bootstrap_fixtures_schema_v1.py"])
    assert result.returncode == 0, result.stderr or result.stdout

    expected = {
        "team_external_context": {"provider", "context_type", "team_id", "provider_team_id", "snapshot_time_utc", "raw_json"},
        "match_external_context": {"provider", "context_type", "fixture_id", "provider_fixture_id", "snapshot_time_utc", "raw_json"},
        "player_external_context": {"provider", "context_type", "player_id", "provider_player_id", "snapshot_time_utc", "raw_json"},
    }

    with db_case.conn.cursor() as cur:
        for table_name, expected_columns in expected.items():
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = %s
                """,
                (table_name,),
            )
            found = {row[0] for row in cur.fetchall()}
            assert expected_columns.issubset(found), f"{table_name} missing columns: {expected_columns - found}"
