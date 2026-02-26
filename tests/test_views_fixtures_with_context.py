# pyright: reportUnknownParameterType=false, reportMissingParameterType=false

from __future__ import annotations

from conftest import TestDbCase, insert_fixture, insert_league, insert_odds_snapshot, insert_team, utc_now


def test_view_returns_latest_pre_match_odds(db_case: TestDbCase) -> None:
    conn = db_case.conn
    with conn.cursor() as cur:
        insert_league(cur, db_case.league_code)
        home_id = insert_team(cur, db_case.league_code, f"Home {db_case.suffix}")
        away_id = insert_team(cur, db_case.league_code, f"Away {db_case.suffix}")

        fixture_id = insert_fixture(
            cur,
            db_case.league_code,
            f"fs_{db_case.suffix}",
            home_id,
            away_id,
            utc_now(),
            "scheduled",
        )

        insert_odds_snapshot(cur, fixture_id, utc_now())
        conn.commit()

        cur.execute(
            """
            SELECT prematch_ou_odds, prematch_1x2_odds, odds_snapshot_time
            FROM v_fixtures_with_context
            WHERE fixture_id = %s
            """,
            (fixture_id,),
        )
        row = cur.fetchone()
        assert row is not None, "v_fixtures_with_context missing fixture row"
        prematch_ou_odds, prematch_1x2_odds, odds_snapshot_time = row
        assert prematch_ou_odds is not None, "prematch_ou_odds missing for latest_pre_match"
        assert odds_snapshot_time is not None, "odds_snapshot_time missing for latest_pre_match"
