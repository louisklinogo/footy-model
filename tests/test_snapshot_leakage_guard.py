# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportUnusedParameter=false

import pytest

from src.features.build_team_premium_snapshots_v1 import build_team_premium_snapshots
from conftest import days_from_now, insert_fixture, insert_league, insert_premium_stats, insert_team


def test_same_kickoff_does_not_leak_future_premium_data(db_case):
    with db_case.conn.cursor() as cur:
        insert_league(cur, db_case.league_code)
        team_1 = insert_team(cur, db_case.league_code, f"Team1 {db_case.suffix}")
        team_2 = insert_team(cur, db_case.league_code, f"Team2 {db_case.suffix}")
        team_3 = insert_team(cur, db_case.league_code, f"Team3 {db_case.suffix}")

        kickoff_p = days_from_now(-2)
        kickoff_t1 = days_from_now(-1)

        fixture_p = insert_fixture(
            cur,
            league_code=db_case.league_code,
            flashscore_id=f"tst_p_{db_case.suffix}",
            home_team_id=team_1,
            away_team_id=team_2,
            kickoff=kickoff_p,
            status="ft",
        )
        fixture_a = insert_fixture(
            cur,
            league_code=db_case.league_code,
            flashscore_id=f"tst_a_{db_case.suffix}",
            home_team_id=team_1,
            away_team_id=team_2,
            kickoff=kickoff_t1,
            status="ft",
        )
        fixture_b = insert_fixture(
            cur,
            league_code=db_case.league_code,
            flashscore_id=f"tst_b_{db_case.suffix}",
            home_team_id=team_3,
            away_team_id=team_1,
            kickoff=kickoff_t1,
            status="ft",
        )

        insert_premium_stats(cur, fixture_p, h_xg=1.40, a_xg=0.30)
        insert_premium_stats(cur, fixture_a, h_xg=9.00, a_xg=0.20)
        insert_premium_stats(cur, fixture_b, h_xg=0.10, a_xg=8.50)

    db_case.conn.commit()

    result = build_team_premium_snapshots(league=db_case.league_code)
    assert result["fixtures_processed"] >= 3

    with db_case.conn.cursor() as cur:
        cur.execute(
            """
            SELECT sample_size, rolling_xg
            FROM team_premium_snapshots
            WHERE fixture_id = %s
              AND team_id = %s
              AND is_home = true
            """,
            (fixture_a, team_1),
        )
        row = cur.fetchone()

    assert row is not None
    assert int(row[0]) == 1
    assert float(row[1]) == pytest.approx(1.40)
