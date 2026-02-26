import pytest
from pathlib import Path
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportUnusedParameter=false

from src.features.build_team_premium_snapshots_v1 import build_team_premium_snapshots
from conftest import (
    days_from_now,
    insert_fixture,
    insert_league,
    insert_odds_snapshot,
    insert_premium_stats,
    insert_team,
    run_script,
)


def test_prediction_upsert_is_idempotent_per_fixture_market(db_case):
    if not Path("src/modeling/evaluation/predict_market_outcomes_fixtures_first.py").exists():
        pytest.skip("fixtures-first scripts removed/moved in this workspace")

    with db_case.conn.cursor() as cur:
        insert_league(cur, db_case.league_code)
        home_team = insert_team(cur, db_case.league_code, f"Home {db_case.suffix}")
        away_team = insert_team(cur, db_case.league_code, f"Away {db_case.suffix}")

        prior_fixture = insert_fixture(
            cur,
            league_code=db_case.league_code,
            flashscore_id=f"tst_hist_{db_case.suffix}",
            home_team_id=home_team,
            away_team_id=away_team,
            kickoff=days_from_now(-1),
            status="ft",
        )
        future_fixture = insert_fixture(
            cur,
            league_code=db_case.league_code,
            flashscore_id=f"tst_future_{db_case.suffix}",
            home_team_id=home_team,
            away_team_id=away_team,
            kickoff=days_from_now(1),
            status="scheduled",
        )

        insert_premium_stats(cur, prior_fixture, h_xg=1.60, a_xg=0.90)
        insert_odds_snapshot(cur, future_fixture, snapshot_time=days_from_now(0))

    db_case.conn.commit()

    _ = build_team_premium_snapshots(league=db_case.league_code)

    first = run_script(
        [
            "src/modeling/evaluation/predict_market_outcomes_fixtures_first.py",
            "--league",
            db_case.league_code,
            "--days",
            "7",
            "--limit",
            "1",
        ]
    )
    assert first.returncode == 0, first.stderr or first.stdout

    with db_case.conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM predictions
            WHERE fixture_id = %s
              AND model_name = 'market_outcome_gbm'
              AND model_version = 'fixtures_first_prematch_v1'
            """,
            (future_fixture,),
        )
        first_count = int(cur.fetchone()[0])

    second = run_script(
        [
            "src/modeling/evaluation/predict_market_outcomes_fixtures_first.py",
            "--league",
            db_case.league_code,
            "--days",
            "7",
            "--limit",
            "1",
        ]
    )
    assert second.returncode == 0, second.stderr or second.stdout

    with db_case.conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM predictions
            WHERE fixture_id = %s
              AND model_name = 'market_outcome_gbm'
              AND model_version = 'fixtures_first_prematch_v1'
            """,
            (future_fixture,),
        )
        second_count = int(cur.fetchone()[0])

    assert first_count > 0
    assert second_count == first_count
