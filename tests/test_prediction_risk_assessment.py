from __future__ import annotations

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportUnusedParameter=false

import json
from pathlib import Path

import pytest

from conftest import (
    days_from_now,
    insert_fixture,
    insert_league,
    insert_odds_snapshot,
    insert_team,
    run_script,
)


def test_assess_prediction_risk_writes_rows_and_odds_sensitive_actions(db_case) -> None:
    if not Path("src/modeling/evaluation/assess_prediction_risk.py").exists():
        pytest.skip("risk assessment script removed/moved in this workspace")

    with db_case.conn.cursor() as cur:
        insert_league(cur, db_case.league_code)
        home_team = insert_team(cur, db_case.league_code, f"RiskHome {db_case.suffix}")
        away_team = insert_team(cur, db_case.league_code, f"RiskAway {db_case.suffix}")

        future_fixture = insert_fixture(
            cur,
            league_code=db_case.league_code,
            flashscore_id=f"tst_risk_future_{db_case.suffix}",
            home_team_id=home_team,
            away_team_id=away_team,
            kickoff=days_from_now(1),
            status="scheduled",
        )
        insert_odds_snapshot(cur, future_fixture, snapshot_time=days_from_now(0))

        cur.execute(
            """
            INSERT INTO predictions (fixture_id, market_code, model_name, model_version, p_model, metadata_json)
            VALUES (%s, 'o25', 'market_outcome_gbm', 'fixtures_first_prematch_v1', %s, %s::jsonb)
            """,
            (
                future_fixture,
                0.72,
                json.dumps(
                    {
                        "features_missing_count": 0,
                        "fallback_used": False,
                        "home_sample_size": 12,
                        "away_sample_size": 11,
                    }
                ),
            ),
        )
        cur.execute(
            """
            INSERT INTO predictions (fixture_id, market_code, model_name, model_version, p_model, metadata_json)
            VALUES (%s, 'c85', 'market_outcome_gbm', 'fixtures_first_prematch_v1', %s, %s::jsonb)
            """,
            (
                future_fixture,
                0.61,
                json.dumps(
                    {
                        "features_missing_count": 3,
                        "fallback_used": True,
                        "home_sample_size": 5,
                        "away_sample_size": 6,
                    }
                ),
            ),
        )
    db_case.conn.commit()

    risk = run_script(
        [
            "src/modeling/evaluation/assess_prediction_risk.py",
            "--league",
            db_case.league_code,
            "--days",
            "7",
        ]
    )
    assert risk.returncode == 0, risk.stderr or risk.stdout

    with db_case.conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                market_code,
                implied_probability,
                edge_raw,
                action,
                risk_flags_json
            FROM prediction_risk_assessments
            WHERE fixture_id = %s
              AND model_name = 'market_outcome_gbm'
              AND model_version = 'fixtures_first_prematch_v1'
            ORDER BY market_code ASC
            """,
            (future_fixture,),
        )
        rows = cur.fetchall()

    assert len(rows) == 2

    by_market = {str(row[0]): row for row in rows}

    o25 = by_market["o25"]
    assert o25[1] is not None
    assert o25[2] is not None
    assert str(o25[3]) in {"pass", "watch", "bet_small", "bet"}

    c85 = by_market["c85"]
    assert c85[1] is None
    assert c85[2] is None
    assert str(c85[3]) == "pass"
    flags = c85[4] if isinstance(c85[4], list) else []
    assert "missing_odds" in flags
