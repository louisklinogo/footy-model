from __future__ import annotations

# pyright: reportMissingTypeStubs=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportUnusedParameter=false, reportImplicitRelativeImport=false

import json
import importlib.util
from pathlib import Path
import sys

import pytest
from psycopg2.errors import UndefinedTable

from src.features.build_team_premium_snapshots_v1 import build_team_premium_snapshots
from src.modeling.layer2_situational.predict_situational_residual import (
    load_prediction_data,
)


def _load_local_test_helpers() -> object:
    root_dir = Path(__file__).resolve().parents[1]
    conftest_path = root_dir / "tests" / "conftest.py"
    spec = importlib.util.spec_from_file_location("_local_conftest", conftest_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load conftest at {conftest_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_helpers = _load_local_test_helpers()
days_from_now = _helpers.days_from_now
insert_fixture = _helpers.insert_fixture
insert_league = _helpers.insert_league
insert_premium_stats = _helpers.insert_premium_stats
insert_team = _helpers.insert_team


def _is_missing_optional_player_table(exc: Exception) -> bool:
    if isinstance(exc, UndefinedTable):
        return True

    text = str(exc)
    if "UndefinedTable" not in text and "does not exist" not in text:
        return False

    optional_tables = (
        "player_availability",
        "fixture_player_stats",
    )
    return any(table_name in text for table_name in optional_tables)


def test_situational_prediction_features_from_db_seeded_state(db_case) -> None:
    # NOTE: some dev DBs have triggers that write league_code into VARCHAR(10)
    # columns. Keep the test league_code <= 10 chars to avoid truncation errors.
    league_code = f"TST_{db_case.suffix[:6]}"
    db_case.league_code = league_code
    cl_fixture: int | None = None

    try:
        with db_case.conn.cursor() as cur:
            insert_league(cur, league_code)
            insert_league(cur, "CL")

            home_team = insert_team(cur, league_code, f"Home {db_case.suffix}")
            away_team = insert_team(cur, league_code, f"Away {db_case.suffix}")

            ft_1 = insert_fixture(
                cur,
                league_code=league_code,
                flashscore_id=f"tst_ft1_{db_case.suffix}",
                home_team_id=home_team,
                away_team_id=away_team,
                kickoff=days_from_now(-12),
                status="ft",
            )
            ft_2 = insert_fixture(
                cur,
                league_code=league_code,
                flashscore_id=f"tst_ft2_{db_case.suffix}",
                home_team_id=away_team,
                away_team_id=home_team,
                kickoff=days_from_now(-8),
                status="ft",
            )
            ft_3 = insert_fixture(
                cur,
                league_code=league_code,
                flashscore_id=f"tst_ft3_{db_case.suffix}",
                home_team_id=home_team,
                away_team_id=away_team,
                kickoff=days_from_now(-4),
                status="ft",
            )

            target_fixture = insert_fixture(
                cur,
                league_code=league_code,
                flashscore_id=f"tst_target_{db_case.suffix}",
                home_team_id=home_team,
                away_team_id=away_team,
                kickoff=days_from_now(1),
                status="scheduled",
            )
            cl_fixture = insert_fixture(
                cur,
                league_code="CL",
                flashscore_id=f"tst_cl_{db_case.suffix}",
                home_team_id=home_team,
                away_team_id=away_team,
                kickoff=days_from_now(2),
                status="scheduled",
            )

            cur.execute(
                """
            INSERT INTO fixture_results (fixture_id, home_goals, away_goals, result_status)
            VALUES (%s, %s, %s, 'ft')
            ON CONFLICT (fixture_id) DO UPDATE
            SET home_goals = EXCLUDED.home_goals,
                away_goals = EXCLUDED.away_goals,
                result_status = EXCLUDED.result_status
            """,
                (ft_1, 2, 0),
            )
            cur.execute(
                """
            INSERT INTO fixture_results (fixture_id, home_goals, away_goals, result_status)
            VALUES (%s, %s, %s, 'ft')
            ON CONFLICT (fixture_id) DO UPDATE
            SET home_goals = EXCLUDED.home_goals,
                away_goals = EXCLUDED.away_goals,
                result_status = EXCLUDED.result_status
            """,
                (ft_2, 1, 1),
            )
            cur.execute(
                """
            INSERT INTO fixture_results (fixture_id, home_goals, away_goals, result_status)
            VALUES (%s, %s, %s, 'ft')
            ON CONFLICT (fixture_id) DO UPDATE
            SET home_goals = EXCLUDED.home_goals,
                away_goals = EXCLUDED.away_goals,
                result_status = EXCLUDED.result_status
            """,
                (ft_3, 3, 1),
            )

            insert_premium_stats(cur, ft_1, h_xg=1.50, a_xg=0.70)
            insert_premium_stats(cur, ft_2, h_xg=1.10, a_xg=1.00)
            insert_premium_stats(cur, ft_3, h_xg=1.80, a_xg=0.95)

            cur.execute(
                """
            INSERT INTO predictions (fixture_id, market_code, model_name, model_version, p_model, metadata_json)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (fixture_id, market_code, model_name, model_version) DO UPDATE
            SET p_model = EXCLUDED.p_model,
                metadata_json = EXCLUDED.metadata_json
            """,
                (
                    target_fixture,
                    "lambda_home",
                    "lambda_xgb",
                    "v_test",
                    0.5,
                    json.dumps({"lambda": 1.2}),
                ),
            )
            cur.execute(
                """
            INSERT INTO predictions (fixture_id, market_code, model_name, model_version, p_model, metadata_json)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (fixture_id, market_code, model_name, model_version) DO UPDATE
            SET p_model = EXCLUDED.p_model,
                metadata_json = EXCLUDED.metadata_json
            """,
                (
                    target_fixture,
                    "lambda_away",
                    "lambda_xgb",
                    "v_test",
                    0.5,
                    json.dumps({"lambda": 0.9}),
                ),
            )

        db_case.conn.commit()

        _ = build_team_premium_snapshots(league=league_code)

        try:
            pred_df = load_prediction_data(days=3, league=league_code)
        except Exception as exc:
            if _is_missing_optional_player_table(exc):
                pytest.skip(
                    "Skipping DB situational prediction regression: optional player tables are missing"
                )
            raise

        row = pred_df[pred_df["fixture_id"] == target_fixture]
        assert len(row) == 1
        sample = row.iloc[0]

        assert int(sample["congestion_flag"]) == 1
        assert int(sample["home_upcoming_tier"]) == 3

        home_form = int(sample["home_form_streak"])
        away_form = int(sample["away_form_streak"])
        assert home_form >= 0
        assert away_form >= 0
        assert home_form > 0 or away_form > 0

        assert sample["points_gap"] == sample["home_points"] - sample["away_points"]
    finally:
        with db_case.conn.cursor() as cur:
            if cl_fixture is not None:
                cur.execute(
                    "DELETE FROM predictions WHERE fixture_id = %s", (cl_fixture,)
                )
                cur.execute(
                    "DELETE FROM team_premium_snapshots WHERE fixture_id = %s",
                    (cl_fixture,),
                )
                cur.execute(
                    "DELETE FROM fixture_stats_premium WHERE fixture_id = %s",
                    (cl_fixture,),
                )
                cur.execute(
                    "DELETE FROM fixture_results WHERE fixture_id = %s", (cl_fixture,)
                )
                cur.execute("DELETE FROM fixtures WHERE fixture_id = %s", (cl_fixture,))

            # Some DBs have triggers that materialize point-in-time standings into
            # team_league_standings on fixture_results upsert. Clear those rows so
            # the shared conftest cleanup can delete teams.
            cur.execute(
                "DELETE FROM team_league_standings WHERE league_code = %s",
                (league_code,),
            )
        db_case.conn.commit()
