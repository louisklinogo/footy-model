# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import pandas as pd

from src.modeling.layer2_situational.situational_utils import (
    add_season_key,
    compute_point_in_time_state,
    latest_lambda_pairs_sql,
)


def test_add_season_key_uses_registry_for_e0_boundary() -> None:
    df = pd.DataFrame(
        {
            "league_code": ["E0", "E0"],
            "match_datetime_utc": ["2025-01-10T12:00:00Z", "2025-08-10T12:00:00Z"],
        }
    )

    out = add_season_key(df)

    assert out["season"].tolist() == ["2024", "2025"]


def test_compute_point_in_time_state_pre_kickoff_points_and_form() -> None:
    fixtures = pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "league_code": "E0",
                "match_datetime_utc": "2025-01-01T12:00:00Z",
                "home_team_id": 10,
                "away_team_id": 20,
                "home_goals": 2,
                "away_goals": 0,
                "status": "ft",
            },
            {
                "fixture_id": 2,
                "league_code": "E0",
                "match_datetime_utc": "2025-01-05T12:00:00Z",
                "home_team_id": 30,
                "away_team_id": 10,
                "home_goals": 0,
                "away_goals": 1,
                "status": "ft",
            },
            {
                "fixture_id": 3,
                "league_code": "E0",
                "match_datetime_utc": "2025-01-07T12:00:00Z",
                "home_team_id": 20,
                "away_team_id": 30,
                "home_goals": 1,
                "away_goals": 0,
                "status": "ft",
            },
            {
                "fixture_id": 4,
                "league_code": "E0",
                "match_datetime_utc": "2025-01-10T12:00:00Z",
                "home_team_id": 10,
                "away_team_id": 20,
                "home_goals": pd.NA,
                "away_goals": pd.NA,
                "status": "scheduled",
            },
            {
                "fixture_id": 5,
                "league_code": "E0",
                "match_datetime_utc": "2025-01-11T12:00:00Z",
                "home_team_id": 20,
                "away_team_id": 10,
                "home_goals": pd.NA,
                "away_goals": pd.NA,
                "status": "scheduled",
            },
        ]
    )

    out = compute_point_in_time_state(fixtures)

    fixture_4 = out.loc[out["fixture_id"] == 4].iloc[0]
    fixture_5 = out.loc[out["fixture_id"] == 5].iloc[0]

    assert int(fixture_4["home_points"]) == 6
    assert int(fixture_4["away_points"]) == 3
    assert int(fixture_4["home_played"]) == 2
    assert int(fixture_4["away_played"]) == 2
    assert int(fixture_4["home_form_streak"]) == 6
    assert int(fixture_4["away_form_streak"]) == 3
    assert int(fixture_4["points_gap"]) == int(fixture_4["home_points"]) - int(
        fixture_4["away_points"]
    )

    assert int(fixture_5["home_points"]) == 3
    assert int(fixture_5["away_points"]) == 6
    assert int(fixture_5["home_played"]) == 2
    assert int(fixture_5["away_played"]) == 2


def test_compute_point_in_time_state_same_kickoff_rows_do_not_leak_ft_updates() -> None:
    fixtures = pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "league_code": "E0",
                "match_datetime_utc": "2025-01-01T12:00:00Z",
                "home_team_id": 20,
                "away_team_id": 101,
                "home_goals": 1,
                "away_goals": 0,
                "status": "ft",
            },
            {
                "fixture_id": 2,
                "league_code": "E0",
                "match_datetime_utc": "2025-01-02T12:00:00Z",
                "home_team_id": 10,
                "away_team_id": 102,
                "home_goals": 1,
                "away_goals": 0,
                "status": "ft",
            },
            {
                "fixture_id": 3,
                "league_code": "E0",
                "match_datetime_utc": "2025-01-10T12:00:00Z",
                "home_team_id": 10,
                "away_team_id": 103,
                "home_goals": 1,
                "away_goals": 0,
                "status": "ft",
            },
            {
                "fixture_id": 4,
                "league_code": "E0",
                "match_datetime_utc": "2025-01-10T12:00:00Z",
                "home_team_id": 20,
                "away_team_id": 104,
                "home_goals": 1,
                "away_goals": 0,
                "status": "ft",
            },
        ]
    )

    out = compute_point_in_time_state(fixtures)
    fixture_4 = out.loc[out["fixture_id"] == 4].iloc[0]

    assert int(fixture_4["home_points"]) == 3
    assert int(fixture_4["home_played"]) == 1
    assert int(fixture_4["home_form_streak"]) == 3
    assert int(fixture_4["home_position"]) == 1


def test_compute_point_in_time_state_adds_lame_duck_support_columns() -> None:
    fixtures = pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "league_code": "E0",
                "match_datetime_utc": "2025-01-01T12:00:00Z",
                "home_team_id": 10,
                "away_team_id": 20,
                "home_goals": 1,
                "away_goals": 0,
                "status": "ft",
            },
            {
                "fixture_id": 2,
                "league_code": "E0",
                "match_datetime_utc": "2025-01-08T12:00:00Z",
                "home_team_id": 30,
                "away_team_id": 10,
                "home_goals": pd.NA,
                "away_goals": pd.NA,
                "status": "scheduled",
            },
        ]
    )

    out = compute_point_in_time_state(fixtures)
    row = out.loc[out["fixture_id"] == 2].iloc[0]

    assert "home_points_to_top" in out.columns
    assert "away_points_to_top" in out.columns
    assert "home_points_to_relegation" in out.columns
    assert "away_points_to_relegation" in out.columns
    assert "league_team_count" in out.columns
    assert int(row["league_team_count"]) >= 4


def test_latest_lambda_pairs_sql_uses_metadata_and_version_pairing() -> None:
    sql = latest_lambda_pairs_sql()
    assert "metadata_json->>'lambda'" in sql
    assert "COUNT(DISTINCT market_code) = 2" in sql
    assert "model_version" in sql
