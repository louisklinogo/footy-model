from __future__ import annotations

import pandas as pd

from src.modeling.evaluation import predict_market_outcomes_fixtures_first as fixtures_first
from src.modeling.layer2_markets import market_outcome_calibrator as legacy_calibrator


def test_add_targets_and_derived_adds_defensive_pressure_missingness_indicators() -> None:
    frame = pd.DataFrame(
        [
            {
                "home_rolling_xg": 1.4,
                "away_rolling_xg_against": 1.1,
                "home_rolling_xgot": 1.2,
                "away_rolling_xgot_against": 1.0,
                "home_rolling_xa": 1.1,
                "away_rolling_xa_against": 0.9,
                "home_rolling_corners": 5.0,
                "away_rolling_corners_against": 4.0,
                "home_sample_size": 10.0,
                "away_sample_size": 9.0,
                "away_rolling_xg": 1.0,
                "odds_over_15": 1.8,
                "odds_under_15": 2.0,
                "odds_over_25": 2.1,
                "odds_under_25": 1.7,
                "odds_over_35": 3.0,
                "odds_under_35": 1.4,
                "odds_c75_over": 1.9,
                "odds_c75_under": 1.9,
                "odds_c85_over": 2.0,
                "odds_c85_under": 1.8,
                "odds_c95_over": 2.2,
                "odds_c95_under": 1.7,
                "odds_c105_over": 2.5,
                "odds_c105_under": 1.6,
                "home_formation": "4-3-3",
                "away_formation": "4-4-2",
                "fixture_id": 1,
                "league_code": "E0",
                "match_datetime_utc": "2026-03-10T12:00:00Z",
                "home_team_id": 10,
                "away_team_id": 20,
                "home_goals": 1,
                "away_goals": 0,
                "total_goals": 1,
                "home_corners": 5,
                "away_corners": 4,
                "total_corners": 9,
                "home_led_by_1_any": 1,
                "away_led_by_1_any": 0,
                "home_led_by_2_any": 0,
                "away_led_by_2_any": 0,
                "home_rolling_errors_lead_to_shot": None,
                "away_rolling_errors_lead_to_shot": 0.2,
                "home_rolling_tackles_pct": 62.0,
                "away_rolling_tackles_pct": None,
                "home_rolling_rest_days": 5.0,
                "away_rolling_rest_days": None,
                "home_fidelity_score": 0.92,
                "away_fidelity_score": None,
                "home_rolling_lead_rate_1up": 0.61,
                "away_rolling_lead_rate_1up": None,
                "home_rolling_lead_rate_2up": 0.21,
                "away_rolling_lead_rate_2up": None,
            }
        ]
    )

    out = legacy_calibrator.add_targets_and_derived(frame)

    assert float(out.loc[0, "home_rolling_errors_lead_to_shot_is_missing"]) == 1.0
    assert float(out.loc[0, "away_rolling_errors_lead_to_shot_is_missing"]) == 0.0
    assert float(out.loc[0, "home_rolling_tackles_pct_is_missing"]) == 0.0
    assert float(out.loc[0, "away_rolling_tackles_pct_is_missing"]) == 1.0
    assert pd.isna(out.loc[0, "lead_rate_1up_diff"])
    assert pd.isna(out.loc[0, "lead_rate_2up_diff"])
    assert pd.isna(out.loc[0, "rest_days_sum"])
    assert pd.isna(out.loc[0, "fidelity_score_sum"])
    assert float(out.loc[0, "away_rolling_rest_days_is_missing"]) == 1.0
    assert float(out.loc[0, "away_fidelity_score_is_missing"]) == 1.0
    assert float(out.loc[0, "away_rolling_lead_rate_1up_is_missing"]) == 1.0
    assert float(out.loc[0, "away_rolling_lead_rate_2up_is_missing"]) == 1.0


def test_add_derived_features_adds_defensive_pressure_missingness_indicators() -> None:
    frame = pd.DataFrame(
        [
            {
                "home_rolling_xg": 1.4,
                "away_rolling_xg_against": 1.1,
                "home_rolling_xgot": 1.2,
                "away_rolling_xgot_against": 1.0,
                "home_rolling_xa": 1.1,
                "away_rolling_xa_against": 0.9,
                "home_rolling_corners": 5.0,
                "away_rolling_corners_against": 4.0,
                "home_sample_size": 10.0,
                "away_sample_size": 9.0,
                "away_rolling_xg": 1.0,
                "odds_over_15": 1.8,
                "odds_under_15": 2.0,
                "odds_over_25": 2.1,
                "odds_under_25": 1.7,
                "odds_over_35": 3.0,
                "odds_under_35": 1.4,
                "odds_c75_over": 1.9,
                "odds_c75_under": 1.9,
                "odds_c85_over": 2.0,
                "odds_c85_under": 1.8,
                "odds_c95_over": 2.2,
                "odds_c95_under": 1.7,
                "odds_c105_over": 2.5,
                "odds_c105_under": 1.6,
                "home_rolling_errors_lead_to_shot": None,
                "away_rolling_errors_lead_to_shot": 0.2,
                "home_rolling_tackles_pct": 62.0,
                "away_rolling_tackles_pct": None,
                "home_rolling_rest_days": 5.0,
                "away_rolling_rest_days": None,
                "home_fidelity_score": 0.92,
                "away_fidelity_score": None,
                "home_rolling_lead_rate_1up": 0.61,
                "away_rolling_lead_rate_1up": None,
                "home_rolling_lead_rate_2up": 0.21,
                "away_rolling_lead_rate_2up": None,
            }
        ]
    )

    out = fixtures_first.add_derived_features(frame)

    assert float(out.loc[0, "home_rolling_errors_lead_to_shot_is_missing"]) == 1.0
    assert float(out.loc[0, "away_rolling_errors_lead_to_shot_is_missing"]) == 0.0
    assert float(out.loc[0, "home_rolling_tackles_pct_is_missing"]) == 0.0
    assert float(out.loc[0, "away_rolling_tackles_pct_is_missing"]) == 1.0
    assert pd.isna(out.loc[0, "lead_rate_1up_diff"])
    assert pd.isna(out.loc[0, "lead_rate_2up_diff"])
    assert pd.isna(out.loc[0, "rest_days_sum"])
    assert pd.isna(out.loc[0, "fidelity_score_sum"])
    assert float(out.loc[0, "away_rolling_rest_days_is_missing"]) == 1.0
    assert float(out.loc[0, "away_fidelity_score_is_missing"]) == 1.0
    assert float(out.loc[0, "away_rolling_lead_rate_1up_is_missing"]) == 1.0
    assert float(out.loc[0, "away_rolling_lead_rate_2up_is_missing"]) == 1.0