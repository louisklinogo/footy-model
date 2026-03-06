from __future__ import annotations

import pandas as pd

from src.modeling.layer2_markets.market_outcome_calibrator import _add_xg_anchor_features


def test_add_xg_anchor_features_uses_only_prior_matches() -> None:
    frame = pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "league_code": "ENG",
                "match_datetime_utc": "2024-01-01T12:00:00Z",
                "home_team_id": 10,
                "away_team_id": 20,
                "home_rolling_xg": 1.0,
                "away_rolling_xg": 0.8,
            },
            {
                "fixture_id": 2,
                "league_code": "ENG",
                "match_datetime_utc": "2024-01-08T12:00:00Z",
                "home_team_id": 10,
                "away_team_id": 20,
                "home_rolling_xg": 1.5,
                "away_rolling_xg": 1.1,
            },
            {
                "fixture_id": 3,
                "league_code": "ENG",
                "match_datetime_utc": "2024-01-15T12:00:00Z",
                "home_team_id": 10,
                "away_team_id": 20,
                "home_rolling_xg": 9.0,
                "away_rolling_xg": 0.6,
            },
        ]
    )

    out = _add_xg_anchor_features(frame)
    row3 = out[out["fixture_id"] == 3].iloc[0]

    # Baseline/recent anchor must not include current-row xG (9.0 here).
    assert abs(float(row3["home_season_baseline_xg"]) - 1.25) < 1e-9
    assert abs(float(row3["home_recent_xg_mean_5"]) - 1.25) < 1e-9
    assert abs(float(row3["home_recent_vs_baseline_zscore"])) < 1e-9

    # Composite anti-recency features are produced.
    assert "regressed_xg_diff" in out.columns
    assert "recent_vs_baseline_gap" in out.columns

