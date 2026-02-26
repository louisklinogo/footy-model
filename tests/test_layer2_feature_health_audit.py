from __future__ import annotations

import pandas as pd

from src.modeling.layer2_situational.audit_feature_health import compute_feature_health


def test_feature_health_detects_zero_variance_and_binary_flags() -> None:
    df = pd.DataFrame(
        {
            "position_gap": [1, 1, 1, 1],
            "congestion_flag": [0, 1, 0, 1],
            "home_rolling_xg": [1.2, 1.5, 1.1, 1.4],
            "odds_model_gap_home": [0.05, -0.02, 0.01, 0.03],
        }
    )
    report = compute_feature_health(
        df,
        [
            "position_gap",
            "congestion_flag",
            "home_rolling_xg",
            "odds_model_gap_home",
        ],
    )

    assert "position_gap" in report["zero_variance_features"]
    binary_flags = {row["feature"] for row in report["binary_flags"]}
    assert "congestion_flag" in binary_flags
    assert "odds_model_gap_home" in report["odds_gap_stats"]
