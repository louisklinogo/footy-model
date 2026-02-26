import json

import numpy as np
import pandas as pd

from src.modeling.evaluation.predict_market_outcomes_fixtures_first import (
    MARKETS,
    build_prediction_rows,
)


class _MockBinaryModel:
    def predict_proba(self, x_row: pd.DataFrame):
        _ = x_row
        return np.array([[0.2, 0.8]], dtype=float)


def _base_scored_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fixture_id": 12345,
                "features_missing_count": 0,
                "home_sample_size": 8.0,
                "away_sample_size": 8.0,
                "home_rolling_xg": 1.45,
                "away_rolling_xg": 1.10,
                "home_rolling_corners": 5.2,
                "away_rolling_corners": 4.4,
                "lambda_home_l1": 1.40,
                "lambda_away_l1": 1.05,
                "adj_lambda_home_final": 1.55,
                "adj_lambda_away_final": 0.98,
                "rule_fired_home": 1,
                "rule_fired_away": 0,
                "odds_snapshot_time_utc": pd.Timestamp("2026-02-26T12:00:00Z"),
                "f1": 1.0,
            }
        ]
    )


def test_build_prediction_rows_all_fallback_when_models_missing():
    scored = _base_scored_df()
    failure_map = {market: "missing_artifact" for market in MARKETS}

    rows, fallback_rows = build_prediction_rows(
        scored=scored,
        features=[],
        models={},
        model_failures=failure_map,
    )

    assert len(rows) == len(MARKETS)
    assert fallback_rows == len(MARKETS)

    for row in rows:
        p_model = float(row[4])
        assert 0.001 <= p_model <= 0.999
        meta = json.loads(row[5])
        assert meta["fallback_used"] is True
        assert meta["fallback_reason"] == "missing_artifact"
        assert meta["fallback_lambda_source"] in {"adj_lambda_final", "lambda_l1", "rolling_xg_proxy"}


def test_build_prediction_rows_uses_model_when_available():
    scored = _base_scored_df()
    failure_map = {market: "missing_artifact" for market in MARKETS if market != "o15"}

    rows, fallback_rows = build_prediction_rows(
        scored=scored,
        features=["f1"],
        models={"o15": _MockBinaryModel()},
        model_failures=failure_map,
    )

    assert len(rows) == len(MARKETS)
    assert fallback_rows == len(MARKETS) - 1

    by_market = {row[1]: row for row in rows}
    o15_meta = json.loads(by_market["o15"][5])
    assert o15_meta["fallback_used"] is False
    assert abs(float(by_market["o15"][4]) - 0.8) < 1e-9
