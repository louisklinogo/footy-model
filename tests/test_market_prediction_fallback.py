import json

import numpy as np
import pandas as pd

from src.modeling.evaluation.predict_market_outcomes_fixtures_first import (
    MARKETS,
    _fallback_market_probabilities,
    build_prediction_rows,
)


class _MockBinaryModel:
    def predict_proba(self, x_row: pd.DataFrame):
        _ = x_row
        return np.array([[0.2, 0.8]], dtype=float)


class _FeatureSubsetBinaryModel:
    feature_names_in_ = np.array(["f1"], dtype=object)

    def predict_proba(self, x_row: pd.DataFrame):
        if list(x_row.columns) != ["f1"]:
            raise ValueError(f"Unexpected columns: {list(x_row.columns)}")
        return np.array([[0.35, 0.65]], dtype=float)


class _MockMulticlassModel:
    classes_ = np.array([0, 1, 2], dtype=int)

    def predict_proba(self, x_row: pd.DataFrame):
        _ = x_row
        return np.array([[0.52, 0.21, 0.27]], dtype=float)


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


def test_build_prediction_rows_aligns_to_model_feature_subset_when_available():
    scored = _base_scored_df()
    scored["f2"] = 2.0
    failure_map = {market: "missing_artifact" for market in MARKETS if market != "o15"}

    rows, fallback_rows = build_prediction_rows(
        scored=scored,
        features=["f1", "f2"],
        models={"o15": _FeatureSubsetBinaryModel()},
        model_failures=failure_map,
    )

    assert len(rows) == len(MARKETS)
    assert fallback_rows == len(MARKETS) - 1

    by_market = {row[1]: row for row in rows}
    o15_meta = json.loads(by_market["o15"][5])
    assert o15_meta["fallback_used"] is False
    assert abs(float(by_market["o15"][4]) - 0.65) < 1e-9


def test_build_prediction_rows_uses_multiclass_for_configured_1x2_dc_markets():
    scored = _base_scored_df()
    failure_map = {market: "missing_artifact" for market in MARKETS}

    rows, fallback_rows = build_prediction_rows(
        scored=scored,
        features=["f1"],
        models={},
        model_failures=failure_map,
        multiclass_model=_MockMulticlassModel(),
        multiclass_markets={"1x2_h", "dc_12"},
    )

    by_market = {row[1]: row for row in rows}
    assert abs(float(by_market["1x2_h"][4]) - 0.52) < 1e-9
    assert abs(float(by_market["dc_12"][4]) - 0.79) < 1e-9

    m_1x2 = json.loads(by_market["1x2_h"][5])
    m_dc12 = json.loads(by_market["dc_12"][5])
    assert m_1x2["fallback_used"] is False
    assert m_dc12["fallback_used"] is False
    assert m_1x2["prediction_model_family"] == "1x2_dc_multiclass"
    assert m_dc12["prediction_model_family"] == "1x2_dc_multiclass"

    assert fallback_rows == len(MARKETS) - 2


def test_fallback_market_probabilities_include_canonical_handicap_contract() -> None:
    scored = _base_scored_df()

    probs, trace = _fallback_market_probabilities(scored.iloc[0])

    assert trace["lambda_source"] in {"adj_lambda_final", "lambda_l1", "rolling_xg_proxy"}
    assert probs["ah2_home_m05"] == probs["ah_h05"]
    assert probs["ah2_away_m15"] == probs["ah_a15"]
    assert probs["eh3_0_1_home"] == probs["eh_h1"]
    assert probs["eh3_1_0_away"] == probs["eh_a1"]
    assert abs(probs["ah2_home_m05"] + probs["ah2_away_p05"] - 1.0) < 0.01
    assert abs(
        probs["eh3_0_1_home"] + probs["eh3_0_1_draw"] + probs["eh3_0_1_away"] - 1.0
    ) < 0.01
