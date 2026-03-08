from __future__ import annotations

import numpy as np
import pandas as pd

from src.modeling.v2.families.scoreline.residual_utils import (
    apply_residual_bundle,
    build_residual_source_frame,
    prepare_residual_feature_frame,
)


class _ConstantProbabilityModel:
    def __init__(self, positive_prob: float) -> None:
        self.positive_prob = float(positive_prob)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        probs = np.full(x.shape[0], self.positive_prob, dtype=float)
        return np.column_stack([1.0 - probs, probs])


def test_prepare_residual_feature_frame_adds_generated_missingness_and_imputation() -> None:
    source = pd.DataFrame(
        [
            {
                "base_lambda_home": 1.2,
                "base_lambda_away": 0.9,
                "base_lambda_total": 2.1,
                "base_1x2_h": 0.45,
                "base_1x2_d": 0.28,
                "base_1x2_a": 0.27,
                "base_dc_1x": 0.73,
                "base_dc_x2": 0.55,
                "base_dc_12": 0.72,
                "base_o15": 0.76,
                "base_u35": 0.79,
                "odds_over_15": None,
                "implied_over15": 0.62,
            }
        ]
    )

    features, columns, medians = prepare_residual_feature_frame(source)

    assert "odds_over_15__missing" in columns
    assert medians["odds_over_15"] == 0.0
    assert features.loc[0, "odds_over_15__missing"] == 1.0


def test_apply_residual_bundle_normalizes_one_x_two_and_derives_double_chance() -> None:
    raw = pd.DataFrame([{"fixture_id": 1, "odds_over_15": 1.4}])
    base_markets = pd.DataFrame(
        [
            {
                "1x2_h": 0.40,
                "1x2_d": 0.30,
                "1x2_a": 0.30,
                "dc_1x": 0.70,
                "dc_x2": 0.60,
                "dc_12": 0.70,
                "o15": 0.72,
                "u35": 0.81,
            }
        ]
    )
    source = build_residual_source_frame(
        raw_frame=raw,
        base_market_frame=base_markets,
        lambda_home=np.array([1.5]),
        lambda_away=np.array([1.0]),
    )
    residual_features, _, _ = prepare_residual_feature_frame(source)
    adjusted = apply_residual_bundle(
        base_market_frame=base_markets,
        residual_features=residual_features,
        residual_bundle={
            "models": {
                "1x2_h": _ConstantProbabilityModel(0.70),
                "1x2_d": _ConstantProbabilityModel(0.20),
                "1x2_a": _ConstantProbabilityModel(0.40),
                "o15": _ConstantProbabilityModel(0.83),
            }
        },
        direct_market_overrides=True,
    )

    assert np.isclose(
        adjusted.loc[0, "1x2_h"] + adjusted.loc[0, "1x2_d"] + adjusted.loc[0, "1x2_a"],
        1.0,
    )
    assert np.isclose(adjusted.loc[0, "dc_1x"], adjusted.loc[0, "1x2_h"] + adjusted.loc[0, "1x2_d"])
    assert adjusted.loc[0, "o15"] > 0.8


def test_apply_residual_bundle_defaults_to_base_market_frame() -> None:
    base_markets = pd.DataFrame([{"1x2_h": 0.40, "1x2_d": 0.30, "1x2_a": 0.30, "o15": 0.72}])
    residual_features = pd.DataFrame([{"base_1x2_h": 0.40}])

    adjusted = apply_residual_bundle(
        base_market_frame=base_markets,
        residual_features=residual_features,
        residual_bundle={"models": {"1x2_h": _ConstantProbabilityModel(0.90)}},
    )

    assert adjusted.equals(base_markets)
