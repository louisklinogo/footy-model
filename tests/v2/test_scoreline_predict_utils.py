from __future__ import annotations

import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd

from src.modeling.v2.calibration.methods import fit_binary_calibrator
from src.modeling.v2.families.scoreline.predict_scoreline import _apply_market_calibrators, load_artifacts
from src.modeling.v2.families.scoreline.residual_utils import apply_residual_bundle
from src.modeling.v2.families.scoreline.total_intensity import apply_total_intensity_correction


class _ConstantRegressor:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.value, dtype=float)


def test_load_artifacts_reads_optional_residual_and_calibrator_payloads() -> None:
    with tempfile.TemporaryDirectory(prefix="scoreline_predict_artifacts_") as td:
        artifact_dir = Path(td)
        joblib.dump(_ConstantRegressor(1.2), artifact_dir / "home_goals_model.pkl")
        joblib.dump(_ConstantRegressor(0.8), artifact_dir / "away_goals_model.pkl")
        (artifact_dir / "features.json").write_text(json.dumps(["home_rolling_xg"]), encoding="utf-8")
        (artifact_dir / "imputation.json").write_text(
            json.dumps({"global_medians": {"home_rolling_xg": 1.0}}), encoding="utf-8"
        )
        (artifact_dir / "total_intensity_correction.json").write_text(
            json.dumps({"method": "mean_total_ratio_v1", "multiplier": 1.05}),
            encoding="utf-8",
        )
        joblib.dump({"trained_markets": ["1x2_h"], "models": {}}, artifact_dir / "residual_models.joblib")
        joblib.dump({"o15": {"method": "identity"}}, artifact_dir / "calibrators.joblib")

        _, _, features, medians, correction, residual_bundle, calibrators = load_artifacts(artifact_dir)

        assert features == ["home_rolling_xg"]
        assert medians == {"home_rolling_xg": 1.0}
        assert correction["multiplier"] == 1.05
        assert residual_bundle is not None
        assert residual_bundle["trained_markets"] == ["1x2_h"]
        assert calibrators["o15"]["method"] == "identity"


def test_load_artifacts_uses_identity_total_intensity_when_sidecar_missing() -> None:
    with tempfile.TemporaryDirectory(prefix="scoreline_predict_artifacts_") as td:
        artifact_dir = Path(td)
        joblib.dump(_ConstantRegressor(1.2), artifact_dir / "home_goals_model.pkl")
        joblib.dump(_ConstantRegressor(0.8), artifact_dir / "away_goals_model.pkl")
        (artifact_dir / "features.json").write_text(json.dumps(["home_rolling_xg"]), encoding="utf-8")
        (artifact_dir / "imputation.json").write_text(
            json.dumps({"global_medians": {"home_rolling_xg": 1.0}}), encoding="utf-8"
        )

        _, _, _, _, correction, _, _ = load_artifacts(artifact_dir)

        assert correction["method"] == "identity"
        assert correction["multiplier"] == 1.0


def test_apply_market_calibrators_updates_probability_column() -> None:
    calibrator = fit_binary_calibrator(
        "sigmoid",
        p_model=np.array([0.15, 0.35, 0.65, 0.85]),
        y_true=np.array([0, 0, 1, 1]),
    )
    market_frame = pd.DataFrame({"o15": [0.2, 0.8], "u35": [0.7, 0.4]})

    adjusted, methods = _apply_market_calibrators(
        market_frame=market_frame,
        calibrators={"o15": calibrator},
    )

    assert methods == {"o15": "sigmoid"}
    assert list(adjusted.columns) == ["o15", "u35"]
    assert not np.allclose(adjusted["o15"].to_numpy(), market_frame["o15"].to_numpy())
    assert np.allclose(adjusted["u35"].to_numpy(), market_frame["u35"].to_numpy())


def test_residual_bundle_is_safe_by_default_for_runtime_surface() -> None:
    market_frame = pd.DataFrame({"1x2_h": [0.4], "1x2_d": [0.3], "1x2_a": [0.3]})
    adjusted = apply_residual_bundle(
        base_market_frame=market_frame,
        residual_features=pd.DataFrame([{"base_1x2_h": 0.4}]),
        residual_bundle={"models": {}, "publish_to_canonical_surface": False},
    )

    assert adjusted.equals(market_frame)


def test_apply_total_intensity_correction_changes_total_not_share() -> None:
    corrected_home, corrected_away = apply_total_intensity_correction(
        lambda_home=np.array([1.8]),
        lambda_away=np.array([1.2]),
        correction={"multiplier": 0.95},
    )

    assert np.allclose(corrected_home / (corrected_home + corrected_away), np.array([0.6]))
    assert np.allclose(corrected_home + corrected_away, np.array([2.85]))
