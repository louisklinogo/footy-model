from __future__ import annotations

import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pytest

from src.modeling.v2.families.corners.predict_corners import load_artifacts


class _ConstantRegressor:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, x) -> np.ndarray:
        return np.full(len(x), self.value, dtype=float)


def _write_base_corner_artifact_dir(artifact_dir: Path) -> None:
    joblib.dump(_ConstantRegressor(10.0), artifact_dir / "total_corners_model.pkl")
    joblib.dump(_ConstantRegressor(0.5), artifact_dir / "home_share_model.pkl")
    (artifact_dir / "features.json").write_text(
        json.dumps(["home_rolling_corners", "away_rolling_corners"]),
        encoding="utf-8",
    )
    (artifact_dir / "imputation.json").write_text(
        json.dumps({"global_medians": {"home_rolling_corners": 5.0, "away_rolling_corners": 4.0}}),
        encoding="utf-8",
    )
    (artifact_dir / "dispersion.json").write_text(
        json.dumps({"total_r": 1.0, "home_r": 1.0, "away_r": 1.0}),
        encoding="utf-8",
    )
    (artifact_dir / "league_regime.json").write_text(json.dumps({}), encoding="utf-8")


def test_load_artifacts_detects_neural_share_sidecar_when_model_config_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(prefix="corners_predict_artifacts_") as td:
        artifact_dir = Path(td)
        _write_base_corner_artifact_dir(artifact_dir)
        (artifact_dir / "neural_share_residual_bundle.pt").write_text("stub", encoding="utf-8")
        monkeypatch.setattr(
            "src.modeling.v2.families.corners.predict_corners.load_neural_share_residual_bundle",
            lambda path: {"loaded_from": path.name},
        )

        models, features, medians, dispersion, path_version, league_regime = load_artifacts(artifact_dir)

        assert path_version == "totals_first_neural_share_residual"
        assert models["neural_share_residual"] == {"loaded_from": "neural_share_residual_bundle.pt"}
        assert features == ["home_rolling_corners", "away_rolling_corners"]
        assert medians["home_rolling_corners"] == 5.0
        assert dispersion["total_r"] == 1.0
        assert league_regime == {}


def test_load_artifacts_surfaces_clear_neural_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(prefix="corners_predict_artifacts_") as td:
        artifact_dir = Path(td)
        _write_base_corner_artifact_dir(artifact_dir)
        (artifact_dir / "model_config.json").write_text(
            json.dumps(
                {
                    "path_version": "totals_first_neural_share_residual",
                    "neural_residual_sidecar": "neural_share_residual_bundle.pt",
                }
            ),
            encoding="utf-8",
        )
        (artifact_dir / "neural_share_residual_bundle.pt").write_text("stub", encoding="utf-8")
        monkeypatch.setattr(
            "src.modeling.v2.families.corners.predict_corners.load_neural_share_residual_bundle",
            lambda path: (_ for _ in ()).throw(
                RuntimeError(
                    f"PyTorch is required for the corners neural residual path for artifact {path}."
                )
            ),
        )

        with pytest.raises(RuntimeError, match="PyTorch is required for the corners neural residual path"):
            load_artifacts(artifact_dir)


def test_load_artifacts_detects_neural_total_share_sidecars_when_model_config_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(prefix="corners_predict_artifacts_") as td:
        artifact_dir = Path(td)
        _write_base_corner_artifact_dir(artifact_dir)
        (artifact_dir / "neural_total_residual_bundle.pt").write_text("stub", encoding="utf-8")
        (artifact_dir / "neural_share_residual_bundle.pt").write_text("stub", encoding="utf-8")
        monkeypatch.setattr(
            "src.modeling.v2.families.corners.predict_corners.load_neural_total_residual_bundle",
            lambda path: {"total_loaded_from": path.name},
        )
        monkeypatch.setattr(
            "src.modeling.v2.families.corners.predict_corners.load_neural_share_residual_bundle",
            lambda path: {"share_loaded_from": path.name},
        )

        models, features, medians, dispersion, path_version, league_regime = load_artifacts(artifact_dir)

        assert path_version == "totals_first_neural_total_share_residual"
        assert models["neural_total_residual"] == {"total_loaded_from": "neural_total_residual_bundle.pt"}
        assert models["neural_share_residual"] == {"share_loaded_from": "neural_share_residual_bundle.pt"}
        assert features == ["home_rolling_corners", "away_rolling_corners"]
        assert medians["home_rolling_corners"] == 5.0
        assert dispersion["total_r"] == 1.0
        assert league_regime == {}


def test_load_artifacts_detects_neural_total_ladder_sidecar_before_legacy_total_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(prefix="corners_predict_artifacts_") as td:
        artifact_dir = Path(td)
        _write_base_corner_artifact_dir(artifact_dir)
        (artifact_dir / "neural_total_market_ladder_bundle.pt").write_text("stub", encoding="utf-8")
        joblib.dump({"legacy": True}, artifact_dir / "total_market_heads.pkl")
        monkeypatch.setattr(
            "src.modeling.v2.families.corners.predict_corners.load_neural_total_market_ladder_bundle",
            lambda path: {"ladder_loaded_from": path.name},
        )

        models, features, medians, dispersion, path_version, league_regime = load_artifacts(artifact_dir)

        assert path_version == "totals_surface_neural_ladder_calibrated"
        assert models["neural_total_market_ladder"] == {"ladder_loaded_from": "neural_total_market_ladder_bundle.pt"}
        assert "total_market_heads" not in models
        assert features == ["home_rolling_corners", "away_rolling_corners"]
        assert medians["home_rolling_corners"] == 5.0
        assert dispersion["total_r"] == 1.0
        assert league_regime == {}


def test_load_artifacts_reads_neural_total_ladder_from_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(prefix="corners_predict_artifacts_") as td:
        artifact_dir = Path(td)
        _write_base_corner_artifact_dir(artifact_dir)
        (artifact_dir / "model_config.json").write_text(
            json.dumps(
                {
                    "path_version": "totals_surface_neural_ladder_calibrated",
                    "neural_total_market_ladder_sidecar": "neural_total_market_ladder_bundle.pt",
                    "total_market_ladder_blend": {"c75": 1.0},
                }
            ),
            encoding="utf-8",
        )
        (artifact_dir / "neural_total_market_ladder_bundle.pt").write_text("stub", encoding="utf-8")
        monkeypatch.setattr(
            "src.modeling.v2.families.corners.predict_corners.load_neural_total_market_ladder_bundle",
            lambda path: {"ladder_loaded_from": path.name},
        )

        models, _, _, _, path_version, league_regime = load_artifacts(artifact_dir)

        assert path_version == "totals_surface_neural_ladder_calibrated"
        assert models["neural_total_market_ladder"] == {"ladder_loaded_from": "neural_total_market_ladder_bundle.pt"}
        assert models["total_market_ladder_blend"]["c75"] == 1.0
        assert league_regime == {}