from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pandas as pd
import pytest
from sklearn.linear_model import PoissonRegressor

from src.modeling.v2.families.anytime import train_anytime
from src.modeling.v2.families.anytime.train_anytime import (
    _aggregate_walkforward_quality,
    _build_regressor,
    _prepare_phase_targets,
    _select_features,
)


def test_anytime_select_features_raises_when_required_missing() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_anytime_contract_") as td:
        contract = Path(td) / "anytime.yaml"
        contract.write_text(
            "\n".join(
                [
                    "family: anytime",
                    "required_features:",
                    "  - lambda_home_l1",
                    "  - lambda_away_l1",
                    "optional_features:",
                    "  - rule_fired_home",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        frame = pd.DataFrame([{"lambda_home_l1": 1.2}])
        with pytest.raises(RuntimeError):
            _select_features(frame, contract)


def test_anytime_select_features_includes_optional_when_present() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_anytime_contract_") as td:
        contract = Path(td) / "anytime.yaml"
        contract.write_text(
            "\n".join(
                [
                    "family: anytime",
                    "required_features:",
                    "  - lambda_home_l1",
                    "optional_features:",
                    "  - rule_fired_home",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        frame = pd.DataFrame([{"lambda_home_l1": 1.2, "rule_fired_home": 1}])
        selected = _select_features(frame, contract)
        assert selected == ["lambda_home_l1", "rule_fired_home"]


def test_anytime_aggregate_walkforward_quality_summarizes_markets() -> None:
    summary = {
        "h_1up": {"auc_mean": 0.60, "brier_mean": 0.23, "n_total": 140},
        "a_1up": {"auc_mean": 0.58, "brier_mean": 0.24, "n_total": 130},
    }
    agg = _aggregate_walkforward_quality(summary)
    assert agg["markets_used"] == 2
    assert abs(float(agg["auc_mean"]) - 0.59) < 1e-9
    assert abs(float(agg["brier_mean"]) - 0.235) < 1e-9
    assert agg["n_total"] == 270


def test_anytime_select_features_excludes_disabled_entries() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_anytime_contract_") as td:
        contract = Path(td) / "anytime.yaml"
        contract.write_text(
            "\n".join(
                [
                    "family: anytime",
                    "required_features:",
                    "  - lambda_home_l1",
                    "  - lambda_away_l1",
                    "optional_features:",
                    "  - home_recent_xg_mean_5",
                    "disabled_features:",
                    "  - home_recent_xg_mean_5",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        frame = pd.DataFrame(
            [
                {
                    "lambda_home_l1": 1.2,
                    "lambda_away_l1": 1.1,
                    "home_recent_xg_mean_5": 1.4,
                }
            ]
        )
        selected = _select_features(frame, contract)
        assert selected == ["lambda_home_l1", "lambda_away_l1"]


def test_anytime_prepare_phase_targets_derives_second_half_goals() -> None:
    frame = pd.DataFrame(
        [
            {
                "home_goals": 3,
                "away_goals": 1,
                "home_goals_p1": 1,
                "away_goals_p1": 1,
            },
            {
                "home_goals": 2,
                "away_goals": 0,
                "home_goals_p1": None,
                "away_goals_p1": None,
            },
        ]
    )
    out = _prepare_phase_targets(frame)
    assert out.loc[0, "home_goals_p2"] == 2.0
    assert out.loc[0, "away_goals_p2"] == 0.0
    assert pd.isna(out.loc[1, "home_goals_p2"])
    assert pd.isna(out.loc[1, "away_goals_p2"])


def test_anytime_build_regressor_uses_custom_poisson_alpha() -> None:
    model = _build_regressor("poisson_glm", poisson_alpha=0.3)
    assert isinstance(model, PoissonRegressor)
    assert model.alpha == pytest.approx(0.3)


def test_load_training_frame_reads_pit_dataset_and_enforces_validation() -> None:
    with tempfile.TemporaryDirectory(prefix="anytime_pit_dataset_") as td:
        root = Path(td)
        dataset_path = root / "dataset.csv"
        pd.DataFrame(
            [{"fixture_id": 2, "prediction_time_utc": "2026-03-01T00:00:00+00:00"}]
        ).to_csv(dataset_path, index=False)
        (root / "pit_validation_report.json").write_text(
            json.dumps({"status": "passed"}), encoding="utf-8"
        )

        frame, source = train_anytime._load_training_frame(dataset_path)

        assert source == str(dataset_path)
        assert list(frame["fixture_id"]) == [2]


def test_load_training_frame_raises_when_validation_failed() -> None:
    with tempfile.TemporaryDirectory(prefix="anytime_pit_invalid_") as td:
        root = Path(td)
        dataset_path = root / "dataset.csv"
        pd.DataFrame([{"fixture_id": 2}]).to_csv(dataset_path, index=False)
        (root / "pit_validation_report.json").write_text(
            json.dumps({"status": "failed"}), encoding="utf-8"
        )

        with pytest.raises(RuntimeError, match="PIT dataset failed validation"):
            train_anytime._load_training_frame(dataset_path)
