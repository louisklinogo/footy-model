from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pandas as pd
import pytest

from src.modeling.v2.families.corners import train_corners
from src.modeling.v2.families.corners.train_corners import (
    _aggregate_walkforward_quality,
    _select_features,
)


def test_corners_select_features_raises_when_required_missing() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_corners_contract_") as td:
        contract = Path(td) / "corners.yaml"
        contract.write_text(
            "\n".join(
                [
                    "family: corners",
                    "required_features:",
                    "  - home_rolling_corners",
                    "  - away_rolling_corners",
                    "optional_features:",
                    "  - odds_c95_over",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        frame = pd.DataFrame([{"home_rolling_corners": 5.1, "odds_c95_over": 1.9}])
        with pytest.raises(RuntimeError):
            _select_features(frame, contract)


def test_corners_select_features_includes_optional_when_available() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_corners_contract_") as td:
        contract = Path(td) / "corners.yaml"
        contract.write_text(
            "\n".join(
                [
                    "family: corners",
                    "required_features:",
                    "  - home_rolling_corners",
                    "optional_features:",
                    "  - odds_c95_over",
                    "missingness_indicators:",
                    "  - odds_c95_over_is_missing",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        frame = pd.DataFrame(
            [
                {
                    "home_rolling_corners": 5.1,
                    "odds_c95_over": 1.9,
                    "odds_c95_over_is_missing": 0,
                }
            ]
        )
        selected = _select_features(frame, contract)
        assert selected == [
            "home_rolling_corners",
            "odds_c95_over",
            "odds_c95_over_is_missing",
        ]


def test_corners_aggregate_walkforward_quality_summarizes_markets() -> None:
    summary = {
        "c85": {"auc_mean": 0.57, "brier_mean": 0.24, "n_total": 200},
        "c95": {"auc_mean": 0.55, "brier_mean": 0.25, "n_total": 180},
    }
    agg = _aggregate_walkforward_quality(summary)
    assert agg["markets_used"] == 2
    assert abs(float(agg["auc_mean"]) - 0.56) < 1e-9
    assert abs(float(agg["brier_mean"]) - 0.245) < 1e-9
    assert agg["n_total"] == 380


def test_corners_select_features_excludes_disabled_entries() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_corners_contract_") as td:
        contract = Path(td) / "corners.yaml"
        contract.write_text(
            "\n".join(
                [
                    "family: corners",
                    "required_features:",
                    "  - home_rolling_corners",
                    "  - away_rolling_corners",
                    "optional_features:",
                    "  - home_rolling_crosses",
                    "disabled_features:",
                    "  - home_rolling_crosses",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        frame = pd.DataFrame(
            [
                {
                    "home_rolling_corners": 5.1,
                    "away_rolling_corners": 4.8,
                    "home_rolling_crosses": 14.2,
                }
            ]
        )
        selected = _select_features(frame, contract)
        assert selected == ["home_rolling_corners", "away_rolling_corners"]


def test_load_training_frame_reads_pit_dataset_and_enforces_validation() -> None:
    with tempfile.TemporaryDirectory(prefix="corners_pit_dataset_") as td:
        root = Path(td)
        dataset_path = root / "dataset.csv"
        pd.DataFrame(
            [{"fixture_id": 1, "prediction_time_utc": "2026-03-01T00:00:00+00:00"}]
        ).to_csv(dataset_path, index=False)
        (root / "pit_validation_report.json").write_text(
            json.dumps({"status": "passed"}), encoding="utf-8"
        )

        frame, source = train_corners._load_training_frame(dataset_path)

        assert source == str(dataset_path)
        assert list(frame["fixture_id"]) == [1]


def test_load_training_frame_raises_when_validation_failed() -> None:
    with tempfile.TemporaryDirectory(prefix="corners_pit_invalid_") as td:
        root = Path(td)
        dataset_path = root / "dataset.csv"
        pd.DataFrame([{"fixture_id": 1}]).to_csv(dataset_path, index=False)
        (root / "pit_validation_report.json").write_text(
            json.dumps({"status": "failed"}), encoding="utf-8"
        )

        with pytest.raises(RuntimeError, match="PIT dataset failed validation"):
            train_corners._load_training_frame(dataset_path)
