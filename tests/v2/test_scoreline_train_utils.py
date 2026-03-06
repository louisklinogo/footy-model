from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.modeling.v2.families.scoreline.train_scoreline import (
    _json_default,
    _load_training_frame,
    _aggregate_walkforward_quality,
    _score_matrix_independent_poisson,
    _select_features,
)


def test_score_matrix_independent_poisson_normalizes() -> None:
    mat = _score_matrix_independent_poisson(1.4, 1.1, 10)
    assert mat.shape == (11, 11)
    assert abs(float(mat.sum()) - 1.0) < 1e-9
    assert float(mat.min()) >= 0.0


def test_select_features_raises_when_required_missing() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_scoreline_contract_") as td:
        contract = Path(td) / "scoreline.yaml"
        contract.write_text(
            "\n".join(
                [
                    "family: scoreline",
                    "required_features:",
                    "  - home_rolling_xg",
                    "  - away_rolling_xg",
                    "optional_features:",
                    "  - odds_over_15",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        frame = pd.DataFrame([{"home_rolling_xg": 1.2, "odds_over_15": 1.8}])
        with pytest.raises(RuntimeError):
            _select_features(frame, contract)


def test_select_features_includes_optional_when_present() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_scoreline_contract_") as td:
        contract = Path(td) / "scoreline.yaml"
        contract.write_text(
            "\n".join(
                [
                    "family: scoreline",
                    "required_features:",
                    "  - home_rolling_xg",
                    "optional_features:",
                    "  - odds_over_15",
                    "missingness_indicators:",
                    "  - odds_over_15_is_missing",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        frame = pd.DataFrame(
            [
                {
                    "home_rolling_xg": 1.2,
                    "odds_over_15": 1.8,
                    "odds_over_15_is_missing": 0,
                }
            ]
        )
        selected = _select_features(frame, contract)
        assert selected == [
            "home_rolling_xg",
            "odds_over_15",
            "odds_over_15_is_missing",
        ]


def test_aggregate_walkforward_quality_summarizes_markets() -> None:
    summary = {
        "o15": {"auc_mean": 0.61, "brier_mean": 0.19, "n_total": 120},
        "u35": {"auc_mean": 0.58, "brier_mean": 0.22, "n_total": 140},
    }
    agg = _aggregate_walkforward_quality(summary)
    assert agg["markets_used"] == 2
    assert abs(float(agg["auc_mean"]) - 0.595) < 1e-9
    assert abs(float(agg["brier_mean"]) - 0.205) < 1e-9
    assert agg["n_total"] == 260


def test_load_training_frame_reads_csv_dataset_path() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_scoreline_dataset_") as td:
        dataset_path = Path(td) / "pit_dataset.csv"
        expected = pd.DataFrame(
            [
                {
                    "fixture_id": 1,
                    "match_datetime_utc": "2026-01-01T00:00:00Z",
                    "prediction_time_utc": "2025-12-31T18:00:00Z",
                }
            ]
        )
        expected.to_csv(dataset_path, index=False)

        frame, source = _load_training_frame(dataset_path)

        assert source == str(dataset_path)
        assert frame.to_dict(orient="records") == expected.to_dict(orient="records")


def test_load_training_frame_uses_legacy_fetch_when_dataset_path_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = pd.DataFrame(
        [
            {
                "fixture_id": 2,
                "match_datetime_utc": "2026-02-01T00:00:00Z",
                "prediction_time_utc": "2026-01-31T18:00:00Z",
            }
        ]
    )

    monkeypatch.setattr(
        "src.modeling.v2.families.scoreline.train_scoreline.legacy_calibrator.fetch_dataset",
        lambda: expected.copy(),
    )

    frame, source = _load_training_frame(None)

    assert source == "legacy_fetch_dataset"
    assert frame.to_dict(orient="records") == expected.to_dict(orient="records")


def test_load_training_frame_rejects_failed_pit_validation_report() -> None:
    with tempfile.TemporaryDirectory(prefix="v2_scoreline_dataset_") as td:
        dataset_path = Path(td) / "pit_dataset.csv"
        dataset_path.write_text("fixture_id\n1\n", encoding="utf-8")
        (Path(td) / "pit_validation_report.json").write_text(
            '{"status": "failed", "violations": [{"code": "late_odds"}]}',
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError, match="PIT dataset failed validation"):
            _load_training_frame(dataset_path)


def test_json_default_serializes_numpy_scalars() -> None:
    assert _json_default(np.int64(7)) == 7
    assert _json_default(np.float64(0.25)) == 0.25
