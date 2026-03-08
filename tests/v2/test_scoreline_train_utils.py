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
from src.modeling.v2.families.scoreline.total_intensity import (
    apply_total_intensity_correction,
    fit_total_intensity_correction,
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


def test_select_features_supports_scoreline_live_parity_plus_contract() -> None:
    contract = Path("model_v2/feature_contracts/experiments/scoreline_live_parity_plus.yaml")
    frame = pd.DataFrame(
        [
            {
                "home_sample_size": 10,
                "away_sample_size": 11,
                "home_rolling_xg": 1.6,
                "home_rolling_xg_against": 1.0,
                "away_rolling_xg": 1.2,
                "away_rolling_xg_against": 1.1,
                "home_rolling_sot": 4.8,
                "away_rolling_sot": 4.1,
                "lambda_home_l1": 1.5,
                "lambda_away_l1": 1.1,
                "adj_lambda_home_final": 1.55,
                "adj_lambda_away_final": 1.05,
                "xg_net_diff": 0.5,
                "goal_diff_proxy": 0.4,
                "rule_fired_home": 1,
                "odds_over_15": 1.55,
                "implied_over15": 0.645,
                "home_availability_known": 1,
                "home_lineup_known": 1,
                "adj_lambda_home_final_is_missing": 0,
            }
        ]
    )

    selected = _select_features(frame, contract)

    assert "adj_lambda_home_final" in selected
    assert "adj_lambda_away_final" in selected
    assert "rule_fired_home" in selected
    assert "odds_over_15" in selected
    assert "implied_over15" in selected
    assert "home_availability_known" in selected
    assert "home_lineup_known" in selected
    assert "adj_lambda_home_final_is_missing" in selected


def test_select_features_supports_scoreline_live_parity_strict_contract() -> None:
    contract = Path("model_v2/feature_contracts/experiments/scoreline_live_parity_strict.yaml")
    frame = pd.DataFrame(
        [
            {
                "home_sample_size": 10,
                "away_sample_size": 11,
                "home_rolling_xg": 1.6,
                "home_rolling_xg_against": 1.0,
                "away_rolling_xg": 1.2,
                "away_rolling_xg_against": 1.1,
                "home_rolling_sot": 4.8,
                "away_rolling_sot": 4.1,
                "lambda_home_l1": 1.5,
                "lambda_away_l1": 1.1,
                "adj_lambda_home_final": 1.55,
                "adj_lambda_away_final": 1.05,
                "xg_net_diff": 0.5,
                "goal_diff_proxy": 0.4,
                "rule_fired_home": 1,
                "odds_over_15": 1.55,
                "implied_over15": 0.645,
                "home_availability_known": 1,
                "home_lineup_known": 1,
                "home_recent_xg_mean_5": 1.4,
                "adj_lambda_home_final_is_missing": 0,
            }
        ]
    )

    selected = _select_features(frame, contract)

    assert "adj_lambda_home_final" in selected
    assert "adj_lambda_away_final" in selected
    assert "rule_fired_home" in selected
    assert "odds_over_15" in selected
    assert "implied_over15" in selected
    assert "home_availability_known" not in selected
    assert "home_lineup_known" not in selected
    assert "home_recent_xg_mean_5" not in selected
    assert "adj_lambda_home_final_is_missing" in selected


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


def test_fit_total_intensity_correction_is_identity_when_totals_match() -> None:
    payload = fit_total_intensity_correction(
        actual_totals=np.array([2.0, 3.0, 4.0]),
        pred_home=np.array([1.0, 1.5, 2.0]),
        pred_away=np.array([1.0, 1.5, 2.0]),
    )

    assert payload["method"] == "mean_total_ratio_v1"
    assert abs(float(payload["multiplier"]) - 1.0) < 1e-9


def test_fit_total_intensity_correction_snaps_de_minimis_multiplier_to_identity() -> None:
    payload = fit_total_intensity_correction(
        actual_totals=np.array([2.999996869809184] * 4),
        pred_home=np.array([1.8] * 4),
        pred_away=np.array([1.2] * 4),
    )

    assert float(payload["multiplier"]) == 1.0
    assert float(payload["corrected_total_goals_mean_pred"]) == pytest.approx(3.0)


def test_fit_total_intensity_correction_moves_up_when_raw_totals_underpredict() -> None:
    payload = fit_total_intensity_correction(
        actual_totals=np.array([3.0, 3.0, 3.0]),
        pred_home=np.array([1.0, 1.0, 1.0]),
        pred_away=np.array([1.0, 1.0, 1.0]),
    )

    assert float(payload["multiplier"]) > 1.0


def test_fit_total_intensity_correction_moves_down_when_raw_totals_overpredict() -> None:
    payload = fit_total_intensity_correction(
        actual_totals=np.array([1.0, 1.0, 1.0]),
        pred_home=np.array([1.0, 1.0, 1.0]),
        pred_away=np.array([1.0, 1.0, 1.0]),
    )

    assert float(payload["multiplier"]) < 1.0


def test_fit_total_intensity_correction_keeps_material_multiplier_changes() -> None:
    payload = fit_total_intensity_correction(
        actual_totals=np.array([2.97, 2.97, 2.97]),
        pred_home=np.array([1.8, 1.8, 1.8]),
        pred_away=np.array([1.2, 1.2, 1.2]),
    )

    assert float(payload["multiplier"]) == pytest.approx(0.99)


def test_apply_total_intensity_correction_preserves_share() -> None:
    home, away = apply_total_intensity_correction(
        lambda_home=np.array([1.2, 2.4]),
        lambda_away=np.array([0.8, 1.6]),
        correction={"multiplier": 1.1},
    )

    assert np.allclose(home / (home + away), np.array([0.6, 0.6]))
    assert np.allclose(home + away, np.array([2.2, 4.4]))


def test_json_default_serializes_numpy_scalars() -> None:
    assert _json_default(np.int64(7)) == 7
    assert _json_default(np.float64(0.25)) == 0.25
