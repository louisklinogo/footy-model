from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from src.modeling.v2.families.scoreline import train_scoreline_o15_stacked


def _synthetic_dataset(rows: int = 120) -> pd.DataFrame:
    payload: list[dict[str, object]] = []
    for idx in range(rows):
        feature = float(idx % 12) / 10.0
        home_goals = int(feature >= 0.4) + int(feature >= 0.8)
        away_goals = int(feature >= 0.6)
        payload.append(
            {
                "fixture_id": idx + 1,
                "match_datetime_utc": f"2026-01-{(idx % 28) + 1:02d}T12:00:00Z",
                "prediction_time_utc": f"2026-01-{(idx % 28) + 1:02d}T06:00:00Z",
                "league_code": "L1" if idx % 2 == 0 else "L2",
                "feature_a": feature,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "target_o15": int((home_goals + away_goals) >= 2),
            }
        )
    return pd.DataFrame(payload)


def test_main_writes_o15_stacked_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_path = tmp_path / "pit_dataset.csv"
    output_dir = tmp_path / "artifacts"
    _synthetic_dataset().to_csv(dataset_path, index=False)

    monkeypatch.setattr(train_scoreline_o15_stacked.legacy_calibrator, "feature_columns", lambda: ["feature_a"])
    monkeypatch.setattr(
        train_scoreline_o15_stacked,
        "parse_args",
        lambda: argparse.Namespace(
            output_dir=output_dir,
            dataset_path=dataset_path,
            model_version="test_o15_stacked_v1",
            model_kind="gbm",
            scoreline_model_type="histgb_poisson",
            max_goals=6,
            max_rows=None,
            folds=3,
            min_fold_test_n=10,
        ),
    )

    train_scoreline_o15_stacked.main()

    holdout = pd.read_csv(output_dir / "holdout_predictions.csv")
    assert set(holdout.columns) >= {"market", "fixture_id", "y_true", "p_model", "scoreline_p_o15"}
    assert set(holdout["market"].unique()) == {"o15"}

    metadata = json.loads((output_dir / "artifact_metadata.json").read_text(encoding="utf-8"))
    assert metadata["family"] == "scoreline"
    assert metadata["model_version"] == "test_o15_stacked_v1"
    assert metadata["market"] == "o15"

    features = json.loads((output_dir / "features.json").read_text(encoding="utf-8"))
    assert "scoreline_p_o15" in features
    assert "scoreline_lambda_total" in features
    assert (output_dir / "stack_scoreline_home_goals_model.pkl").exists()
    assert (output_dir / "stack_scoreline_away_goals_model.pkl").exists()

    report = json.loads((output_dir / "training_report.json").read_text(encoding="utf-8"))
    assert report["train_rows_stacked"] < report["train_rows_total"]
    assert report["scoreline_oof_rows"] > 0


def test_main_raises_when_expected_feature_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_path = tmp_path / "pit_dataset.csv"
    _synthetic_dataset().drop(columns=["feature_a"]).to_csv(dataset_path, index=False)

    monkeypatch.setattr(train_scoreline_o15_stacked.legacy_calibrator, "feature_columns", lambda: ["feature_a"])
    monkeypatch.setattr(
        train_scoreline_o15_stacked,
        "parse_args",
        lambda: argparse.Namespace(
            output_dir=tmp_path / "artifacts",
            dataset_path=dataset_path,
            model_version="test_o15_stacked_v1",
            model_kind="gbm",
            scoreline_model_type="histgb_poisson",
            max_goals=6,
            max_rows=None,
            folds=3,
            min_fold_test_n=10,
        ),
    )

    with pytest.raises(RuntimeError, match="Feature contract mismatch"):
        train_scoreline_o15_stacked.main()