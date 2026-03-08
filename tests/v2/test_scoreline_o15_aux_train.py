from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from src.modeling.v2.families.scoreline import train_scoreline_o15_aux


def _synthetic_dataset(rows: int = 120) -> pd.DataFrame:
    payload: list[dict[str, object]] = []
    for idx in range(rows):
        feature = float(idx % 12) / 10.0
        payload.append(
            {
                "fixture_id": idx + 1,
                "match_datetime_utc": f"2026-01-{(idx % 28) + 1:02d}T12:00:00Z",
                "prediction_time_utc": f"2026-01-{(idx % 28) + 1:02d}T06:00:00Z",
                "league_code": "L1" if idx % 2 == 0 else "L2",
                "feature_a": feature,
                "target_o15": int(feature >= 0.6),
            }
        )
    return pd.DataFrame(payload)


def test_main_writes_o15_aux_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_path = tmp_path / "pit_dataset.csv"
    output_dir = tmp_path / "artifacts"
    _synthetic_dataset().to_csv(dataset_path, index=False)

    monkeypatch.setattr(
        train_scoreline_o15_aux.legacy_calibrator,
        "feature_columns",
        lambda: ["feature_a"],
    )
    monkeypatch.setattr(
        train_scoreline_o15_aux,
        "parse_args",
        lambda: argparse.Namespace(
            output_dir=output_dir,
            dataset_path=dataset_path,
            model_version="test_o15_aux_v1",
            model_kind="gbm",
            max_rows=None,
            folds=3,
            min_fold_test_n=10,
        ),
    )

    train_scoreline_o15_aux.main()

    holdout = pd.read_csv(output_dir / "holdout_predictions.csv")
    assert set(holdout.columns) >= {"market", "fixture_id", "y_true", "p_model", "league_code"}
    assert set(holdout["market"].unique()) == {"o15"}

    metadata = json.loads((output_dir / "artifact_metadata.json").read_text(encoding="utf-8"))
    assert metadata["family"] == "scoreline"
    assert metadata["model_version"] == "test_o15_aux_v1"
    assert metadata["market"] == "o15"

    holdout_metrics = json.loads((output_dir / "metrics_holdout.json").read_text(encoding="utf-8"))
    assert len(holdout_metrics) == 1
    assert holdout_metrics[0]["market"] == "o15"
    assert (output_dir / "o15_model.pkl").exists()


def test_main_raises_when_expected_feature_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_path = tmp_path / "pit_dataset.csv"
    _synthetic_dataset().drop(columns=["feature_a"]).to_csv(dataset_path, index=False)

    monkeypatch.setattr(
        train_scoreline_o15_aux.legacy_calibrator,
        "feature_columns",
        lambda: ["feature_a"],
    )
    monkeypatch.setattr(
        train_scoreline_o15_aux,
        "parse_args",
        lambda: argparse.Namespace(
            output_dir=tmp_path / "artifacts",
            dataset_path=dataset_path,
            model_version="test_o15_aux_v1",
            model_kind="gbm",
            max_rows=None,
            folds=3,
            min_fold_test_n=10,
        ),
    )

    with pytest.raises(RuntimeError, match="Feature contract mismatch"):
        train_scoreline_o15_aux.main()