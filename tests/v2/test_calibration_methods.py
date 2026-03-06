from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from src.modeling.v2.calibration.methods import evaluate_market_calibration


def _market_frame(rows_per_class: int = 60) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2026, 1, 1, tzinfo=UTC)
    fixture_id = 1
    for block in range(2):
        for _ in range(rows_per_class):
            rows.append(
                {
                    "market": "o15",
                    "fixture_id": fixture_id,
                    "match_datetime_utc": (start + timedelta(days=fixture_id)).isoformat(),
                    "y_true": 0,
                    "p_model": 0.4,
                }
            )
            fixture_id += 1
        for _ in range(rows_per_class):
            rows.append(
                {
                    "market": "o15",
                    "fixture_id": fixture_id,
                    "match_datetime_utc": (start + timedelta(days=fixture_id)).isoformat(),
                    "y_true": 1,
                    "p_model": 0.6,
                }
            )
            fixture_id += 1
    return pd.DataFrame(rows)


def test_evaluate_market_calibration_selects_non_identity_when_probabilities_are_sharpenable() -> None:
    report, calibrator = evaluate_market_calibration(
        _market_frame(),
        fit_fraction=0.5,
        min_fit_rows=80,
        min_eval_rows=80,
        max_auc_drop=0.01,
    )
    assert report["status"] == "calibrated"
    assert report["selected_method"] in {"sigmoid", "isotonic"}
    assert calibrator is not None
    assert report["calibrated_eval_metrics"]["brier"] < report["raw_eval_metrics"]["brier"]


def test_evaluate_market_calibration_returns_insufficient_rows_when_holdout_is_too_small() -> None:
    report, calibrator = evaluate_market_calibration(
        _market_frame(rows_per_class=10),
        fit_fraction=0.5,
        min_fit_rows=30,
        min_eval_rows=30,
        max_auc_drop=0.01,
    )
    assert report["status"] == "insufficient_rows"
    assert calibrator is None