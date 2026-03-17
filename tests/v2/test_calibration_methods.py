from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from src.modeling.v2.calibration.methods import (
    apply_binary_calibrator,
    evaluate_market_calibration,
)


def test_one_sided_anchor_shrink_reduces_low_base_rate_overconfidence() -> None:
    calibrator = {"method": "one_sided_anchor", "anchor": 0.2, "alpha": 0.5}
    values = np.array([0.05, 0.20, 0.40, 0.90], dtype=float)
    adjusted = apply_binary_calibrator(calibrator, values)
    np.testing.assert_allclose(adjusted[:2], values[:2], atol=1e-9)
    np.testing.assert_allclose(adjusted[2:], np.array([0.30, 0.55]), atol=1e-9)


def test_one_sided_anchor_shrink_lifts_high_base_rate_underconfidence() -> None:
    calibrator = {"method": "one_sided_anchor", "anchor": 0.8, "alpha": 0.5}
    values = np.array([0.05, 0.40, 0.80, 0.95], dtype=float)
    adjusted = apply_binary_calibrator(calibrator, values)
    np.testing.assert_allclose(adjusted[2:], values[2:], atol=1e-9)
    np.testing.assert_allclose(adjusted[:2], np.array([0.425, 0.60]), atol=1e-9)


def test_evaluate_market_calibration_reports_one_sided_anchor_candidate() -> None:
    rows = []
    for idx in range(240):
        y_true = 1 if (idx % 5) == 0 else 0
        if idx < 120:
            p_model = 0.7 if y_true == 0 else 0.6
        else:
            p_model = 0.75 if y_true == 0 else 0.65
        rows.append(
            {
                "fixture_id": idx,
                "market": "ah2_home_m15",
                "y_true": y_true,
                "p_model": p_model,
                "match_datetime_utc": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=idx),
            }
        )
    frame = pd.DataFrame(rows)

    report, calibrator = evaluate_market_calibration(
        frame,
        fit_fraction=0.5,
        min_fit_rows=80,
        min_eval_rows=80,
        max_auc_drop=0.01,
    )

    candidate = report["candidate_metrics"]["one_sided_anchor"]
    assert candidate["anchor"] == 0.2
    assert 0.0 < float(candidate["alpha"]) <= 1.0
    assert float(candidate["log_loss"]) < float(report["raw_eval_metrics"]["log_loss"])
    assert calibrator is not None


def test_evaluate_market_calibration_limits_one_sided_anchor_to_target_markets() -> None:
    frame = _market_frame()
    report, _ = evaluate_market_calibration(
        frame,
        fit_fraction=0.5,
        min_fit_rows=80,
        min_eval_rows=80,
        max_auc_drop=0.01,
    )

    assert "one_sided_anchor" not in report["candidate_metrics"]


def _market_frame(rows_per_class: int = 60) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = datetime(2026, 1, 1, tzinfo=UTC)
    fixture_id = 1
    for _block in range(2):
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
    assert report["selected_method"] in {"sigmoid", "isotonic", "one_sided_anchor"}
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
