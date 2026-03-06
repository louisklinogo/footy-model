from __future__ import annotations

import pandas as pd

from src.modeling.v2.data.build_pit_dataset import validate_pit_leakage


def test_validate_pit_leakage_passes_for_valid_timestamps() -> None:
    frame = pd.DataFrame(
        [
            {
                "fixture_id": 1,
                "match_datetime_utc": "2026-03-10T18:00:00Z",
                "prediction_time_utc": "2026-03-10T12:00:00Z",
                "odds_snapshot_time_utc": "2026-03-10T11:45:00Z",
            }
        ]
    )
    report = validate_pit_leakage(frame)
    assert report["status"] == "passed"
    assert report["violations"] == []


def test_validate_pit_leakage_flags_post_prediction_odds() -> None:
    frame = pd.DataFrame(
        [
            {
                "fixture_id": 2,
                "match_datetime_utc": "2026-03-10T18:00:00Z",
                "prediction_time_utc": "2026-03-10T12:00:00Z",
                "odds_snapshot_time_utc": "2026-03-10T12:15:00Z",
            }
        ]
    )
    report = validate_pit_leakage(frame)
    assert report["status"] == "failed"
    codes = {row["code"] for row in report["violations"]}
    assert "odds_snapshot_after_prediction_time" in codes


def test_validate_pit_leakage_flags_prediction_after_match() -> None:
    frame = pd.DataFrame(
        [
            {
                "fixture_id": 3,
                "match_datetime_utc": "2026-03-10T18:00:00Z",
                "prediction_time_utc": "2026-03-10T19:00:00Z",
                "odds_snapshot_time_utc": "2026-03-10T17:30:00Z",
            }
        ]
    )
    report = validate_pit_leakage(frame)
    assert report["status"] == "failed"
    codes = {row["code"] for row in report["violations"]}
    assert "prediction_time_after_match_time" in codes

