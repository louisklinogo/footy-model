from __future__ import annotations

import pandas as pd
import pytest

from src.modeling.layer2_markets import market_outcome_calibrator as legacy_calibrator
from src.modeling.v2.data.build_pit_dataset import build_pit_frame, validate_pit_leakage


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


def test_build_pit_frame_requests_asof_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, int | None] = {"prediction_lead_hours": None}

    def fake_fetch_dataset(prediction_lead_hours: int | None = None) -> pd.DataFrame:
        captured["prediction_lead_hours"] = prediction_lead_hours
        return pd.DataFrame(
            [
                {
                    "fixture_id": 1,
                    "match_datetime_utc": "2026-03-10T18:00:00Z",
                    "home_goals": 1,
                    "away_goals": 0,
                }
            ]
        )

    monkeypatch.setattr(
        "src.modeling.v2.data.build_pit_dataset.legacy_calibrator.fetch_dataset",
        fake_fetch_dataset,
    )
    monkeypatch.setattr(
        "src.modeling.v2.data.build_pit_dataset.legacy_calibrator.add_targets_and_derived",
        lambda df: df.assign(target_o15=0),
    )

    frame = build_pit_frame(6)

    assert captured["prediction_lead_hours"] == 6
    assert frame.loc[0, "prediction_time_utc"].isoformat() == "2026-03-10T12:00:00+00:00"


def test_fetch_dataset_caps_lambda_predictions_by_prediction_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {"query": None, "closed": False}

    class DummyConn:
        def close(self) -> None:
            captured["closed"] = True

    def fake_read_sql(query: str, conn: object) -> pd.DataFrame:
        captured["query"] = query
        return pd.DataFrame()

    monkeypatch.setattr(legacy_calibrator, "connect_db", lambda: DummyConn())
    monkeypatch.setattr(legacy_calibrator.pd, "read_sql", fake_read_sql)

    legacy_calibrator.fetch_dataset(prediction_lead_hours=6)

    query = captured["query"]
    assert isinstance(query, str)
    cutoff = "AND p.created_at <= (f.match_datetime_utc - INTERVAL '6 hours')"
    assert query.count(cutoff) == 4
    assert captured["closed"] is True


def test_fetch_dataset_leaves_lambda_predictions_uncapped_without_prediction_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {"query": None}

    class DummyConn:
        def close(self) -> None:
            return None

    def fake_read_sql(query: str, conn: object) -> pd.DataFrame:
        captured["query"] = query
        return pd.DataFrame()

    monkeypatch.setattr(legacy_calibrator, "connect_db", lambda: DummyConn())
    monkeypatch.setattr(legacy_calibrator.pd, "read_sql", fake_read_sql)

    legacy_calibrator.fetch_dataset()

    query = captured["query"]
    assert isinstance(query, str)
    assert "AND p.created_at <=" not in query

