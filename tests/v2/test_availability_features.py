from __future__ import annotations

import pandas as pd
import pytest

from src.modeling.availability_features import (
    AvailabilityFeatureConfig,
    availability_feature_select_and_join,
)
from src.modeling.evaluation import predict_market_outcomes_fixtures_first as fixtures_first
from src.modeling.layer2_markets import market_outcome_calibrator as legacy_calibrator


def test_availability_feature_sql_uses_prediction_cutoff_and_role_counts() -> None:
    select_sql, join_sql = availability_feature_select_and_join(
        fixture_alias="f",
        match_time_expr="f.match_datetime_utc",
        cutoff_expr="(f.match_datetime_utc - INTERVAL '6 hours')",
        config=AvailabilityFeatureConfig(True, "event_recorded_at", True),
    )

    assert "home_availability_known" in select_sql
    assert "home_lineup_freshness_hours" in select_sql
    assert "player_availability pa" in join_sql
    assert "LEFT JOIN players pl" in join_sql
    assert "upper(COALESCE(pl.position, '')) IN ('D', 'DEF', 'DEFENDER')" in join_sql
    assert "pa.event_recorded_at <= (f.match_datetime_utc - INTERVAL '6 hours')" in join_sql


def test_availability_feature_sql_falls_back_when_table_or_timing_missing() -> None:
    select_sql, join_sql = availability_feature_select_and_join(
        fixture_alias="f",
        match_time_expr="f.match_datetime_utc",
        cutoff_expr="f.match_datetime_utc",
        config=AvailabilityFeatureConfig(False, None, False),
    )

    assert join_sql == ""
    assert "0::int AS home_availability_known" in select_sql
    assert "NULL::double precision AS home_availability_freshness_hours" in select_sql
    assert "NULL::timestamptz AS home_availability_last_seen_utc" in select_sql


def test_fetch_dataset_includes_availability_cutoff(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {"query": None, "closed": False}

    class DummyConn:
        def close(self) -> None:
            captured["closed"] = True

    def fake_read_sql(query: str, conn: object) -> pd.DataFrame:
        captured["query"] = query
        return pd.DataFrame()

    monkeypatch.setattr(legacy_calibrator, "connect_db", lambda: DummyConn())
    monkeypatch.setattr(
        legacy_calibrator,
        "resolve_availability_feature_config",
        lambda conn: AvailabilityFeatureConfig(True, "event_recorded_at", True),
    )
    monkeypatch.setattr(legacy_calibrator.pd, "read_sql", fake_read_sql)

    legacy_calibrator.fetch_dataset(prediction_lead_hours=6)

    query = captured["query"]
    assert isinstance(query, str)
    assert "home_availability_known" in query
    assert "home_rolling_errors_lead_to_shot" in query
    assert "away_rolling_tackles_pct" in query
    assert "home_rolling_rest_days" in query
    assert "away_fidelity_score" in query
    assert "player_availability pa" in query
    assert "pa.event_recorded_at <= (f.match_datetime_utc - INTERVAL '6 hours')" in query
    assert captured["closed"] is True


def test_fetch_candidate_fixtures_includes_availability_features(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {"query": None, "params": None, "closed": False}

    class DummyConn:
        def close(self) -> None:
            captured["closed"] = True

    def fake_read_sql(query: str, conn: object, params: tuple[object, ...]) -> pd.DataFrame:
        captured["query"] = query
        captured["params"] = params
        return pd.DataFrame()

    monkeypatch.setattr(fixtures_first, "connect_db", lambda: DummyConn())
    monkeypatch.setattr(
        fixtures_first,
        "resolve_availability_feature_config",
        lambda conn: AvailabilityFeatureConfig(True, "event_recorded_at", True),
    )
    monkeypatch.setattr(fixtures_first.pd, "read_sql", fake_read_sql)

    fixtures_first.fetch_candidate_fixtures(days=3, league="EPL", limit=10, backfill_days=None)

    query = captured["query"]
    assert isinstance(query, str)
    assert "f.home_team_id" in query
    assert "home_lineup_known" in query
    assert "home_rolling_errors_lead_to_shot" in query
    assert "away_rolling_tackles_pct" in query
    assert "home_rolling_rest_days" in query
    assert "away_fidelity_score" in query
    assert "pa.event_recorded_at <= f.match_datetime_utc" in query
    assert captured["params"] == (3, "EPL", 10)
    assert captured["closed"] is True