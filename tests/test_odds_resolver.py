from __future__ import annotations

from datetime import datetime, timezone

from src.betting.odds_resolver import resolve_odds_for_market


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def test_odds_resolver_prefers_latest_pre_match() -> None:
    kickoff = _dt("2026-01-02T18:00:00Z")
    odds_rows = [
        {
            "provider": "sofascore",
            "snapshot_type": "closing",
            "snapshot_time_utc": _dt("2026-01-02T17:55:00Z"),
            "market_code": "ou",
            "line_num": 1.5,
            "odds_json": {"prices_latest": {"over": 1.77, "under": 2.0}},
        },
        {
            "provider": "sofascore",
            "snapshot_type": "latest_pre_match",
            "snapshot_time_utc": _dt("2026-01-02T17:40:00Z"),
            "market_code": "ou",
            "line_num": 1.5,
            "odds_json": {"prices_latest": {"over": 1.9, "under": 1.9}},
        },
        {
            "provider": "sofascore",
            "snapshot_type": "latest_pre_match",
            "snapshot_time_utc": _dt("2026-01-02T18:02:00Z"),
            "market_code": "ou",
            "line_num": 1.5,
            "odds_json": {"prices_latest": {"over": 1.55, "under": 2.4}},
        },
    ]

    resolved = resolve_odds_for_market(kickoff, odds_rows, "o15")

    assert resolved.odds_used == 1.9
    assert resolved.snapshot_type == "latest_pre_match"
    assert resolved.snapshot_time_utc is not None
    assert resolved.snapshot_time_utc <= kickoff
    assert resolved.fallback_used is False


def test_odds_resolver_falls_back_eh_to_ah() -> None:
    kickoff = _dt("2026-01-02T18:00:00Z")
    odds_rows = [
        {
            "provider": "sofascore",
            "snapshot_type": "latest_pre_match",
            "snapshot_time_utc": _dt("2026-01-02T17:30:00Z"),
            "market_code": "ah",
            "line_num": -1.5,
            "odds_json": {"prices_latest": {"home": 2.05, "away": 1.8}},
        }
    ]

    resolved = resolve_odds_for_market(kickoff, odds_rows, "eh_h1")

    assert resolved.odds_used == 2.05
    assert isinstance(resolved.fallback_used, str)
    assert "selection_fallback" in str(resolved.fallback_used)


def test_odds_resolver_away_ah_uses_minus_line_first() -> None:
    kickoff = _dt("2026-01-02T18:00:00Z")
    odds_rows = [
        {
            "provider": "sofascore",
            "snapshot_type": "latest_pre_match",
            "snapshot_time_utc": _dt("2026-01-02T17:30:00Z"),
            "market_code": "ah",
            "line_num": -0.5,
            "odds_json": {"prices_latest": {"away": 3.6, "home": 1.27}},
        },
        {
            "provider": "sofascore",
            "snapshot_type": "latest_pre_match",
            "snapshot_time_utc": _dt("2026-01-02T17:29:00Z"),
            "market_code": "ah",
            "line_num": 0.5,
            "odds_json": {"prices_latest": {"away": 1.8, "home": 1.94}},
        },
    ]

    resolved = resolve_odds_for_market(kickoff, odds_rows, "ah_a05")

    assert resolved.odds_used == 3.6
    assert resolved.line_num == -0.5
    assert resolved.fallback_used is False


def test_odds_resolver_supports_canonical_eh_draw_selection() -> None:
    kickoff = _dt("2026-01-02T18:00:00Z")
    odds_rows = [
        {
            "provider": "sofascore",
            "snapshot_type": "latest_pre_match",
            "snapshot_time_utc": _dt("2026-01-02T17:30:00Z"),
            "market_code": "eh",
            "line_num": -1.0,
            "odds_json": {"prices_latest": {"home": 3.7, "draw": 3.8, "away": 1.8}},
        }
    ]

    resolved = resolve_odds_for_market(kickoff, odds_rows, "eh3_0_1_draw")

    assert resolved.odds_used == 3.8
    assert resolved.odds_field == "draw"
    assert resolved.line_num == -1.0


def test_odds_resolver_supports_team_corners_aliases() -> None:
    kickoff = _dt("2026-01-02T18:00:00Z")
    odds_rows = [
        {
            "provider": "sofascore",
            "snapshot_type": "latest_pre_match",
            "snapshot_time_utc": _dt("2026-01-02T17:30:00Z"),
            "market_code": "corners_home_ou",
            "line_num": 2.5,
            "odds_json": {"prices_latest": {"over": 1.86, "under": 1.96}},
        }
    ]

    resolved = resolve_odds_for_market(kickoff, odds_rows, "hc25")

    assert resolved.odds_used == 1.86
    assert resolved.odds_field == "over"

def test_odds_resolver_ho15_no_fallback_to_25() -> None:
    kickoff = _dt("2026-01-02T18:00:00Z")
    odds_rows = [
        {
            "provider": "sofascore",
            "snapshot_type": "latest_pre_match",
            "snapshot_time_utc": _dt("2026-01-02T17:30:00Z"),
            "market_code": "home_ou",
            "line_num": 2.5,
            "odds_json": {"prices_latest": {"over": 2.1, "under": 1.7}},
        }
    ]

    resolved = resolve_odds_for_market(kickoff, odds_rows, "ho15")

    assert resolved.odds_used is None
    assert resolved.fallback_used is False

