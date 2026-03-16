from src.ingest.ingest_external_context import (
    _build_fixture_scope_clause,
    _extract_match_form_metrics,
    _extract_player_attributes_metrics,
    _extract_standing_metrics,
    _form_points,
    _form_sequence_to_text,
    _summarize_h2h_results,
    _summarize_performance_graph,
)


def test_form_helpers() -> None:
    form = ["W", "D", "L", "W", "W"]
    assert _form_sequence_to_text(form) == "WDLWW"
    assert _form_points(form) == 10


def test_extract_standing_metrics() -> None:
    row = {
        "position": 3,
        "points": 61,
        "matches": 30,
        "wins": 18,
        "draws": 7,
        "losses": 5,
        "scoresFor": 55,
        "scoresAgainst": 28,
        "scoreDiffFormatted": "+27",
    }
    metrics = _extract_standing_metrics(row)
    assert metrics["rank"] == 3
    assert metrics["goal_diff"] == 27
    assert metrics["goals_for"] == 55


def test_summarize_performance_graph() -> None:
    payload = {
        "graphData": [
            {"points": 1, "goalDiff": -1},
            {"points": 3, "goalDiff": 2},
            {"points": 0, "goalDiff": -2},
        ]
    }
    summary = _summarize_performance_graph(payload)
    assert summary["performance_graph_samples"] == 3
    assert summary["performance_graph_points_avg"] == 4 / 3
    assert summary["performance_graph_goal_diff_avg"] == -1 / 3


def test_extract_match_form_metrics_and_h2h_summary() -> None:
    form_payload = {
        "homeTeam": {"avgRating": "6.9", "position": 5, "value": "10", "form": ["W", "W", "D", "L", "W"]},
        "awayTeam": {"avgRating": "6.7", "position": 9, "value": "7", "form": ["L", "D", "W", "L", "D"]},
    }
    metrics = _extract_match_form_metrics(form_payload)
    assert metrics["home_form_points_last5"] == 10
    assert metrics["away_form_sequence"] == "LDWLD"

    h2h_payload = {
        "events": [
            {
                "homeTeam": {"id": 100},
                "awayTeam": {"id": 200},
                "homeScore": {"current": 2},
                "awayScore": {"current": 1},
            },
            {
                "homeTeam": {"id": 200},
                "awayTeam": {"id": 100},
                "homeScore": {"current": 0},
                "awayScore": {"current": 0},
            },
            {
                "homeTeam": {"id": 200},
                "awayTeam": {"id": 100},
                "homeScore": {"current": 1},
                "awayScore": {"current": 3},
            },
        ]
    }
    summary = _summarize_h2h_results(h2h_payload, "100", "200")
    assert summary["h2h_home_wins_last_n"] == 2
    assert summary["h2h_draws_last_n"] == 1
    assert summary["h2h_away_wins_last_n"] == 0
    assert summary["h2h_matches_count"] == 3


def test_extract_player_attributes_metrics() -> None:
    payload = {
        "playerAttributeOverviews": [
            {
                "attacking": 72,
                "technical": 68,
                "tactical": 58,
                "defending": 30,
                "creativity": 64,
                "position": "F",
            }
        ]
    }
    metrics = _extract_player_attributes_metrics(payload, "M", 12500000)
    assert metrics["position_group"] == "F"
    assert metrics["attribute_attacking"] == 72
    assert metrics["market_value_euro_snapshot"] == 12500000


def test_build_fixture_scope_clause() -> None:
    sql, params, order_direction = _build_fixture_scope_clause("scheduled", 0, 72, "fx")
    assert "fx.status = %s" in sql
    assert "fx.match_datetime_utc >= NOW()" in sql
    assert params == ["scheduled", 0, 72]
    assert order_direction == "ASC"

    sql, params, order_direction = _build_fixture_scope_clause("ft", 0, 72, "fx")
    assert sql == "fx.status = %s"
    assert params == ["ft"]
    assert order_direction == "DESC"

    sql, params, order_direction = _build_fixture_scope_clause("all", 0, 72, "fx")
    assert sql == ""
    assert params == []
    assert order_direction == "DESC"
