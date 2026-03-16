from __future__ import annotations

from src.modeling.v2.audit_sofascore_db_reuse import (
    _feature_eligibility_registry,
    _recursive_key_hits,
)


def test_recursive_key_hits_finds_nested_sparse_field_aliases() -> None:
    payload = {
        "statistics": [
            {
                "groups": [
                    {
                        "statisticsItems": [
                            {"name": "Recovered balls", "homeValue": 31, "awayValue": 28},
                            {"name": "Yellow cards", "homeValue": 2, "awayValue": 3},
                        ]
                    }
                ]
            }
        ]
    }

    recoveries = _recursive_key_hits(payload, {"Recovered balls", "ballRecoveries"})
    yellows = _recursive_key_hits(payload, {"Yellow cards", "yellowCards"})

    assert recoveries == 1
    assert yellows == 1


def test_feature_eligibility_registry_marks_parser_gap_fields_ready_after_backfill() -> None:
    table_surface = [
        {"table_name": "fixture_stats_premium", "latest_timestamp": "2026-03-15T08:00:00+00:00"},
        {"table_name": "player_availability", "latest_timestamp": "2026-03-15T08:00:00+00:00"},
        {"table_name": "fixture_player_stats", "latest_timestamp": "2026-03-15T08:00:00+00:00"},
        {"table_name": "team_league_standings", "latest_timestamp": "2026-03-15T08:00:00+00:00"},
        {"table_name": "fixture_incident_lead_states", "latest_timestamp": "2026-03-15T08:00:00+00:00"},
    ]
    premium_coverage = [
        {
            "field_group": "ball_recoveries_surface",
            "source_table": "fixture_stats_premium",
            "columns": ["h_ball_recoveries", "a_ball_recoveries"],
            "total_rows": 100,
            "covered_rows": 0,
            "coverage_ratio": 0.0,
            "latest_timestamp": "2026-03-15T08:00:00+00:00",
        },
        {
            "field_group": "final_third_entries_surface",
            "source_table": "fixture_stats_premium",
            "columns": ["h_final_third_entries", "a_final_third_entries"],
            "total_rows": 100,
            "covered_rows": 40,
            "coverage_ratio": 0.4,
            "latest_timestamp": "2026-03-15T08:00:00+00:00",
        },
    ]
    sparse_diagnostics = [
        {
            "field_group": "ball_recoveries_surface",
            "covered_rows": 0,
            "payload_key_hits": 12,
            "diagnosis": "parser_gap_likely",
        }
    ]
    external_context_surface = [
        {
            "table_name": "team_external_context",
            "context_type": "standings_total",
            "provider_entities": 500,
            "canonical_entities": 500,
            "latest_ingested_at": "2026-03-16T00:42:00+00:00",
        },
        {
            "table_name": "team_external_context",
            "context_type": "standings_home",
            "provider_entities": 500,
            "canonical_entities": 500,
            "latest_ingested_at": "2026-03-16T00:42:00+00:00",
        },
        {
            "table_name": "team_external_context",
            "context_type": "standings_away",
            "provider_entities": 500,
            "canonical_entities": 500,
            "latest_ingested_at": "2026-03-16T00:42:00+00:00",
        },
        {
            "table_name": "match_external_context",
            "context_type": "pre_match_form",
            "provider_entities": 2000,
            "canonical_entities": 2000,
            "latest_ingested_at": "2026-03-16T00:42:00+00:00",
        },
    ]

    registry = _feature_eligibility_registry(table_surface, premium_coverage, sparse_diagnostics, external_context_surface)
    by_group = {row["feature_group"]: row for row in registry}

    assert by_group["anytime_player_availability_value_burden"]["readiness_status"] == "ready_now"
    assert by_group["scoreline_standings_strength"]["readiness_status"] == "ready_now"
    assert by_group["incident_lead_states"]["readiness_status"] == "diagnostics_only"
    assert by_group["ball_recoveries_surface"]["readiness_status"] == "ready_after_backfill"
    assert by_group["final_third_entries_surface"]["readiness_status"] == "ready_now"
    assert by_group["external_team_standings"]["readiness_status"] == "ready_now"
    assert by_group["external_match_pre_match_form"]["readiness_status"] == "ready_now"
    assert by_group["pre_match_form"]["readiness_status"] == "missing_requires_new_table"
    assert by_group["win_probability"]["readiness_status"] == "monitoring_only"
