from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.db.db_utils import connect_db


TABLE_SPECS: dict[str, dict[str, str | None]] = {
    "fixture_stats_premium": {"timestamp_col": "ingested_at"},
    "fixture_incidents_sofascore": {"timestamp_col": "updated_at"},
    "fixture_incident_lead_states": {"timestamp_col": "updated_at"},
    "fixture_player_stats": {"timestamp_col": "created_at"},
    "player_availability": {"timestamp_col": "last_refreshed_at"},
    "fixture_formations": {"timestamp_col": "updated_at"},
    "fixture_odds_markets": {"timestamp_col": "snapshot_time_utc"},
    "team_league_standings": {"timestamp_col": "computed_at"},
    "team_external_context": {"timestamp_col": "ingested_at"},
    "match_external_context": {"timestamp_col": "ingested_at"},
    "player_external_context": {"timestamp_col": "ingested_at"},
}

PREMIUM_FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "classic_xg_surface": ("h_xg", "a_xg"),
    "classic_box_touches_surface": ("h_box_touches", "a_box_touches"),
    "classic_crosses_surface": ("h_crosses", "a_crosses"),
    "classic_sot_surface": ("h_sot", "a_sot"),
    "classic_corners_surface": ("h_corners", "a_corners"),
    "final_third_entries_surface": ("h_final_third_entries", "a_final_third_entries"),
    "long_balls_surface": (
        "h_accurate_long_balls",
        "a_accurate_long_balls",
        "h_total_long_balls",
        "a_total_long_balls",
    ),
    "duels_surface": ("h_duels_won", "a_duels_won"),
    "ground_duels_surface": ("h_ground_duels_won", "a_ground_duels_won"),
    "aerial_duels_surface": ("h_aerial_duels_won", "a_aerial_duels_won"),
    "dribbles_surface": (
        "h_dribbles_success",
        "a_dribbles_success",
        "h_dribbles_total",
        "a_dribbles_total",
    ),
    "dispossessed_surface": ("h_dispossessed", "a_dispossessed"),
    "woodwork_surface": ("h_hit_woodwork", "a_hit_woodwork"),
    "offsides_surface": ("h_offsides", "a_offsides"),
    "fouls_surface": ("h_fouls", "a_fouls"),
    "ball_recoveries_surface": ("h_ball_recoveries", "a_ball_recoveries"),
    "yellow_cards_surface": ("h_yellow_cards", "a_yellow_cards"),
    "red_cards_surface": ("h_red_cards", "a_red_cards"),
}

SPARSE_PREMIUM_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "ball_recoveries_surface": ("Recovered balls", "Recovered Balls", "ballRecoveries", "ball_recoveries"),
    "yellow_cards_surface": ("Yellow cards", "Yellow Cards", "yellowCards", "yellow_cards"),
    "red_cards_surface": ("Red cards", "Red Cards", "redCards", "red_cards"),
}

MISSING_TABLE_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("pre_match_form", "missing_requires_new_table", ("anytime", "scoreline")),
    ("team_streaks", "missing_requires_new_table", ("anytime", "scoreline")),
    ("home_away_split_standings", "missing_requires_new_table", ("scoreline",)),
    ("performance_graph", "missing_requires_new_table", ("scoreline", "corners")),
    ("h2h_results", "missing_requires_new_table", ("scoreline",)),
    ("win_probability", "monitoring_only", ("scoreline", "anytime")),
    ("votes", "monitoring_only", ("scoreline", "anytime")),
    ("shotmap", "missing_requires_new_table", ("corners", "scoreline")),
    ("heatmap", "missing_requires_new_table", ("corners",)),
    ("transfer_snapshots", "missing_requires_new_table", ("scoreline", "anytime")),
)

EXTERNAL_CONTEXT_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "feature_group": "external_team_standings",
        "source_table": "team_external_context",
        "context_types": ("standings_total", "standings_home", "standings_away"),
        "family_applicability": ("scoreline", "corners"),
        "min_types_required": 3,
    },
    {
        "feature_group": "external_team_overview",
        "source_table": "team_external_context",
        "context_types": ("team_overview",),
        "family_applicability": ("scoreline", "anytime"),
        "min_types_required": 1,
    },
    {
        "feature_group": "external_team_league_stats",
        "source_table": "team_external_context",
        "context_types": ("league_stats",),
        "family_applicability": ("scoreline", "corners"),
        "min_types_required": 1,
    },
    {
        "feature_group": "external_team_performance_graph",
        "source_table": "team_external_context",
        "context_types": ("performance_graph",),
        "family_applicability": ("scoreline", "corners"),
        "min_types_required": 1,
    },
    {
        "feature_group": "external_match_pre_match_form",
        "source_table": "match_external_context",
        "context_types": ("pre_match_form",),
        "family_applicability": ("scoreline", "anytime"),
        "min_types_required": 1,
    },
    {
        "feature_group": "external_match_team_streaks",
        "source_table": "match_external_context",
        "context_types": ("team_streaks",),
        "family_applicability": ("scoreline", "anytime"),
        "min_types_required": 1,
    },
    {
        "feature_group": "external_match_h2h_results",
        "source_table": "match_external_context",
        "context_types": ("h2h_results",),
        "family_applicability": ("scoreline",),
        "min_types_required": 1,
    },
    {
        "feature_group": "external_player_overview",
        "source_table": "player_external_context",
        "context_types": ("player_overview",),
        "family_applicability": ("scoreline", "anytime"),
        "min_types_required": 1,
    },
    {
        "feature_group": "external_player_attributes",
        "source_table": "player_external_context",
        "context_types": ("attributes",),
        "family_applicability": ("scoreline", "anytime"),
        "min_types_required": 1,
    },
    {
        "feature_group": "external_player_league_stats",
        "source_table": "player_external_context",
        "context_types": ("league_stats",),
        "family_applicability": ("scoreline", "anytime"),
        "min_types_required": 1,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit live SofaScore-derived DB surface for model reuse.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to artifacts/reports/db_reuse_surface/<timestamp>/",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=250,
        help="Max fixture_stats_premium raw_json rows to inspect for sparse field diagnosis.",
    )
    return parser.parse_args()


def _json_default(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()  # type: ignore[no-any-return]
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _field_group_coverage(conn, field_group: str, columns: tuple[str, ...]) -> dict[str, Any]:
    conditions = " AND ".join(f"{col} IS NOT NULL" for col in columns)
    query = f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(*) FILTER (WHERE {conditions}) AS covered_rows,
            MAX(ingested_at) AS latest_timestamp
        FROM fixture_stats_premium
    """
    with conn.cursor() as cur:
        cur.execute(query)
        total_rows, covered_rows, latest_timestamp = cur.fetchone()
    coverage_ratio = float(covered_rows) / float(total_rows) if total_rows else 0.0
    return {
        "field_group": field_group,
        "source_table": "fixture_stats_premium",
        "columns": list(columns),
        "total_rows": int(total_rows or 0),
        "covered_rows": int(covered_rows or 0),
        "coverage_ratio": coverage_ratio,
        "latest_timestamp": latest_timestamp,
    }


def _table_surface_report(conn) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for table_name, spec in TABLE_SPECS.items():
        timestamp_col = spec.get("timestamp_col")
        query = f"SELECT COUNT(*) AS row_count{', MAX(' + timestamp_col + ') AS latest_timestamp' if timestamp_col else ''} FROM {table_name}"
        with conn.cursor() as cur:
            cur.execute(query)
            row = cur.fetchone()
        row_count = int(row[0] or 0)
        latest_timestamp = row[1] if timestamp_col else None
        report.append(
            {
                "table_name": table_name,
                "row_count": row_count,
                "latest_timestamp": latest_timestamp,
            }
        )
    return report


def _external_context_surface_report(conn) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    specs = (
        ("team_external_context", "team_id", "provider_team_id"),
        ("match_external_context", "fixture_id", "provider_fixture_id"),
        ("player_external_context", "player_id", "provider_player_id"),
    )
    with conn.cursor() as cur:
        for table_name, canonical_col, provider_col in specs:
            cur.execute(
                f"""
                SELECT
                    context_type,
                    COUNT(*) AS rows_total,
                    COUNT(DISTINCT {provider_col}) AS provider_entities,
                    COUNT(DISTINCT {canonical_col}) FILTER (WHERE {canonical_col} IS NOT NULL) AS canonical_entities,
                    COUNT(*) FILTER (WHERE {canonical_col} IS NULL) AS null_canonical_rows,
                    MAX(snapshot_time_utc) AS latest_snapshot_time,
                    MAX(ingested_at) AS latest_ingested_at
                FROM {table_name}
                GROUP BY context_type
                ORDER BY context_type
                """
            )
            cols = [d[0] for d in cur.description]
            for row in cur.fetchall():
                payload = dict(zip(cols, row))
                payload["table_name"] = table_name
                reports.append(payload)
    return reports


def _recursive_key_hits(value: object, aliases: set[str]) -> int:
    hits = 0
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in aliases:
                hits += 1
            if (
                isinstance(child, str)
                and child in aliases
                and str(key).lower() in {"name", "key", "label", "metric"}
            ):
                hits += 1
            hits += _recursive_key_hits(child, aliases)
        return hits
    if isinstance(value, list):
        for child in value:
            hits += _recursive_key_hits(child, aliases)
    return hits


def _sparse_field_diagnosis(conn, *, sample_limit: int) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    query = """
        SELECT fixture_id, raw_json
        FROM fixture_stats_premium
        WHERE raw_json IS NOT NULL
        ORDER BY ingested_at DESC NULLS LAST, fixture_id DESC
        LIMIT %s
    """
    raw_rows = pd.read_sql(query, conn, params=(int(sample_limit),))
    parsed_payloads: list[dict[str, Any]] = []
    for raw_json in raw_rows.get("raw_json", pd.Series(dtype=object)).tolist():
        if isinstance(raw_json, dict):
            parsed_payloads.append(raw_json)
            continue
        try:
            payload = json.loads(raw_json)
            if isinstance(payload, dict):
                parsed_payloads.append(payload)
        except Exception:
            continue
    for field_group, aliases in SPARSE_PREMIUM_KEY_ALIASES.items():
        coverage = _field_group_coverage(conn, field_group, PREMIUM_FIELD_GROUPS[field_group])
        alias_set = set(aliases)
        payload_hits = sum(1 for payload in parsed_payloads if _recursive_key_hits(payload, alias_set) > 0)
        if coverage["covered_rows"] > 0:
            diagnosis = "already_populated"
        elif payload_hits > 0:
            diagnosis = "parser_gap_likely"
        else:
            diagnosis = "payload_absent_likely"
        diagnostics.append(
            {
                **coverage,
                "sample_rows_checked": int(len(parsed_payloads)),
                "payload_key_hits": int(payload_hits),
                "diagnosis": diagnosis,
            }
        )
    return diagnostics


def _feature_eligibility_registry(
    table_surface: list[dict[str, Any]],
    premium_coverage: list[dict[str, Any]],
    sparse_diagnostics: list[dict[str, Any]],
    external_context_surface: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest_by_table = {row["table_name"]: row.get("latest_timestamp") for row in table_surface}
    sparse_by_group = {row["field_group"]: row for row in sparse_diagnostics}
    external_by_table_type = {
        (str(row["table_name"]), str(row["context_type"])): row for row in external_context_surface
    }
    registry: list[dict[str, Any]] = []

    registry.extend(
        [
            {
                "feature_group": "anytime_player_availability_value_burden",
                "source_table": "player_availability+players",
                "family_applicability": ["anytime", "scoreline"],
                "readiness_status": "ready_now",
                "coverage_ratio": None,
                "latest_timestamp": latest_by_table.get("player_availability"),
                "notes": "Live availability table is populated and players table has market_value_euro.",
            },
            {
                "feature_group": "anytime_attack_concentration",
                "source_table": "player_availability+fixture_player_stats",
                "family_applicability": ["anytime"],
                "readiness_status": "ready_now",
                "coverage_ratio": None,
                "latest_timestamp": latest_by_table.get("fixture_player_stats"),
                "notes": "Historical player stats are populated enough to compute prior attacking burden from listed players.",
            },
            {
                "feature_group": "scoreline_standings_strength",
                "source_table": "team_league_standings",
                "family_applicability": ["scoreline"],
                "readiness_status": "ready_now",
                "coverage_ratio": None,
                "latest_timestamp": latest_by_table.get("team_league_standings"),
                "notes": "Simple league standings table already exists and is fresh.",
            },
            {
                "feature_group": "incident_lead_states",
                "source_table": "fixture_incident_lead_states",
                "family_applicability": ["anytime", "scoreline", "corners"],
                "readiness_status": "diagnostics_only",
                "coverage_ratio": None,
                "latest_timestamp": latest_by_table.get("fixture_incident_lead_states"),
                "notes": "Useful for targets and diagnostics, not prematch features.",
            },
        ]
    )

    for row in premium_coverage:
        field_group = str(row["field_group"])
        sparse = sparse_by_group.get(field_group)
        if sparse:
            status = "ready_after_backfill" if sparse["diagnosis"] == "parser_gap_likely" else "missing_requires_new_table"
            if sparse["diagnosis"] == "already_populated":
                status = "ready_now"
            notes = f"Sparse field diagnosis={sparse['diagnosis']} sample_hits={sparse['payload_key_hits']}."
        else:
            status = "ready_now" if float(row["coverage_ratio"]) > 0.2 else "ready_after_backfill"
            notes = "Premium fixture stat surface from live DB."
        registry.append(
            {
                "feature_group": field_group,
                "source_table": "fixture_stats_premium",
                "family_applicability": ["corners", "scoreline"],
                "readiness_status": status,
                "coverage_ratio": row["coverage_ratio"],
                "latest_timestamp": row["latest_timestamp"],
                "notes": notes,
            }
        )

    for spec in EXTERNAL_CONTEXT_GROUPS:
        source_table = str(spec["source_table"])
        context_types = tuple(str(x) for x in spec["context_types"])
        found = [external_by_table_type.get((source_table, context_type)) for context_type in context_types]
        found_nonnull = [row for row in found if row is not None]
        min_types_required = int(spec["min_types_required"])
        if len(found_nonnull) >= min_types_required:
            provider_entities = min(int(row["provider_entities"] or 0) for row in found_nonnull)
            canonical_entities = min(int(row["canonical_entities"] or 0) for row in found_nonnull)
            readiness_status = "ready_now" if canonical_entities > 0 else "ready_after_backfill"
            latest_candidates = [row["latest_ingested_at"] for row in found_nonnull if row["latest_ingested_at"] is not None]
            latest_timestamp = max(latest_candidates) if latest_candidates else None
            notes = (
                f"External context types present: {', '.join(context_types)}; "
                f"min canonical entities across required types={canonical_entities}; "
                f"min provider entities across required types={provider_entities}."
            )
            coverage_ratio = None
        else:
            readiness_status = "missing_requires_new_table"
            latest_timestamp = None
            notes = (
                f"Missing required external context types. Found "
                f"{[row['context_type'] for row in found_nonnull]} but require {list(context_types)}."
            )
            coverage_ratio = 0.0
        registry.append(
            {
                "feature_group": spec["feature_group"],
                "source_table": source_table,
                "family_applicability": list(spec["family_applicability"]),
                "readiness_status": readiness_status,
                "coverage_ratio": coverage_ratio,
                "latest_timestamp": latest_timestamp,
                "notes": notes,
            }
        )

    for field_group, readiness_status, families in MISSING_TABLE_GROUPS:
        registry.append(
            {
                "feature_group": field_group,
                "source_table": None,
                "family_applicability": list(families),
                "readiness_status": readiness_status,
                "coverage_ratio": 0.0,
                "latest_timestamp": None,
                "notes": "No dedicated live DB table found during schema audit.",
            }
        )
    return registry


def _output_dir_from_args(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return ROOT_DIR / "artifacts" / "reports" / "db_reuse_surface" / timestamp


def main() -> None:
    args = parse_args()
    out_dir = _output_dir_from_args(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = connect_db()
    try:
        table_surface = _table_surface_report(conn)
        premium_coverage = [
            _field_group_coverage(conn, field_group, columns)
            for field_group, columns in PREMIUM_FIELD_GROUPS.items()
        ]
        sparse_diagnostics = _sparse_field_diagnosis(conn, sample_limit=int(args.sample_limit))
        external_context_surface = _external_context_surface_report(conn)
    finally:
        conn.close()

    registry = _feature_eligibility_registry(
        table_surface,
        premium_coverage,
        sparse_diagnostics,
        external_context_surface,
    )
    audit_report = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "table_surface": table_surface,
        "premium_field_coverage": premium_coverage,
        "sparse_field_diagnostics": sparse_diagnostics,
        "external_context_surface": external_context_surface,
    }

    (out_dir / "db_audit_report.json").write_text(
        json.dumps(audit_report, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (out_dir / "feature_eligibility_registry.json").write_text(
        json.dumps(registry, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (out_dir / "sparse_premium_field_diagnosis.json").write_text(
        json.dumps(sparse_diagnostics, indent=2, default=_json_default),
        encoding="utf-8",
    )

    print(f"Wrote DB reuse audit to {out_dir}")


if __name__ == "__main__":
    main()
